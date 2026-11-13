"""Each cluster's evidence in plain words: what scView's rules read, and what they decided.

A per-cluster probability model (type / low quality / random markers / doublet) failed its
pre-registered checks against the hard rules it softened (2026-11-12, research/cluster-probabilities:
worse Brier, 7 of 12 visible doublets against 12 of 12), so the decisions stay rules (cluster QC
flags, ``pipeline.decide_clusters``) and this module only says why, never as a probability:

- markers vs random genes: the list's best marker-database p against the p that 1 in 10 random
  lists of the same number of database genes reach (``core/marker_randomness.py``, the annotator's
  own check, on the annotator's own lists);
- QC: median genes and % mito against the typical cluster (the median across clusters), with the
  cluster QC flags that fired;
- doublet-like: the two clusters ``cluster_qc.doublet_like_pair`` picked, the markers they share
  and the RNA against the larger of the two;
- typing, on author or curated groupings only (not cluster numbers): the marker databases' types
  as shares of the evidence, softmax(-log10 p / ``TYPE_TAU``) over the best ``TYPE_TOP`` types,
  summed up the Cell Ontology. On Leiden clusters the top share was over-confident (0.97 stated,
  0.80 on the truth's path); on author groups it held;
- the decision: what the latest AI annotation of the grouping recorded, else what it would decide.

Reads stored results only (markers, cluster QC, the databases and ontology on disk: never
downloads, never runs markers), cached per loaded dataset object.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from scview.core import cluster_qc, marker_randomness, marker_selection
from scview.core.object_cache import ObjectCache

logger = logging.getLogger(__name__)

TAG = "result:evidence"
N_MARKERS = 25  # the annotator's list (pipeline._top_markers_per_group)
TYPE_TAU = 3.37  # fitted on author groups of 6 datasets (research/cluster-probabilities, LODO 3.06-3.37)
TYPE_TOP = 30
TYPE_MIN_SHARE = 0.15  # a deeper type is named from this share of the evidence
TYPE_SHOWN = 3
CALL_SHARE = 0.5
GENERIC_SHARE = 0.25  # CL terms above this share of all terms say nothing about cell type
_NUMBERS = re.compile(r"^\d+(?:[,._-]\d+)*$")

FLAG_WORDS = {
    "low_quality": "low quality (few genes and high mito)",
    "high_mito": "high mito",
    "low_genes": "few genes",
    "doublet_rich": "doublet-rich",
    "doublet_like": "doublet-like",
    "stress_driven": "stress-driven",
    "hb_ambient": "ambient haemoglobin",
    "cycling": "cycling",
    "low_malat1": "low MALAT1",
    "small": "very small",
    "nuclei_mito": "mito RNA in nuclei",
    "low_content": "low RNA for its identity",
    "noncoding_markers": "non-coding markers",
    "sex_markers": "sex genes lead its markers",
}
_DAMAGE = ("low_quality", "high_mito")


class Line(BaseModel):
    kind: Literal["markers", "qc", "doublet", "content", "makeup", "typing", "decision"]
    text: str
    fired: bool = False  # this is why a rule fired


class ClusterEvidence(BaseModel):
    group: str
    n_cells: int = 0
    label: str | None = None  # the label scView gave (or would give) itself, without the AI
    decided: bool = False  # True: recorded by an AI annotation run; False: what the rules would do
    flags: list[str] = []
    lines: list[Line] = []


class EvidenceReport(BaseModel):
    column: str
    n_groups: int = 0
    markers: bool = False  # the column has stored markers
    typing: bool = False  # database typing shown (author or curated grouping)
    annotation: str | None = None  # the output column of the recorded AI annotation, if any
    notes: list[str] = []
    clusters: list[ClusterEvidence] = []


# ---------------------------------------------------------------------------
# Words


def p_text(log10p: float) -> str:
    """-log10 p as the app writes small p values in evidence: 1e-3.4."""
    if log10p >= 300:
        return "below 1e-300"
    if log10p < 0.05:
        return "1"
    return f"1e-{log10p:.1f}"


def _n(v: float) -> str:
    return f"{v:,.0f}"


def _share(m: float) -> str:
    return f"{round(100 * m)}%"


def is_curated(labels: list[str]) -> bool:
    """Named groupings (an author's cell types, curated labels) get typing lines; cluster numbers
    ("3", a subcluster's "3,1") do not, whatever the column is called (Kang's cell types are 'cluster')."""
    return not all(_NUMBERS.match(str(v)) for v in labels)


def markers_line(group: str, check: marker_randomness.RandomCheck | None, has_list: bool) -> Line | None:
    if check is None:
        return None
    if not has_list:
        return Line(kind="markers", text="No stored markers pass the marker filters, so they cannot be compared.")
    sc = check.scores.get(group)
    if not sc:
        return Line(kind="markers", text="None of its markers are in the marker databases, so they cannot be "
                                         "compared with random genes.")
    best, kind, n = sc["best_log10p"], sc["best_type"], sc["named"]
    cut = sc.get("null_cut")
    if cut is None:
        return Line(kind="markers", text=(
            f"Only {n} of its markers are in the marker databases (best match {kind}, p {p_text(best)}): too few "
            f"to compare with random genes (that needs {marker_randomness.MIN_NAMED})."))
    if group in check.flagged:
        return Line(kind="markers", fired=True, text=(
            f"Markers match no database type better than random genes: best p {p_text(best)} ({kind}), while 1 in "
            f"10 random lists of {n} database genes reach {p_text(cut)}."))
    return Line(kind="markers", text=(
        f"Markers match {kind} better than random genes: best p {p_text(best)}, while 1 in 10 random lists of "
        f"{n} database genes reach {p_text(cut)}."))


def qc_line(row: cluster_qc.ClusterQC, typical: dict[str, float]) -> Line:
    parts = []
    if row.median_genes is not None and typical.get("median_genes"):
        parts.append(f"{_n(row.median_genes)} genes vs {_n(typical['median_genes'])} typical")
    if row.median_pct_mt is not None and typical.get("median_pct_mt") is not None:
        mito = f"{cluster_qc.mito_text(row.median_pct_mt)} vs {cluster_qc.mito_text(typical['median_pct_mt'])} mito"
        if typical["median_pct_mt"] < cluster_qc.NUCLEI_MITO_MEDIAN and row.mt_detected is not None \
                and typical.get("mt_detected") is not None:
            # Nuclei: the median cell has no mito read, so say how many cells have one
            mito += (f" (mito reads in {row.mt_detected:.0%} of its cells vs {typical['mt_detected']:.0%}, mean "
                     f"{cluster_qc.mito_text(row.mean_pct_mt or 0.0)} vs "
                     f"{cluster_qc.mito_text(typical.get('mean_pct_mt') or 0.0)})")
        parts.append(mito)
    if "doublet_rich" in row.flags and row.doublet_fraction is not None and typical.get("doublet_fraction") is not None:
        parts.append(f"{row.doublet_fraction:.0%} vs {typical['doublet_fraction']:.0%} predicted doublets")
    text = "; ".join(parts) or f"{row.n_cells:,} cells"
    if row.flags:
        text += ". Cluster QC flags it " + ", ".join(FLAG_WORDS.get(f, f.replace("_", " ")) for f in row.flags) + "."
    else:
        text += ". No cluster QC flag."
    # nuclei_mito is a caution read from these numbers: the line is why it fired
    return Line(kind="qc", text=text, fired=any(f in (*_DAMAGE, "nuclei_mito") for f in row.flags))


def doublet_line(row: cluster_qc.ClusterQC, rows: dict[str, cluster_qc.ClusterQC]) -> Line | None:
    if "doublet_like" not in row.flags or len(row.doublet_like_of) != 2:
        return None
    a, b = row.doublet_like_of
    ra, rb = rows.get(a), rows.get(b)
    shared = ""
    if len(row.doublet_like_shared) == 4:
        in_a, in_b, either, k = row.doublet_like_shared
        shared = f": {either} of its {k} top markers are theirs ({in_a} of {a}'s, {in_b} of {b}'s)"
    rna = ""
    if ra and rb and row.median_counts and ra.median_counts and rb.median_counts:
        big = ra if ra.median_counts >= rb.median_counts else rb
        rna = (f", and its cells carry {row.median_counts / big.median_counts:.1f}x the RNA of {big.group} "
               f"({_n(row.median_counts)} vs {_n(big.median_counts)} median counts)")
    return Line(kind="doublet", fired=True, text=f"Looks like a mix of {a} and {b}{shared}{rna}.")


def content_line(row: cluster_qc.ClusterQC, rows: dict[str, cluster_qc.ClusterQC]) -> Line | None:
    """``low_content``: the cluster whose identity it shares, and how much less RNA it carries."""
    o = rows.get(row.low_content_of or "")
    if "low_content" not in row.flags or o is None or len(row.low_content_shared) != 2:
        return None
    shared, k = row.low_content_shared
    if not (row.median_counts and o.median_counts and row.median_genes and o.median_genes):
        return None
    return Line(kind="content", fired=True, text=(
        f"Looks like {o.group} with less RNA: {shared} of its {k} top markers are {o.group}'s, but its cells carry "
        f"{row.median_counts / o.median_counts:.2f}x the counts ({_n(row.median_counts)} vs {_n(o.median_counts)} "
        f"median) and {row.median_genes / o.median_genes:.2f}x the genes. Damaged cells, nuclei losing RNA or "
        "ambient droplets split off by depth; its markers against "
        f"{o.group} are likely ambient genes of other types."))


def makeup_line(row: cluster_qc.ClusterQC) -> Line | None:
    """``sex_markers`` / ``noncoding_markers``: what leads the marker list, in words."""
    parts = []
    if "sex_markers" in row.flags and row.sex_markers:
        parts.append(f"Sex genes lead its markers ({', '.join(row.sex_markers)}): it may be one sex's cells "
                     "(or one donor's) of a type split by sex, not a type of its own.")
    if "noncoding_markers" in row.flags and row.noncoding_markers:
        parts.append(f"Its top markers are mostly non-coding or unnamed ({', '.join(row.noncoding_markers[:6])}): "
                     "little to name a cell type by, typical of a technical cluster.")
    return Line(kind="makeup", fired=True, text=" ".join(parts)) if parts else None


# ---------------------------------------------------------------------------
# Typing: database types as shares of the evidence, up the Cell Ontology


class _Hierarchy:
    """Cell Ontology ancestors-or-self without the generic terms."""

    def __init__(self, index):
        self.index = index
        self.terms = index.terms
        self._anc: dict[str, frozenset] = {}
        counts: dict[str, int] = {}
        for cid in self.terms:
            for a in self.up(cid):
                counts[a] = counts.get(a, 0) + 1
        self.generic = frozenset(a for a, n in counts.items() if n > GENERIC_SHARE * len(self.terms))

    def up(self, cid: str) -> frozenset:
        hit = self._anc.get(cid)
        if hit is None:
            seen, stack = {cid}, [cid]
            while stack:
                for p in self.terms[stack.pop()].parents:
                    if p in self.terms and p not in seen:
                        seen.add(p)
                        stack.append(p)
            hit = self._anc[cid] = frozenset(seen)
        return hit

    def anc(self, cid: str | None) -> frozenset:
        return self.up(cid) - self.generic if cid in self.terms else frozenset()


_HIER: dict[int, _Hierarchy] = {}


def _hierarchy(index) -> _Hierarchy | None:
    if index is None:
        return None
    h = _HIER.get(id(index))
    if h is None or h.index is not index:
        h = _HIER[id(index)] = _Hierarchy(index)
    return h


def type_candidates(scorer, genes: list[str], index=None) -> list[dict]:
    """The ``TYPE_TOP`` database types by hypergeometric score (full sets, as the marker-DB
    annotation), each placed on the Cell Ontology by its own id, else a strict name match."""
    from scview.core import cell_ontology as co

    Q = scorer.query(genes)
    if not Q:
        return []
    s, k, _ = scorer.scores(Q, scorer.open_names, full=True)
    out = []
    for i in np.argsort(-s, kind="stable")[:TYPE_TOP]:
        if k[i] <= 0 or s[i] <= 0:
            continue
        name = scorer.open_names[i]
        cl = None
        if index is not None:
            cl = co.clean_id(scorer.sdb.types[name].get("cl") or "")
            if cl not in index.terms:
                cl = co.match_label(name, index)[0]
        out.append({"cell_type": name, "cl_id": cl, "score": float(s[i])})
    return out


def type_shares(cands: list[dict], hier: _Hierarchy | None) -> tuple[list[tuple[str, float]], bool]:
    """([(name, share)], nested): the deepest Cell Ontology term with at least ``CALL_SHARE`` of
    the evidence and the database types below it with at least ``TYPE_MIN_SHARE``, general to
    specific (nested True); else the best types by share (nested False)."""
    if not cands:
        return [], False
    s = np.array([c["score"] for c in cands]) / TYPE_TAU
    p = np.exp(s - s.max())
    p /= p.sum()
    if hier is not None:
        mass: dict[str, float] = {}
        for c, pi in zip(cands, p):
            for a in hier.anc(c["cl_id"]):
                mass[a] = mass.get(a, 0.0) + float(pi)
        ok = [(len(hier.anc(t)), m, t) for t, m in mass.items() if m >= CALL_SHARE]
        if ok:
            _, m, call = max(ok)
            below = {c["cl_id"] for c in cands if c["cl_id"] in hier.terms and c["cl_id"] != call
                     and call in hier.up(c["cl_id"]) and mass.get(c["cl_id"], 0.0) >= TYPE_MIN_SHARE}
            deeper = sorted(below, key=lambda t: (-mass[t], -len(hier.anc(t))))[: TYPE_SHOWN - 1]
            return [(hier.index.name(t) or t, mass[t]) for t in [call, *deeper]], True
    by: dict[str, float] = {}
    for c, pi in zip(cands, p):
        by[c["cell_type"]] = by.get(c["cell_type"], 0.0) + float(pi)
    top = sorted(by.items(), key=lambda kv: -kv[1])[:TYPE_SHOWN]
    return [(n, m) for n, m in top if m >= TYPE_MIN_SHARE] or top[:1], False


def typing_line(shares: list[tuple[str, float]], nested: bool) -> Line | None:
    if not shares:
        return None
    (n0, m0), rest = shares[0], shares[1:]
    head = f"{n0} ({_share(m0)} of the evidence)"
    more = ", ".join(f"{n} ({_share(m)})" for n, m in rest)
    if nested:  # the term holding most of the evidence, then the types below it
        return Line(kind="typing", text=f"Marker databases: {head}" + (f" > {more}." if more else "."))
    return Line(kind="typing", text="Marker databases, split: " + ", ".join([head, *([more] if more else [])]) + ".")


# ---------------------------------------------------------------------------
# The decision


def recorded_annotation(adata, column: str) -> dict | None:
    """The latest AI annotation step on ``column``: {target, mapping, record} (None: none, or the
    grouping's latest annotation was another method)."""
    from scview.core import provenance

    active = adata.uns.get("scview_active_clustering")
    for h in reversed(provenance.read_provenance(adata).get("history", [])):
        if h.get("step") != "cell_type_annotation":
            continue
        p = h.get("params") or {}
        if (p.get("annotation_groupby") or active) != column:
            continue
        target = p.get("annotation_target") or "cell_type"
        mapping = adata.uns.get(f"{target}_llm_mapping")
        if not hasattr(mapping, "keys"):
            return None
        return {"target": target, "mapping": {str(k): str(v) for k, v in dict(mapping).items()}, "record": p}
    return None


def _ids(v) -> list[str]:
    return [str(x) for x in v] if isinstance(v, list | tuple | np.ndarray) else []


def decision_line(group: str, rec: dict | None, rules: dict[str, list[str]],
                  check: marker_randomness.RandomCheck | None) -> tuple[Line | None, str | None, bool]:
    """(line, scView's own label, recorded): from the annotation's record when there is one,
    else what ``pipeline.decide_clusters`` would decide now."""
    from scview.core.pipeline import LOW_QUALITY_LABEL, UNIDENTIFIED_LABEL

    best = (check.best.get(group) if check else None) or "a type"
    src = rec["record"] if rec else rules
    low, unid, to_ai = (_ids(src.get(k)) for k in ("llm_decided_low_quality", "llm_random_markers",
                                                    "llm_high_mito_type_to_ai"))
    label = LOW_QUALITY_LABEL if group in low else UNIDENTIFIED_LABEL if group in unid else None
    if rec:
        where = f"'{rec['target']}'"
        if label:
            return Line(kind="decision", fired=True, text=(
                f"scView labelled this {label} itself in {where}, without asking the AI.")), label, True
        named = rec["mapping"].get(group)
        if group in to_ai:
            return Line(kind="decision", text=(
                f"High mito, but its markers best match {best}, a cell type with much mitochondrial RNA by nature, "
                f"so the AI was asked{f' and named it {named}' if named else ''} in {where}.")), None, True
        if named:
            return Line(kind="decision", text=f"The AI named it {named} in {where}."), None, True
        return None, None, False
    if label:
        return Line(kind="decision", fired=True, text=(
            f"AI annotation would label this {label} itself, without asking the AI.")), label, False
    if group in to_ai:
        return Line(kind="decision", text=(
            f"High mito, but its markers best match {best}, a cell type with much mitochondrial RNA by nature: "
            "AI annotation would still ask the AI.")), None, False
    return None, None, False


# ---------------------------------------------------------------------------
# All together


def build(adata, column: str, report: cluster_qc.ClusterQCReport, db=None, index=None) -> EvidenceReport:
    """Every cluster's evidence lines (call in a worker thread: reads the dataset, changes nothing).
    ``db`` / ``index``: the marker databases and the Cell Ontology on disk, or None."""
    from scview.core import marker_db as mdb
    from scview.core.pipeline import decide_clusters

    out = EvidenceReport(column=column, n_groups=report.n_groups)
    markers = marker_selection.group_markers(adata, column, n=N_MARKERS)
    out.markers = bool(markers)
    check = None
    if not markers:
        out.notes.append(f"No marker genes for '{column}': compute them (Unified View > Markers) to compare "
                         "each cluster's markers with the marker databases and with random genes.")
    elif db is None:
        out.notes.append("The marker databases are not downloaded yet, so markers are not compared with them "
                         "(they download on first use of the marker database check or annotation).")
    else:
        try:
            check = marker_randomness.check(adata, markers, db)
            if check.skipped:
                out.notes.append(f"Markers not compared with the databases: {check.skipped}.")
                check = None
        except Exception as exc:  # noqa: BLE001 - best effort, as in the annotator
            logger.info("Evidence: random-marker check skipped: %s", exc)
            check = None
    rules = decide_clusters(check or marker_randomness.RandomCheck(), report, markers)[1] if markers else {}
    rec = recorded_annotation(adata, column)
    out.annotation = rec["target"] if rec else None

    groups = [r.group for r in report.clusters]
    scorer = hier = None
    if check is not None and is_curated(groups):
        try:
            sp, _ = mdb._species(adata, None)
            scorer = mdb.Scorer(db.species(sp, db.view("annotate", sp)), mdb.measured_genes(adata))
            hier = _hierarchy(index)
            out.typing = True
        except Exception as exc:  # noqa: BLE001
            logger.info("Evidence: typing skipped: %s", exc)
    elif check is not None:
        out.notes.append("Database types are shown as shares of the evidence on named groupings only: on "
                         "clusters they read more certain than they are.")

    rows = {r.group: r for r in report.clusters}
    for r in report.clusters:
        g = r.group
        ev = ClusterEvidence(group=g, n_cells=r.n_cells, flags=list(r.flags))
        if (m := markers_line(g, check, bool(markers.get(g)))) is not None:
            ev.lines.append(m)
        ev.lines.append(qc_line(r, report.typical))
        if (d := doublet_line(r, rows)) is not None:
            ev.lines.append(d)
        for extra in (content_line(r, rows), makeup_line(r)):
            if extra is not None:
                ev.lines.append(extra)
        if scorer is not None and markers.get(g):
            shares, nested = type_shares(type_candidates(scorer, markers[g], index), hier)
            if (t := typing_line(shares, nested)) is not None:
                ev.lines.append(t)
        line, label, recorded = decision_line(g, rec, rules, check)
        if line is not None:
            ev.lines.append(line)
            ev.label, ev.decided = label, recorded
        out.clusters.append(ev)
    return out


_CACHE = ObjectCache()


def _key(adata, column: str, db, index) -> tuple:
    from scview.core import provenance
    from scview.core.ora_background import resolve_markers_key

    codes = pd.factorize(adata.obs[column].astype(str))[0]
    labels = hashlib.sha1(np.ascontiguousarray(codes).tobytes()).hexdigest()
    mkey = resolve_markers_key(adata.uns, column)
    return ("evidence", column, labels, mkey, id(adata.uns.get(mkey)) if mkey else 0,
            id(adata.uns.get(provenance.UNS_KEY)), tuple(map(str, adata.obs.columns)), id(db), id(index))


def _on_disk():
    """(marker databases, Cell Ontology index) from disk, never fetched; None for each missing."""
    from scview.core import cell_ontology
    from scview.core import marker_db as mdb

    try:
        db = mdb.load(mdb.marker_db_dir())
    except Exception as exc:  # noqa: BLE001
        logger.info("Evidence: marker databases unreadable: %s", exc)
        db = None
    try:
        index = cell_ontology.loaded() if db is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.info("Evidence: Cell Ontology unreadable: %s", exc)
        index = None
    return db, index


def for_adaptor(adaptor, column: str | None = None) -> EvidenceReport:
    """``build`` for a loaded dataset, answered from memory while nothing it reads has changed.
    Raises ``ValueError`` for a column cluster QC cannot group by."""
    adata = adaptor.adata
    col = column or cluster_qc.default_grouping(adata)
    if not col:
        raise ValueError("No clustering yet: run clustering, or pick a grouping column.")
    if col not in adata.obs.columns:
        raise ValueError(f"Column '{col}' not found in obs.")
    db, index = _on_disk()
    key = _key(adata, col, db, index)
    hit = _CACHE.get(adata, key)
    if hit is None:
        hit = build(adata, col, cluster_qc.for_adaptor(adaptor, col), db, index)
        _CACHE.put(adata, hit, key)
    return hit


def cached(adaptor, column: str | None = None) -> EvidenceReport | None:
    """The cached report for ``column`` (default grouping), or None: never computes (the co-pilot
    builds its facts on the event loop; ``api/v1/assistant`` warms this in a thread)."""
    adata = adaptor.adata
    col = column or cluster_qc.default_grouping(adata)
    if not col or col not in adata.obs.columns:
        return None
    db, index = _on_disk()
    return _CACHE.get(adata, _key(adata, col, db, index))


def evidence_tag(column: str, group: str) -> str:
    return f"{TAG}:{column}:{group}"


def grounding_block(report: EvidenceReport, max_clusters: int = 20) -> tuple[list[str], list[tuple[str, str]]]:
    """Prompt lines + ``(tag, detail)`` citations: every cluster with a decision, a flag or a line
    that fired, then one line for the rest."""
    if not report.clusters:
        return [], []
    listed = [c for c in report.clusters if c.label or c.flags or any(ln.fired for ln in c.lines)]
    head = f"{TAG}:{report.column}"
    lines = [f"\n## Why each cluster was labelled or flagged (evidence for scView's rules, no probabilities; "
             f"column '{report.column}') [{head}]"]
    cites: list[tuple[str, str]] = [(head, f"{report.column}: evidence for {len(report.clusters)} clusters")]
    for c in listed[:max_clusters]:
        tag = evidence_tag(report.column, c.group)
        lines.append(f"- **{c.group}** [{tag}]: " + " ".join(ln.text for ln in c.lines))
        cites.append((tag, f"{c.group}: " + (c.label or ", ".join(c.flags) or "evidence")))
    if len(listed) > max_clusters:
        lines.append(f"- {len(listed) - max_clusters} more clusters have a flag or a decided label (Data Assessment "
                     f"> Cluster QC) [{head}]")
    if rest := len(report.clusters) - len(listed):
        lines.append(f"- The other {rest} clusters: no cluster QC flag and no label decided by scView [{head}]")
    for n in report.notes:
        lines.append(f"- Note: {n}")
    return lines, cites
