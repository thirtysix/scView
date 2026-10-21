"""The cluster report (C064): the evidence scView has for one group of one column.

Sections, in order: identity (size, per donor and condition), labels (what the
other annotations call these cells, label confidence), markers (the shared marker
choice and the marker-database verdict), enrichment (ORA against HALLMARK), QC
(medians vs the rest, the cluster QC flags, doublets, QC states, cells the
filtering removed), composition (the cluster's share per condition and per donor)
and methods. ``core/report.py`` has the section shape and the not-computed rule.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from scview.core import report as R

logger = logging.getLogger(__name__)

N_MARKERS = 20
ORA_N = 100  # query genes: the enrichment endpoints' default
ORA_COLLECTION = "h.all"
N_TERMS = 10
TOP_LABELS = 3
MAX_LABEL_COLUMNS = 8
MAX_LABEL_VALUES = 100
MIN_SAMPLE_CELLS = 10  # a donor sample with fewer cells gives a share that is mostly noise


def build(adaptor, column: str, value: str, *, dataset_name: str | None = None,
          msigdb_loader=None, marker_db_dir=None) -> dict:
    """The report for group ``value`` of ``column`` (call in a worker thread)."""
    t0 = time.perf_counter()
    adata = adaptor.adata
    obs = adata.obs
    lab = R.grouping(obs, column)
    value = str(value)
    mask = (lab == value).to_numpy()
    if not mask.any():
        raise R.ReportError(f"'{value}' is not a group of '{column}'.", 404)
    condition, donor = R.design(obs, column)
    ctx = {"adaptor": adaptor, "adata": adata, "column": column, "value": value, "mask": mask,
           "labels": lab, "condition": condition, "donor": donor}

    sections = [
        R.guarded("identity", "Identity", lambda: _identity(ctx)),
        R.guarded("labels", "Labels", lambda: _labels(ctx)),
        R.guarded("markers", "Marker genes", lambda: _markers(ctx, marker_db_dir)),
        R.guarded("enrichment", "Pathway enrichment", lambda: _enrichment(ctx, msigdb_loader)),
        R.guarded("qc", "Quality control", lambda: _qc(ctx)),
        R.guarded("composition", "Composition", lambda: _composition(ctx)),
    ]
    sections.append(R.guarded("methods", "Methods",
                              lambda: R.methods_section(adaptor, R.sections_methods(sections))))
    head = R.meta(adaptor, dataset_name)
    return {
        "kind": "cluster",
        "title": f"{head['dataset']['name']} - cluster {value} report",
        "column": column,
        "value": value,
        "n_cells": int(mask.sum()),
        "design": {"condition": condition, "donor": donor},
        **head,
        "sections": sections,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }


# ---------------------------------------------------------------------------
# 1. Identity
# ---------------------------------------------------------------------------


def _identity(ctx: dict) -> dict:
    obs, mask, column, value = ctx["adata"].obs, ctx["mask"], ctx["column"], ctx["value"]
    condition, donor = ctx["condition"], ctx["donor"]
    n, total = int(mask.sum()), int(mask.size)
    data: dict = {"n_cells": n, "n_total": total, "fraction": R.num(n / total),
                  "n_groups": len(R.group_order(obs[column])), "condition": condition, "donor": donor,
                  "by_condition": None, "by_donor": None, "n_donors": None, "conditions": None}
    cond = R.labels(obs[condition])[mask] if condition else None
    if condition:
        order = R.group_order(obs[condition])
        counts = cond.value_counts()
        data["conditions"] = order
        data["by_condition"] = [{"value": c, "n_cells": int(counts.get(c, 0)),
                                 "fraction": R.num(counts.get(c, 0) / n)} for c in order]
    if donor:
        d = R.labels(obs[donor])[mask]
        frame = pd.DataFrame({"d": d.to_numpy(), "c": cond.to_numpy() if cond is not None else ""})
        frame = frame[frame["d"].notna()]
        donors = [x for x in R.group_order(obs[donor]) if x in set(frame["d"])]
        per = frame.groupby("d", observed=True).size()
        by_cond = (frame.groupby(["d", "c"], observed=True).size() if condition else None)
        rows = []
        for x in donors:
            row = {"donor": x, "n_cells": int(per.get(x, 0)), "fraction": R.num(per.get(x, 0) / n)}
            if by_cond is not None:
                row["by_condition"] = {c: int(by_cond.get((x, c), 0)) for c in data["conditions"]}
            rows.append(row)
        data["n_donors"] = len(rows)
        data["by_donor"] = rows[: R.MAX_ROWS]
        data["by_donor_total"] = len(rows)
    caption = f"{value}: {n:,} of {total:,} cells ({R.pct(n / total)}) in '{column}'"
    if donor:
        caption += f", from {R.plural(data['n_donors'], 'donor')}"
    if condition:
        present = sum(1 for r in data["by_condition"] if r["n_cells"])
        caption += f" in {present} of {len(data['conditions'])} '{condition}' groups"
    caption += "."
    found = []
    if donor:
        found.append(f"the donor column '{donor}'")
    if condition:
        found.append(f"the condition column '{condition}'")
    method = (f"Cells counted from '{column}'"
              + (f"; {' and '.join(found)} found by name as the composition test finds them "
                 "(a donor-like name with 2+ donors per condition, a condition-like name with 2 to 8 "
                 "values)." if found else "; no donor or condition column was found by name."))
    return R.section("identity", "Identity", caption=caption, method=method, data=data)


# ---------------------------------------------------------------------------
# 2. Labels
# ---------------------------------------------------------------------------

def _is_label(series: pd.Series) -> bool:
    return (isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object
            or pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_integer_dtype(series.dtype))


def label_columns(adaptor, column: str, skip: set[str]) -> list[tuple[str, str]]:
    """Other columns that name the cells, with what made them: author cell types,
    CellTypist / LLM / marker-score annotations, saved consensus columns, hand
    labels, other label-like columns, then clusterings; at most
    ``MAX_LABEL_COLUMNS``."""
    from scview.core import annotation_confidence, annotation_crosscheck
    from scview.core.anndata_adaptor import CLUSTERING_CANDIDATES
    from scview.core.sample_pseudobulk import _CELLTYPE_RE, default_celltype_column

    adata = adaptor.adata
    obs = adata.obs
    found: dict[str, str] = {}

    def add(c, kind: str) -> None:
        c = str(c)
        if c in found or c == column or c in skip or c not in obs.columns:
            return
        if annotation_confidence.is_output(c) and not c.endswith("_unknown"):
            return  # flags and alternatives, not labels
        s = obs[c]
        if _is_label(s) and 2 <= s.nunique(dropna=True) <= MAX_LABEL_VALUES:
            found[c] = kind

    typed = default_celltype_column(obs)
    if typed:
        add(typed, "author cell types")
    active = getattr(adaptor, "active_clustering_column", lambda: None)()
    for c, method in annotation_crosscheck.annotation_columns(adata, None).items():
        add(c, method)
    for entry in annotation_crosscheck.registry(adata).values():
        if isinstance(entry, dict):
            add((entry.get("columns") or {}).get("consensus", ""), "consensus")
    users = getattr(adaptor, "user_annotation_columns", lambda: [])()
    for c in users:
        add(c, "hand labels")
    for c in obs.columns:
        if _CELLTYPE_RE.search(str(c)) and not str(c).startswith("scview_conf_"):
            add(c, "labels")
    if active:
        add(active, "clustering")
    for c in CLUSTERING_CANDIDATES:
        add(c, "clustering")
    for c in obs.columns:
        if str(c).startswith(("scview_leiden", "scview_louvain")):
            add(c, "clustering")
    return list(found.items())[:MAX_LABEL_COLUMNS]


def _labels(ctx: dict) -> dict:
    from scview.core import annotation_confidence as conf

    adaptor, adata, mask = ctx["adaptor"], ctx["adata"], ctx["mask"]
    column, value = ctx["column"], ctx["value"]
    obs = adata.obs
    n = int(mask.sum())
    skip = {c for c in (ctx["donor"], ctx["condition"]) if c}
    cols = []
    for c, kind in label_columns(adaptor, column, skip):
        lab = R.labels(obs[c])[mask]
        counts = lab.value_counts()
        top = [{"label": str(k), "n_cells": int(v), "fraction": R.num(v / n)}
               for k, v in counts.head(TOP_LABELS).items()]
        cols.append({"column": c, "kind": kind, "top": top, "n_labels": int(counts.size),
                     "n_unlabelled": int(n - counts.sum())})

    checks = []
    for col, entry in conf.registry(adata).items():
        if not isinstance(entry, dict):
            continue
        flag = (entry.get("columns") or {}).get("flag")
        if not flag or flag not in obs.columns or col not in obs.columns:
            continue
        f = obs[flag].astype(str).to_numpy()[mask]
        counts = {k: int((f == k).sum()) for k in (*conf.FLAGS, conf.NOT_CHECKED)}
        summary = conf.summary(adata, col) or {}
        own = None
        if col == column:
            own = next((r for r in entry.get("labels", []) if str(r.get("label")) == value), None)
        checks.append({"column": col, "alpha": entry.get("alpha"), "classifier": entry.get("classifier"),
                       "counts": counts, "n_cells": n, "stale": bool(summary.get("stale")),
                       "own_label": own})
    confidence = ({"status": R.OK, "checks": checks, "note": conf.NOTE} if checks else
                  {"status": R.NOT_COMPUTED, "checks": [],
                   "note": f"Not computed yet: check how well labels fit in {R.WHERE['confidence']}."})
    data = {"columns": cols, "confidence": confidence}
    if not cols:
        return R.section(
            "labels", "Labels", status=R.NOT_COMPUTED, data=data,
            caption=f"No other label column names the {n:,} cells of {value}.",
            note=(f"Nothing to compare with yet: run CellTypist or the LLM annotator in "
                  f"{R.WHERE['annotation']}."),
            method="Label columns are found by name and by the annotation runs scView recorded.")
    return R.section(
        "labels", "Labels", data=data,
        caption=(f"How {R.plural(len(cols), 'other label column')} "
                 f"{'names' if len(cols) == 1 else 'name'} the {n:,} cells of {value}: "
                 f"the top {TOP_LABELS} labels {'' if len(cols) == 1 else 'of each '}as a share of the cells."),
        method=("Cross-tabulation of the cells' labels; columns found by name and from the recorded "
                "CellTypist, LLM, marker-score, consensus and hand-label runs"
                + ("; label confidence from the stored split-conformal check." if checks else ".")))


# ---------------------------------------------------------------------------
# 3. Markers (+ the marker databases)
# ---------------------------------------------------------------------------


def _markers(ctx: dict, marker_db_dir) -> dict:
    from scview.core.marker_selection import group_markers

    adata, column, value, mask = ctx["adata"], ctx["column"], ctx["value"], ctx["mask"]
    key, why = R.stored_markers(adata, column)
    if key is None:
        return R.section("markers", "Marker genes", status=R.NOT_COMPUTED, note=why,
                         caption=f"No marker genes are stored for '{column}'.",
                         data={"database": {"status": R.NOT_COMPUTED,
                                            "note": "The database check needs the cluster's marker genes."}})
    passing = group_markers(adata, column, n=10**6, key=key).get(value, [])
    top = passing[:N_MARKERS]
    stats = R.marker_stats(adata, column, key, value, top)
    params = R.marker_params(adata, key)
    rows = [stats[g] for g in top]
    n = int(mask.sum())
    filt = R.marker_filter_text()
    if top:
        caption = (f"Top {len(top)} of {len(passing):,} marker genes of {value} ({n:,} cells) against the "
                   f"rest of '{column}' ({int(mask.size - n):,} cells), {filt}; by log fold change.")
    else:
        caption = f"No gene of {value} ({n:,} cells) passes the marker filters ({filt})."
    data = {"genes": rows, "n_passing": len(passing), "n_shown": len(top), "test": params,
            "database": _database(adata, column, value, key, marker_db_dir)}
    return R.section(
        "markers", "Marker genes", caption=caption, data=data,
        method=(f"Stored {params['method']} test of each group against the rest (scanpy "
                f"rank_genes_groups, {params['corr_method']} adjusted p), filtered and ordered by "
                "scView's shared marker choice (the one the co-pilot, the database check and "
                "enrichment use)."))


def _database(adata, column: str, value: str, key: str, marker_db_dir) -> dict:
    from scview.core import marker_db
    from scview.core.marker_selection import group_markers

    db, why = R.marker_db_status(marker_db_dir)
    if db is None:
        return {"status": R.NOT_COMPUTED, "note": why}
    if len(R.group_order(adata.obs[column])) < 2:
        return {"status": R.NOT_APPLICABLE, "note": "The database check compares groups; this column has one."}
    sp, sp_note = marker_db._species(adata, None)
    verdict_from, suggest_from = db.view("verdict", sp), db.view("suggest", sp)
    sdb = db.species(sp, verdict_from)
    if not sdb.types:
        return {"status": R.NOT_APPLICABLE, "note": f"The marker databases have no {sp} rows."}
    measured = marker_db.measured_genes(adata)
    sc = marker_db.Scorer(sdb, measured)
    sc_suggest = sc if suggest_from == verdict_from else marker_db.Scorer(db.species(sp, suggest_from), measured)
    markers = group_markers(adata, column, n=marker_db.N_MARKERS, key=key).get(value, [])
    row = marker_db._label_row(sc, sc_suggest, value, markers)
    return {
        "status": R.OK,
        "verdict": row["verdict"], "verdict_text": row["verdict_text"],
        "db_type": row["db_type"], "cl_id": row["cl_id"], "match": row["match"]["how_text"],
        "p": R.p_value(row["p"]) if row["p"] is not None else None,
        "overlap": row["overlap"], "n_db_genes": row["n_db_genes"], "n_markers": len(markers),
        "suggestions": [{"cell_type": s["cell_type"], "p": R.p_value(s["p"]), "overlap": s["overlap"],
                         "sources": s["sources"]} for s in row["suggestions"]],
        "species": sp, "species_note": sp_note,
        "verdict_sources": db.labels(verdict_from), "suggestion_sources": db.labels(suggest_from),
        "note": marker_db.NOTE,
    }


# ---------------------------------------------------------------------------
# 4. Enrichment
# ---------------------------------------------------------------------------


def _enrichment(ctx: dict, loader) -> dict:
    from scview.core.marker_selection import ora_query_genes
    from scview.core.ora_background import get_background_genes
    from scview.core.species import detect_species, human_genesets_warning

    adata, column, value = ctx["adata"], ctx["column"], ctx["value"]
    title = "Pathway enrichment"
    key, why = R.stored_markers(adata, column)
    if key is None:
        return R.section("enrichment", title, status=R.NOT_COMPUTED,
                         caption="Enrichment tests the cluster's marker genes, and none are stored.",
                         note=why)
    if loader is None or not loader.get_collection_as_dict(ORA_COLLECTION):
        return R.section("enrichment", title, status=R.NOT_COMPUTED,
                         caption="The HALLMARK gene sets are not available.",
                         note=("MSigDB is not on disk: scView downloads it into the msigdb volume on "
                               "first start (MSIGDB_DIR must point at it)."))
    query = ora_query_genes(adata, column, value, ORA_N)
    background = get_background_genes(adata, column, value) or []
    warning = human_genesets_warning(detect_species(adata)) or None
    method = (f"Over-representation (gseapy hypergeometric test, Benjamini-Hochberg adjusted p) of up to "
              f"{ORA_N} of the cluster's marker genes ({R.marker_filter_text()}, by log fold change) "
              f"against the MSigDB HALLMARK sets, with the {len(background):,} genes the marker test "
              "ranked as the background, as the Enrichment tab does.")
    if not query:
        return R.section("enrichment", title, method=method, note=warning,
                         caption=f"No gene of {value} passes the marker filters, so there is nothing to test.",
                         data={"terms": [], "n_query": 0, "n_background": len(background)})
    from scview.api.v1.enrichment import _ora_records

    records = _ora_records(query, background or None, [ORA_COLLECTION], loader, None)
    n_sets = len(loader.get_collection_as_dict(ORA_COLLECTION))
    n_sig = sum(1 for r in records if r.get("adjusted_pvalue", 1.0) < 0.05)
    top = [{"term": r["term"], "p": R.p_value(r["pvalue"]), "padj": R.p_value(r["adjusted_pvalue"]),
            "overlap": int(r["overlap_count"]), "set_size": int(r["gene_count"]), "genes": r["genes"]}
           for r in records[:N_TERMS]]
    caption = (f"{R.plural(n_sig, 'HALLMARK set')} of {n_sets} at adjusted p < 0.05 for {len(query)} marker "
               f"genes of {value} against {len(background):,} background genes"
               + (f"; the top {len(top)} of the {len(records)} sets that share a gene with them, by adjusted p."
                  if records else "; no set shares a gene with them."))
    return R.section("enrichment", title, caption=caption, method=method, note=warning,
                     data={"terms": top, "n_sets": n_sets, "n_tested": len(records), "n_significant": n_sig,
                           "n_query": len(query), "n_background": len(background),
                           "collection": "MSigDB HALLMARK"})


# ---------------------------------------------------------------------------
# 5. QC
# ---------------------------------------------------------------------------


def _median(values, m: np.ndarray) -> float | None:
    if values is None:
        return None
    a = np.asarray(values, dtype=np.float64)[m]
    a = a[np.isfinite(a)]
    return R.num(np.median(a), 2) if a.size else None


def _qc(ctx: dict) -> dict:
    from scview.core import cluster_qc
    from scview.core.anndata_adaptor import per_cell_qc

    adaptor, adata, mask, column, value = (ctx["adaptor"], ctx["adata"], ctx["mask"],
                                           ctx["column"], ctx["value"])
    obs = adata.obs
    fn = getattr(adaptor, "qc_per_cell", None)
    cells = fn(ribo=True) if callable(fn) else per_cell_qc(adata, ribo=True)
    rest = ~mask
    has_mt = bool(cells.get("has_mt", True)) and cells.get("pct_mt") is not None
    ribo = cells.get("pct_ribo")
    medians = [
        {"metric": "n_genes", "label": "Genes per cell", "cluster": _median(cells["n_genes"], mask),
         "rest": _median(cells["n_genes"], rest), "note": None},
        {"metric": "total_counts", "label": "UMIs per cell", "cluster": _median(cells["total_counts"], mask),
         "rest": _median(cells["total_counts"], rest), "note": None},
        {"metric": "pct_mt", "label": "% mitochondrial",
         "cluster": _median(cells["pct_mt"], mask) if has_mt else None,
         "rest": _median(cells["pct_mt"], rest) if has_mt else None,
         "note": None if has_mt else "unmeasurable: no mitochondrial genes"},
        {"metric": "pct_ribo", "label": "% ribosomal", "cluster": _median(ribo, mask),
         "rest": _median(ribo, rest),
         "note": None if ribo is not None else "not measured: no counts or no ribosomal genes"},
    ]

    try:
        rep = cluster_qc.for_adaptor(adaptor, column)
        row = next((c for c in rep.clusters if c.group == value), None)
        flags = {"status": R.OK if row is not None else R.NOT_APPLICABLE,
                 "flags": [{"flag": f, "reason": (row.reasons[i] if i < len(row.reasons) else ""),
                            "help": cluster_qc.FLAG_HELP.get(f, "")} for i, f in enumerate(row.flags)]
                 if row else [],
                 "notes": list(row.notes) if row else [], "n_groups": rep.n_groups,
                 "note": rep.note or None}
    except ValueError as exc:
        flags = {"status": R.NOT_APPLICABLE, "flags": [], "notes": [], "note": str(exc)}

    calls = cluster_qc.doublet_calls(obs)
    score = (pd.to_numeric(obs["doublet_score"], errors="coerce").to_numpy(dtype=float)
             if "doublet_score" in obs.columns else None)
    if calls is None and score is None:
        doublets = {"status": R.NOT_COMPUTED,
                    "note": f"Not computed yet: run doublet detection in {R.WHERE['doublets']}."}
    else:
        doublets = {"status": R.OK,
                    "n_doublets": int(calls[mask].sum()) if calls is not None else None,
                    "fraction": R.num(calls[mask].mean()) if calls is not None else None,
                    "fraction_rest": R.num(calls[rest].mean()) if calls is not None and rest.any() else None,
                    "median_score": _median(score, mask), "median_score_rest": _median(score, rest)}

    states = _qc_states(adata, mask)
    removed = _removed(adata, column, value, int(mask.sum()))
    n = int(mask.sum())
    flag_names = [f["flag"] for f in flags["flags"]]
    caption = (f"Median QC values of the {n:,} cells of {value} against the other {int(rest.sum()):,} cells"
               + (f"; flagged {', '.join(flag_names)}." if flag_names else
                  "; no cluster QC flag." if flags["status"] == R.OK else "."))
    method = ("Medians of the per-cell QC values (from obs, else counted from the counts matrix); flags "
              "from the cluster QC table (each cluster against the median cluster, MAD-scaled, with "
              "absolute floors)" + (", which compares the groups of this column." if flags["status"] == R.OK
                                    else "."))
    return R.section("qc", "Quality control", caption=caption, method=method,
                     data={"medians": medians, "has_mt": has_mt, "flags": flags, "doublets": doublets,
                           "states": states, "removed": removed})


def _qc_states(adata, mask: np.ndarray) -> dict:
    from scview.core import qc_states

    cols = [c for c in qc_states.written_columns(adata) if c in adata.obs.columns]
    if not cols and qc_states.DEFAULT_NAME in adata.obs.columns:
        cols = [qc_states.DEFAULT_NAME]
    if not cols:
        return {"status": R.NOT_COMPUTED,
                "note": "No QC state column: label the states in Data Assessment > QC states."}
    col = cols[-1]
    n = int(mask.sum())
    counts = R.labels(adata.obs[col])[mask].value_counts()
    return {"status": R.OK, "column": col,
            "states": [{"state": str(k), "n_cells": int(v), "fraction": R.num(v / n)} for k, v in counts.items()]}


def _removed(adata, column: str, value: str, n_kept: int) -> dict:
    """What the last filtering step removed of this group, from its record."""
    from scview.core import qc_filter

    rec = adata.uns.get(qc_filter.RECORD_KEY)
    if not isinstance(rec, dict):
        return {"status": R.NOT_APPLICABLE,
                "note": ("No filtering record: the cells were not filtered in scView since it began "
                         "keeping one (v0.28.0), or not filtered at all.")}
    try:
        if int(rec["n_kept"]) != int(adata.n_obs):
            return {"status": R.NOT_APPLICABLE,
                    "note": (f"The filtering record kept {int(rec['n_kept']):,} cells but the dataset has "
                             f"{int(adata.n_obs):,} now: cells changed since, so it no longer adds up.")}
        entries = list((rec.get("columns") or {}).values())
        entry = next((e for e in entries if str(e["name"]) == column), None)
        if entry is None:
            kept = ", ".join(f"'{e['name']}'" for e in entries) or "none"
            return {"status": R.NOT_APPLICABLE,
                    "note": f"The filtering record does not keep '{column}' (it keeps {kept})."}
        cats = [str(x) for x in np.atleast_1d(np.asarray(entry["categories"], dtype=object))]
        codes = np.asarray(entry["codes"], dtype=np.int64)
        if value not in cats:
            sel = np.zeros(codes.size, dtype=bool)
        else:
            sel = codes == cats.index(value)
        bits = np.asarray(rec.get("criterion_bits"), dtype=np.uint8)
        criteria = [str(c) for c in np.atleast_1d(np.asarray(rec.get("criteria"), dtype=object))]
        by = {c: int((((bits >> b) & 1) > 0)[sel].sum()) for b, c in enumerate(criteria)} if bits.size else {}
        k = int(sel.sum())
        return {"status": R.OK, "n_removed": k, "n_kept": n_kept,
                "fraction_lost": R.num(k / (k + n_kept)) if k + n_kept else None,
                "by_criterion": {c: v for c, v in by.items() if v},
                "truncated": bool(rec.get("truncated", False)), "mode": str(rec.get("mode") or ""),
                "timestamp": str(rec.get("timestamp") or "")}
    except Exception as exc:  # a malformed record only loses this part
        logger.warning("QC removal record unreadable: %s", exc)
        return {"status": R.NOT_APPLICABLE, "note": "The filtering record is unreadable."}


# ---------------------------------------------------------------------------
# 6. Composition
# ---------------------------------------------------------------------------


def _composition(ctx: dict) -> dict:
    from scview.core import composition, paired_design

    adaptor, adata, mask = ctx["adaptor"], ctx["adata"], ctx["mask"]
    column, value, condition, donor = ctx["column"], ctx["value"], ctx["condition"], ctx["donor"]
    obs = adata.obs
    title = "Composition"
    if condition is None and donor is None:
        return R.section("composition", title, status=R.NOT_APPLICABLE,
                         caption=f"{value}'s share of the cells cannot be split here.",
                         note="No condition or donor column was found, so there is nothing to split the share by.")
    cond = R.labels(obs[condition]) if condition else pd.Series("", index=obs.index)
    don = R.labels(obs[donor]) if donor else None
    conditions = R.group_order(obs[condition]) if condition else []
    by_condition = None
    if condition:
        by_condition = []
        for c in conditions:
            in_c = (cond == c).to_numpy()
            total = int(in_c.sum())
            k = int((in_c & mask).sum())
            by_condition.append({"condition": c, "n_cells": k, "n_total": total,
                                 "share": R.num(k / total) if total else None})
    samples = None
    paired = None
    if donor:
        frame = pd.DataFrame({"d": don.to_numpy(), "c": cond.to_numpy(), "m": mask})
        frame = frame[frame["d"].notna() & frame["c"].notna()]
        agg = frame.groupby(["d", "c"], observed=True)["m"].agg(["size", "sum"])
        dpos = {x: i for i, x in enumerate(R.group_order(obs[donor]))}
        cpos = {x: i for i, x in enumerate(conditions)}
        samples = sorted(
            ({"donor": d, "condition": c, "n_total": int(r["size"]), "n_cells": int(r["sum"]),
              "share": R.num(r["sum"] / r["size"]) if r["size"] else None,
              "few_cells": bool(r["size"] < MIN_SAMPLE_CELLS)}
             for (d, c), r in agg.iterrows()),
            key=lambda s: (cpos.get(s["condition"], 0), dpos.get(s["donor"], 0)))
        if condition:
            design = paired_design.detect_design({(s["donor"], s["condition"]) for s in samples}, conditions)
            n_donors = len({s["donor"] for s in samples})
            paired = {**design, "n_donors": n_donors,
                      "every_donor": bool(design["paired"] and design["n_pairs"] == n_donors)}
    stored = None
    r = composition._last_result(adaptor, obs)
    if r is not None and r.group_key == column:
        g = next((x for x in r.groups if x.group == value), None)
        if g is not None:
            stored = {"condition_key": r.condition_key, "sample_key": r.sample_key,
                      "conditions": r.conditions, "tested": r.tested, "test": r.test,
                      "design": r.design, "props": {k: R.num(v) for k, v in g.props.items()},
                      "fold_change": R.num(g.fold_change), "p": R.p_value(g.pval), "fdr": R.p_value(g.fdr),
                      "n_samples": r.n_samples}
    n_samples = len(samples) if samples else 0
    parts = []
    if condition:
        parts.append(f"per '{condition}' group ({len(conditions)})")
    if donor:
        parts.append(f"per donor sample ({n_samples}, one point each)")
    caption = f"Share of all cells that are {value}, " + " and ".join(parts) + "."
    note = None
    if paired and paired["every_donor"]:
        note = (f"Every donor was sampled in each '{condition}' group (paired design, "
                f"{paired['n_pairs']} donors): compare each donor's points, not the pooled shares.")
    elif paired and paired["paired"]:
        note = f"{paired['n_pairs']} donors were sampled in every '{condition}' group (partly paired)."
    if stored is None:
        tail = (f"No composition test is stored for '{column}': {R.WHERE['composition']} tests the shares "
                "with the donors as replicates.")
        note = f"{note} {tail}" if note else tail
    method = ("Proportions only: the cluster's cells over all cells of each condition or donor sample; no "
              "test is made here" + ("; the stored composition test (propeller-style, donors as "
                                     "replicates) is cited as it was run." if stored else "."))
    return R.section("composition", title, caption=caption, method=method, note=note,
                     data={"condition": condition, "donor": donor, "conditions": conditions,
                           "by_condition": by_condition, "samples": samples, "paired": paired,
                           "stored_test": stored, "min_sample_cells": MIN_SAMPLE_CELLS})
