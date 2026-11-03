"""Each sample's sex from its own expression: XIST (female) against Y-chromosome
genes (male), with the evidence behind every call.

Why it matters: a sample whose expression says the other sex than its metadata
is a likely sample swap or mislabel; a sample with both signals in many cells
pools donors of both sexes (or carries cross-donor doublets); and when sex
lines up with the condition, sex-linked genes top every comparison and a sex
difference reads as a condition effect.

Per sample (the donor / sample column ``qc_groups.sample_grouping`` picks, as
the per-sample QC table uses), from the raw counts:

- the share of cells with any female-marker count (XIST, TSIX) and with any
  male-marker count (RPS4Y1, DDX3Y, KDM5D, UTY, EIF1AY, ZFY, USP9Y; mouse Xist /
  Tsix against Ddx3y, Eif2s3y, Kdm5d, Uty), and the share with both;
- each marker set's pseudobulk CPM (its summed counts per million of the
  sample's counts).

The call reads the shares of cells: a marker set is "on" in at least
``ON_SHARE`` of the cells and "off" in at most ``OFF_SHARE`` (ambient RNA from
other donors in a pooled run gives a male sample a little XIST). Female = female
markers on and male off; male the reverse; mixed = both on; ambiguous = in
between, or neither on. With one marker set missing (genes filtered out, as in
some Kang releases with no Y genes) the other alone decides, at low confidence,
and with both missing the sample is undetermined. Fewer than ``MIN_CELLS`` cells
are too few to call.

Species from ``core/species.py`` picks the gene list (both when unknown).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import qc_groups
from scview.core.confounding import condition_columns
from scview.core.species import detect_species, match_genes

FEMALE_GENES = {"human": ("XIST", "TSIX"), "mouse": ("Xist", "Tsix")}
MALE_GENES = {
    "human": ("RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY", "ZFY", "USP9Y"),
    "mouse": ("Ddx3y", "Eif2s3y", "Kdm5d", "Uty"),
}

ON_SHARE = 0.10  # a marker set is "on" in at least this share of a sample's cells
OFF_SHARE = 0.02  # and "off" at or below this share (ambient RNA stays under it)
MIN_CELLS = 10  # as pseudobulk and the per-donor plots
# A per-sample marker share this spread-out across samples (max/min) is not "one level".
UNIFORM_RATIO = 1.5
# A sample with Y genes in this share of its cells sets the dataset's male Y-gene level.
MALE_LIKE_SHARE = 0.5
# A marker set's CPM under this share of the other sex's samples' level is ambient RNA.
AMBIENT_SHARE = 0.1

CALLS = ("female", "male", "mixed", "ambiguous", "undetermined")

# Metadata columns that say the donor's sex: sex, Sex, donor_sex, gender,
# sex_ontology_term_id (cellxgene), reported_sex ...
SEX_COLUMN_RE = re.compile(r"(^|[^a-z])(sex|gender)([^a-z]|$)", re.I)
_FEMALE_WORDS = {"f", "female", "woman", "w", "fem", "xx", "pato:0000383"}
_MALE_WORDS = {"m", "male", "man", "xy", "pato:0000384"}


def normalise_sex(value) -> str | None:
    """'female' / 'male' for the usual spellings of a sex label, else None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    v = str(value).strip().lower()
    if v in _FEMALE_WORDS:
        return "female"
    if v in _MALE_WORDS:
        return "male"
    return None


def sex_metadata_column(obs: pd.DataFrame) -> str | None:
    """The first obs column named like sex/gender whose values read as female/male."""
    for col in obs.columns:
        if not SEX_COLUMN_RE.search(str(col)):
            continue
        values = obs[col].dropna().astype(str).unique()[:50]
        if any(normalise_sex(v) for v in values):
            return str(col)
    return None


def _genes_for(species: str) -> tuple[list[str], list[str]]:
    if species in FEMALE_GENES:
        return list(FEMALE_GENES[species]), list(MALE_GENES[species])
    # Unknown: both lists; matching ignores case, so XIST/Xist count once.
    return ([*FEMALE_GENES["human"], *FEMALE_GENES["mouse"]],
            [*MALE_GENES["human"], *MALE_GENES["mouse"]])


def _matrix(adata) -> tuple[object, pd.Index, str, bool]:
    """(cells x genes matrix, gene names, where from, holds counts). Raw counts
    when there are any; else the log values (.raw, then X), which still say
    whether a gene was detected in a cell, but give no CPM."""
    from scview.core import large_layout
    from scview.core.pseudobulk import PseudobulkError, raw_counts

    if large_layout.is_large(adata):
        # Matrices on disk: the few genes from the gene-major copy of the log values
        # (the counts are stored cell by cell, so a gene would need every cell read).
        return adata.X, pd.Index(adata.var_names), "X (log values)", False
    try:
        M, names, source = raw_counts(adata)
        return M, pd.Index(names), source, True
    except PseudobulkError:
        pass
    if adata.raw is not None:
        return adata.raw.X, pd.Index(adata.raw.var_names), ".raw (log values)", False
    return adata.X, pd.Index(adata.var_names), "X (normalised values)", False


def _columns(M, idx: Sequence[int]) -> np.ndarray:
    """Dense cells x len(idx) block of ``M``."""
    if not idx:
        return np.zeros((M.shape[0], 0))
    block = M[:, list(idx)]
    block = block.toarray() if sp.issparse(block) else np.asarray(block)
    return np.asarray(block, dtype=np.float64)


def _totals(M) -> np.ndarray:
    return np.asarray(M.sum(axis=1), dtype=np.float64).ravel()


def _share(x: float) -> str:
    return f"{x:.0%}" if x >= 0.1 or x == 0 else f"{x:.1%}"


def call_sex(n_cells: int, female: float | None, male: float | None) -> tuple[str, str]:
    """(call, confidence) from the shares of cells with female / male markers
    (None = that marker set is not in the data)."""
    if n_cells < MIN_CELLS:
        return "undetermined", "none"
    if female is None and male is None:
        return "undetermined", "none"
    f_on = female is not None and female >= ON_SHARE
    m_on = male is not None and male >= ON_SHARE
    f_off = female is not None and female <= OFF_SHARE
    m_off = male is not None and male <= OFF_SHARE
    both = female is not None and male is not None
    if f_on and m_on:
        return "mixed", "high" if min(female, male) >= 2 * ON_SHARE else "low"
    if f_on and (m_off or male is None):
        return "female", "high" if both else "low"
    if m_on and (f_off or female is None):
        return "male", "high" if both else "low"
    if f_off and male is None:
        return "male", "low"  # XIST absent, Y genes not measured
    if m_off and female is None:
        return "female", "low"  # Y genes absent, XIST not measured
    return "ambiguous", "low"


def ambient_recall(rows: list[dict]) -> None:
    """Re-call a mixed or ambiguous sample whose second marker set is at an ambient level.

    In pooled lanes (Liu 2021: 5' v1, donors pooled and demultiplexed) a female sample's
    cells pick up Y-gene reads from the male donors' ambient RNA: Y genes in 5-14% of its
    cells, so a cell-share rule calls it mixed or ambiguous, though at 12-36 CPM against the
    male samples' 676-863. Measured against the dataset's own samples (those with Y genes in
    half their cells or more for the male level; the others with XIST on for the female
    level), a level under ``AMBIENT_SHARE`` of it is ambient: the call is the other sex.
    Needs counts (CPM) and samples of the other sex to compare with; changes ``rows``."""
    def level(values: list[float]) -> float | None:
        return float(np.median(values)) if values else None

    sized = [r for r in rows if r["n_cells"] >= MIN_CELLS]
    ref_m = level([r["cpm_male"] for r in sized if r["cpm_male"] is not None
                   and (r["frac_male"] or 0) >= MALE_LIKE_SHARE])
    ref_f = level([r["cpm_female"] for r in sized if r["cpm_female"] is not None
                   and (r["frac_male"] or 0) < MALE_LIKE_SHARE and (r["frac_female"] or 0) >= ON_SHARE])
    for r in sized:
        if r["call"] not in ("mixed", "ambiguous"):
            continue
        f, m = r["frac_female"] or 0, r["frac_male"] or 0
        if ref_m and r["cpm_male"] is not None and f >= ON_SHARE and r["cpm_male"] < AMBIENT_SHARE * ref_m:
            r["call"], r["confidence"] = "female", "high" if f >= 2 * ON_SHARE else "low"
            r["evidence"] = r["evidence"][:-1] + (
                f". The Y genes' {r['cpm_male']:,.0f} CPM is {r['cpm_male'] / ref_m:.0%} of the male "
                f"samples' {ref_m:,.0f}: ambient RNA from male donors in the same run, not a male donor.")
        elif ref_f and r["cpm_female"] is not None and m >= ON_SHARE and r["cpm_female"] < AMBIENT_SHARE * ref_f:
            r["call"], r["confidence"] = "male", "high" if m >= 2 * ON_SHARE else "low"
            r["evidence"] = r["evidence"][:-1] + (
                f". XIST's {r['cpm_female']:,.0f} CPM is {r['cpm_female'] / ref_f:.0%} of the female "
                f"samples' {ref_f:,.0f}: ambient RNA from female donors in the same run, not a female donor.")


def _evidence(row: dict, f_genes: list[str], m_genes: list[str]) -> str:
    bits = []
    if f_genes:
        cpm = f", {row['cpm_female']:,.0f} CPM" if row["cpm_female"] is not None else ""
        bits.append(f"{'/'.join(f_genes)} in {_share(row['frac_female'])} of cells{cpm}")
    else:
        bits.append("no XIST in the data")
    if m_genes:
        cpm = f", {row['cpm_male']:,.0f} CPM" if row["cpm_male"] is not None else ""
        bits.append(f"Y genes in {_share(row['frac_male'])}{cpm}")
    else:
        bits.append("no Y-chromosome genes in the data")
    if f_genes and m_genes and row["frac_both"] > 0:
        bits.append(f"both in {_share(row['frac_both'])}")
    return "; ".join(bits) + f" ({row['n_cells']:,} cells)."


def _metadata_per_group(obs: pd.DataFrame, column: str, codes: np.ndarray,
                        k: int) -> list[dict]:
    """Each group's metadata sex: its main value and that value's share."""
    labels = obs[column].map(normalise_sex).to_numpy()
    out = []
    for i in range(k):
        vals = labels[codes == i]
        vals = vals[pd.notna(vals)]
        if vals.size == 0:
            out.append({"value": None, "share": None})
            continue
        uniq, n = np.unique(vals.astype(str), return_counts=True)
        top = int(np.argmax(n))
        share = float(n[top] / vals.size)
        out.append({"value": str(uniq[top]) if share >= 0.9 else "mixed", "share": round(share, 3)})
    return out


def _sex_of_rows(rows: list[dict]) -> dict[str, str]:
    """group -> female/male: the expression call, else the metadata value."""
    out = {}
    for r in rows:
        if r["call"] in ("female", "male"):
            out[r["name"]] = r["call"]
        elif r.get("metadata") in ("female", "male"):
            out[r["name"]] = r["metadata"]
    return out


def sex_confounding(obs: pd.DataFrame, column: str, codes: np.ndarray, names: list[str],
                    sexes: dict[str, str]) -> list[dict]:
    """For each condition column: the sexes of the samples in each of its levels
    (a sample counts in a level with at least MIN_CELLS of its cells there).

    ``confounded``: each level holds samples of one sex and the levels differ, so
    a condition difference is also a sex difference. ``imbalanced``: some level
    (2+ samples) holds one sex while others hold both."""
    out = []
    if len(set(sexes.values())) < 2:
        return out  # one sex in the data (or none known): nothing to confound
    for cond in condition_columns(obs):
        if cond == column:
            continue
        lab = obs[cond].astype(str).where(obs[cond].notna()).to_numpy()
        per_level: dict[str, dict[str, int]] = {}
        order = qc_groups.group_codes(obs[cond])[0]
        for lvl in order:
            in_lvl = lab == lvl
            counts = np.bincount(codes[in_lvl & (codes >= 0)], minlength=len(names))
            tally = {"female": 0, "male": 0}
            for i, name in enumerate(names):
                if counts[i] >= MIN_CELLS and name in sexes:
                    tally[sexes[name]] += 1
            if tally["female"] + tally["male"]:
                per_level[lvl] = tally
        if len(per_level) < 2:
            continue
        single = {lvl: ("female" if t["male"] == 0 else "male")
                  for lvl, t in per_level.items() if (t["female"] == 0) != (t["male"] == 0)}
        desc = "; ".join(f"{lvl}: {t['female']} female, {t['male']} male"
                         for lvl, t in per_level.items())
        if len(single) == len(per_level) and len(set(single.values())) > 1:
            which = ", ".join(f"{lvl} {s}" for lvl, s in single.items())
            out.append({
                "condition": cond, "kind": "confounded", "levels": per_level,
                "message": (f"Sex is confounded with '{cond}' ({which}): every '{cond}' "
                            "difference is also a sex difference, so sex-linked genes (XIST, "
                            "Y genes) will top any comparison and no analysis can separate "
                            f"the two. Samples by '{column}': {desc}."),
            })
            continue
        lopsided = [lvl for lvl, s in single.items()
                    if per_level[lvl]["female"] + per_level[lvl]["male"] >= 2]
        if lopsided and len(single) < len(per_level):
            which = ", ".join(f"'{lvl}' only {single[lvl]}" for lvl in lopsided)
            out.append({
                "condition": cond, "kind": "imbalanced", "levels": per_level,
                "message": (f"Sex is unbalanced across '{cond}' ({which}, while other groups "
                            "have both sexes): sex-linked genes can show up in comparisons "
                            f"between them; consider sex as a covariate. Samples by '{column}': "
                            f"{desc}."),
            })
    return out


# The last expression-based calls per (dataset, sample column): sample -> female / male.
# Process memory: a DE between groups reads them when there is no sex metadata column,
# and never runs the expression check itself.
_calls: dict[tuple[str, str], dict[str, str]] = {}


def remember(dataset_id: str, result: dict) -> None:
    """Keep a sex check's per-sample calls for later comparisons (``sample_sexes``)."""
    if result.get("column"):
        _calls[(dataset_id, str(result["column"]))] = _sex_of_rows(result.get("groups") or [])


def sample_sexes(obs: pd.DataFrame, sample_key: str,
                 dataset_id: str | None = None) -> tuple[dict[str, str], str] | None:
    """Each sample's sex (female / male) and where it came from: the sex metadata column
    (a sample's value when 90%+ of its cells agree), else the last sex check of this
    dataset by ``sample_key``; None when neither exists."""
    meta = sex_metadata_column(obs)
    if meta is not None:
        names, codes = qc_groups.group_codes(obs[sample_key])
        per = _metadata_per_group(obs, meta, codes, len(names))
        sexes = {n: p["value"] for n, p in zip(names, per) if p["value"] in ("female", "male")}
        return sexes, f"metadata column '{meta}'"
    stored = _calls.get((dataset_id or "", sample_key))
    if stored:
        return dict(stored), "the sex check (from expression)"
    return None


def compare_sides(obs: pd.DataFrame, sample_key: str, mask_a: np.ndarray, mask_b: np.ndarray,
                  label_a: str, label_b: str, sexes: dict[str, str], source: str) -> dict | None:
    """Whether the two sides of a comparison differ in their samples' sexes (a sample
    counts on a side with at least MIN_CELLS of its cells there), as ``sex_confounding``
    judges the levels of a condition: ``confounded`` (each side one sex, not the same)
    or ``imbalanced`` (a side of 2+ samples all one sex, the other side both). None
    when they do not, or when a side has no sample of known sex."""
    samples = obs[sample_key].astype(str).where(obs[sample_key].notna()).to_numpy()
    tallies = []
    for mask in (mask_a, mask_b):
        names, counts = np.unique(samples[mask & pd.notna(samples)], return_counts=True)
        t = {"female": 0, "male": 0}
        for name, n in zip(names, counts):
            if n >= MIN_CELLS and str(name) in sexes:
                t[sexes[str(name)]] += 1
        tallies.append(t)
    if not all(t["female"] + t["male"] for t in tallies):
        return None
    single = [("female" if t["male"] == 0 else "male") if (t["female"] == 0) != (t["male"] == 0) else None
              for t in tallies]
    desc = "; ".join(f"{lab}: {t['female']} female, {t['male']} male"
                     for lab, t in zip((label_a, label_b), tallies))
    tail = f" (samples by '{sample_key}', sex from {source})"
    if single[0] and single[1] and single[0] != single[1]:
        kind = "confounded"
        msg = (f"Sex differs completely between the sides ({desc}){tail}: every difference here is "
               "also a sex difference, so XIST, TSIX and Y-chromosome genes will top the list and no "
               "test can separate sex from the comparison.")
    elif any(s and (t["female"] + t["male"]) >= 2 for s, t in zip(single, tallies)) and None in single:
        kind = "imbalanced"
        msg = (f"Sex differs between the sides ({desc}){tail}: XIST, TSIX and Y-chromosome genes "
               "will show up in this comparison for that reason, not the condition's; read sex-linked "
               "genes as sex.")
    else:
        return None
    return {"kind": kind, "message": msg, "sample_key": sample_key, "source": source,
            "a": tallies[0], "b": tallies[1]}


def sex_check(adata, column: str | None = None) -> dict:
    """Per-sample sex calls from expression, compared with any sex metadata, and
    warnings for mixed samples, mismatches and sex x condition confounding.

    ``column`` names the samples (default: detected). With no sample column the
    dataset is one group, "All cells". Raises KeyError for a missing column and
    ValueError for one with too many groups."""
    obs = adata.obs
    if column:
        if column not in obs.columns:
            raise KeyError(f"Column '{column}' not found in obs.")
        reason = "requested"
    else:
        column, reason = qc_groups.sample_grouping(adata)
    species = detect_species(adata)
    M, names, source, is_counts = _matrix(adata)
    f_list, m_list = _genes_for(species.species)
    f_genes, m_genes = match_genes(f_list, names), match_genes(m_list, names)

    if column:
        group_names, codes = qc_groups.group_codes(obs[column])
        if len(group_names) > qc_groups.MAX_GROUPS:
            raise ValueError(f"'{column}' has {len(group_names)} groups; the sex check takes at "
                             f"most {qc_groups.MAX_GROUPS}.")
    else:
        group_names, codes = ["All cells"], np.zeros(adata.n_obs, dtype=int)
    k = len(group_names)

    F = _columns(M, [names.get_loc(g) for g in f_genes])
    Y = _columns(M, [names.get_loc(g) for g in m_genes])
    f_any = (F > 0).any(axis=1) if f_genes else np.zeros(adata.n_obs, bool)
    m_any = (Y > 0).any(axis=1) if m_genes else np.zeros(adata.n_obs, bool)
    f_sum, m_sum = F.sum(axis=1), Y.sum(axis=1)
    totals = _totals(M) if is_counts else None

    meta_col = sex_metadata_column(obs)
    meta = _metadata_per_group(obs, meta_col, codes, k) if meta_col else None

    rows: list[dict] = []
    for i, name in enumerate(group_names):
        m = codes == i
        n = int(m.sum())
        row: dict = {"name": name, "n_cells": n}
        row["frac_female"] = round(float(f_any[m].mean()), 4) if n and f_genes else None
        row["frac_male"] = round(float(m_any[m].mean()), 4) if n and m_genes else None
        row["frac_both"] = round(float((f_any & m_any)[m].mean()), 4) if n else 0.0
        tot = float(totals[m].sum()) if totals is not None else 0.0
        row["cpm_female"] = (round(float(f_sum[m].sum()) / tot * 1e6, 1)
                             if tot > 0 and f_genes else None)
        row["cpm_male"] = (round(float(m_sum[m].sum()) / tot * 1e6, 1)
                           if tot > 0 and m_genes else None)
        row["call"], row["confidence"] = call_sex(n, row["frac_female"], row["frac_male"])
        row["evidence"] = (f"Only {n} cells: too few to call." if n < MIN_CELLS
                           else _evidence(row, f_genes, m_genes))
        row["metadata"] = meta[i]["value"] if meta else None
        rows.append(row)
    ambient_recall(rows)
    for r in rows:
        r["mismatch"] = bool(r["call"] in ("female", "male") and r["metadata"] in ("female", "male")
                             and r["metadata"] != r["call"])

    warnings: list[str] = []
    for r in rows:
        if r["mismatch"]:
            warnings.append(
                f"'{r['name']}' is labelled {r['metadata']} in '{meta_col}' but its expression "
                f"looks {r['call']} ({r['evidence'][:-1]}): a possible sample swap or mislabel.")
    for r in rows:
        if r["call"] == "mixed":
            warnings.append(
                f"'{r['name']}' has both female and male markers in many cells "
                f"({r['evidence'][:-1]}): it may pool donors of both sexes (demultiplex them "
                "before per-donor analysis), hold doublets across donors, or carry ambient RNA "
                "from other donors in a pooled run (then its CPM of the second set is far below "
                "the other samples').")
    confounding = sex_confounding(obs, column, codes, group_names, _sex_of_rows(rows)) if column else []
    warnings += [c["message"] for c in confounding]

    summary = {c: sum(r["call"] == c for r in rows) for c in CALLS}
    notes: list[str] = []
    if not column:
        notes.append("No sample, donor or batch column found, so all cells are judged as one "
                     "group.")
    if not f_genes and not m_genes:
        notes.append("None of the sex marker genes (XIST, TSIX, Y-chromosome genes) are in the "
                     "data, so sex can't be inferred (often filtered out before upload).")
    elif not m_genes:
        notes.append("No Y-chromosome genes are in the data (filtered out?), so calls rest on "
                     f"{'/'.join(f_genes)} alone: a male call means only that it is absent.")
    elif not f_genes:
        notes.append("XIST and TSIX are not in the data, so calls rest on the Y genes alone.")
    elif not any(g.upper() == "XIST" for g in f_genes):
        # Jäkel 2019's CELLxGENE version: no XIST, so both metadata females came out
        # ambiguous with nothing saying why.
        top = max((r["frac_female"] or 0 for r in rows), default=0.0)
        notes.append(f"XIST is not in the data, only {'/'.join(f_genes)}, which is much sparser (in at "
                     f"most {_share(top)} of a sample's cells here): a sample without Y genes is left "
                     "ambiguous rather than called female, as it could also be a male sample too "
                     "shallow for its Y genes to show.")
    if not is_counts:
        notes.append(f"No raw counts, so detection is read from {source} and no CPM is given.")
    called = [r for r in rows if r["n_cells"] >= MIN_CELLS and r["frac_female"] is not None]
    if len(called) >= 3 and summary["ambiguous"] == len(called):
        fr = [r["frac_female"] for r in called]
        if min(fr) > 0 and max(fr) / min(fr) <= UNIFORM_RATIO:
            notes.append(
                f"{'/'.join(f_genes)} is detected at a similar low level in every sample "
                f"({_share(min(fr))} to {_share(max(fr))} of cells): the samples likely share "
                "one sex, but the marker is too sparse at this depth to say which.")
    return {
        "column": column, "reason": reason, "n_cells": int(adata.n_obs),
        "species": species.species, "species_evidence": species.evidence,
        "female_genes": f_genes, "male_genes": m_genes,
        "female_genes_missing": [g for g in f_list if g.upper() not in {x.upper() for x in f_genes}],
        "male_genes_missing": [g for g in m_list if g.upper() not in {x.upper() for x in m_genes}],
        "source": source, "counts": is_counts,
        "thresholds": {"on": ON_SHARE, "off": OFF_SHARE, "min_cells": MIN_CELLS},
        "metadata_column": meta_col,
        "groups": rows, "summary": summary, "confounding": confounding,
        "warnings": warnings, "notes": notes,
    }


def report_lines(result: dict) -> tuple[str, list[str]]:
    """(one fact line for the QC report, its concerns) from ``sex_check``'s result."""
    s = result["summary"]
    rows = result["groups"]
    where = f"by '{result['column']}'" if result["column"] else "over all cells"
    if not result["female_genes"] and not result["male_genes"]:
        return ("Sex can't be inferred from expression: no XIST or Y-chromosome genes in "
                "the data.", [])
    counts = ", ".join(f"{s[c]} {c}" for c in CALLS if s[c])
    head = f"Inferred sex from expression {where} ({len(rows)} groups): {counts}."
    if result["metadata_column"]:
        n_bad = sum(r["mismatch"] for r in rows)
        head += (f" Compared with '{result['metadata_column']}': "
                 + (f"{n_bad} disagree." if n_bad else "no disagreement."))
    extra = [n for n in result["notes"] if "similar low level" in n or "Y-chromosome genes are" in n]
    if extra:
        head += " " + " ".join(extra)
    return head, list(result["warnings"])
