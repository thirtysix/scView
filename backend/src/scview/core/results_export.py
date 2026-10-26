"""Stored results as plain tables: ``.tsv.gz`` per result, and a zip of them all.

Each result kind has its table (``tables``): the DE genes, a markers result's genes per
group, the enrichment terms, the ligand-receptor interactions, ... A kind without its own
reader falls back to the longest list of records it holds. The ``.tsv.gz`` is a header and
rows, no comment lines, so it reads straight into R or pandas; numbers are written to 6
significant digits and p-values as ``%.3e``, so float32 storage shows no noise digits. The
parameters travel in the zip's ``manifest.json``.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import zipfile
from typing import Any

import numpy as np
import pandas as pd

from scview.core import results_store

MARKER_FIELDS = ("names", "scores", "logfoldchanges", "pvals", "pvals_adj")


def _cell(v: Any) -> Any:
    if isinstance(v, (list, tuple, set)):
        return ";".join(map(str, v))
    if isinstance(v, dict):
        return json.dumps(v, default=str)
    if isinstance(v, np.ndarray):
        return ";".join(map(str, v.tolist()))
    return v


def _flat(r: dict) -> dict:
    """A record's values as cells; a dict of plain values (a composition group's share per
    condition) becomes one column per key, ``props_CTRL``, ``props_STIM``."""
    out: dict = {}
    for k, v in r.items():
        if isinstance(v, dict) and v and all(not isinstance(x, (dict, list, tuple)) for x in v.values()):
            out.update({f"{k}_{kk}": x for kk, x in v.items()})
        else:
            out[k] = _cell(v)
    return out


def _records(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([_flat(r) for r in rows])


def _markers(entry: dict, value: dict) -> pd.DataFrame:
    rgg = value["value"]
    names = rgg["names"]
    groups = list(names.dtype.names)
    pts, rest = rgg.get("pts"), rgg.get("pts_rest")
    frames = []
    for g in groups:
        df = pd.DataFrame({"group": g, "rank": np.arange(1, len(names) + 1)})
        for f in MARKER_FIELDS:
            if f in rgg:
                df["gene" if f == "names" else f] = np.asarray(rgg[f][g])
        if isinstance(pts, pd.DataFrame) and g in pts.columns:
            df["pct_in"] = pts[g].reindex(df["gene"]).to_numpy()
        if isinstance(rest, pd.DataFrame) and g in rest.columns:
            df["pct_rest"] = rest[g].reindex(df["gene"]).to_numpy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _de(entry: dict, r: Any) -> pd.DataFrame:
    df = pd.DataFrame({"gene": r.genes, "logfoldchange": r.logfoldchanges, "pval": r.pvals})
    if r.pvals_adj is not None:
        df["pval_adj"] = r.pvals_adj
    if np.isfinite(r.stats).any():
        df["stat"] = r.stats
    return df


def _enrichment(entry: dict, value: dict) -> pd.DataFrame:
    raw = value["value"]
    return _records(json.loads(raw) if isinstance(raw, str) else list(raw or []))


def _ccc(entry: dict, r: Any) -> pd.DataFrame:
    return r.table.drop(columns=[c for c in r.table.columns if str(c).startswith("_")])


def _composition(entry: dict, r: Any) -> pd.DataFrame:
    return _records([g.__dict__ for g in r.groups])


def _milo(entry: dict, r: Any) -> pd.DataFrame:
    df = pd.DataFrame({"nhood": np.arange(len(r.index_cells)), "index_cell": r.index_cells,
                       "size": r.sizes, "logFC": r.log_fc, "pval": r.pval, "spatial_fdr": r.spatial_fdr})
    for cond, counts in (r.counts or {}).items():
        df[f"cells_{cond}"] = counts
    if r.labels is not None and r.label_fraction is not None:
        df["label"] = r.labels
        df["label_fraction"] = r.label_fraction
    return df


def _cytokine(entry: dict, value: dict) -> pd.DataFrame:
    r = value["result"]
    rows = []
    for i, g in enumerate(r["groups"]):
        for j, c in enumerate(r["cytokines"]):
            rows.append({"group": g, "cytokine": c, "z": r["z"][i][j], "pval": r["p"][i][j], "coef": r["coef"][i][j]})
    return pd.DataFrame(rows)


def _longest_records(value: Any) -> pd.DataFrame | None:
    """The longest list of records in a result (a fallback for kinds without a reader)."""
    best: list | None = None

    def walk(v: Any, depth: int) -> None:
        nonlocal best
        if depth > 3:
            return
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            if best is None or len(v) > len(best):
                best = v
        elif isinstance(v, dict):
            for x in v.values():
                walk(x, depth + 1)
        elif hasattr(v, "__dataclass_fields__"):
            for name in v.__dataclass_fields__:
                walk(getattr(v, name), depth + 1)

    if isinstance(value, pd.DataFrame):
        return value
    walk(value, 0)
    return _records(best) if best else None


READERS = {
    "markers": _markers, "de": _de, "enrichment": _enrichment, "ccc": _ccc, "ccc_compare": _ccc,
    "composition": _composition, "milo": _milo, "cytokine": _cytokine,
    "offtarget": lambda e, v: _records(v["genes"]),
    "merge_check": lambda e, v: _records(v["pairs"]),
    "count_split": lambda e, v: _records(v["groups"]),
    "ppi": lambda e, v: _records(v),
}


def table(entry: dict, value: Any) -> pd.DataFrame | None:
    """The result as one table, or None when it holds none."""
    reader = READERS.get(entry["kind"])
    try:
        df = reader(entry, value) if reader else _longest_records(value)
    except (KeyError, TypeError, AttributeError, ValueError):
        df = _longest_records(value)
    return df if df is not None and len(df.columns) else None


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if s.dtype.kind == "f":
            spec = "%.3e" if results_store.is_p_column(col) else "%.6g"
            out[col] = [("" if not np.isfinite(v) else spec % v) for v in s.to_numpy(dtype=np.float64)]
    return out


def tsv_gz(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    _fmt(df).to_csv(buf, sep="\t", index=False, lineterminator="\n")
    return gzip.compress(buf.getvalue().encode(), 6, mtime=0)


def file_stem(entry: dict) -> str:
    stem = re.sub(r"[^A-Za-z0-9._ +-]+", "_", entry.get("label") or entry["kind"]).strip(" ._")[:80]
    return f"{stem or entry['kind']}_{entry['id'][:8]}"


def manifest_row(entry: dict, file: str | None) -> dict:
    return {k: entry.get(k) for k in ("id", "kind", "key", "label", "params", "version", "created_at",
                                      "fingerprint", "status", "stale")} | {"file": file}


def results_zip(dataset_id: str, out: Any, adaptor: Any = None, h5ad_path: Any = None,
                h5ad_name: str | None = None) -> int:
    """Write every stored result as ``results/{label}.tsv.gz`` + ``manifest.json`` (each
    result's parameters, version, fingerprint and stale flag) to ``out`` (a path or file);
    with ``h5ad_path``, the h5ad beside them. Returns how many results it holds."""
    rows = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:  # members are gzip already
        for e in results_store.entries(dataset_id, adaptor=adaptor):
            got = results_store.load(dataset_id, e["id"])
            df = table(e, got[1]) if got else None
            name = None
            if df is not None:
                name = f"results/{file_stem(e)}.tsv.gz"
                zf.writestr(name, tsv_gz(df))
            rows.append(manifest_row(e, name))
        zf.writestr("manifest.json", json.dumps({"dataset_id": dataset_id, "results": rows}, indent=1,
                                                default=str))
        if h5ad_path:
            zf.write(h5ad_path, h5ad_name or "dataset.h5ad")
    return len(rows)


# ---------------------------------------------------------------------------
# Compare with current
# ---------------------------------------------------------------------------

def _top_markers(rgg: Any, n: int) -> dict[str, list[str]]:
    names = rgg["names"]
    return {g: [str(x) for x in np.asarray(names[g])[:n]] for g in names.dtype.names}


def compare_markers(old: dict, current: Any, n: int = 50) -> dict:
    """Per group: how many of the top ``n`` genes the earlier and current markers share."""
    a, b = _top_markers(old, n), _top_markers(current, n)
    groups = []
    for g in a:
        if g not in b:
            groups.append({"group": g, "shared": None, "only_earlier": a[g][:5], "only_current": []})
            continue
        sa, sb = set(a[g]), set(b[g])
        groups.append({"group": g, "shared": len(sa & sb), "jaccard": round(len(sa & sb) / len(sa | sb), 3),
                       "only_earlier": [x for x in a[g] if x not in sb][:5],
                       "only_current": [x for x in b[g] if x not in sa][:5]})
    return {"n": n, "groups": groups,
            "earlier": dict(old.get("params") or {}), "current": dict(current.get("params") or {})}


def compare_enrichment(old: Any, current: Any, n: int = 20, fdr: float = 0.05) -> dict:
    """The top ``n`` terms of each, and the significant terms they share."""
    def recs(raw):
        rows = json.loads(raw) if isinstance(raw, str) else list(raw or [])
        return sorted(rows, key=lambda r: r.get("adjusted_pvalue", 1.0))
    a, b = recs(old), recs(current)
    sig_a = {r["term"] for r in a if r.get("adjusted_pvalue", 1) < fdr}
    sig_b = {r["term"] for r in b if r.get("adjusted_pvalue", 1) < fdr}
    return {"n": n, "fdr": fdr, "n_sig_earlier": len(sig_a), "n_sig_current": len(sig_b),
            "shared": len(sig_a & sig_b),
            "jaccard": round(len(sig_a & sig_b) / len(sig_a | sig_b), 3) if sig_a | sig_b else None,
            "top_earlier": [r["term"] for r in a[:n]], "top_current": [r["term"] for r in b[:n]],
            "only_earlier": sorted(sig_a - sig_b)[:10], "only_current": sorted(sig_b - sig_a)[:10]}
