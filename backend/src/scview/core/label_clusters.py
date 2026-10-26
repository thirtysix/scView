"""Which clusters each cell-type label spans, for the co-pilot.

A label is often split over several clusters (Kang's CD14 Mono is Leiden 1, its
control cells, and Leiden 2, its IFN-beta cells). With only a cluster -> label
map, the co-pilot said "the same cells are Leiden cluster 1". This fact lists,
for each label of a cell-type-like column, every cluster holding a real share of
its cells, so "this label = clusters 1 and 2" is the obvious reading.
"""

from __future__ import annotations

import re

import pandas as pd

TAG = "result:labels"  # + ":<column>"
MAX_COLUMNS = 2
MAX_LABELS = 40
MIN_SHARE = 0.05   # a cluster is listed when it holds >= 5% of the label's cells ...
MIN_CELLS = 20     # ... and at least this many
MIN_PURITY = 0.6   # the column must follow the clusters: a condition or donor column does not

_NOT_LABELS = re.compile(
    r"condition|stim|treat|disease|status|genotype|timepoint|time_point|infect|perturb|"
    r"batch|library|lane|plate|chip|^dataset$|^run$|donor|patient|sample|replicate|ident|"
    r"channel|sex|tissue|phase|guide|^vdj_|^hash_|^qc_|^scview_|doublet|mixscape|clon",
    re.I,
)


def _label_columns(obs: pd.DataFrame, clustering: str) -> list[str]:
    clusters = obs[clustering].astype(str)
    out: list[tuple[float, str]] = []
    for col in obs.columns:
        if col == clustering or _NOT_LABELS.search(str(col)):
            continue
        s = obs[col]
        if not (isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object):
            continue
        values = s.dropna().astype(str)
        n_unique = values.nunique()
        if not 2 <= n_unique <= MAX_LABELS or all(v.isdigit() for v in values.unique()):
            continue
        ct = pd.crosstab(clusters[values.index], values)
        purity = float(ct.max(axis=1).sum() / max(1, ct.to_numpy().sum()))
        if purity >= MIN_PURITY:
            out.append((purity, str(col)))
    out.sort(key=lambda t: -t[0])
    return [c for _, c in out[:MAX_COLUMNS]]


def grounding_lines(adaptor) -> tuple[list[str], list[tuple[str, str, list[str]]]]:
    """Prompt lines and, per label column, ``(tag, detail, its lines)``."""
    clustering = adaptor.active_clustering_column()
    obs = adaptor.adata.obs
    if not clustering or clustering not in obs.columns:
        return [], []
    lines: list[str] = []
    cites: list[tuple[str, str, list[str]]] = []
    for col in _label_columns(obs, clustering):
        labels = obs[col].astype(str).where(obs[col].notna())
        ct = pd.crosstab(labels, obs[clustering].astype(str))
        tag = f"{TAG}:{col}"
        n_split = 0
        sized: list[tuple[int, str]] = []
        for label, counts in ct.iterrows():
            total = int(counts.sum())
            if total == 0:
                continue
            keep = sorted(((c, int(n)) for c, n in counts.items()
                           if n >= MIN_CELLS and n / total >= MIN_SHARE), key=lambda t: -t[1])
            rest = total - sum(n for _, n in keep)
            n_split += len(keep) > 1
            where = ", ".join(f"{c} ({n:,})" for c, n in keep) or "no cluster holds 5% of them"
            sized.append((total, f"- **{label}** ({total:,} cells): "
                          + ("cluster " if len(keep) == 1 else "clusters ") + where
                          + (f"; {rest:,} more in other clusters" if rest >= MIN_CELLS else "")))
        rows = [r for _, r in sorted(sized, key=lambda t: -t[0])]
        head = (f"\n## Labels in '{col}' across the clusters of '{clustering}' (two different columns: "
                f"{len(rows)} labels, {ct.shape[1]} clusters; {n_split} labels span several clusters) "
                f"[{tag}]")
        lines += [head, *rows]
        cites.append((tag, f"{col} over {clustering}: {n_split} of {len(rows)} labels span "
                           "several clusters", rows))
    return lines, cites
