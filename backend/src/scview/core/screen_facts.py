"""What the user has on screen, for the co-pilot: "Explain this result".

The view context names the open Unified View subtab (``subtab``), the Compare
tab and, in ``on_screen``, the result shown there:

- ``{"kind": "de", "id": <result id>, "label": ...}``: a DE result, read back from
  ``core/de_results.py`` by its id;
- ``{"kind": "enrichment", "column": ..., "group": ...}``: the enrichment of one
  group's markers, read from ``uns`` (``enrichment__<column>__<group>``);
- ``{"kind": "compare", "tab": ...}``: the open Compare tab, whose results are in
  the groups facts already.

``view_lines`` says what is shown in the "currently viewing" block; ``fact_lines``
gives the on-screen result's numbers under the ``[result:screen]`` tag.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

TAG = "result:screen"
SIG_PADJ = 0.05
SIG_LFC = 1.0
TOP_GENES = 8
TOP_TERMS = 5

SUBTAB_NAMES = {
    "markers": "Markers", "expression": "Expression", "protein": "Protein",
    "genesets": "Gene sets", "enrichment": "Enrichment", "de": "DE", "activity": "Activity",
    "genelists": "Gene lists", "communication": "Communication", "report": "Report",
    "programs": "Programs", "cnv": "Copy number", "clonotypes": "Clonotypes",
    "perturbation": "Perturbation",
}
# The Compare panel's tabs (frontend ComparePanel.tsx); Milo and miloDE sit under Composition.
COMPARE_TABS = {"joint": "Joint embedding", "composition": "Composition", "planner": "Plan a study"}


def _screen(vc: dict | None) -> dict:
    scr = (vc or {}).get("on_screen")
    return scr if isinstance(scr, dict) else {}


def _short(value: Any, limit: int = 120) -> str:
    s = str(value or "").strip()
    return s[:limit]


def view_lines(vc: dict | None) -> list[str]:
    """Lines for the "currently viewing" block: the open subtab or Compare tab and
    the result on screen."""
    vc = vc or {}
    out = []
    sub = vc.get("subtab")
    if sub:
        out.append(f"- Unified View tab open: {SUBTAB_NAMES.get(str(sub), _short(sub, 40))}")
    scr = _screen(vc)
    kind = str(scr.get("kind") or "")
    if kind == "compare" and scr.get("tab"):
        out.append(f"- Compare tab open: {COMPARE_TABS.get(str(scr['tab']), _short(scr['tab'], 40))}")
    elif kind == "de" and (scr.get("label") or scr.get("id")):
        out.append(f"- On screen: the DE result \"{_short(scr.get('label')) or 'unnamed'}\" "
                   f"(facts under [{TAG}] when given)")
    elif kind == "enrichment" and scr.get("group"):
        out.append(f"- On screen: the enrichment of {_short(scr.get('column'), 60)} = "
                   f"{_short(scr.get('group'), 60)}'s markers (facts under [{TAG}] when given)")
    return out


def fact_lines(adaptor, vc: dict | None) -> tuple[list[str], str]:
    """The on-screen result's facts (header line first), and a one-line detail.
    Empty when nothing readable is on screen."""
    scr = _screen(vc)
    kind = scr.get("kind")
    try:
        if kind == "de" and scr.get("id"):
            return _de_lines(adaptor, str(scr["id"]))
        if kind == "enrichment" and scr.get("column") and scr.get("group") is not None:
            return _enrichment_lines(adaptor, str(scr["column"]), str(scr["group"]))
    except Exception as exc:  # pragma: no cover - defensive: never break an answer
        logger.debug("screen facts failed: %s", exc)
    return [], ""


def _de_lines(adaptor, result_id: str) -> tuple[list[str], str]:
    from scview.core import de_results, results_store

    dataset_id = results_store.dataset_id_of(adaptor)
    r = de_results.recall(dataset_id, result_id) if dataset_id else None
    if r is None:
        return [], ""
    c = r.comparison or {}
    lfc = np.asarray(r.logfoldchanges, dtype=np.float64)
    padj = (np.asarray(r.pvals_adj, dtype=np.float64) if r.pvals_adj is not None
            else np.full(lfc.shape, np.nan))
    ok = np.isfinite(padj) & (padj < SIG_PADJ)
    up = np.flatnonzero(ok & (lfc >= SIG_LFC))
    down = np.flatnonzero(ok & (lfc <= -SIG_LFC))
    up = up[np.argsort(-lfc[up], kind="stable")][:TOP_GENES]
    down = down[np.argsort(lfc[down], kind="stable")][:TOP_GENES]

    def genes(idx) -> str:
        return ", ".join(f"{r.genes[i]} ({lfc[i]:+.2f})" for i in idx) or "none"

    if c.get("mode") == "groups":
        a, b = c.get("group_a"), c.get("group_b")
        what = f"{a} vs {b} in '{c.get('column')}'"
        if c.get("within_column") and c.get("within_value") is not None:
            what += f", within {c['within_column']} = {c['within_value']}"
    else:
        a, b = c.get("label") or "the selection", "the rest"
        what = f"{a} vs the rest"
    sizes = ""
    if isinstance(c.get("n_a"), int) and isinstance(c.get("n_b"), int):
        sizes = f"; {c['n_a']:,} vs {c['n_b']:,} cells"
    if c.get("replicate_key"):
        sizes += f", donors/samples from '{c['replicate_key']}'"
    n_up, n_down = int((ok & (lfc >= SIG_LFC)).sum()), int((ok & (lfc <= -SIG_LFC)).sum())
    sig = (f"{n_up} genes up and {n_down} down in {a} (adjusted p < {SIG_PADJ:g}, |log2FC| >= "
           f"{SIG_LFC:g})" if r.pvals_adj is not None else
           "no adjusted p-values (logistic regression gives coefficients only)")
    lines = [
        f"\n## The DE result on screen [{TAG}]",
        f"- {r.label}: {what}{sizes}; {len(r.genes):,} genes tested; {sig} [{TAG}]",
        f"- Most up in {a} (log2FC): {genes(up)} [{TAG}]",
        f"- Most down in {a}, i.e. up in {b} (log2FC): {genes(down)} [{TAG}]",
    ]
    if r.n_skipped:
        lines.append(f"- {r.n_skipped:,} genes were not tested (detected in too few cells of either "
                     f"group) [{TAG}]")
    return lines, f"{r.label}: {n_up} up, {n_down} down"


def _enrichment_lines(adaptor, column: str, group: str) -> tuple[list[str], str]:
    from scview.core.uns_keys import enrichment_key

    raw = adaptor.adata.uns.get(enrichment_key(column, group))
    if raw is None:
        return [], ""
    rows = json.loads(raw) if isinstance(raw, str) else list(raw)
    terms = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        term = row.get("term", row.get("Term"))
        padj = row.get("adjusted_pvalue", row.get("Adjusted P-value"))
        if term is None or padj is None:
            continue
        terms.append((float(padj), str(term), row.get("overlap_count")))
    terms.sort(key=lambda t: t[0])
    n_sig = sum(1 for p, _, _ in terms if p < SIG_PADJ)
    top = "; ".join(f"{t} (adj p {p:.2g}" + (f", {n} genes" if isinstance(n, int) else "") + ")"
                    for p, t, n in terms[:TOP_TERMS]) or "none"
    lines = [
        f"\n## The enrichment on screen [{TAG}]",
        f"- {column} = {group} markers: {len(terms)} gene sets tested, {n_sig} with adjusted p < "
        f"{SIG_PADJ:g}; top: {top} [{TAG}]",
    ]
    return lines, f"enrichment of {column} = {group}: {n_sig} gene sets"
