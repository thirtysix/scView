"""A cluster or gene report (``core/report.py``) as one self-contained HTML file.

Inline CSS, inline SVG bars and dots, no script, no external font or image, so the
file opens anywhere, can be emailed and prints cleanly (``@media print``). Every
string that comes from the dataset (cluster, gene, label, donor and column names,
notes quoting them) goes through ``html.escape``: a label can hold ``<script>``.
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable, Iterable

ACCENT = "#2563eb"  # bars
INK = "#334155"  # points, text in plots
MUTED = "#94a3b8"
LABEL_CHARS = 28

STATUS_TEXT = {"not_computed": "Not computed", "not_applicable": "Not applicable"}

CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f8fafc; color: #0f172a;
  font: 13px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 880px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 0 0 6px; display: flex; gap: 8px; align-items: baseline; }
h3 { font-size: 12px; margin: 12px 0 4px; color: #334155; }
.sub { color: #475569; margin: 0 0 12px; }
nav { font-size: 12px; color: #475569; margin-bottom: 16px; }
nav a { color: #1d4ed8; text-decoration: none; margin-right: 10px; white-space: nowrap; }
section { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 16px;
  margin: 0 0 12px; break-inside: avoid; }
.caption { color: #475569; font-size: 12px; margin: 0 0 8px; }
.method { color: #64748b; font-size: 11px; margin: 8px 0 0; }
.note { background: #f8fafc; border-left: 3px solid #cbd5e1; padding: 6px 8px; font-size: 12px;
  color: #334155; margin: 8px 0 0; }
.status { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em;
  color: #92400e; background: #fef3c7; border-radius: 4px; padding: 1px 6px; }
.status.na { color: #475569; background: #f1f5f9; }
table { border-collapse: collapse; width: 100%; font-size: 12px; margin: 4px 0; }
th, td { text-align: left; padding: 3px 6px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }
th { color: #475569; font-weight: 600; border-bottom: 1px solid #e2e8f0; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.genes { color: #475569; font-size: 11px; }
.flag { font-weight: 600; color: #92400e; }
.cap { color: #64748b; font-size: 11px; margin: 2px 0 0; }
svg { display: block; max-width: 100%; height: auto; margin: 6px 0; }
footer { color: #64748b; font-size: 11px; margin-top: 20px; }
@media print {
  body { background: #fff; font-size: 11px; }
  main { max-width: none; padding: 0; }
  nav { display: none; }
  section { border: none; border-top: 1px solid #cbd5e1; border-radius: 0; padding: 10px 0; }
  a { color: inherit; text-decoration: none; }
}
"""


def e(value) -> str:
    """Escaped text (None as an empty string)."""
    return "" if value is None else html.escape(str(value), quote=True)


def fnum(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:,.{digits}f}"


def fpct(v, digits: int = 1) -> str:
    return "–" if v is None else f"{100.0 * v:.{digits}f}%"


def fp(v) -> str:
    if v is None:
        return "–"
    if v == 0:
        return "< 1e-300"  # below double precision
    return f"{v:.2e}" if v < 1e-3 else f"{v:.3f}"


def short(text: str, n: int = LABEL_CHARS) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def table(headers: list[tuple[str, str, bool]], rows: Iterable[list[str]]) -> str:
    """``headers``: (label, tooltip, numeric); ``rows``: cells as ALREADY ESCAPED html."""
    head = "".join(f'<th{" class=n" if num else ""} title="{e(tip)}">{e(label)}</th>'
                   for label, tip, num in headers)
    body = "".join(
        "<tr>" + "".join(f'<td{" class=n" if headers[i][2] else ""}>{cell}</td>'
                         for i, cell in enumerate(row)) + "</tr>"
        for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def hbars(items: list[tuple[str, float | None, str]], *, max_value: float | None = None,
          label: str = "") -> str:
    """Horizontal bars: (label, value, value text) per row, one accent colour."""
    rows = [(lab, v, txt) for lab, v, txt in items]
    if not rows:
        return ""
    top = max_value or max((v for _, v, _ in rows if v is not None), default=0.0) or 1.0
    lw, bw, rh = 170, 330, 18
    width, height = lw + bw + 90, rh * len(rows) + 6
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
             f'height="{height}" role="img" aria-label="{e(label)}">']
    for i, (lab, v, txt) in enumerate(rows):
        y = 3 + i * rh
        w = 0.0 if v is None or v <= 0 else bw * min(v / top, 1.0)
        parts.append(f'<text x="{lw - 6}" y="{y + 12}" font-size="11" text-anchor="end" fill="{INK}">'
                     f"<title>{e(lab)}</title>{e(short(lab))}</text>")
        parts.append(f'<rect x="{lw}" y="{y + 3}" width="{bw}" height="{rh - 6}" fill="#f1f5f9"/>')
        parts.append(f'<rect x="{lw}" y="{y + 3}" width="{w:.1f}" height="{rh - 6}" fill="{ACCENT}"/>')
        parts.append(f'<text x="{lw + bw + 6}" y="{y + 12}" font-size="11" fill="{INK}">{e(txt)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def dots(rows: list[tuple[str, list[tuple[float, str]]]], *, max_value: float | None = None,
         fmt: Callable[[float], str] = fpct, label: str = "") -> str:
    """One row per label, one dot per value (label shown on hover), on a shared axis
    from 0 to the largest value."""
    if not rows:
        return ""
    vals = [v for _, pts in rows for v, _ in pts if v is not None]
    top = max_value or (max(vals) if vals else 1.0) or 1.0
    lw, aw, rh = 170, 380, 18
    width, height = lw + aw + 60, rh * len(rows) + 22
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
             f'height="{height}" role="img" aria-label="{e(label)}">']
    for i, (lab, pts) in enumerate(rows):
        y = 3 + i * rh + rh / 2
        parts.append(f'<text x="{lw - 6}" y="{y + 4}" font-size="11" text-anchor="end" fill="{INK}">'
                     f"<title>{e(lab)}</title>{e(short(lab))}</text>")
        parts.append(f'<line x1="{lw}" x2="{lw + aw}" y1="{y}" y2="{y}" stroke="#e2e8f0"/>')
        for v, name in pts:
            if v is None:
                continue
            x = lw + aw * min(max(v / top, 0.0), 1.0)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{INK}" fill-opacity="0.75">'
                         f"<title>{e(name)}: {e(fmt(v))}</title></circle>")
    ay = 3 + len(rows) * rh + 4
    parts.append(f'<line x1="{lw}" x2="{lw + aw}" y1="{ay}" y2="{ay}" stroke="{MUTED}"/>')
    for frac in (0.0, 0.5, 1.0):
        x = lw + aw * frac
        parts.append(f'<text x="{x:.1f}" y="{ay + 13}" font-size="10" text-anchor="middle" '
                     f'fill="{MUTED}">{e(fmt(top * frac))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _sub(title: str, part: dict | None, body: Callable[[dict], str]) -> str:
    """A sub-part with its own status (the marker database, doublets, ...)."""
    if not part:
        return ""
    head = f"<h3>{e(title)}</h3>"
    if part.get("status") != "ok":
        return head + f'<p class="note">{e(part.get("note") or STATUS_TEXT.get(part.get("status"), ""))}</p>'
    return head + body(part)


# ---------------------------------------------------------------------------
# Cluster sections
# ---------------------------------------------------------------------------


def _identity(d: dict) -> str:
    out = table([("Cells", "Cells in this group", True), ("Share of dataset", "Cells / all cells", True),
                 ("Groups in column", "Groups of the column", True)],
                [[fnum(d.get("n_cells")), fpct(d.get("fraction")), fnum(d.get("n_groups"))]])
    if d.get("by_condition"):
        out += f"<h3>Per {e(d['condition'])}</h3>"
        out += table([(str(d["condition"]), "Condition group", False), ("Cells", "Cells of this cluster", True),
                      ("Share of cluster", "Cells in the condition / cells of the cluster", True)],
                     ([e(r["value"]), fnum(r["n_cells"]), fpct(r["fraction"])] for r in d["by_condition"]))
    if d.get("by_donor"):
        conds = d.get("conditions") or []
        heads = [(str(d["donor"]), "Donor or sample", False), ("Cells", "Cells of this cluster", True),
                 ("Share", "Donor's cells / cells of the cluster", True)]
        heads += [(str(c), f"Cells of this cluster from the donor in {c}", True) for c in conds]
        out += f"<h3>Per {e(d['donor'])}</h3>" + table(heads, (
            [e(r["donor"]), fnum(r["n_cells"]), fpct(r["fraction"])]
            + [fnum((r.get("by_condition") or {}).get(c)) for c in conds] for r in d["by_donor"]))
        if d.get("by_donor_total", 0) > len(d["by_donor"]):
            out += f'<p class="cap">First {len(d["by_donor"])} of {d["by_donor_total"]:,} donors.</p>'
    return out


def _labels(d: dict) -> str:
    out = ""
    if d.get("columns"):
        out += table([("Column", "Another label column", False), ("Made by", "How the labels were made", False),
                      ("Top labels", "The commonest labels among these cells, with their share", False),
                      ("Unlabelled", "Cells without a label in this column", True)],
                     ([f'<span class="mono">{e(c["column"])}</span>', e(c["kind"]),
                       "; ".join(f"{e(t['label'])} ({fpct(t['fraction'])}, {fnum(t['n_cells'])})" for t in c["top"]),
                       fnum(c["n_unlabelled"])] for c in d["columns"]))

    def conf(part: dict) -> str:
        rows = []
        for c in part["checks"]:
            k = c["counts"]
            rows.append([f'<span class="mono">{e(c["column"])}</span>' + (" (stale)" if c.get("stale") else ""),
                         fnum(c.get("alpha")), fnum(k.get("confident")), fnum(k.get("ambiguous")),
                         fnum(k.get("unsure")), fnum(k.get("doesn't fit")), fnum(k.get("not checked"))])
        return table([("Checked column", "Label column the check ran on", False),
                      ("alpha", "Error rate of the prediction sets", True),
                      ("Confident", "Set = the cell's own label", True),
                      ("Ambiguous", "Own label and others", True), ("Unsure", "Empty set", True),
                      ("Doesn't fit", "Other labels, not its own", True), ("Not checked", "Too small a label", True)],
                     rows) + f'<p class="cap">{e(part.get("note"))}</p>'

    out += _sub("Label confidence", d.get("confidence"), conf)
    return out


def _markers(d: dict) -> str:
    out = ""
    if d.get("genes"):
        out += table([("Gene", "Marker gene", False), ("log2FC", "Log2 fold change vs the rest", True),
                      ("% in group", "Share of the group's cells expressing it", True),
                      ("% rest", "Share of the other cells expressing it", True),
                      ("Adj. p", "Adjusted p-value of the stored test", True),
                      ("Rank", "Place in the stored score order", True)],
                     ([f'<span class="mono">{e(g["gene"])}</span>', fnum(g["logfc"]), fpct(g["pct_in"]),
                       fpct(g["pct_out"]), fp(g["padj"]), fnum(g["rank"])] for g in d["genes"]))

    def db(part: dict) -> str:
        line = (f"<p><b>{e(part['verdict_text'])}</b>"
                + (f" for {e(part['db_type'])}" if part.get("db_type") else "")
                + (f" (p {fp(part['p'])}; overlap {e(', '.join(part['overlap']))})" if part.get("overlap") else "")
                + (f"; name match: {e(part['match'])}" if part.get("match") else "") + ".</p>")
        sugg = table([("Best database types", "From the group's top marker genes", False),
                      ("p", "Hypergeometric p", True), ("Shared genes", "Markers in the type's list", False)],
                     ([e(s["cell_type"]), fp(s["p"]), f'<span class="genes">{e(", ".join(s["overlap"]))}</span>']
                      for s in part.get("suggestions") or [])) if part.get("suggestions") else ""
        src = (f'<p class="cap">Verdict from {e(", ".join(part.get("verdict_sources") or []))}; suggestions from '
               f'{e(", ".join(part.get("suggestion_sources") or []))}. {e(part.get("note"))}</p>')
        return line + sugg + src

    out += _sub("Marker databases", d.get("database"), db)
    return out


def _enrichment(d: dict) -> str:
    if not d.get("terms"):
        return ""
    return table([("HALLMARK set", "MSigDB HALLMARK gene set", False),
                  ("Overlap", "Query genes in the set / set size in the background", True),
                  ("Adj. p", "Benjamini-Hochberg adjusted p", True),
                  ("Genes", "Query genes in the set", False)],
                 ([e(t["term"]), f"{t['overlap']}/{t['set_size']}", fp(t["padj"]),
                   f'<span class="genes">{e(", ".join(t["genes"]))}</span>'] for t in d["terms"]))


def _digits(metric: dict) -> int:
    """Percentages to 2 places, gene and UMI counts as whole numbers."""
    return 2 if str(metric.get("metric", "")).startswith("pct") else 0


def _qc(d: dict) -> str:
    out = table([("Metric", "Per-cell QC value", False), ("Cluster", "Median over this cluster's cells", True),
                 ("Rest", "Median over the other cells", True), ("", "", False)],
                ([e(m["label"]), fnum(m["cluster"], _digits(m)), fnum(m["rest"], _digits(m)), e(m.get("note") or "")]
                 for m in d.get("medians") or []))

    def flags(part: dict) -> str:
        items = "".join(f'<li><span class="flag">{e(f["flag"])}</span>: {e(f["reason"])}</li>'
                        for f in part.get("flags") or [])
        notes = "".join(f"<li>{e(n)}</li>" for n in part.get("notes") or [])
        body = f"<ul>{items}{notes}</ul>" if items or notes else "<p>No flag: this cluster does not stand out.</p>"
        return body + (f'<p class="cap">{e(part["note"])}</p>' if part.get("note") else "")

    def doublets(part: dict) -> str:
        bits = []
        if part.get("fraction") is not None:
            bits.append(f"{fnum(part['n_doublets'])} predicted doublets ({fpct(part['fraction'])}; rest "
                        f"{fpct(part.get('fraction_rest'))})")
        if part.get("median_score") is not None:
            bits.append(f"median doublet score {fnum(part['median_score'])} (rest {fnum(part.get('median_score_rest'))})")
        return f"<p>{e('; '.join(bits))}.</p>"

    def states(part: dict) -> str:
        return table([(str(part["column"]), "QC state", False), ("Cells", "Cells of this cluster", True),
                      ("Share", "Share of the cluster", True)],
                     ([e(s["state"]), fnum(s["n_cells"]), fpct(s["fraction"])] for s in part["states"]))

    def removed(part: dict) -> str:
        crit = ", ".join(f"{k} {v:,}" for k, v in (part.get("by_criterion") or {}).items())
        txt = (f"The filtering step removed {'at least ' if part.get('truncated') else ''}{part['n_removed']:,} "
               f"cells of this group ({fpct(part.get('fraction_lost'))} of it)" + (f": {crit}" if crit else "") + ".")
        return f"<p>{e(txt)}</p>"

    out += _sub("Cluster QC flags", d.get("flags"), flags)
    out += _sub("Doublets", d.get("doublets"), doublets)
    out += _sub("QC states", d.get("states"), states)
    out += _sub("Removed by filtering", d.get("removed"), removed)
    return out


def _composition(d: dict) -> str:
    out = ""
    if d.get("by_condition"):
        out += f"<h3>Share per {e(d['condition'])}</h3>"
        out += hbars([(r["condition"], r["share"], f"{fpct(r['share'])} of {r['n_total']:,}")
                      for r in d["by_condition"]], label="Share per condition")
    if d.get("samples"):
        conds = d.get("conditions") or [""]
        rows = []
        for c in conds:
            pts = [(s["share"], s["donor"]) for s in d["samples"] if s["condition"] == c]
            if pts:
                rows.append((c or "All cells", pts))
        out += f"<h3>Share per {e(d['donor'])} (one point per donor)</h3>"
        out += dots(rows, label="Share per donor")
        few = [s for s in d["samples"] if s.get("few_cells")]
        if few:
            out += (f'<p class="cap">{len(few)} donor samples have fewer than {d.get("min_sample_cells")} '
                    "cells: their share is mostly noise.</p>")
    st = d.get("stored_test")
    if st:
        props = ", ".join(f"{k} {fpct(v)}" for k, v in (st.get("props") or {}).items())
        out += (f"<h3>Stored composition test</h3><p>{e(st.get('test') or 'not tested')} on '{e(st['condition_key'])}' "
                f"with samples from '{e(st.get('sample_key'))}' ({e(st.get('design'))}): mean shares {e(props)}; "
                f"FDR {fp(st.get('fdr'))}.</p>")
    return out


def _methods(d: dict) -> str:
    out = "".join(f"<p>{e(line)}</p>" for line in d.get("summary") or [])
    if d.get("steps"):
        out += "<h3>Recorded analysis steps</h3><ol>" + "".join(f"<li>{e(s)}</li>" for s in d["steps"]) + "</ol>"
    if d.get("sections"):
        out += "<h3>This report</h3><ul>" + "".join(f"<li>{e(s)}</li>" for s in d["sections"]) + "</ul>"
    return out


# ---------------------------------------------------------------------------
# Gene sections
# ---------------------------------------------------------------------------


def _expression(d: dict) -> str:
    groups = d.get("groups") or []
    top = max((g["mean"] or 0.0 for g in groups), default=0.0)
    out = hbars([(g["group"], g["mean"], f"{fnum(g['mean'])} · {fpct(g['fraction'])}") for g in groups],
                max_value=top or 1.0, label="Mean expression per group")
    out += table([("Group", "Group of the column", False), ("Cells", "Cells in the group", True),
                  ("Mean", "Mean over all the group's cells", True),
                  ("Mean (expressing)", "Mean over the cells above 0", True),
                  ("% expressing", "Share of cells above 0", True)],
                 ([e(g["group"]), fnum(g["n_cells"]), fnum(g["mean"]), fnum(g["mean_expressing"]),
                   fpct(g["fraction"])] for g in groups))
    return out


def _gene_markers(d: dict) -> str:
    if not d.get("groups"):
        return ""
    return table([("Group", "Group where it passes the marker filters", False),
                  ("log2FC", "Log2 fold change vs the rest", True), ("% in group", "Expressing in the group", True),
                  ("% rest", "Expressing in the other cells", True), ("Adj. p", "Adjusted p-value", True),
                  ("Rank", "Place among the group's markers by fold change", True)],
                 ([e(g["group"]), fnum(g["logfc"]), fpct(g["pct_in"]), fpct(g["pct_out"]), fp(g["padj"]),
                   f"{g['marker_rank']} of {g['n_markers']}"] for g in d["groups"]))


def _databases(d: dict) -> str:
    if not d.get("types"):
        return ""
    out = table([("Cell type", "Database cell type listing the gene", False),
                 ("Rank", "The gene's place in the type's list (1 = best evidenced)", True),
                 ("List size", "Genes the type lists", True), ("Sources", "Databases naming it", False)],
                ([e(t["cell_type"]), fnum(t["rank"]), fnum(t["n_genes"]), e(", ".join(t["sources"]))]
                 for t in d["types"]))
    if d.get("n_types", 0) > len(d["types"]):
        out += f'<p class="cap">First {len(d["types"])} of {d["n_types"]:,} cell types by rank.</p>'
    return out


def _per_donor(d: dict) -> str:
    out = ""
    summary = d.get("summary") or []
    if summary:
        rows = []
        for g in d.get("groups") or []:
            for c in d.get("conditions") or [""]:
                pts = [(p["value"], p["sample"]) for p in d.get("points") or []
                       if p["group"] == g and p["condition"] == c]
                if pts:
                    rows.append((f"{g} · {c}" if c else g, pts))
        out += dots(rows[:80], fmt=lambda v: f"{v:.2f}", label="Per-donor values")
        if len(rows) > 80:
            out += f'<p class="cap">First 80 of {len(rows)} group and condition rows; the table has all.</p>'
        out += table([("Group", "Group", False), ("Condition", "Condition", False),
                      ("Donors", "Donor samples with enough cells", True), ("Median", "Median over donors", True),
                      ("Range", "Lowest to highest donor", True)],
                     ([e(s["group"]), e(s["condition"]), fnum(s["n_donors"]), fnum(s["median"]),
                       f"{fnum(s['min'])} to {fnum(s['max'])}"] for s in summary))
    tests = d.get("paired_tests") or {}
    if tests:
        out += "<h3>Paired tests</h3>" + table(
            [("Group", "Group", False), ("Pairs", "Donors in both conditions", True),
             ("Mean change", "Second minus first condition", True), ("Wilcoxon p", "Signed-rank test", True),
             ("Paired t p", "Paired t-test", True)],
            ([e(g), fnum(t.get("n_pairs")), fnum(t.get("mean_diff")), fp(t.get("wilcoxon_p")), fp(t.get("ttest_p"))]
             for g, t in tests.items()))
    return out


def _coexpression(d: dict) -> str:
    def rows(items):
        return ([f'<span class="mono">{e(g["gene"])}</span>', fnum(g["r"]), fpct(g["pct_expressing"] / 100.0)]
                for g in items)

    heads = [("Gene", "Correlated gene", False), ("r", "Pearson correlation over cells", True),
             ("% expressing", "Cells expressing it", True)]
    out = ""
    if d.get("positive"):
        out += "<h3>Most positive</h3>" + table(heads, rows(d["positive"]))
    if d.get("negative"):
        out += "<h3>Most negative</h3>" + table(heads, rows(d["negative"]))
    return out


def _hvg(d: dict) -> str:
    if not d.get("stats"):
        return ""
    return table([("Statistic", "Stored with the selection", False), ("Value", "", True)],
                 ([e(s["label"]), fnum(s["value"], 3)] for s in d["stats"]))


def _gene_sets(d: dict) -> str:
    out = ""
    if d.get("by_collection"):
        out += table([("Collection", "MSigDB collection", False), ("Sets", "Sets holding the gene", True)],
                     ([e(c["collection_name"]), fnum(c["n_sets"])] for c in d["by_collection"]))
    if d.get("sets"):
        out += "<ul>" + "".join(f"<li>{e(s['name'])} <span class='genes'>({e(s['collection_name'])})</span></li>"
                                for s in d["sets"]) + "</ul>"
    return out


def _protein(d: dict) -> str:
    out = ""
    for p in d.get("proteins") or []:
        out += (f"<h3>{e(p['protein'])}</h3><p>Spearman with RNA: {fnum(p.get('spearman_rna'))}"
                + (" (isotype control)" if p.get("control") else "") + "</p>")
        out += hbars([(g["group"], g["mean_clr"], fnum(g["mean_clr"])) for g in p["groups"]],
                     label="Mean CLR per group")
    return out


BODIES: dict[str, Callable[[dict], str]] = {
    "identity": _identity, "labels": _labels, "enrichment": _enrichment, "qc": _qc,
    "composition": _composition, "methods": _methods, "expression": _expression,
    "databases": _databases, "per_donor": _per_donor, "coexpression": _coexpression, "hvg": _hvg,
    "gene_sets": _gene_sets, "protein": _protein,
}


def _section(s: dict, kind: str) -> str:
    status = s.get("status", "ok")
    badge = ("" if status == "ok" else
             f'<span class="status{" na" if status == "not_applicable" else ""}">{e(STATUS_TEXT.get(status, status))}</span>')
    body = ""
    if status == "ok":
        fn = (_gene_markers if kind == "gene" else _markers) if s["id"] == "markers" else BODIES.get(s["id"])
        body = fn(s.get("data") or {}) if fn else ""
    elif s["id"] == "markers" and kind == "cluster":
        body = _markers(s.get("data") or {})
    out = [f'<section id="s-{e(s["id"])}"><h2>{e(s["title"])} {badge}</h2>']
    if s.get("caption"):
        out.append(f'<p class="caption">{e(s["caption"])}</p>')
    out.append(body)
    if s.get("note"):
        out.append(f'<p class="note">{e(s["note"])}</p>')
    if s.get("method"):
        out.append(f'<p class="method">Method: {e(s["method"])}</p>')
    out.append("</section>")
    return "".join(out)


def render(report: dict) -> str:
    """The report as a complete HTML document."""
    kind = report.get("kind", "cluster")
    ds = report.get("dataset") or {}
    if kind == "cluster":
        sub = (f"{ds.get('name')}: group '{report.get('value')}' of column '{report.get('column')}', "
               f"{report.get('n_cells', 0):,} of {ds.get('n_cells', 0):,} cells.")
    else:
        sub = (f"{ds.get('name')}: gene {report.get('gene')}, cells grouped by '{report.get('groupby')}' "
               f"({ds.get('n_cells', 0):,} cells x {ds.get('n_genes', 0):,} genes).")
    sections = report.get("sections") or []
    nav = "".join(f'<a href="#s-{e(s["id"])}">{e(s["title"])}</a>' for s in sections)
    date = str(report.get("generated_at") or "")
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{e(report.get('title'))}</title><style>{CSS}</style></head><body><main>"
        f"<header><h1>{e(report.get('title'))}</h1><p class=\"sub\">{e(sub)}</p></header>"
        f"<nav>{nav}</nav>"
        + "".join(_section(s, kind) for s in sections)
        + f"<footer>Made with scView {e(report.get('scview_version'))} on {e(date[:10])} ({e(date)}). "
          "Every number comes from the dataset as it was when the report was made; captions give the cells "
          "each rests on.</footer></main></body></html>\n"
    )
