"""The gene report (C064): the evidence scView has for one gene.

Sections, in order: expression per group (mean, share expressing, n), marker
status (groups where the shared marker choice keeps it), marker databases listing
it, per-donor pseudobulk by condition, co-expressed genes, highly-variable-gene
status, MSigDB sets containing it, its protein counterpart (CITE-seq) and methods.
``core/report.py`` has the section shape and the not-computed rule.
"""

from __future__ import annotations

import difflib
import logging
import re
import threading
import time
import weakref
from collections import defaultdict

import numpy as np

from scview.core import report as R

logger = logging.getLogger(__name__)

MAX_DB_TYPES = 15
N_COEXPRESSED = 10
# Co-expression over at most this many cells (a seeded random subset beyond): over all
# 13,836 Kang cells ISG15 took 1.1-2.5 s on 2 cores, most of the report; over 10,000,
# 0.7-0.9 s with the same five top genes (r within 0.01).
MAX_COEXPRESSION_CELLS = 10_000
COEXPRESSION_SEED = 0
N_GENE_SETS = 10
MAX_DONOR_POINTS = 3000

# Antibody target (as CITE-seq panels name it) -> the genes that encode it, human then
# mouse. Keep in step with TARGET_GENES in frontend/src/lib/protein.ts.
TARGET_GENES: dict[str, list[str]] = {
    "CD3": ["CD3E", "CD3D", "CD3G", "Cd3e"], "CD4": ["CD4", "Cd4"], "CD8": ["CD8A", "Cd8a"],
    "CD8A": ["CD8A", "Cd8a"], "CD8B": ["CD8B", "Cd8b1"], "CD11B": ["ITGAM", "Itgam"],
    "CD11C": ["ITGAX", "Itgax"], "CD14": ["CD14", "Cd14"], "CD15": ["FUT4", "Fut4"],
    "CD16": ["FCGR3A", "FCGR3B", "Fcgr3"], "CD19": ["CD19", "Cd19"], "CD20": ["MS4A1", "Ms4a1"],
    "CD25": ["IL2RA", "Il2ra"], "CD27": ["CD27", "Cd27"], "CD28": ["CD28", "Cd28"],
    "CD34": ["CD34", "Cd34"], "CD38": ["CD38", "Cd38"], "CD45": ["PTPRC", "Ptprc"],
    "CD45RA": ["PTPRC", "Ptprc"], "CD45RO": ["PTPRC", "Ptprc"], "CD56": ["NCAM1", "Ncam1"],
    "CD57": ["B3GAT1", "B3gat1"], "CD62L": ["SELL", "Sell"], "CD69": ["CD69", "Cd69"],
    "CD127": ["IL7R", "Il7r"], "CD161": ["KLRB1", "Klrb1"], "CD197": ["CCR7", "Ccr7"],
    "CCR7": ["CCR7", "Ccr7"], "CD279": ["PDCD1", "Pdcd1"], "PD-1": ["PDCD1", "Pdcd1"],
    "PD1": ["PDCD1", "Pdcd1"], "PD-L1": ["CD274", "Cd274"], "TIGIT": ["TIGIT", "Tigit"],
    "HLA-DR": ["HLA-DRA"], "IGD": ["IGHD"], "IGM": ["IGHM"],
}
_TARGET_PREFIX = re.compile(r"^(anti[-_ ]?|human[-_ ]|mouse[-_ ]|hu[._]|m[._])", re.I)
_ANTI = re.compile(r"^anti[-_ ]?", re.I)


def build(adaptor, gene: str, groupby: str | None, *, dataset_name: str | None = None,
          msigdb_loader=None, marker_db_dir=None) -> dict:
    """The report for ``gene``, its expression split by ``groupby`` (default: the
    active clustering). Call in a worker thread."""
    t0 = time.perf_counter()
    adata = adaptor.adata
    gene = resolve_gene(adaptor, gene)
    if not groupby:
        groupby = getattr(adaptor, "active_clustering_column", lambda: None)()
        if not groupby:
            raise R.ReportError("Pick a column to group the cells by (no clustering is recorded).", 422)
    lab = R.grouping(adata.obs, groupby, "column (groupby)")
    condition, donor = R.design(adata.obs, groupby)
    ctx = {"adaptor": adaptor, "adata": adata, "gene": gene, "groupby": groupby, "labels": lab,
           "condition": condition, "donor": donor}
    sections = [
        R.guarded("expression", "Expression by group", lambda: _expression(ctx)),
        R.guarded("markers", "Marker status", lambda: _markers(ctx)),
        R.guarded("databases", "Marker databases", lambda: _databases(ctx, marker_db_dir)),
        R.guarded("per_donor", "Per donor", lambda: _per_donor(ctx)),
        R.guarded("coexpression", "Co-expressed genes", lambda: _coexpression(ctx)),
        R.guarded("hvg", "Highly variable gene", lambda: _hvg(ctx)),
        R.guarded("gene_sets", "Gene sets", lambda: _gene_sets(ctx, msigdb_loader)),
        R.guarded("protein", "Protein", lambda: _protein(ctx)),
    ]
    sections.append(R.guarded("methods", "Methods",
                              lambda: R.methods_section(adaptor, R.sections_methods(sections))))
    head = R.meta(adaptor, dataset_name)
    return {
        "kind": "gene",
        "title": f"{head['dataset']['name']} - gene {gene} report",
        "gene": gene,
        "groupby": groupby,
        "design": {"condition": condition, "donor": donor},
        **head,
        "sections": sections,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }


def resolve_gene(adaptor, gene: str) -> str:
    """``gene`` when the dataset has it, else a 404 with the closest name (a
    case-only difference first)."""
    gene = (gene or "").strip()
    if not gene:
        raise R.ReportError("Name a gene.", 422)
    names = adaptor.gene_names()
    if gene in set(names):
        return gene
    upper = {}
    for n in names:
        upper.setdefault(str(n).upper(), str(n))
    suggestion = upper.get(gene.upper())
    if suggestion is None:
        close = difflib.get_close_matches(gene.upper(), list(upper), n=1, cutoff=0.8)
        suggestion = upper[close[0]] if close else None
    raise R.ReportError(
        f"Gene '{gene}' not found in this dataset." + (f" Did you mean {suggestion}?" if suggestion else ""),
        404, suggestion=suggestion)


# ---------------------------------------------------------------------------
# Expression per group
# ---------------------------------------------------------------------------


def _layer_label(adaptor, layer: str) -> str:
    for entry in adaptor.available_expression_layers():
        if entry["key"] == layer:
            return f"{entry['label']}"
    return layer


def _expression(ctx: dict) -> dict:
    adaptor, gene, groupby, lab = ctx["adaptor"], ctx["gene"], ctx["groupby"], ctx["labels"]
    layer = adaptor.default_expression_layer()
    eg = adaptor.expression_by_group([gene], groupby, layer)
    counts = lab.value_counts()
    rows = [{"group": g, "n_cells": int(counts.get(g, 0)), "mean": R.num(eg["mean"][i][0]),
             "mean_expressing": R.num(eg["mean_expressing"][i][0]), "fraction": R.num(eg["fraction"][i][0])}
            for i, g in enumerate(eg["groups"])]
    values = adaptor.get_expression([gene], layer=layer)[:, 0]
    n = int(values.size)
    overall = {"n_cells": n, "mean": R.num(values.mean()) if n else None,
               "fraction": R.num((values > 0).mean()) if n else None,
               "n_expressing": int((values > 0).sum())}
    label = _layer_label(adaptor, layer)
    n_unlabelled = int(lab.isna().sum())
    caption = (f"{gene} in the {len(rows)} groups of '{groupby}': mean expression and share of cells "
               f"expressing it; {overall['n_expressing']:,} of {n:,} cells express it ({R.pct(overall['fraction'])}).")
    if n_unlabelled:
        caption += f" {n_unlabelled:,} cells have no '{groupby}' value."
    return R.section(
        "expression", "Expression by group", caption=caption,
        method=(f"Per group, the mean of the '{label}' values over all its cells (so it mixes how many "
                "cells express the gene with how much), the mean over the expressing cells (value above 0) "
                "and the share of cells above 0."),
        data={"layer": layer, "layer_label": label, "groups": rows, "overall": overall,
              "scale_range": eg.get("scale_range")})


# ---------------------------------------------------------------------------
# Marker status
# ---------------------------------------------------------------------------


def _markers(ctx: dict) -> dict:
    from scview.core.marker_selection import group_markers

    adata, gene, groupby = ctx["adata"], ctx["gene"], ctx["groupby"]
    title = "Marker status"
    key, why = R.stored_markers(adata, groupby)
    if key is None:
        return R.section("markers", title, status=R.NOT_COMPUTED, note=why,
                         caption=f"No marker genes are stored for '{groupby}'.")
    passing = group_markers(adata, groupby, n=10**6, key=key)
    rows = []
    for group, genes in passing.items():
        if gene in genes:
            s = R.marker_stats(adata, groupby, key, group, [gene])[gene]
            rows.append({**s, "group": group, "marker_rank": genes.index(gene) + 1, "n_markers": len(genes)})
    rows.sort(key=lambda r: -(r["logfc"] if r["logfc"] is not None else -np.inf))
    params = R.marker_params(adata, key)
    caption = (f"{gene} is a marker of {R.plural(len(rows), 'group')} of the {len(passing)} in '{groupby}' "
               f"({R.marker_filter_text()})" + ("." if rows else ": it passes in none."))
    return R.section(
        "markers", title, caption=caption, data={"groups": rows, "n_groups": len(passing), "test": params},
        method=(f"The stored {params['method']} test of each group against the rest ({params['corr_method']} "
                "adjusted p), filtered by scView's shared marker choice; the rank is the gene's place among "
                "the group's markers by log fold change."))


# ---------------------------------------------------------------------------
# Marker databases
# ---------------------------------------------------------------------------

_DB_INDEX: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_DB_LOCK = threading.Lock()


def _db_index(sdb) -> dict[str, list[tuple[str, int]]]:
    """Upper-case gene -> [(database type, the gene's best rank in its list)], once per
    merged species database. A list can spell one gene two ways (Gata1, GATA1): the
    type is listed once."""
    with _DB_LOCK:
        hit = _DB_INDEX.get(sdb)
        if hit is not None:
            return hit
        index: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for t in sdb.names:
            seen: set[str] = set()
            for i, g in enumerate(sdb.types[t]["genes"]):
                key = str(g).upper()
                if key not in seen:
                    seen.add(key)
                    index[key].append((t, i + 1))
        _DB_INDEX[sdb] = dict(index)
        return _DB_INDEX[sdb]


def _databases(ctx: dict, marker_db_dir) -> dict:
    from scview.core import marker_db

    adata, gene = ctx["adata"], ctx["gene"]
    title = "Marker databases"
    db, why = R.marker_db_status(marker_db_dir)
    if db is None:
        return R.section("databases", title, status=R.NOT_COMPUTED, note=why,
                         caption=f"Which database cell types list {gene} is not known yet.")
    sp, sp_note = marker_db._species(adata, None)
    parts = db.view("suggest", sp)
    sdb = db.species(sp, parts)
    hits = sorted(_db_index(sdb).get(gene.upper(), []), key=lambda h: (h[1], h[0]))
    rows = [{"cell_type": t, "rank": rank, "n_genes": len(sdb.types[t]["genes"]),
             "in_top": rank <= marker_db.TOP_DB, "cl_id": sdb.types[t]["cl"] or None,
             "sources": [marker_db.PART_LABELS.get(s, s) for s in sdb.types[t]["sources"]]}
            for t, rank in hits]
    sources = ", ".join(db.labels(parts))
    caption = (f"{R.plural(len(rows), 'cell type')} of the {len(sdb.types):,} {sp} types in {sources} list "
               f"{gene} among their markers" + (f"; the first {min(len(rows), MAX_DB_TYPES)} by how high it "
                                                 "ranks in each type's list." if rows else "."))
    return R.section(
        "databases", title, caption=caption, note=sp_note,
        method=(f"Looked up in the curated marker databases ({sources}), merged by cell-type name as for the "
                f"marker-database check's suggestions; rank 1 is a type's best-evidenced gene, and the check "
                f"uses each type's top {marker_db.TOP_DB}."),
        data={"types": rows[:MAX_DB_TYPES], "n_types": len(rows), "n_db_types": len(sdb.types),
              "species": sp, "sources": db.labels(parts)})


# ---------------------------------------------------------------------------
# Per donor
# ---------------------------------------------------------------------------


def _per_donor(ctx: dict) -> dict:
    from scview.core.donor_expression import DonorExpressionError, per_donor_expression

    adata, gene, groupby, condition, donor = (ctx["adata"], ctx["gene"], ctx["groupby"],
                                              ctx["condition"], ctx["donor"])
    title = "Per donor"
    if donor is None:
        return R.section("per_donor", title, status=R.NOT_APPLICABLE,
                         caption=f"{gene} per donor cannot be shown.",
                         note=("No donor or sample column was found (one value per person or sample, "
                               "2+ donors in every condition)."))
    try:
        res = per_donor_expression(adata, gene, groupby, sample_key=donor, condition_key=condition)
    except DonorExpressionError as exc:
        return R.section("per_donor", title, status=R.NOT_APPLICABLE, caption=f"{gene} per donor.",
                         note=str(exc))
    points = res["points"]
    summary = []
    for g in res["groups"]:
        for c in res["conditions"]:
            vals = [p["value"] for p in points if p["group"] == g and p["condition"] == c]
            if vals:
                summary.append({"group": g, "condition": c, "n_donors": len(vals),
                                "median": R.num(np.median(vals)), "min": R.num(min(vals)),
                                "max": R.num(max(vals))})
    tests = {g: {k: (R.p_value(v) if k.endswith("_p") else v) for k, v in t.items()}
             for g, t in (res.get("paired_tests") or {}).items()}
    n_donors = len({p["sample"] for p in points})
    cond_text = f" in each '{condition}' group" if condition else ""
    caption = (f"{gene} per donor{cond_text}: one point per donor sample of at least {res['min_cells']} cells "
               f"in each '{groupby}' group ({R.plural(len(points), 'sample')} from "
               f"{R.plural(n_donors, 'donor')}).")
    if res["n_dropped"]:
        caption += f" {R.plural(res['n_dropped'], 'sample')} with fewer cells left out."
    design = res["design"]
    note = " ".join(res.get("warnings") or []) or None
    if design.get("paired"):
        paired = (f"Paired design: {design['n_pairs']} donors were sampled in every '{condition}' group, so "
                  "each donor is its own control" + ("; the table gives a Wilcoxon signed-rank and a paired "
                                                      "t-test per group." if tests else "."))
        note = f"{paired} {note}" if note else paired
    return R.section(
        "per_donor", title, caption=caption, note=note,
        method=(f"Pseudobulk per donor sample: the {res['value_label']} ({res['source']}); donors from "
                f"'{res['sample_key']}'."),
        data={"sample_key": res["sample_key"], "condition_key": res["condition_key"],
              "groups": res["groups"], "conditions": res["conditions"], "summary": summary,
              "points": points[:MAX_DONOR_POINTS], "n_points": len(points), "design": design,
              "paired_tests": tests, "value_label": res["value_label"], "min_cells": res["min_cells"]})


# ---------------------------------------------------------------------------
# Co-expression
# ---------------------------------------------------------------------------


def _coexpression(ctx: dict) -> dict:
    from scview.core.coexpression import CoexpressionError, coexpression

    adata, gene = ctx["adata"], ctx["gene"]
    title = "Co-expressed genes"
    n = int(adata.n_obs)
    rows = None
    if n > MAX_COEXPRESSION_CELLS:
        rng = np.random.default_rng(COEXPRESSION_SEED)
        rows = np.sort(rng.choice(n, MAX_COEXPRESSION_CELLS, replace=False))
    try:
        res = coexpression(adata, gene, rows=rows, top_n=N_COEXPRESSED)
    except CoexpressionError as exc:
        return R.section("coexpression", title, status=R.NOT_APPLICABLE,
                         caption=f"No co-expression for {gene}.", note=str(exc))
    cells = (f"a random {res['n_cells']:,} of the {n:,} cells (seed {COEXPRESSION_SEED}); the "
             "Co-expression tool in the Expression tab uses every cell" if rows is not None
             else f"all {n:,} cells")
    caption = (f"The {len(res['positive'])} genes most positively and {len(res['negative'])} most negatively "
               f"correlated with {gene} over {cells}; {res['n_genes_tested']:,} genes tested.")
    return R.section(
        "coexpression", title, caption=caption,
        method=(f"Pearson correlation over cells of the {res['values']} values ({res['layer']}); a gene needs "
                f"at least {res['min_expressing'] or 0:,} expressing cells to be tested."),
        data={"positive": res["positive"], "negative": res["negative"], "n_cells": res["n_cells"],
              "n_cells_total": n, "subsampled": rows is not None, "n_genes_tested": res["n_genes_tested"],
              "layer": res["layer"], "values": res["values"]})


# ---------------------------------------------------------------------------
# Highly variable gene
# ---------------------------------------------------------------------------

_HVG_FIELDS = (("means", "mean"), ("dispersions", "dispersion"), ("dispersions_norm", "normalised dispersion"),
               ("variances", "variance"), ("variances_norm", "normalised variance"),
               ("highly_variable_rank", "rank"))


def _hvg(ctx: dict) -> dict:
    adata, gene = ctx["adata"], ctx["gene"]
    title = "Highly variable gene"
    var = adata.var
    if "highly_variable" not in var.columns:
        return R.section("hvg", title, status=R.NOT_COMPUTED,
                         caption="No highly-variable-gene selection is stored.",
                         note=f"Not computed yet: select highly variable genes in {R.WHERE['hvg']}.")
    hv = var["highly_variable"].astype(str).str.lower().isin(("true", "1"))
    n_hvg, n_vars = int(hv.sum()), int(var.shape[0])
    flavor = None
    try:
        flavor = str(adata.uns["hvg"].get("flavor")) if "hvg" in adata.uns else None
    except AttributeError:
        flavor = None
    stats: list[dict] = []
    if gene in var.index:
        is_hvg = bool(hv.loc[gene])
        for col, label in _HVG_FIELDS:
            if col in var.columns:
                stats.append({"field": col, "label": label, "value": R.num(var.loc[gene, col])})
        note = None
    else:
        is_hvg = False
        n_raw = int(adata.raw.n_vars) if adata.raw is not None else None
        note = (f"{gene} is not among X's {n_vars:,} genes" + (f" (.raw keeps all {n_raw:,})" if n_raw else "")
                + ", so it was not selected.")
    caption = (f"{gene} is {'' if is_hvg else 'not '}one of the {n_hvg:,} highly variable genes of the "
               f"{n_vars:,} genes in X.")
    return R.section(
        "hvg", title, caption=caption, note=note,
        method=("The stored highly-variable-gene selection (var['highly_variable']"
                + (f", flavour {flavor}" if flavor and flavor != "None" else "") + ") and its statistics."),
        data={"is_hvg": is_hvg, "n_hvg": n_hvg, "n_genes": n_vars, "flavor": flavor, "stats": stats})


# ---------------------------------------------------------------------------
# Gene sets
# ---------------------------------------------------------------------------

_GS_INDEX: dict[tuple, tuple[list[tuple[str, str]], dict[str, np.ndarray]]] = {}
_GS_LOCK = threading.Lock()


def _gene_set_collections(loader) -> list[str]:
    from scview.core.msigdb_loader import DEFAULT_COLLECTIONS

    return [c for c in DEFAULT_COLLECTIONS if loader._find_file(c) is not None]


def _gs_index(loader, collections: list[str]) -> tuple[list[tuple[str, str]], dict[str, np.ndarray]]:
    """(the sets as (collection, name), upper-case gene -> indices of the sets
    holding it), built once per MSigDB directory and collection list."""
    key = (str(loader.dir), tuple(collections))
    with _GS_LOCK:
        hit = _GS_INDEX.get(key)
        if hit is not None:
            return hit
        sets: list[tuple[str, str]] = []
        index: dict[str, list[int]] = defaultdict(list)
        for cid in collections:
            for name, genes in loader.get_collection_as_dict(cid).items():
                j = len(sets)
                sets.append((cid, name))
                for g in set(str(x).upper() for x in genes):
                    index[g].append(j)
        built = (sets, {g: np.asarray(v, dtype=np.int32) for g, v in index.items()})
        _GS_INDEX.clear()  # one directory / selection at a time
        _GS_INDEX[key] = built
        return built


def _gene_sets(ctx: dict, loader) -> dict:
    from scview.core.msigdb_loader import MSIGDB_COLLECTIONS
    from scview.core.species import detect_species, human_genesets_warning

    adata, gene = ctx["adata"], ctx["gene"]
    title = "Gene sets"
    collections = _gene_set_collections(loader) if loader is not None else []
    if not collections:
        return R.section("gene_sets", title, status=R.NOT_COMPUTED,
                         caption=f"The MSigDB sets holding {gene} are not known.",
                         note=("MSigDB is not on disk: scView downloads it into the msigdb volume on first "
                               "start (MSIGDB_DIR must point at it)."))
    sets, index = _gs_index(loader, collections)
    hits = index.get(gene.upper(), np.zeros(0, dtype=np.int32))
    order = {c: i for i, c in enumerate(collections)}
    named = sorted((sets[j] for j in hits), key=lambda s: (order[s[0]], s[1]))
    names = {c["id"]: c["name"] for c in MSIGDB_COLLECTIONS}
    per = defaultdict(int)
    for cid, _ in named:
        per[cid] += 1
    warning = human_genesets_warning(detect_species(adata)) or None
    caption = (f"{gene} is in {R.plural(len(named), 'set')} of the {len(sets):,} in the default MSigDB "
               f"collections" + (f"; the first {min(N_GENE_SETS, len(named))}, HALLMARK first." if named else "."))
    return R.section(
        "gene_sets", title, caption=caption, note=warning,
        method=("Membership in the MSigDB collections the Enrichment tab uses by default ("
                + ", ".join(names.get(c, c) for c in collections) + "), matched ignoring case."),
        data={"sets": [{"collection": c, "collection_name": names.get(c, c), "name": n}
                       for c, n in named[:N_GENE_SETS]],
              "n_sets": len(named), "n_searched": len(sets),
              "by_collection": [{"collection": c, "collection_name": names.get(c, c), "n_sets": per[c]}
                                for c in collections if per[c]],
              "collections": collections})


# ---------------------------------------------------------------------------
# Protein (CITE-seq)
# ---------------------------------------------------------------------------


def target_genes(protein_name: str) -> list[str]:
    """Genes that may encode an antibody's target, most likely first (the name's
    own spelling last), as ``targetGenes`` in frontend/src/lib/protein.ts."""
    key = _TARGET_PREFIX.sub("", protein_name).upper()
    own = _ANTI.sub("", protein_name)
    return list(dict.fromkeys([*TARGET_GENES.get(key, []), own, own.upper()]))


def _protein(ctx: dict) -> dict:
    from scipy import stats

    from scview.core import protein

    adaptor, adata, gene, groupby, lab = (ctx["adaptor"], ctx["adata"], ctx["gene"], ctx["groupby"],
                                          ctx["labels"])
    title = "Protein"
    if not protein.has_protein(adata):
        return R.section("protein", title, status=R.NOT_APPLICABLE,
                         caption=f"No protein measurement for {gene}.",
                         note="This dataset has no protein (CITE-seq antibody) measurements.")
    g = gene.upper()
    matches = [p for p in protein.protein_names(adata) if g in {t.upper() for t in target_genes(p)}]
    if not matches:
        return R.section("protein", title, status=R.NOT_APPLICABLE,
                         caption=f"No antibody in the panel targets {gene}'s product.",
                         note=f"The panel has {len(protein.protein_names(adata))} proteins; none maps to {gene}.")
    rna = adaptor.get_expression([gene])[:, 0]
    groups = R.group_order(adata.obs[groupby])
    labels = lab.to_numpy()
    out = []
    for p in matches:
        clr = protein.values(adata, p).astype(np.float64)
        rho = None
        if np.ptp(clr) > 0 and np.ptp(rna) > 0:
            rho = R.num(stats.spearmanr(clr, rna).statistic)
        out.append({"protein": p, "control": protein.is_control(p), "spearman_rna": rho,
                    "groups": [{"group": grp, "mean_clr": R.num(clr[labels == grp].mean())
                                if (labels == grp).any() else None} for grp in groups]})
    caption = (f"{R.plural(len(out), 'antibody', 'antibodies')} measure the protein {gene} encodes: mean CLR "
               f"per '{groupby}' group, and the rank correlation with its RNA over {rna.size:,} cells.")
    return R.section(
        "protein", title, caption=caption,
        method=("Antibody counts normalised by centred log-ratio per cell (Seurat CLR); antibodies matched to the "
                "gene by target name; Spearman correlation of protein and RNA over cells."),
        data={"proteins": out})
