"""Collapse a redundant enrichment result into a term-similarity network.

Gene-set collections are heavily redundant: GO:BP ships 7,538 terms arranged in a
DAG where a child's genes are by definition also its parent's. A top-N chart is a
poor summary of that, because the largest parent terms carry the most statistical
power and so crowd the top slots with near-duplicates of one signal. This builds
the graph those duplicates imply (terms as nodes, gene overlap as edges) and lets
community detection group them, so redundancy becomes the signal rather than noise.

This is the EnrichmentMap method (Merico et al. 2010, PLoS ONE; protocol in
Reimand et al. 2019, Nat Protoc), reimplemented rather than integrated: every
mature implementation (EnrichmentMap/Cytoscape, aPEAR, clusterProfiler::emapplot)
is Java or R and would sit on the wrong side of a runtime boundary from a Python
API feeding a React client. The parameters below are EnrichmentMap's; only the
code is local.

Two choices differ from the popular blog write-ups of this method:

1. Edges use EnrichmentMap's COMBINED coefficient (0.5*Jaccard + 0.5*overlap) at
   0.375, not bare Jaccard at 0.1. Jaccard punishes size asymmetry, so a 15-gene
   term fully contained in a 500-gene term scores 15/500 = 0.03 and gets no edge
   at all, despite being the most redundant pair possible. Measured on the GO:BP
   collection bundled with scView (5,268 terms after a 10-500 gene filter), 33% of
   pairs with overlap coefficient >= 0.9 fall below Jaccard 0.1. The overlap
   coefficient (|A&B| / min(|A|,|B|)) is 1.0 for containment at any size, so the
   combination catches parent/child redundancy that Jaccard alone misses.

2. Similarity is computed on each term's OVERLAP genes (the query genes driving
   the enrichment), not its full annotation. That makes the graph reflect this
   experiment rather than the ontology's shape.

Two caveats to carry to whoever reads the output:

Cluster SIZE tracks how finely curators subdivided that branch of the ontology
(immune is annotated far more densely than most metabolism), not biological
importance. Read cluster membership and existence; do not read cluster size as
effect size. Nor does this correct the multiple-testing dependence between nested
terms; it only makes it visible.

The cluster COUNT is a parameter, not a discovery. Popular write-ups of this
method report that an interpretation "emerged from the network structure itself",
which overstates it: sweeping only the resolution, on identical input, moves the
answer from a handful of programs to well over a hundred (34 -> 144 across
resolution 0.2 -> 0.9 on the bundled ovary data). The metric, threshold,
resolution and seed are all choices; they are better choices than per-term
cherry-picking because they are global and disclosed, but they are not an absence
of choice. Tune it and say what you tuned it to.

Clustering uses CPM, not RB/modularity. leidenalg's default
RBConfigurationVertexPartition penalises a community by its size, so above
resolution ~1 it shatters cliques: on the ovary sample data six terms whose
overlap-gene sets were byte-identical (similarity exactly 1.0) were assigned to
six different clusters, which is precisely the redundancy this feature exists to
collapse. CPM's resolution is a density threshold instead, so a clique survives at
any setting, and the parameter lands on the same 0-1 scale as the similarity
weights (Traag et al. 2019 recommend CPM for exactly this reason).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Iterable

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)

# EnrichmentMap's published defaults for the combined coefficient.
DEFAULT_METRIC = "combined"
DEFAULT_MIN_SIMILARITY = 0.375

# Resolution is a CPM density threshold, so it lives on the same 0-1 scale as the
# similarity weights: a community must hold together above roughly this mean
# internal similarity. Slightly above the edge threshold is a sensible default.
# NOT comparable to a modularity/RB resolution, which is unbounded and unitless.
DEFAULT_RESOLUTION = 0.4
DEFAULT_FDR = 0.05
DEFAULT_SEED = 0

# Conventional ORA size bounds. GO:BP contains terms from 1 to ~2000 genes, and
# scView's enrichment applies no size filter at all, so without this the network
# is dominated by enormous vague parents ("homeostatic process", n=1543) that
# overlap everything and cluster into uninformative blobs.
DEFAULT_MIN_TERM_SIZE = 10
DEFAULT_MAX_TERM_SIZE = 500

METRICS = ("combined", "jaccard", "overlap")


def _similarity(
    sets: list[set[str]], metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense pairwise similarity between term gene sets.

    Vectorised rather than the O(n^2) Python loop in gseapy's DotPlot.to_edgelist,
    which is the only comparable helper already vendored here.
    """
    vocab = sorted({g for s in sets for g in s})
    index = {g: i for i, g in enumerate(vocab)}
    rows, cols = [], []
    for r, s in enumerate(sets):
        for g in s:
            rows.append(r)
            cols.append(index[g])
    n = len(sets)
    M = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, max(len(vocab), 1)),
    )
    inter = np.asarray((M @ M.T).todense(), dtype=np.float64)
    size = np.asarray([len(s) for s in sets], dtype=np.float64)

    union = size[:, None] + size[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, inter / union, 0.0)
        smaller = np.minimum(size[:, None], size[None, :])
        overlap = np.where(smaller > 0, inter / smaller, 0.0)

    if metric == "jaccard":
        sim = jaccard
    elif metric == "overlap":
        sim = overlap
    else:
        sim = 0.5 * jaccard + 0.5 * overlap

    np.fill_diagonal(sim, 0.0)
    np.fill_diagonal(inter, 0.0)
    return sim, inter, jaccard


def _leiden(
    n: int, edges: list[tuple[int, int]], weights: list[float], resolution: float, seed: int
) -> tuple[list[int], list[tuple[float, float]]]:
    """Community assignment and layout. Falls back gracefully if igraph is absent."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError:  # pragma: no cover - igraph ships in the backend image
        logger.warning("igraph/leidenalg unavailable; returning a single cluster")
        return [0] * n, [(0.0, 0.0)] * n

    g = ig.Graph(n=n, edges=edges)
    g.es["weight"] = weights

    # Leiden and Fruchterman-Reingold are both stochastic; seed for reproducibility.
    prev = ig.set_random_number_generator(random.Random(seed))
    try:
        part = leidenalg.find_partition(
            g,
            leidenalg.CPMVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=seed,
        )
        membership = list(part.membership)
        layout = g.layout_fruchterman_reingold(weights="weight") if n > 1 else [(0.0, 0.0)]
        coords = [(float(x), float(y)) for x, y in layout]
    finally:
        try:
            ig.set_random_number_generator(prev if prev is not None else random)
        except Exception:  # pragma: no cover - restoring RNG must never fail a request
            pass
    return membership, coords


def build_network(
    results: Iterable[dict[str, Any]],
    *,
    fdr: float = DEFAULT_FDR,
    metric: str = DEFAULT_METRIC,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    resolution: float = DEFAULT_RESOLUTION,
    seed: int = DEFAULT_SEED,
    min_term_size: int = DEFAULT_MIN_TERM_SIZE,
    max_term_size: int = DEFAULT_MAX_TERM_SIZE,
    max_terms: int | None = 1500,
) -> dict[str, Any]:
    """Build a term-similarity network from enrichment results.

    ``results`` are EnrichmentResult-shaped dicts; ``genes`` must hold the overlap
    genes (gseapy's ``Genes`` column), which is what scView already stores.

    Returns nodes (with cluster, layout position and hub flag), edges, and one
    entry per cluster labelled by its most significant term. ``truncated`` and
    ``n_size_filtered`` report anything dropped, so a bounded view is never
    mistaken for a complete one.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    # CPM's resolution is a density threshold against similarity weights, which
    # cannot exceed 1. At resolution >= 1 no community can ever satisfy it and every
    # term comes back a singleton, silently. Reject rather than return that.
    if not 0.0 < resolution < 1.0:
        raise ValueError(
            f"resolution must be between 0 and 1 exclusive (a CPM density threshold "
            f"on similarity weights), got {resolution!r}"
        )

    sig = [
        r
        for r in results
        if r.get("adjusted_pvalue") is not None
        and float(r["adjusted_pvalue"]) < fdr
        and r.get("genes")
    ]

    before_size_filter = len(sig)
    sig = [
        r
        for r in sig
        if min_term_size <= int(r.get("gene_count") or 0) <= max_term_size
    ]
    n_size_filtered = before_size_filter - len(sig)

    sig.sort(key=lambda r: float(r["adjusted_pvalue"]))

    truncated = 0
    if max_terms is not None and len(sig) > max_terms:
        truncated = len(sig) - max_terms
        logger.info(
            "Enrichment network truncated to the %d most significant of %d terms",
            max_terms,
            len(sig),
        )
        sig = sig[:max_terms]

    if not sig:
        return {
            "nodes": [], "edges": [], "clusters": [],
            "n_terms": 0, "n_clusters": 0, "truncated": truncated,
            "n_size_filtered": n_size_filtered,
            "params": {"fdr": fdr, "metric": metric, "min_similarity": min_similarity,
                       "resolution": resolution, "seed": seed,
                       "min_term_size": min_term_size, "max_term_size": max_term_size},
        }

    sets = [set(r["genes"]) for r in sig]
    sim, inter, jaccard = _similarity(sets, metric)

    iu = np.triu_indices(len(sig), 1)
    keep = sim[iu] >= min_similarity
    src, dst = iu[0][keep], iu[1][keep]
    weights = sim[iu][keep]

    membership, coords = _leiden(
        len(sig), list(zip(src.tolist(), dst.tolist())), weights.tolist(), resolution, seed
    )

    degree = np.zeros(len(sig))
    for s, d, w in zip(src, dst, weights):
        degree[s] += w
        degree[d] += w

    clusters: dict[int, list[int]] = {}
    for i, c in enumerate(membership):
        clusters.setdefault(c, []).append(i)

    # Label each cluster by its most significant term. Weighted degree is the
    # intuitive choice and is wrong here: in a redundancy graph the best-connected
    # term is the most GENERIC one, because generic terms overlap everything. On
    # the bundled Kang data it labelled a 114-term monocyte cluster
    # "negative regulation of immune system process", where significance picks
    # "antigen processing and presentation of exogenous peptide antigen".
    # Smallest-term-size was also tried and is worse still: it surfaces incidental
    # tiny terms ("microglial cell mediated cytotoxicity" in monocytes).
    # Degree is retained per node so a caller can rank within a cluster.
    hubs: dict[int, int] = {}
    for c, members in clusters.items():
        hubs[c] = min(members, key=lambda i: (float(sig[i]["adjusted_pvalue"]), -degree[i]))

    nodes = [
        {
            "id": i,
            "term": sig[i].get("term", ""),
            "adjusted_pvalue": float(sig[i]["adjusted_pvalue"]),
            "overlap_count": int(sig[i].get("overlap_count", len(sets[i]))),
            "gene_count": int(sig[i].get("gene_count", 0)),
            "collection": sig[i].get("collection", ""),
            "genes": sorted(sets[i]),
            "cluster": int(membership[i]),
            "is_hub": hubs[membership[i]] == i,
            "degree": float(degree[i]),
            "x": coords[i][0],
            "y": coords[i][1],
        }
        for i in range(len(sig))
    ]

    edges = [
        {
            "source": int(s),
            "target": int(d),
            "weight": float(w),
            "jaccard": float(jaccard[s, d]),
            "n_shared": int(inter[s, d]),
            "same_cluster": bool(membership[s] == membership[d]),
        }
        for s, d, w in zip(src.tolist(), dst.tolist(), weights.tolist())
    ]

    # Rank programs by strength, not by size. Sorting by size contradicts the
    # caveat above and misleads in practice: on the bundled ovary data it put a
    # 57-term cell-type-signature program (adj.P 1e-13) above a translation
    # program at 1e-101, purely because the ontology subdivides one branch more
    # finely than the other. Size stays available per row as a property.
    cluster_rows = sorted(
        (
            {
                "cluster": int(c),
                "label": sig[hubs[c]].get("term", ""),
                "n_terms": len(members),
                "best_adjusted_pvalue": min(float(sig[i]["adjusted_pvalue"]) for i in members),
                "genes": sorted({g for i in members for g in sets[i]}),
                # Members strongest-first, so expanding a program leads with its
                # best evidence rather than an arbitrary order.
                "terms": [
                    sig[i].get("term", "")
                    for i in sorted(members, key=lambda i: float(sig[i]["adjusted_pvalue"]))
                ],
            }
            for c, members in clusters.items()
        ),
        key=lambda r: (r["best_adjusted_pvalue"], -r["n_terms"]),
    )

    return {
        "nodes": nodes,
        "edges": edges,
        "clusters": cluster_rows,
        "n_terms": len(sig),
        "n_clusters": len(cluster_rows),
        "truncated": truncated,
        "n_size_filtered": n_size_filtered,
        "params": {
            "fdr": fdr, "metric": metric, "min_similarity": min_similarity,
            "resolution": resolution, "seed": seed,
            "min_term_size": min_term_size, "max_term_size": max_term_size,
        },
    }
