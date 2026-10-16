"""Gene sets for an analysis, drawn from MSigDB and the user's custom collections.

The enrichment endpoints and the pipeline's enrichment step take a list of
collection ids. MSigDB ids ("h.all") come from ``MSigDBLoader``; custom ids
("custom-...") from ``CustomGeneSetStore``. This module merges them into the
single {term: genes} dict gseapy wants, remembers which collection each term
came from, and labels results so a custom collection is shown by its name, or
as deleted once it is gone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from scview.core.custom_genesets import CustomGeneSetStore, coverage, is_custom_id

logger = logging.getLogger(__name__)

# gseapy calls a gene list upper-case when at least 90% of it is.
_UPPER_FRACTION = 0.9


@dataclass
class ResolvedGeneSets:
    gene_sets: dict[str, list[str]] = field(default_factory=dict)
    # term -> id of the collection it came from
    term_collection: dict[str, str] = field(default_factory=dict)
    # terms from custom collections (their genes are stored upper-case)
    custom_terms: set[str] = field(default_factory=set)
    # custom ids that were asked for but no longer exist
    missing_custom: list[str] = field(default_factory=list)
    # custom id -> its name when resolved
    custom_names: dict[str, str] = field(default_factory=dict)
    # MSigDB ids that were asked for while MSigDB is not configured
    msigdb_unavailable: bool = False


def resolve_gene_sets(
    collection_ids: Iterable[str],
    msigdb_loader: Any,
    custom_store: CustomGeneSetStore | None,
) -> ResolvedGeneSets:
    """Merge the requested collections into one gseapy gene-set dict.

    MSigDB collections go first whatever the request order, so an MSigDB term
    always keeps its plain name: a custom set reusing it (a custom copy of
    Hallmark, say) is kept under "<term> (<collection>)" instead of silently
    replacing it.
    """
    out = ResolvedGeneSets()
    seen: set[str] = set()
    ids = list(collection_ids)
    ordered = [c for c in ids if not is_custom_id(c)] + [c for c in ids if is_custom_id(c)]
    for cid in ordered:
        if cid in seen:
            continue
        seen.add(cid)
        custom = is_custom_id(cid)
        if custom:
            meta = custom_store.get_meta(cid) if custom_store else None
            sets = custom_store.get_sets(cid) if (custom_store and meta) else None
            if meta is None or sets is None:
                out.missing_custom.append(cid)
                continue
            label = meta["name"]
            out.custom_names[cid] = label
        else:
            if msigdb_loader is None:
                out.msigdb_unavailable = True
                continue
            sets = msigdb_loader.get_collection_as_dict(cid)
            label = cid
        for term, genes in sets.items():
            key = term
            if key in out.gene_sets:
                key = f"{term} ({label})"
                n = 2
                while key in out.gene_sets:
                    key = f"{term} ({label} {n})"
                    n += 1
            out.gene_sets[key] = genes
            out.term_collection[key] = cid
            if custom:
                out.custom_terms.add(key)
    return out


def _mostly_upper(genes: Iterable[str]) -> bool:
    flags = [str(g).isupper() for g in genes]
    return bool(flags) and sum(flags) / len(flags) >= _UPPER_FRACTION


def match_query_case(
    resolved: ResolvedGeneSets, query: Iterable[str], universe: Iterable[str] | None
) -> dict[str, list[str]]:
    """The gene-set dict with custom genes spelled the way gseapy will compare them.

    gseapy compares genes exactly when the query is mostly upper-case (human),
    and upper-cases the query and background when it is not (mouse) and the
    sets are upper-case, as MSigDB's and the stored custom ones are. So for a
    mouse query the stored spelling already matches. For a human query the few
    mixed-case symbols (C1orf112, stored as C1ORF112) are mapped back to the
    dataset's own spelling, which is what makes the match case-insensitive.
    """
    if not resolved.custom_terms or not universe or not _mostly_upper(query):
        return resolved.gene_sets
    mixed = {g.upper(): g for g in (str(u) for u in universe) if not g.isupper()}
    if not mixed:
        return resolved.gene_sets
    out = dict(resolved.gene_sets)
    for term in resolved.custom_terms:
        out[term] = [mixed.get(g, g) for g in out[term]]
    return out


def tag_records(records: list[dict], resolved: ResolvedGeneSets) -> list[dict]:
    """Record each result's collection (and a custom one's name) when it is computed."""
    for rec in records:
        cid = resolved.term_collection.get(rec.get("term", ""), "")
        rec["collection"] = cid
        if cid in resolved.custom_names:
            rec["collection_name"] = resolved.custom_names[cid]
    return records


def label_records(records: list[dict], custom_store: CustomGeneSetStore | None) -> list[dict]:
    """Copies of result records with custom collections labelled as they are now.

    A renamed collection shows its new name; a deleted one keeps the name it
    had when the result was computed, marked "(deleted)", so a cached result
    never silently points at nothing.
    """
    out = []
    names: dict[str, str | None] = {}
    for rec in records:
        cid = rec.get("collection", "")
        if not is_custom_id(cid):
            out.append(rec)
            continue
        if cid not in names:
            meta = custom_store.get_meta(cid) if custom_store else None
            names[cid] = meta["name"] if meta else None
        current = names[cid]
        rec = dict(rec)
        if current is not None:
            rec["collection_name"] = current
            rec["collection_deleted"] = False
        else:
            was = rec.get("collection_name") or "Custom collection"
            rec["collection_name"] = f"{was} (deleted)"
            rec["collection_deleted"] = True
        out.append(rec)
    return out


def dataset_gene_names(adaptor: Any) -> set[str]:
    """Every gene a dataset measures: its expression source plus var_names."""
    names = {str(g) for g in adaptor.gene_names()}
    names.update(str(g) for g in adaptor.adata.var_names)
    return names


def custom_collections_for_dataset(adaptor: Any, custom_store: CustomGeneSetStore) -> list[dict]:
    """Custom collections' metadata plus how much of each the dataset measures."""
    metas = custom_store.list()
    if not metas:
        return []
    genes = dataset_gene_names(adaptor)
    out = []
    for meta in metas:
        sets = custom_store.get_sets(meta["id"]) or {}
        out.append({**meta, **coverage(sets, genes)})
    return out
