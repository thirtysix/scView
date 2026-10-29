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
from scview.core.msigdb_loader import collection_species

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
    # MSigDB ids written for another species than the dataset's (genes_for_species ran)
    cross_species: list[str] = field(default_factory=list)
    # term -> (source id, details URL) from MSigDB's exactSource / externalDetailsURL
    term_source: dict[str, tuple[str, str]] = field(default_factory=dict)


def genes_for_species(
    sets: dict[str, list[str]], set_species: str, dataset_species: str
) -> dict[str, list[str]]:
    """THE place a gene-set library written for one species is made to fit a dataset
    of another. Human sets on a mouse dataset are mapped through the HCOP ortholog
    table (``core/orthologs.py``, decoupler's rules); a set with no mapped gene is
    dropped. Without the table (offline, switched off), and for mouse sets on a human
    dataset, the genes are kept and matched ignoring case downstream
    (``match_query_case``, gseapy): a gene whose symbol differs is then missed."""
    if not (set_species == "human" and dataset_species == "mouse"):
        return sets
    from scview.core import orthologs

    try:
        table = orthologs.load()
    except orthologs.OrthologsUnavailable as e:
        logger.warning("Human gene sets on a mouse dataset matched by case: %s", e)
        return sets
    mapped, _ = orthologs.map_gene_sets(sets, table=table)
    return {name: genes for name, genes in mapped.items() if genes}


def resolve_gene_sets(
    collection_ids: Iterable[str],
    msigdb_loader: Any,
    custom_store: CustomGeneSetStore | None,
    species: str | None = None,
) -> ResolvedGeneSets:
    """Merge the requested collections into one gseapy gene-set dict.

    MSigDB collections go first whatever the request order, so an MSigDB term
    always keeps its plain name: a custom set reusing it (a custom copy of
    Hallmark, say) is kept under "<term> (<collection>)" instead of silently
    replacing it. With the dataset's ``species`` (human or mouse), that species'
    collections go first (MH and H share their set names, so on a mouse dataset
    MH keeps the plain ones) and another species' pass through
    ``genes_for_species``.
    """
    out = ResolvedGeneSets()
    seen: set[str] = set()
    ids = list(collection_ids)
    msig = [c for c in ids if not is_custom_id(c)]
    if species in ("human", "mouse"):
        msig = ([c for c in msig if collection_species(c) in (species, None)]
                + [c for c in msig if collection_species(c) not in (species, None)])
    ordered = msig + [c for c in ids if is_custom_id(c)]
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
            get_sources = getattr(msigdb_loader, "get_sources", None)
            sources = get_sources(cid) if get_sources else {}
            label = cid
            set_species = collection_species(cid)
            if species in ("human", "mouse") and set_species and set_species != species:
                sets = genes_for_species(sets, set_species, species)
                out.cross_species.append(cid)
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
            elif term in sources:
                out.term_source[key] = sources[term]
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

    gseapy decides that from the first 10-20 sets only, so a selection mixing
    spellings (mouse MSigDB beside human MSigDB or custom sets, or a library
    passed through ``genes_for_species``) would lose one side. Then every set is
    spelled the dataset's way (each gene by the universe's or query's name that
    matches it ignoring case), and gseapy compares exactly.
    """
    mixed_species = resolved.cross_species or any(
        collection_species(c) == "mouse" for c in set(resolved.term_collection.values())
    )
    if mixed_species and universe:
        lookup: dict[str, str] = {}
        for g in (*query, *universe):
            lookup.setdefault(str(g).upper(), str(g))
        return {
            term: [lookup.get(str(g).upper(), g) for g in genes]
            for term, genes in resolved.gene_sets.items()
        }
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
        rec["species"] = collection_species(cid) or ""
        if cid in resolved.custom_names:
            rec["collection_name"] = resolved.custom_names[cid]
        sid, url = resolved.term_source.get(rec.get("term", ""), ("", ""))
        rec["source_id"], rec["source_url"] = sid, url
    return records


def label_records(
    records: list[dict], custom_store: CustomGeneSetStore | None, msigdb_loader: Any = None
) -> list[dict]:
    """Copies of result records with custom collections labelled as they are now.

    A renamed collection shows its new name; a deleted one keeps the name it
    had when the result was computed, marked "(deleted)", so a cached result
    never silently points at nothing. With ``msigdb_loader``, an MSigDB record
    cached before records carried their source (R-HSA-..., WP..., N...) gets it.
    """
    out = []
    names: dict[str, str | None] = {}
    sources: dict[str, dict[str, tuple[str, str]]] = {}
    for rec in records:
        cid = rec.get("collection", "")
        if not is_custom_id(cid):
            if "species" not in rec and collection_species(cid):  # cached before species
                rec = {**rec, "species": collection_species(cid)}
            if "source_id" not in rec and msigdb_loader is not None and collection_species(cid):
                if cid not in sources:
                    get_sources = getattr(msigdb_loader, "get_sources", None)
                    sources[cid] = get_sources(cid) if get_sources else {}
                sid, url = sources[cid].get(rec.get("term", ""), ("", ""))
                rec = {**rec, "source_id": sid, "source_url": url}
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
