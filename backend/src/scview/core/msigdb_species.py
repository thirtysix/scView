"""Which MSigDB collections fit a dataset's species, and what to say when they don't.

A mouse dataset (``core/species.py``) gets MSigDB's mouse collections by default
(MH, M2 Reactome/WikiPathways, M5 GO, M8: mouse symbols); the human collections
stay selectable after them, matched ignoring case. A human or unknown dataset is
unchanged: the human collections, in the order they always had. The warning about
human symbols on mouse genes shows only when a human collection (or a human-only
Enrichr library) is actually used on a mouse dataset.
"""

from __future__ import annotations

from collections.abc import Iterable

from scview.core.msigdb_loader import ALL_COLLECTIONS, collection_species, defaults_for

_BY_ID = {c["id"]: c for c in ALL_COLLECTIONS}


def dataset_species(adata) -> str:
    """"human", "mouse" or "unknown" (``species.detect_species``)."""
    from scview.core.species import detect_species

    return detect_species(adata).species


def resolve(collections: Iterable[str] | None, species: str | None) -> list[str]:
    """The request's collections, or the species' defaults when it sent none."""
    if collections is None:
        return defaults_for(species)
    return list(collections)


def cross_species(collections: Iterable[str], species: str | None) -> list[str]:
    """MSigDB ids whose species is not the dataset's (none when the dataset's is unknown)."""
    if species not in ("human", "mouse"):
        return []
    return [c for c in collections if (s := collection_species(c)) and s != species]


def species_used(collections: Iterable[str]) -> list[str]:
    """The species of the MSigDB collections in ``collections``, each once."""
    return list(dict.fromkeys(s for c in collections if (s := collection_species(c))))


def short_label(collection_id: str) -> str:
    """"MH", "M2 Reactome", "C5 GO Biological Process": the category, plus the
    collection's name when the category holds several."""
    c = _BY_ID.get(str(collection_id).lower())
    if c is None:
        return str(collection_id)
    return c["category"] if not c["subcategory"] else f"{c['category']} {c['name']}"


def describe(collections: Iterable[str], species: str | None = None) -> str:
    """One phrase naming the gene sets by species, e.g. "mouse MSigDB MH, M2 Reactome;
    human MSigDB H (human symbols, case-matched)"; custom collections are counted."""
    ids = list(dict.fromkeys(collections))
    parts = []
    for sp in ("mouse", "human"):
        # In the registry's order (Hallmark first), whatever order the ids came in.
        mine = [short_label(c["id"]) for c in ALL_COLLECTIONS if c["species"] == sp and c["id"] in ids]
        if mine:
            note = " (human symbols, case-matched)" if sp == "human" and species == "mouse" else ""
            note = " (mouse symbols, case-matched)" if sp == "mouse" and species == "human" else note
            parts.append(f"{sp} MSigDB {', '.join(mine)}{note}")
    n_custom = sum(1 for c in ids if collection_species(c) is None)
    if n_custom:
        parts.append(f"{n_custom} custom collection{'' if n_custom == 1 else 's'}")
    return "; ".join(parts)


def warning(species: str | None, collections: Iterable[str] = (), *, human_library: bool = False) -> str:
    """A note when the gene sets are written for another species than the dataset's:
    human MSigDB collections (or a human-only Enrichr library, ``human_library``) on a
    mouse dataset, or mouse collections on a human one. Empty otherwise."""
    from scview.core.species import SpeciesCall, human_genesets_warning

    other = cross_species(collections, species)
    if species == "mouse" and (other or human_library):
        named = ", ".join(short_label(c) for c in other)
        text = human_genesets_warning(SpeciesCall(species="mouse"))
        if named:
            text = text.replace("but the gene sets are human gene symbols",
                                f"but {named} {'is' if len(other) == 1 else 'are'} human gene symbols")
        return text + " The mouse MSigDB collections (MH, M2, M5, M8) use mouse symbols."
    if species == "human" and other:
        return ("This looks like a human dataset, but " + ", ".join(short_label(c) for c in other)
                + (" is" if len(other) == 1 else " are") + " mouse gene symbols, matched ignoring case: "
                "a human gene whose symbol differs from the mouse one is missed. Use the human "
                "collections for human data.")
    return ""


def enrichment_fact(adata) -> str:
    """For the co-pilot: which species' gene sets the stored enrichment results used,
    from their provenance sidecars ("mouse MSigDB MH, M5 GO Biological Process"),
    plus a caution when they do not fit the dataset. Empty without sidecars."""
    import json

    from scview.core.uns_keys import META_PREFIX

    used: list[str] = []
    for key in adata.uns:
        if not str(key).startswith(META_PREFIX):
            continue
        raw = adata.uns[key]
        try:
            meta = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (TypeError, ValueError):
            continue
        used.extend(str(c) for c in meta.get("collections") or [])
    used = list(dict.fromkeys(used))
    if not used:
        return ""
    species = dataset_species(adata)
    other = cross_species(used, species)
    caution = (f" ({', '.join(short_label(c) for c in other)}: another species' symbols, matched ignoring case)"
               if other else "")
    return f"Gene sets used: {describe(used, species)}{caution}"
