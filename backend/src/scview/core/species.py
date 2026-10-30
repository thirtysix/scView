"""Which species a dataset's genes come from: human, mouse or unknown.

Species matters wherever a gene list is written as human symbols (cell-cycle and
stress signatures, MSigDB), and the LLM advisor and co-pilot are better grounded
when they know it. The call reads gene names only, never expression:

1. Ensembl gene ids, in var_names or an id column such as ``gene_ids``: ENSG… is
   human, ENSMUSG… mouse, any other ENS<species>G prefix another species.
2. Otherwise the case of symbol-like names (human CD3E, mouse Cd3e), with the
   mitochondrial prefix (MT- vs mt-) as a second, independent sign. Rat symbols
   are title case too, so a call from case alone says it rests on case.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

Species = Literal["human", "mouse", "unknown"]

_ENSEMBL_RE = re.compile(r"^ENS([A-Z]*)G\d{6,}")
_ENSEMBL_SPECIES = {"": "human", "MUS": "mouse"}
# var columns that hold Ensembl ids next to symbol var_names (10x, Seurat, cellxgene).
_ID_COLUMNS = ("gene_ids", "gene_id", "ensembl_id", "ensembl", "gene_ensembl", "feature_id", "id")
_SAMPLE = 5000  # names looked at per source; evenly spaced, so chromosome order is no bias
_MIN_SYMBOLS = 50  # fewer symbol-like names than this: too few to judge their case
_CASE_SHARE = 0.8  # share of symbol-like names in one case needed for a call
# Symbol-like: a letter, then letters/digits. Leaves out clone names (AC114488.2,
# RP11-206L10.2), which have no case convention, and human open reading frames
# (C1orf112), the one common human symbol with lower-case letters.
_SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]+$")
_ORF_RE = re.compile(r"^C(\d+|X|Y)orf\d+$")


class SpeciesCall(BaseModel):
    """The species call and the evidence it rests on, in one plain sentence."""

    species: Species = "unknown"
    method: Literal["ensembl", "symbol_case", "mito_prefix", "recorded", "none"] = "none"
    evidence: str = ""


def _sample(values) -> list[str]:
    arr = np.asarray(values, dtype=object)
    if arr.size > _SAMPLE:
        arr = arr[np.linspace(0, arr.size - 1, _SAMPLE).astype(int)]
    return [str(v) for v in arr if v is not None]


def _ensembl_counts(values) -> dict[str, int] | None:
    """Ensembl ids per species prefix, or None when the values are not mostly ids."""
    vals = _sample(values)
    counts: dict[str, int] = {}
    for v in vals:
        m = _ENSEMBL_RE.match(v)
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    hits = sum(counts.values())
    return counts if hits >= max(10, 0.5 * len(vals)) else None


def _from_ensembl(counts: dict[str, int], where: str) -> SpeciesCall:
    total = sum(counts.values())
    prefix, n = max(counts.items(), key=lambda kv: kv[1])
    share = n / total
    if share >= 0.9 and prefix in _ENSEMBL_SPECIES:
        species = _ENSEMBL_SPECIES[prefix]
        return SpeciesCall(species=species, method="ensembl", evidence=(
            f"Ensembl gene ids in {where}: {share:.0%} are {species} (ENS{prefix}G)."
        ))
    if share >= 0.9:
        return SpeciesCall(method="ensembl", evidence=(
            f"Ensembl gene ids in {where} are from another species (ENS{prefix}G), "
            "not human or mouse."
        ))
    mix = ", ".join(f"ENS{p}G {c / total:.0%}" for p, c in sorted(counts.items(), key=lambda kv: -kv[1]))
    return SpeciesCall(method="ensembl", evidence=(
        f"Ensembl gene ids in {where} mix species ({mix}): a multi-species reference?"
    ))


def _is_title(v: str) -> bool:
    rest = v[1:]
    return v[0].isupper() and any(c.islower() for c in rest) and not any(c.isupper() for c in rest)


def _from_symbols(names) -> SpeciesCall:
    idx = pd.Index(names).astype(str)
    mt_upper = int(idx.str.startswith("MT-").sum())
    mt_lower = int(idx.str.startswith("mt-").sum())
    mito: str | None = None
    if mt_upper and not mt_lower:
        mito = "human"
    elif mt_lower and not mt_upper:
        mito = "mouse"
    mito_text = (
        f"mitochondrial genes use 'MT-' ({mt_upper})" if mito == "human"
        else f"mitochondrial genes use 'mt-' ({mt_lower})" if mito == "mouse"
        else "no mitochondrial genes found" if not (mt_upper or mt_lower)
        else "mitochondrial genes use both 'MT-' and 'mt-'"
    )

    vals = [v for v in _sample(idx) if _SYMBOL_RE.match(v) and not _ORF_RE.match(v)]
    n = len(vals)
    upper = [v for v in vals if v.isupper()]
    title = [v for v in vals if _is_title(v)]
    case: str | None = None
    if n >= _MIN_SYMBOLS:
        if len(upper) >= _CASE_SHARE * n:
            case = "human"
        elif len(title) >= _CASE_SHARE * n:
            case = "mouse"

    if case and mito and case != mito:
        return SpeciesCall(evidence=(
            f"Gene symbols look {case} by case, but {mito_text}: conflicting signs."
        ))
    if case:
        style, share, example = (
            ("upper", len(upper) / n, upper[0]) if case == "human"
            else ("title", len(title) / n, title[0])
        )
        note = "" if case == "human" else " (rat symbols look the same)"
        return SpeciesCall(species=case, method="symbol_case", evidence=(
            f"Gene symbols are {style} case ({share:.0%} of {n:,} symbol-like names, "
            f"e.g. {example}){note}; {mito_text}."
        ))
    if mito:
        return SpeciesCall(species=mito, method="mito_prefix", evidence=(
            f"Gene-name case is not decisive, but {mito_text}."
        ))
    if n >= _MIN_SYMBOLS and sum(v.islower() for v in vals) >= _CASE_SHARE * n:
        why = "Gene names are lower case (zebrafish style), neither human nor mouse."
    elif n < _MIN_SYMBOLS:
        why = f"Too few gene symbols to tell ({n})."
    else:
        why = "Gene-name case is mixed, so neither human nor mouse."
    return SpeciesCall(evidence=f"{why} No Ensembl gene ids; {mito_text}.")


def _recorded_organism(adata) -> SpeciesCall | None:
    info = getattr(adata, "uns", {}).get("scview_spatial")
    organism = info.get("organism") if isinstance(info, Mapping) else None
    if organism not in ("human", "mouse"):
        return None
    return SpeciesCall(species=organism, method="recorded",
                       evidence=f"The panel's organism in experiment.xenium: {organism}.")


def detect_species(adata) -> SpeciesCall:
    """Human, mouse or unknown: what an import recorded (a Xenium panel's ``panel_organism``),
    else from the dataset's gene ids and names."""
    recorded = _recorded_organism(adata)
    if recorded is not None:
        return recorded
    names = adata.var_names
    raw = getattr(adata, "raw", None)
    if raw is not None and len(raw.var_names) > len(names):
        names = raw.var_names  # X may hold only the variable genes
    counts = _ensembl_counts(names)
    if counts:
        return _from_ensembl(counts, "var_names")
    for col in adata.var.columns:
        s = adata.var[col]
        if str(col).lower() in _ID_COLUMNS and not pd.api.types.is_numeric_dtype(s):
            counts = _ensembl_counts(s.to_numpy())
            if counts:
                return _from_ensembl(counts, f"var '{col}'")
    return _from_symbols(names)


def match_genes(
    genes: Iterable[str], names: Iterable[str], aliases: dict[str, tuple[str, ...]] | None = None
) -> list[str]:
    """The dataset's own spelling of each gene in ``genes`` that it has.

    Matched ignoring case, so a human symbol list finds mouse genes whose symbol
    is the same but title-cased (FOS -> Fos). ``aliases`` gives other names to try
    for a gene that was renamed. Order follows ``genes``; no name appears twice.
    """
    lookup: dict[str, str] = {}
    for n in names:
        lookup.setdefault(str(n).upper(), str(n))
    aliases = aliases or {}
    out: list[str] = []
    for g in genes:
        for cand in (g, *aliases.get(g, ())):
            hit = lookup.get(cand.upper())
            if hit is not None:
                if hit not in out:
                    out.append(hit)
                break
    return out


def human_genesets_warning(call: SpeciesCall) -> str:
    """A note for enrichment on a mouse dataset: MSigDB (and the default Enrichr
    libraries) are human gene symbols. Ortholog mapping is a separate item."""
    if call.species != "mouse":
        return ""
    return (
        "This looks like a mouse dataset, but the gene sets are human gene symbols. "
        "Genes are matched ignoring case, so a mouse gene whose symbol is the human one "
        "title-cased (Cd3e, CD3E) is found, but one whose mouse symbol differs is silently "
        "missed. Treat the results as approximate: no ortholog mapping is done."
    )
