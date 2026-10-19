"""Matching free-text cell-type labels to marker-database cell types.

Ported from the marker-database assessment (research/marker-db-assessment,
``names.py``; 2026-10-19). Three cumulative modes; scView uses C:

  A  token normalisation: lower case, Greek letters spelled out, stop words and
     plurals dropped, roman numerals -> digits. The exact key, else the nearest
     database key whose tokens are a subset or superset of the label's (Jaccard;
     ties go to the key holding the label's head noun, its last word).
  B  A + Cell Ontology (CL names + EXACT synonyms; CellMarker rows carry their own
     CL ids): a label that resolves to a CL term matches every database type with
     that term, tried before the nearest-key step; when nothing matches, the
     nearest is_a ancestor (<= 3 steps, not a generic term) with database types,
     reported as a weaker "via parent term" match.
  C  B + a small abbreviation table (Mono -> monocyte, Ery -> erythroid, ...) and
     leading cluster numbers stripped ("14Mo" -> "Mo"), the shorthand Seurat /
     Azimuth / paul15 labels use.

Measured coverage of PanglaoDB + CellMarker 2.0 names in modes A/B/C: HPA
0.90/0.93/0.93, Tabula Sapiens 0.97/1.00/1.00, Kang 0.85/0.85/1.00, paul15
0.40/0.50/1.00. Fixes over the assessment's version: Greek letters are spelled
out (it dropped them, so "T-cells" matched "γδ T cell"), ties prefer the head
noun (it took the alphabetically last key, so "Intestinal goblet cells" matched
"Intestinal cell"), and "Microglial" reads as "Microglia".
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

STOP = {"cell", "cells", "type", "positive", "of", "the", "and", "like"}
ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}
GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ζ": "zeta",
    "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu",
    "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "φ": "phi",
    "χ": "chi", "ψ": "psi", "ω": "omega",
}
ABBR = {  # label token -> expansion (mode C)
    "mono": "monocyte", "monos": "monocyte", "mo": "monocyte", "eryth": "erythroid",
    "ery": "erythroid", "erythro": "erythroid", "mk": "megakaryocyte", "megak": "megakaryocyte",
    "baso": "basophil", "neu": "neutrophil", "neut": "neutrophil", "eos": "eosinophil",
    "lymph": "lymphocyte", "mep": "megakaryocyte erythroid progenitor",
    "gmp": "granulocyte monocyte progenitor", "cmp": "common myeloid progenitor",
    "dc": "dendritic", "pdc": "plasmacytoid dendritic", "cdc": "conventional dendritic",
    "mac": "macrophage", "macro": "macrophage", "treg": "regulatory t", "fib": "fibroblast",
    "endo": "endothelial", "epi": "epithelial", "hsc": "hematopoietic stem",
    "mpp": "multipotent progenitor", "ilc": "innate lymphoid", "plt": "platelet",
    "pc": "plasma", "smc": "smooth muscle", "nkt": "natural killer t",
    "opc": "oligodendrocyte precursor", "ec": "endothelial", "microglial": "microglia",
}
GENERIC_DESC = 400  # CL terms with more descendants than this are too generic for an ancestor match
MAX_ANCESTOR_STEPS = 3

# How a label reached its database types, weakest last.
HOW_TEXT = {
    "exact": "same name",
    "cl": "same Cell Ontology term",
    "subset": "partial name match",
    "cl-ancestor": "matched via parent term",
}


def _greek(s: str) -> str:
    return "".join(f" {GREEK[c]} " if c in GREEK else c for c in s)


def toks(name: str, abbr: bool = False) -> list[str]:
    s = _greek(str(name).lower())
    if abbr:
        s = re.sub(r"^\d+\s*", "", s)  # paul15-style "14Mo"
        s = re.sub(r"([a-z])\+", r"\1 ", s)
    t = re.findall(r"[a-z0-9]+", s)
    if abbr:
        t = " ".join(ABBR.get(x, x) for x in t).split()
    return [ROMAN.get(x, x) for x in t]


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def norm(name: str, abbr: bool = False) -> frozenset[str]:
    return frozenset(_stem(t) for t in toks(name, abbr) if t not in STOP)


def head(name: str, abbr: bool = False) -> str | None:
    """The label's head noun: its last word that is not a stop word."""
    words = [_stem(t) for t in toks(name, abbr) if t not in STOP]
    return words[-1] if words else None


@dataclass
class Ontology:
    """Cell Ontology terms: CL id -> (name, EXACT synonyms, is_a parents)."""

    terms: dict[str, tuple[str, list[str], list[str]]]
    key: dict[frozenset, set[str]] = field(default_factory=dict)
    _anc: dict[str, frozenset] = field(default_factory=dict)
    _ndesc: dict[str, int] | None = None

    def __post_init__(self):
        key: dict[frozenset, set[str]] = defaultdict(set)
        for cid, (name, syns, _parents) in self.terms.items():
            for n in (name, *syns):
                k = norm(n)
                if k:
                    key[k].add(cid)
        self.key = dict(key)

    def parents(self, cid: str) -> list[str]:
        t = self.terms.get(cid)
        return t[2] if t else []

    def name(self, cid: str) -> str:
        t = self.terms.get(cid)
        return t[0] if t else cid

    def ancestors(self, cid: str) -> frozenset:
        if cid not in self._anc:
            out: set[str] = set()
            todo = list(self.parents(cid))
            while todo:
                x = todo.pop()
                if x not in out:
                    out.add(x)
                    todo += self.parents(x)
            self._anc[cid] = frozenset(out)
        return self._anc[cid]

    def n_desc(self, cid: str) -> int:
        if self._ndesc is None:
            counts: dict[str, int] = defaultdict(int)
            for t in self.terms:
                for a in self.ancestors(t):
                    counts[a] += 1
            self._ndesc = dict(counts)
        return self._ndesc.get(cid, 0)

    def ancestors_by_step(self, cid: str, max_step: int = MAX_ANCESTOR_STEPS) -> list[set[str]]:
        out, frontier, seen = [], {cid}, {cid}
        for _ in range(max_step):
            frontier = {p for x in frontier for p in self.parents(x)} - seen
            seen |= frontier
            out.append(frontier)
        return out

    def label_ids(self, name: str, abbr: bool = False) -> set[str]:
        return set(self.key.get(norm(name, abbr), ()))


def parse_obo(text: str) -> dict[str, tuple[str, list[str], list[str]]]:
    """The non-obsolete CL terms of an OBO file: id -> (name, EXACT synonyms, is_a parents)."""
    terms: dict[str, tuple[str, list[str], list[str]]] = {}
    cur: dict | None = None

    def flush(c):
        if c and c.get("id", "").startswith("CL:") and "name" in c and not c.get("obs"):
            terms[c["id"]] = (c["name"], c["syn"], sorted(c["is_a"]))

    for line in text.splitlines():
        if line in ("[Term]", "[Typedef]"):
            flush(cur)
            cur = {"is_a": set(), "syn": []} if line == "[Term]" else None
        elif cur is not None and ": " in line:
            k, v = line.split(": ", 1)
            if k in ("id", "name"):
                cur[k] = v
            elif k == "is_a":
                p = v.split(" ! ")[0].split(" {")[0].strip()
                if p.startswith("CL:"):
                    cur["is_a"].add(p)
            elif k == "synonym" and " EXACT" in v and v.count('"') >= 2:
                cur["syn"].append(v.split('"')[1])
            elif k == "is_obsolete" and v.strip() == "true":
                cur["obs"] = True
    flush(cur)
    return terms


class Matcher:
    """Index over one species' database cell types (name -> CL id or "")."""

    def __init__(self, type_cl: dict[str, str], ontology: Ontology | None = None):
        self.cl = ontology
        self.by_key: dict[frozenset, list[str]] = defaultdict(list)
        self.by_cl: dict[str, list[str]] = defaultdict(list)
        for name, cid in type_cl.items():
            k = norm(name)
            if k:
                self.by_key[k].append(name)
            ids = {cid} if cid else (self.cl.label_ids(name) if self.cl else set())
            for i in ids:
                self.by_cl[i].append(name)
        self.type_cl = {n: i for i, ns in self.by_cl.items() for n in ns}

    def _nearest(self, k: frozenset, head_noun: str | None) -> list[str]:
        near = [
            (len(k & d) / len(k | d), head_noun in d, tuple(sorted(d)))
            for d in self.by_key if k <= d or d <= k
        ]
        if not near:
            return []
        best = max(near)
        return list(self.by_key[frozenset(best[2])])

    def match(self, label: str, mode: str = "C") -> tuple[list[str], str]:
        """(matched database type names, how): how is exact | cl | subset | cl-ancestor | none."""
        abbr = mode == "C"
        use_cl = mode in ("B", "C") and self.cl is not None
        k = norm(label, abbr)
        if not k:
            return [], "none"
        if k in self.by_key:
            return list(self.by_key[k]), "exact"
        ids = self.cl.label_ids(label, abbr) if use_cl else set()
        hit = sorted({n for i in ids for n in self.by_cl.get(i, ())})
        if hit:
            return hit, "cl"
        near = self._nearest(k, head(label, abbr))
        if near:
            return near, "subset"
        for i in sorted(ids):
            for step in self.cl.ancestors_by_step(i):
                hit = sorted({
                    n for a in step if self.cl.n_desc(a) <= GENERIC_DESC for n in self.by_cl.get(a, ())
                })
                if hit:
                    return hit, "cl-ancestor"
        return [], "none"
