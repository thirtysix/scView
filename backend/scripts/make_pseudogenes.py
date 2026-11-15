"""Build the pseudogene symbol list shipped in ``src/scview/resources/`` (core/gene_filters.py).

One gzip text file, one gene symbol per line, human and mouse together (symbols are matched
with their case: human PTGES3P1, mouse Ptges3-ps). A symbol is listed when, in at least one of
the GENCODE releases below, every gene of that name has a pseudogene type (``*_pseudogene``,
except ``polymorphic_pseudogene``: coding in some people or strains), and in none of them is a
gene of that name of another type (a name that is coding anywhere stays a marker). A long
non-coding locus of the same name does not count against it (GENCODE v44 has CES1P1 both as a
pseudogene and as an lncRNA); a coding one, a small RNA or an immune segment does. Names with
no name (Ensembl ids, clone ids: ``gene_filters.is_unnamed``) are left out; they are skipped
already.

The releases are the ones 10x builds its references from, so the names match what Cell Ranger
counts: GENCODE v32 (GRCh38-2020-A) and v44 (GRCh38-2024-A), vM23 (mm10-2020-A) and vM33
(GRCm39-2024-A); each one's basic GTF (gene records are the same as in the full file).

    python backend/scripts/make_pseudogenes.py \\
        gencode.v32.basic.annotation.gtf.gz gencode.v44.basic.annotation.gtf.gz \\
        gencode.vM23.basic.annotation.gtf.gz gencode.vM33.basic.annotation.gtf.gz

From https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_{human,mouse}/release_*/ (GENCODE
data has no restrictions on use). Only the standard library and gene_filters are used.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from scview.core.gene_filters import is_unnamed  # noqa: E402

OUT = SRC / "scview" / "resources" / "pseudogenes.txt.gz"
_ATTR = re.compile(r'(\w+) "([^"]*)"')
KEPT_TYPES = {"polymorphic_pseudogene"}
NEUTRAL_TYPES = {"lncRNA", "TEC"}  # another locus of these types leaves a pseudogene name listed


def is_pseudo(gene_type: str) -> bool:
    return gene_type.endswith("pseudogene") and gene_type not in KEPT_TYPES


def read(gtf: Path) -> dict[str, set[str]]:
    """Gene name -> its gene types in one release."""
    types: dict[str, set[str]] = defaultdict(set)
    with gzip.open(gtf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split("\t", 9)
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = dict(_ATTR.findall(parts[8]))
            types[attrs["gene_name"]].add(attrs["gene_type"])
    return types


def build(gtfs: list[Path]) -> Path:
    pseudo: set[str] = set()
    other: set[str] = set()
    lines = []
    for gtf in gtfs:
        types = read(gtf)
        p = {n for n, t in types.items() if all(is_pseudo(x) for x in t)}
        pseudo |= p
        other |= {n for n, t in types.items() if any(not is_pseudo(x) and x not in NEUTRAL_TYPES for x in t)}
        sha = hashlib.sha256(gtf.read_bytes()).hexdigest()
        lines.append(f"# {gtf.name} sha256 {sha}: {len(p):,} pseudogene names")
        print(lines[-1])
    names = sorted(n for n in pseudo - other if not is_unnamed(n))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0: the same GTFs give the same bytes (and sha256).
    with open(OUT, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(b"# Pseudogene symbols (backend/scripts/make_pseudogenes.py)\n")
        for line in lines:
            gz.write((line + "\n").encode())
        gz.write(f"# {len(names):,} names (names of another type than pseudogene, lncRNA or TEC "
                 "in any release left out)\n".encode())
        for n in names:
            gz.write((n + "\n").encode())
    print(f"{OUT.name}: {len(names):,} names, {OUT.stat().st_size / 1e3:.0f} kB, "
          f"sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
    return OUT


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    build([Path(a) for a in sys.argv[1:]])
