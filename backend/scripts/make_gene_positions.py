"""Build the gene position tables shipped in ``src/scview/resources/`` (core/cnv.py).

One gzip TSV per genome: symbol, Ensembl gene id, chromosome (``chr`` + the
Ensembl name), start, end (1-based, inclusive, as in the GTF), for every
``gene`` feature on the primary chromosomes (1-22 or 1-19, X, Y, MT).
Ensembl data has no restrictions on use (https://www.ensembl.org/info/about/legal/).

    python backend/scripts/make_gene_positions.py \\
        Homo_sapiens.GRCh38.113.gtf.gz Mus_musculus.GRCm39.113.gtf.gz

The GTFs come from https://ftp.ensembl.org/pub/release-113/gtf/{homo_sapiens,mus_musculus}/.
Only the standard library is used, so it runs anywhere.
"""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "scview" / "resources"
GENOMES = {"Homo_sapiens.GRCh38": ("human", 22), "Mus_musculus.GRCm39": ("mouse", 19)}
_ATTR = re.compile(r'(\w+) "([^"]*)"')


def build(gtf: Path) -> Path:
    stem = next((k for k in GENOMES if gtf.name.startswith(k)), None)
    if stem is None:
        raise SystemExit(f"Not a GRCh38/GRCm39 Ensembl GTF: {gtf.name}")
    species, n_auto = GENOMES[stem]
    release = gtf.name[len(stem) + 1:].split(".")[0]
    keep = {str(i) for i in range(1, n_auto + 1)} | {"X", "Y", "MT"}
    rows = []
    with gzip.open(gtf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene" or parts[0] not in keep:
                continue
            attrs = dict(_ATTR.findall(parts[8]))
            rows.append((attrs.get("gene_name", ""), attrs["gene_id"], f"chr{parts[0]}",
                         int(parts[3]), int(parts[4])))
    order = {f"chr{c}": i for i, c in enumerate([str(i) for i in range(1, n_auto + 1)] + ["X", "Y", "MT"])}
    rows.sort(key=lambda r: (order[r[2]], r[3], r[1]))
    out = OUT / f"gene_positions_{species}.tsv.gz"
    OUT.mkdir(parents=True, exist_ok=True)
    # mtime=0: the same GTF gives the same bytes (and sha256).
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(f"# Ensembl release {release} {stem.split('.')[1]} gene features; {gtf.name}\n".encode())
        gz.write(b"symbol\tgene_id\tchromosome\tstart\tend\n")
        for r in rows:
            gz.write(("\t".join(map(str, r)) + "\n").encode())
    print(f"{out.name}: {len(rows):,} genes, {out.stat().st_size / 1e6:.2f} MB")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for arg in sys.argv[1:]:
        build(Path(arg))
