"""Gene positions: where each gene starts, ends and which way it reads.

ATAC gene activity and TSS enrichment both need gene bodies. A 10x multiome h5 has an
``interval`` per gene, but it is not the gene body (FAM138A: chr1:36080-36081, 1 bp), so
positions come from the GENCODE release 10x built its 2020-A references from: v32 for
GRCh38, vM23 for mm10. The GTF is fetched on first use, its sha256 pinned, and reduced to
``DATA_DIR/annotation/<genome>.genes.tsv.gz``. ``ANNOTATION_AUTO_DOWNLOAD=false`` turns
fetching off; ``seed_from`` reads a GTF already on disk.

Coordinates are 0-based half open, as in BED and 10x fragments: GTF start 11869 -> 11868.
``tss`` is the first transcribed base: ``start`` on +, ``end - 1`` on -.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

SUBDIR = "annotation"
COLUMNS = ["gene_id", "gene_name", "chrom", "start", "end", "strand", "gene_type", "tss"]
MAX_DOWNLOAD_BYTES = 200_000_000
TIMEOUT = httpx.Timeout(120.0, connect=20.0)


@dataclass(frozen=True)
class Release:
    genome: str
    source: str
    url: str
    sha256: str


RELEASES = {
    "GRCh38": Release(
        "GRCh38", "GENCODE v32 basic",
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_32/gencode.v32.basic.annotation.gtf.gz",
        "a61ab50fc8b5d63f76d50e1f89bc0f5bba0b64fb29d2ac8cfa1b1ef8d4ef0c91",
    ),
    "mm10": Release(
        "mm10", "GENCODE vM23 basic",
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M23/gencode.vM23.basic.annotation.gtf.gz",
        "1751488e29871316ed7bbea4e38cbe229d67edc714b811012fcb9fe6131970b7",
    ),
}
_ALIASES = {"grch38": "GRCh38", "hg38": "GRCh38", "mm10": "mm10", "grcm38": "mm10"}
_lock = threading.Lock()
_cache: dict[tuple[str, str], pd.DataFrame] = {}


class AnnotationError(RuntimeError):
    """The sentence says which genome, and what to do."""


def canonical_genome(name: str | None) -> str:
    """'GRCh38' or 'mm10' for a reference name such as 'GRCh38-2020-A' or 'mm10-2020-A'."""
    key = re.split(r"[-_.\s]", str(name or "").strip().lower(), maxsplit=1)[0]
    genome = _ALIASES.get(key)
    if genome is None:
        raise AnnotationError(
            f"scView has gene positions for GRCh38 and mm10, not {name!r}. "
            "Peaks from another reference cannot be matched to genes."
        )
    return genome


def _attribute(attrs: str, key: str) -> str:
    m = re.search(rf'{key} "([^"]+)"', attrs)
    return m.group(1) if m else ""


def parse_gtf(path: str | Path) -> pd.DataFrame:
    """The ``gene`` records of a GTF (plain or gzip), versions stripped from gene ids,
    PAR_Y copies dropped (their ids repeat the X gene's)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            raw_id = _attribute(parts[8], "gene_id")
            if raw_id.endswith("_PAR_Y"):
                continue
            start, end = int(parts[3]) - 1, int(parts[4])
            strand = parts[6]
            rows.append((raw_id.split(".")[0], _attribute(parts[8], "gene_name"), parts[0], start, end, strand,
                         _attribute(parts[8], "gene_type"), end - 1 if strand == "-" else start))
    return pd.DataFrame(rows, columns=COLUMNS)


def _table_path(data_dir: str | Path, genome: str) -> Path:
    return Path(data_dir) / SUBDIR / f"{genome}.genes.tsv.gz"


def seed_from(gtf: str | Path, data_dir: str | Path, genome: str) -> Path:
    """Build the table from a GTF on disk."""
    out = _table_path(data_dir, canonical_genome(genome))
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(f".{out.name}.part")
    parse_gtf(gtf).to_csv(part, sep="\t", index=False, compression="gzip")
    part.replace(out)
    return out


def _fetch(release: Release, data_dir: str | Path) -> None:
    folder = Path(data_dir) / SUBDIR
    folder.mkdir(parents=True, exist_ok=True)
    part = folder / f".{release.genome}.gtf.gz.part"
    digest = hashlib.sha256()
    size = 0
    try:
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client, client.stream("GET", release.url) as resp:
            resp.raise_for_status()
            with open(part, "wb") as fh:
                for chunk in resp.iter_bytes():
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise AnnotationError(f"The {release.source} download is larger than expected.")
                    digest.update(chunk)
                    fh.write(chunk)
        if digest.hexdigest() != release.sha256:
            raise AnnotationError(f"The {release.source} download is not the release scView expects.")
        gz = folder / f".{release.genome}.part.gtf.gz"
        part.replace(gz)
        try:
            seed_from(gz, data_dir, release.genome)
        finally:
            gz.unlink(missing_ok=True)
    except httpx.HTTPError as e:
        raise AnnotationError(
            f"Could not download {release.source} gene positions ({e}). Check the network and try again."
        ) from e
    finally:
        part.unlink(missing_ok=True)


def load(genome: str | None, data_dir: str | Path, auto_download: bool = True) -> pd.DataFrame:
    """Gene table of a genome (COLUMNS), fetching it the first time. Not to be called on
    the event loop: the first call downloads and parses."""
    name = canonical_genome(genome)
    key = (str(data_dir), name)
    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached
        path = _table_path(data_dir, name)
        if not path.exists():
            if not auto_download:
                raise AnnotationError(
                    f"Gene positions for {name} are not on this machine and ANNOTATION_AUTO_DOWNLOAD is off."
                )
            _fetch(RELEASES[name], data_dir)
        table = pd.read_csv(path, sep="\t", dtype={"chrom": str, "gene_id": str, "gene_name": str, "strand": str,
                                                   "gene_type": str})
        _cache[key] = table
        return table


def source_of(genome: str | None) -> str:
    return RELEASES[canonical_genome(genome)].source


def clear_cache() -> None:
    with _lock:
        _cache.clear()
