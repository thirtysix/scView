"""Which human build ATAC peaks and fragments are on: GRCh38, or hg19 (GRCh37).

hg19 and GRCh38 name their chromosomes alike, so names alone take hg19 input for GRCh38
and give it GRCh38 gene bodies, motifs, links and TSS positions, all shifted. scView has
gene positions (``gene_annotation``) and genome sequence (``genome_seq``) for GRCh38 and
mm10 only, so input known to be hg19 is recorded as ``hg19`` and every genome-dependent
step refuses it with ``HG19_REFUSAL`` (``gene_annotation.canonical_genome`` raises it).

The signals, each enough on its own:

- a reference the input names: a 10x h5's ``features/genome`` ("hg19"), a fragments
  file's ``# reference_path=.../refdata-cellranger-atac-hg19-1.2.0`` header line;
- coordinates past GRCh38's chromosome ends that hg19's chromosomes still hold. hg19's
  chr9 is 2.8 Mb longer, chr8 1.2 Mb, chr21 1.4 Mb, chr19 0.5 Mb, and those ends are
  gene-dense (9q34.3, 8q24.3, 19q13.43), so a whole-genome peak set on hg19 has peaks
  there. A GRCh38 set never does. Input past the ends of both is neither (another species).

Where nothing says, the current behaviour stays (GRCh38 from Ensembl ids or chromosome names).
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

HG19 = "hg19"

HG19_REFUSAL = ("This ATAC input is on hg19 (GRCh37), and scView has gene positions and genome sequence for "
                "GRCh38 and mm10 only, so gene activity, nearest genes, motifs, peak-gene links and TSS "
                "enrichment are off: lift the peaks over to GRCh38, or run Cell Ranger ARC on GRCh38, to use them.")

# UCSC chrom.sizes, autosomes and X/Y.
HG38_SIZES = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555, "5": 181538259, "6": 170805979,
    "7": 159345973, "8": 145138636, "9": 138394717, "10": 133797422, "11": 135086622, "12": 133275309,
    "13": 114364328, "14": 107043718, "15": 101991189, "16": 90338345, "17": 83257441, "18": 80373285,
    "19": 58617616, "20": 64444167, "21": 46709983, "22": 50818468, "X": 156040895, "Y": 57227415,
}
HG19_SIZES = {
    "1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276, "5": 180915260, "6": 171115067,
    "7": 159138663, "8": 146364022, "9": 141213431, "10": 135534747, "11": 135006516, "12": 133851895,
    "13": 115169878, "14": 107349540, "15": 102531392, "16": 90354753, "17": 81195210, "18": 78077248,
    "19": 59128983, "20": 63025520, "21": 48129895, "22": 51304566, "X": 155270560, "Y": 59373566,
}

# hg19 / GRCh37 / b37 / hs37d5 as a word in a reference name or path.
_HG19_NAME = re.compile(r"(?<![a-z0-9])(hg19|grch37|b37|hs37d5|hg37)(?![a-z0-9])", re.IGNORECASE)
_CHROM = re.compile(r"^(?:chr)?([0-9]+|X|Y)$", re.IGNORECASE)
# Header keys of a Cell Ranger ATAC / ARC fragments file that name the reference.
_REFERENCE_KEYS = ("reference_path", "reference", "genome", "reference_genome", "reference_assembly")
HEADER_LINES = 500
WINDOW = 1 << 14  # a tabix linear-index window


def names_hg19(name: object) -> bool:
    """A reference name or path that says hg19 / GRCh37."""
    return bool(name) and bool(_HG19_NAME.search(str(name)))


def _key(chrom: object) -> str | None:
    m = _CHROM.match(str(chrom))
    return m.group(1).upper() if m else None


def from_positions(chroms: Iterable[object], ends: Iterable[object]) -> str | None:
    """``hg19`` when some positions lie past GRCh38's end of their chromosome but none past
    hg19's; None otherwise (GRCh38, another species, or no evidence either way)."""
    past_38 = past_19 = 0
    for chrom, end in zip(chroms, ends):
        k = _key(chrom)
        try:
            e = int(end)
        except (TypeError, ValueError):
            continue
        if k is None or e < 0:
            continue
        if e > HG19_SIZES[k]:
            past_19 += 1
        if e > HG38_SIZES[k]:
            past_38 += 1
    return HG19 if past_38 and not past_19 else None


def from_intervals(parsed: Iterable[tuple[str, int, int] | None]) -> str | None:
    """``from_positions`` over parsed peak intervals (chrom, start, end), None skipped."""
    kept = [p for p in parsed if p]
    return from_positions((p[0] for p in kept), (p[2] for p in kept))


def read_header(path: str | Path, max_lines: int = HEADER_LINES) -> dict[str, str]:
    """``# key=value`` lines at the head of a fragments file (gzip or BGZF), first value per
    key; {} when it has none or cannot be read."""
    out: dict[str, str] = {}
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= max_lines or not line.startswith("#"):
                    break
                body = line[1:].strip()
                if "=" in body:
                    k, v = body.split("=", 1)
                    out.setdefault(k.strip().lower(), v.strip())
    except (OSError, EOFError, ValueError):
        return {}
    return out


def from_header(header: dict[str, str]) -> str | None:
    """``hg19`` when the fragments header names an hg19 / GRCh37 reference."""
    return HG19 if any(names_hg19(header.get(k)) for k in _REFERENCE_KEYS) else None


def from_index(names: Sequence[str], linear: Sequence[Sequence[int]]) -> str | None:
    """``from_positions`` on a tabix index: a chromosome's linear index has a window up to
    its last record, so the last window's start is a position the file reaches."""
    return from_positions(names, [max(0, len(lin) - 1) * WINDOW for lin in linear])
