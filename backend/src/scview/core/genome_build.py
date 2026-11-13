"""Which build ATAC peaks and fragments are on: GRCh38 or hg19 (GRCh37) for human, mm10, mm9
(NCBIM37) or mm39 (GRCm39) for mouse.

Each other build names its chromosomes as GRCh38 / mm10 do, so names alone take hg19 input
for GRCh38 (mm9 or mm39 for mm10) and give it GRCh38's (mm10's) gene bodies, motifs, links and
TSS positions, all shifted. scView has gene positions (``gene_annotation``) and genome sequence
(``genome_seq``) for GRCh38 and mm10 only, so input known to be hg19, mm9 or mm39 is recorded
as ``hg19`` / ``mm9`` / ``mm39`` and every genome-dependent step refuses it with
``HG19_REFUSAL`` / ``MM9_REFUSAL`` / ``MM39_REFUSAL`` (``gene_annotation.canonical_genome``
raises it).

The signals, each enough on its own:

- a reference the input names: a 10x h5's ``features/genome`` ("hg19", "mm9",
  "GRCm39-2024-A"), a fragments file's ``# reference_path=.../refdata-cellranger-atac-hg19-1.2.0``
  header line;
- coordinates past the newer build's chromosome ends that the older build's chromosomes
  still hold. hg19's chr9 is 2.8 Mb longer than GRCh38's, chr8 1.2 Mb, chr21 1.4 Mb, chr19
  0.5 Mb, and those ends are gene-dense (9q34.3, 8q24.3, 19q13.43). mm9's chr7 is 7.1 Mb
  longer than mm10's, chr8 2.3 Mb, chr1 1.7 Mb, chr12 1.1 Mb, and distal chr7 holds the
  H19/Igf2 and Kcnq1 clusters. So a whole-genome peak set on the older build has peaks there;
  one on the newer build never does. Input past the ends of both, or on a chromosome the
  older build lacks, is neither (another species). mm39 is shorter than mm10 on most
  chromosomes but longer on chr8 (0.73 Mb), chr13 (0.46 Mb), chr4 (0.35 Mb), chr17 (0.31 Mb)
  and chr14 (0.24 Mb); less the telomere gaps that leaves about 1.5 Mb of sequence past mm10's
  ends, holding Itgb1 (chr8), Mettl4 (chr17) and Fgf14 (chr14), so a whole-genome mm39 peak set
  has peaks there and one on mm10 never does. mm39 is checked before mm9: positions that fit
  both (chr8, chr14, chr17 just past mm10's ends) are taken as mm39, the build current 10x
  references use; a whole mm9 set reaches past mm39's ends (chr7 to 152.5 Mb, chr8 to 131.7 Mb)
  and fails mm39's check, and a whole mm39 set reaches past mm9's (chrX, chr4) and fails mm9's.

Where nothing says, the current behaviour stays (GRCh38 or mm10 from Ensembl ids or
chromosome names).
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

HG19 = "hg19"
MM9 = "mm9"
MM39 = "mm39"

HG19_REFUSAL = ("This ATAC input is on hg19 (GRCh37), and scView has gene positions and genome sequence for "
                "GRCh38 and mm10 only, so gene activity, nearest genes, motifs, peak-gene links and TSS "
                "enrichment are off: lift the peaks over to GRCh38, or run Cell Ranger ARC on GRCh38, to use them.")
MM9_REFUSAL = ("This ATAC input is on mm9 (NCBIM37), and scView has gene positions and genome sequence for "
               "GRCh38 and mm10 only, so gene activity, nearest genes, motifs, peak-gene links and TSS "
               "enrichment are off: lift the peaks over to mm10, or run Cell Ranger ARC on mm10, to use them.")
MM39_REFUSAL = ("This ATAC input is on mm39 (GRCm39), and scView has gene positions and genome sequence for "
                "GRCh38 and mm10 only, so gene activity, nearest genes, motifs, peak-gene links and TSS "
                "enrichment are off: lift the peaks over to mm10, or run Cell Ranger ARC on mm10, to use them.")
REFUSALS = {HG19: HG19_REFUSAL, MM9: MM9_REFUSAL, MM39: MM39_REFUSAL}
# The build scView has positions for, of each other build's species.
NEWER = {HG19: "GRCh38", MM9: "mm10", MM39: "mm10"}

# UCSC chrom.sizes, autosomes and X/Y:
# hgdownload.soe.ucsc.edu/goldenPath/{hg38,hg19,mm10,mm9,mm39}/bigZips/*.chrom.sizes.
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
MM10_SIZES = {
    "1": 195471971, "2": 182113224, "3": 160039680, "4": 156508116, "5": 151834684, "6": 149736546,
    "7": 145441459, "8": 129401213, "9": 124595110, "10": 130694993, "11": 122082543, "12": 120129022,
    "13": 120421639, "14": 124902244, "15": 104043685, "16": 98207768, "17": 94987271, "18": 90702639,
    "19": 61431566, "X": 171031299, "Y": 91744698,
}
MM9_SIZES = {
    "1": 197195432, "2": 181748087, "3": 159599783, "4": 155630120, "5": 152537259, "6": 149517037,
    "7": 152524553, "8": 131738871, "9": 124076172, "10": 129993255, "11": 121843856, "12": 121257530,
    "13": 120284312, "14": 125194864, "15": 103494974, "16": 98319150, "17": 95272651, "18": 90772031,
    "19": 61342430, "X": 166650296, "Y": 15902555,
}
MM39_SIZES = {
    "1": 195154279, "2": 181755017, "3": 159745316, "4": 156860686, "5": 151758149, "6": 149588044,
    "7": 144995196, "8": 130127694, "9": 124359700, "10": 130530862, "11": 121973369, "12": 120092757,
    "13": 120883175, "14": 125139656, "15": 104073951, "16": 98008968, "17": 95294699, "18": 90720763,
    "19": 61420004, "X": 169476592, "Y": 91455967,
}
# (other build, sizes of the build scView has, the other build's sizes), human first; mm39
# before mm9 (see the module docstring).
_BUILDS = ((HG19, HG38_SIZES, HG19_SIZES), (MM39, MM10_SIZES, MM39_SIZES), (MM9, MM10_SIZES, MM9_SIZES))

# hg19 / GRCh37 / b37 / hs37d5 as a word in a reference name or path.
_HG19_NAME = re.compile(r"(?<![a-z0-9])(hg19|grch37|b37|hs37d5|hg37)(?![a-z0-9])", re.IGNORECASE)
# mm9 / NCBIM37 (Ensembl) / MGSCv37 (NCBI) likewise.
_MM9_NAME = re.compile(r"(?<![a-z0-9])(mm9|ncbim37|mgscv37)(?![a-z0-9])", re.IGNORECASE)
# mm39 / GRCm39 likewise (10x: GRCm39-2024-A, refdata-cellranger-arc-GRCm39-2024-A).
_MM39_NAME = re.compile(r"(?<![a-z0-9])(mm39|grcm39)(?![a-z0-9])", re.IGNORECASE)
_CHROM = re.compile(r"^(?:chr)?([0-9]+|X|Y)$", re.IGNORECASE)
# Header keys of a Cell Ranger ATAC / ARC fragments file that name the reference.
_REFERENCE_KEYS = ("reference_path", "reference", "genome", "reference_genome", "reference_assembly")
HEADER_LINES = 500
WINDOW = 1 << 14  # a tabix linear-index window


def names_hg19(name: object) -> bool:
    """A reference name or path that says hg19 / GRCh37."""
    return bool(name) and bool(_HG19_NAME.search(str(name)))


def names_mm9(name: object) -> bool:
    """A reference name or path that says mm9 / NCBIM37."""
    return bool(name) and bool(_MM9_NAME.search(str(name)))


def names_mm39(name: object) -> bool:
    """A reference name or path that says mm39 / GRCm39."""
    return bool(name) and bool(_MM39_NAME.search(str(name)))


def old_build(name: object) -> str | None:
    """``hg19``, ``mm9`` or ``mm39`` (a build scView has no positions for) when a reference
    name or path says so, else None."""
    return HG19 if names_hg19(name) else MM9 if names_mm9(name) else MM39 if names_mm39(name) else None


def refusal(name: object) -> str | None:
    """The sentence a genome-dependent step refuses a named other build with, else None."""
    return REFUSALS.get(old_build(name) or "")


def _key(chrom: object) -> str | None:
    m = _CHROM.match(str(chrom))
    return m.group(1).upper() if m else None


def _older(pairs: list[tuple[str, int]], newer: dict[str, int], older: dict[str, int]) -> bool:
    """Some positions past the newer build's end of their chromosome, none past the older
    build's, and none on a chromosome the older build lacks."""
    past_new = 0
    for k, e in pairs:
        if k not in older or e > older[k]:
            return False
        past_new += e > newer[k]
    return past_new > 0


def from_positions(chroms: Iterable[object], ends: Iterable[object]) -> str | None:
    """``hg19`` (or ``mm39``, ``mm9``) when some positions lie past GRCh38's (mm10's) end of
    their chromosome but none past hg19's (mm39's, mm9's); None otherwise (GRCh38 or mm10,
    another species, or no evidence either way)."""
    pairs = []
    for chrom, end in zip(chroms, ends):
        k = _key(chrom)
        try:
            e = int(end)
        except (TypeError, ValueError):
            continue
        if k is not None and e >= 0:
            pairs.append((k, e))
    return next((build for build, newer, older in _BUILDS if _older(pairs, newer, older)), None)


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
    """``hg19`` (``mm9``, ``mm39``) when the fragments header names an hg19 / GRCh37 (mm9,
    mm39) reference."""
    return next((b for b in map(old_build, (header.get(k) for k in _REFERENCE_KEYS)) if b), None)


def from_index(names: Sequence[str], linear: Sequence[Sequence[int]]) -> str | None:
    """``from_positions`` on a tabix index: a chromosome's linear index has a window up to
    its last record, so the last window's start is a position the file reaches."""
    return from_positions(names, [max(0, len(lin) - 1) * WINDOW for lin in linear])
