"""Which build ATAC peaks and fragments are on: GRCh38 or hg19 (GRCh37) for human, mm10, mm39
(GRCm39) or mm9 (NCBIM37) for mouse, or something scView has no positions for.

scView has gene positions (``gene_annotation``) and genome sequence (``genome_seq``) for
GRCh38, mm10 and mm39 (``SUPPORTED``). hg19 and mm9 name their chromosomes as GRCh38 / mm10
do, so names alone take hg19 input for GRCh38 (mm9 for mm10) and give it GRCh38's (mm10's)
gene bodies, motifs, links and TSS positions, all shifted. Input known to be hg19 or mm9 is
recorded as ``hg19`` / ``mm9`` and every genome-dependent step refuses it with
``HG19_REFUSAL`` / ``MM9_REFUSAL`` (``gene_annotation.canonical_genome`` raises it). mm39 also
names its chromosomes as mm10 does; input known to be mm39 is recorded as ``mm39`` and uses
mm39's own gene positions and sequence.

The signals, each enough on its own:

- a reference the input names: a 10x h5's ``features/genome`` ("hg19", "mm9",
  "GRCm39-2024-A"), a fragments file's ``# reference_path=.../refdata-cellranger-atac-hg19-1.2.0``
  header line;
- coordinates past the assumed build's chromosome ends that the other build's chromosomes
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

Other species and builds (T2T-CHM13, rat, zebrafish, fly...) get one sentence
(``unsupported``) naming the builds scView has and saying RNA work is unaffected. They are
told by a reference name (``other_build``), by chromosome names no human or mouse build has
(more than 22 numbered, 2L/3R, Z/W, roman numerals: ``other_from_names``) or by positions past
every human and mouse build's end of that chromosome (``other_from_positions``: T2T-CHM13's
chr9 runs to 150.6 Mb, a rat chr1 to 260 Mb). Such input is recorded as ``other`` with its
sentence.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

HG19 = "hg19"
MM9 = "mm9"
MM39 = "mm39"
OTHER = "other"  # recorded for input told by its chromosomes to be another species or build
SUPPORTED = ("GRCh38", "mm10", "mm39")

SUPPORTED_TEXT = "GRCh38 (human) and mm10 or mm39 (mouse)"
_STEPS = "gene activity, nearest genes, motifs, peak-gene links and TSS enrichment"
_UNAFFECTED = ("RNA analysis and the peak counts themselves (QC, LSI, the ATAC UMAP and clusters, differential "
               "peaks) do not need a genome and work as usual.")

HG19_REFUSAL = ("This ATAC input is on hg19 (GRCh37), and scView has gene positions and genome sequence for "
                f"{SUPPORTED_TEXT} only, so {_STEPS} are off: lift the peaks over to GRCh38, or run Cell "
                "Ranger ARC on GRCh38, to use them.")
MM9_REFUSAL = ("This ATAC input is on mm9 (NCBIM37), and scView has gene positions and genome sequence for "
               f"{SUPPORTED_TEXT} only, so {_STEPS} are off: lift the peaks over to mm10 or mm39, or run Cell "
               "Ranger ARC on a current mouse reference, to use them.")
REFUSALS = {HG19: HG19_REFUSAL, MM9: MM9_REFUSAL}
# The build each one is taken for when nothing names it: they share its chromosome names.
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
# Each chromosome's longest end over every human and mouse build: past it is another genome.
_LONGEST = {k: max(t.get(k, 0) for t in (HG38_SIZES, HG19_SIZES, MM10_SIZES, MM9_SIZES, MM39_SIZES))
            for k in {*HG38_SIZES, *MM10_SIZES}}

# hg19 / GRCh37 / b37 / hs37d5 as a word in a reference name or path.
_HG19_NAME = re.compile(r"(?<![a-z0-9])(hg19|grch37|b37|hs37d5|hg37)(?![a-z0-9])", re.IGNORECASE)
# mm9 / NCBIM37 (Ensembl) / MGSCv37 (NCBI) likewise.
_MM9_NAME = re.compile(r"(?<![a-z0-9])(mm9|ncbim37|mgscv37)(?![a-z0-9])", re.IGNORECASE)
# mm39 / GRCm39 likewise (10x: GRCm39-2024-A, refdata-cellranger-arc-GRCm39-2024-A).
_MM39_NAME = re.compile(r"(?<![a-z0-9])(mm39|grcm39)(?![a-z0-9])", re.IGNORECASE)
# The builds scView has, likewise (10x: GRCh38-2020-A, refdata-cellranger-arc-mm10-2020-A-2.0.0).
_GRCH38_NAME = re.compile(r"(?<![a-z0-9])(grch38|hg38)(?![a-z0-9])", re.IGNORECASE)
_MM10_NAME = re.compile(r"(?<![a-z0-9])(mm10|grcm38)(?![a-z0-9])", re.IGNORECASE)
# Exact reference names (the part before the first - _ . or space).
_ALIASES = {"grch38": "GRCh38", "hg38": "GRCh38", "mm10": "mm10", "grcm38": "mm10", "mm39": "mm39",
            "grcm39": "mm39"}
# Other species and builds by UCSC, Ensembl or NCBI name, as a word start (versions follow:
# T2T-CHM13v2.0, mRatBN7.2, GRCz11).
_OTHER_NAMES = tuple((re.compile(rf"(?<![a-z0-9])(?:{rx})(?![0-9])", re.IGNORECASE), label) for rx, label in (
    (r"chm13|t2t|hs1", "T2T-CHM13 (human)"),
    (r"rn[4-8]|rnor[_-]?\d|mratbn\d", "a rat genome"),
    (r"danrer\d+|grcz\d+|zv\d+", "a zebrafish genome"),
    (r"dm[36]|bdgp\d", "a fly (Drosophila) genome"),
    (r"ce1[01]|wbcel\d+", "a worm (C. elegans) genome"),
    (r"galgal\d|grcg\d", "a chicken genome"),
    (r"rhemac\d+|mmul[_-]?\d+|macfas\d", "a macaque genome"),
    (r"pantro\d|caljac\d", "a primate genome other than human"),
    (r"susscr\d+|sscrofa\d+", "a pig genome"),
    (r"bostau\d+|ars[_-]ucd\d", "a cow genome"),
    (r"canfam\d", "a dog genome"),
    (r"saccer\d|r64", "a yeast genome"),
    (r"tair\d+", "an Arabidopsis genome"),
    (r"xentro\d+|xenlae\d+", "a Xenopus genome"),
))
_ROMAN = re.compile(r"^(?:chr)?(I|II|III|IV|V|VI|VII|VIII|IX|XI|XII|XIII|XIV|XV|XVI)$", re.IGNORECASE)
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
    """``hg19``, ``mm9`` or ``mm39`` (a build that names chromosomes as GRCh38 / mm10 do) when a
    reference name or path says so, else None."""
    return HG19 if names_hg19(name) else MM9 if names_mm9(name) else MM39 if names_mm39(name) else None


def refusal(name: object) -> str | None:
    """The sentence a genome-dependent step refuses a named hg19 / mm9 with, else None."""
    return REFUSALS.get(old_build(name) or "")


def supported(name: object) -> str | None:
    """'GRCh38', 'mm10' or 'mm39' for a reference name such as 'GRCh38-2020-A', 'GRCm39-2024-A' or
    a path naming one; None for anything else (hg19, mm9, other species, nothing)."""
    text = str(name or "").strip()
    key = re.split(r"[-_.\s]", text.lower(), maxsplit=1)[0]
    if key in _ALIASES:
        return _ALIASES[key]
    if old_build(text) in REFUSALS:
        return None
    return ("mm39" if names_mm39(text) else "GRCh38" if _GRCH38_NAME.search(text)
            else "mm10" if _MM10_NAME.search(text) else None)


def other_build(name: object) -> str | None:
    """'a rat genome', 'T2T-CHM13 (human)' ... when a reference name says another species or
    build, else None."""
    text = str(name or "")
    return next((label for rx, label in _OTHER_NAMES if rx.search(text)), None) if text else None


def other_sentence(what: str) -> str:
    return (f"This ATAC input is on {what}, and scView has gene positions and genome sequence for {SUPPORTED_TEXT} "
            f"only, so {_STEPS} are off. {_UNAFFECTED}")


def unsupported(name: object) -> str | None:
    """The one sentence a genome-dependent step refuses ``name`` with; None for a build scView
    has. hg19 and mm9 say how to get to a build scView has."""
    if supported(name):
        return None
    if (sentence := refusal(name)):
        return sentence
    text = str(name or "").strip()
    if (label := other_build(text)):
        return other_sentence(f"{label} ({text})" if text.lower() not in label.lower() else label)
    if text == OTHER:
        return other_sentence("another species or build")
    if not text or text.lower() in ("none", "nan"):
        return (f"This ATAC input does not say which genome it is on, and scView has gene positions and genome "
                f"sequence for {SUPPORTED_TEXT} only, so {_STEPS} are off. {_UNAFFECTED}")
    return other_sentence(f"{text!r}, a reference scView does not know")


def note(genome: object, stored: str | None = None) -> str | None:
    """The note a dataset shows for its recorded genome, worked out again from the name so a
    build scView has since learned (mm39, refused before v0.77) shows none. ``stored`` is kept
    for input recorded as ``other``: its sentence says what the chromosomes showed. None when
    nothing is recorded (the steps then say why on their own)."""
    if not genome:
        return None
    if str(genome) == OTHER and stored:
        return stored
    return unsupported(genome)


def other_from_names(names: Iterable[object]) -> str | None:
    """What a set of chromosome names says when no human or mouse build has them: more than 22
    numbered chromosomes, fly arms (2L, 3R), bird Z / W, roman numerals (worm, yeast); else None."""
    keys = [re.sub(r"^chr", "", str(n), flags=re.IGNORECASE) for n in names]
    upper = {k.upper() for k in keys}
    if upper & {"2L", "2R", "3L", "3R"}:
        return "a fly (Drosophila) genome (chromosomes 2L, 2R, 3L, 3R)"
    if upper & {"Z", "W"}:
        return "a bird genome (chromosome Z or W)"
    if any(_ROMAN.match(k) for k in keys):
        return "a worm or yeast genome (chromosomes I, II, III and on)"
    numbers = [int(k) for k in keys if k.isdigit()]
    if numbers and max(numbers) > 22:
        return f"a genome with chromosomes up to chr{max(numbers)} (human has 22, mouse 19)"
    return None


STANDARD_TEXT = "chr1-22, X, Y (mouse chr1-19, X, Y)"


def standard_chromosomes(chrom):
    """Per peak, True on a placed chromosome: 1-22, X or Y, with or without "chr" (Signac's
    ``keepStandardChromosomes``; mouse has 1-19). False on chrM / MT, unplaced and unlocalised
    scaffolds (``JH584304.1``, ``KI270728.1``, ``chrUn_...``) and alt or random contigs. A peak
    without a chromosome name stays (nothing says otherwise), and so does every peak when the
    names are not a human or mouse set (a chromosome numbered above 22, 2L, Z, roman numerals,
    or none of these names at all): there is no list to apply. Self-contained: the replay script
    copies it."""
    import numpy as np

    names = np.asarray(chrom, dtype=object).astype(str)
    keys = {n: re.sub(r"^chr", "", n, flags=re.IGNORECASE).upper() for n in set(names.tolist()) - {"", "None", "nan"}}
    numbered = [int(k) for k in keys.values() if k.isdigit()]
    ok = {n: bool(re.fullmatch(r"[1-9]|1[0-9]|2[0-2]|X|Y", k)) for n, k in keys.items()}
    if not any(ok.values()) or (numbered and max(numbered) > 22) or set(keys.values()) & {"2L", "2R", "3L", "3R", "Z", "W"}:
        return np.ones(names.size, dtype=bool)
    return np.fromiter((ok.get(n, True) for n in names.tolist()), dtype=bool, count=names.size)


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


def other_from_positions(chroms: Iterable[object], ends: Iterable[object]) -> str | None:
    """What positions say when one lies past every human and mouse build's end of its
    chromosome (T2T-CHM13's chr9, a rat chr1), else None."""
    for chrom, end in zip(chroms, ends):
        k = _key(chrom)
        try:
            e = int(end)
        except (TypeError, ValueError):
            continue
        if k is not None and k in _LONGEST and e > _LONGEST[k]:
            return (f"a genome whose chr{k} reaches {e / 1e6:,.1f} Mb, past every human and mouse build's end "
                    "(T2T-CHM13 or another species)")
    return None


def other_from_chroms(chroms: Sequence[object], ends: Sequence[object]) -> str | None:
    """``other_from_names`` then ``other_from_positions``."""
    return other_from_names(set(map(str, chroms))) or other_from_positions(chroms, ends)


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


def other_from_header(header: dict[str, str]) -> str | None:
    """The sentence for a fragments header that names another species or build (and no human or
    mouse build), else None."""
    values = [v for v in (header.get(k) for k in _REFERENCE_KEYS) if v]
    if any(supported(v) or old_build(v) for v in values):
        return None
    label = next((b for b in map(other_build, values) if b), None)
    return other_sentence(label) if label else None


def index_ends(linear: Sequence[Sequence[int]]) -> list[int]:
    """A position each chromosome of a tabix index reaches: its linear index has a window up to
    its last record, so the last window's start."""
    return [max(0, len(lin) - 1) * WINDOW for lin in linear]


def from_index(names: Sequence[str], linear: Sequence[Sequence[int]]) -> str | None:
    """``from_positions`` on a tabix index (``index_ends``)."""
    return from_positions(names, index_ends(linear))
