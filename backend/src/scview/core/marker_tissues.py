"""The tissues of the marker databases' cell types, for annotating a dataset of a known tissue.

The marker-score annotator picked the database type a cluster's markers fit best from every
tissue: on kidney cortex 11 of 21 clusters became Astrocytes, Syncytiotrophoblast, Colonocyte,
Beta cells or Pigmented ciliary body cells (research/v075-qa/wilson2022). The databases say where
each type was described: PanglaoDB's ``organ``, CellMarker's ``tissue_class`` / ``tissue_type``,
ACT's per-tissue files. Both sides are read into one coarse organ vocabulary (``ORGANS``), and
when the dataset's tissue is known (the user's hint, else an obs ``tissue`` column as CELLxGENE
files carry, else the title) the annotator keeps the types of that organ, the ones found in every
tissue (immune, blood, vessels, stroma: ``EVERYWHERE``) and types with no tissue named. Shares of
a type's rows, not any row: CellMarker puts astrocytes in blood and heart now and then.

The index is built once from the raw files under ``DATA_DIR/marker_db/raw`` (CellMarker 3.0's
zips are read as a stream, ~10 s) and kept as ``tissues.json.gz`` beside the tables, rebuilt
when a raw file changes. Without the raw files nothing is restricted.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import re
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

INDEX_FILE = "tissues.json.gz"
VERSION = 2

# Phrases taken first and removed, so their words do not also count ("bone marrow" is not bone,
# "smooth muscle" not skeletal muscle).
_FIRST: tuple[tuple[str, str], ...] = (
    (r"bone marrow", "immune"),
    (r"smooth muscle", "stroma"),
    (r"skeletal muscle|muscle organ|muscle tissue|vastus|orbitalis muscle", "muscle"),
    (r"cord blood|umbilical cord blood", "blood"),
    (r"adrenal medulla", "endocrine"),
)
ORGANS: dict[str, str] = {
    "kidney": r"kidney|renal|nephr|glomerul",
    "urinary": r"(?<!gall )(?<!gall)bladder|ureter|urethra|urine|urinary",
    "liver": r"liver|hepat|bile|biliary|gall ?bladder",
    "pancreas": r"pancrea|islet",
    "gut": r"intestin|colon|colorect|\bgut\b|gastr(?!ul)|stomach|duoden|ileum|jejun|rect(um|al)|append|caecum|cecum"
           r"|esophag|oesophag|digestive|gi tract|anorectal|peyer|fundic|pyloric|lieberkuhn",
    "lung": r"lung|alveol|bronch|trache|airway|respirat|pleura|sputum",
    "nervous": r"brain|cerebr|cerebell|hippocamp|hypothalam|thalam|striatum|spinal cord|\bnerve|neural|neuron"
               r"|mening|meninx|ganglion|nodose|telencephal|prefrontal|entorhinal|temporal lobe|olfactory",
    "eye": r"\beye|retina|cornea|ciliary|\biris\b|choroid|sclera|lacrimal|lens",
    "heart": r"heart|cardiac|myocard|endocard|epicard|atri(um|al)|sinoatrial",
    "vessels": r"vascul|vessel|arter|aorta|\bvein|venous|umbilical (artery|vein)",
    "immune": r"immune|lymph|spleen|splen|thymus|thymic|tonsil",
    "blood": r"blood|pbmc|plasma|serum",
    "bone": r"\bbone|cartilag|joint|articul|ligament|tendon|synovi|tooth|teeth|periodon|intervertebral|menisc"
            r"|knee|\brib\b|vertebra|femur|calvaria|periosteum|dental",
    "muscle": r"\bmuscle",
    "stroma": r"connective tissue|soft tissue|mesenchym|stroma|adventitia|granulation tissue",
    "skin": r"skin|epiderm|dermis|scalp|hair|hypodermis|\bnail",
    "adipose": r"adipose|\bfat\b|fat pad|omentum",
    "breast": r"breast|mammary",
    "reproductive": r"reproduct|gonad|testis|testes|ovar|\buter|endometri|myometri|oviduct|fallopian|prostate"
                    r"|cervix|endocervi|ectocervi|vagina|semen|seminal|genital|penis|sperm|corpus luteum",
    "placenta": r"placent|decidua|amnio|chorion|trophoblast|umbilical cord(?! blood)|yolk sac",
    "endocrine": r"adrenal|suprarenal|thyroid|pituitar",
    "oral": r"\boral|mouth|tongue|saliva|taste|pharyn|head and neck|gingiv|labial|parotid|submandibular",
    "embryo": r"embryo|zygote|blastocyst|gastrul|pluripotent|primitive streak",
}
# Types found here are kept whatever the dataset's tissue: every tissue has blood and immune cells,
# vessels and stroma.
EVERYWHERE = frozenset({"immune", "blood", "vessels", "stroma"})
_COMPILED = [(k, re.compile(v, re.I)) for k, v in ORGANS.items()]
_FIRST_COMPILED = [(re.compile(p, re.I), k) for p, k in _FIRST]


def organs(text: Any) -> set[str]:
    """The coarse organs a tissue name names ("cortex of kidney" -> {"kidney"}); empty for none."""
    s = str(text or "").strip().lower()
    if not s or s in ("na", "nan", "none", "undefined", "unknown"):
        return set()
    out: set[str] = set()
    for rx, key in _FIRST_COMPILED:
        if rx.search(s):
            out.add(key)
            s = rx.sub(" ", s)
    out.update(k for k, rx in _COMPILED if rx.search(s))
    return out


# ---------------------------------------------------------------------------
# The databases' side


def _panglao(path: Path) -> dict[str, dict[str, Counter]]:
    out: dict[str, dict[str, Counter]] = {"human": defaultdict(Counter), "mouse": defaultdict(Counter)}
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            o = organs(r.get("organ"))
            sp = (r.get("species") or "").split()
            for code, species in (("Hs", "human"), ("Mm", "mouse")):
                if code in sp:
                    out[species][(r.get("cell type") or "").strip()].update(o)
    return out


def _cellmarker3(root: Path) -> tuple[dict, dict]:
    """(all rows, curated rows) of the CellMarker 3.0 zips: species -> cell name -> organs."""
    import zipfile

    every: dict[str, dict[str, Counter]] = {"human": defaultdict(Counter), "mouse": defaultdict(Counter)}
    curated: dict[str, dict[str, Counter]] = {"human": defaultdict(Counter), "mouse": defaultdict(Counter)}
    seen: dict[tuple, set] = {}
    for zpath in sorted(root.glob("*.zip")):
        with zipfile.ZipFile(zpath) as z:
            for member in z.infolist():
                if not member.filename.endswith(".txt"):
                    continue
                with z.open(member) as raw:
                    reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=""),
                                            delimiter="\t")
                    for r in reader:
                        sp = (r.get("species") or "").strip().lower()
                        if sp not in every or r.get("disease_category") != "Normal":
                            continue
                        name = (r.get("cell_name") or "").strip()
                        key = (r.get("tissue_class") or "", r.get("tissue_type") or "")
                        o = seen.get(key)
                        if o is None:
                            o = seen[key] = organs(key[0]) | organs(key[1])
                        every[sp][name].update(o)
                        if r.get("marker_source") != "Method":
                            curated[sp][name].update(o)
    return every, curated


def _cellmarker2(path: Path) -> dict[str, dict[str, Counter]]:
    import openpyxl

    out: dict[str, dict[str, Counter]] = {"human": defaultdict(Counter), "mouse": defaultdict(Counter)}
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = wb.active.iter_rows(values_only=True)
        head = [str(h or "").strip() for h in next(rows)]
        if not {"species", "cell_name"} <= set(head):
            return out
        i = {c: head.index(c) for c in ("species", "cell_name", "tissue_class", "tissue_type") if c in head}
        for r in rows:
            sp = str(r[i["species"]] or "").strip().lower()
            if sp not in out:
                continue
            o = set()
            for c in ("tissue_class", "tissue_type"):
                if c in i:
                    o |= organs(r[i[c]])
            out[sp][str(r[i["cell_name"]] or "").strip()].update(o)
    finally:
        wb.close()
    return out


def _act(root: Path) -> dict[str, dict[str, Counter]]:
    out: dict[str, dict[str, Counter]] = {"human": defaultdict(Counter), "mouse": defaultdict(Counter)}
    for f in sorted(root.glob("*/*.txt")):
        with open(f, encoding="latin-1", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                sp = (r.get("Species") or "").strip().lower()
                if sp in out:
                    out[sp][(r.get("CellType") or "").strip()].update(organs(r.get("Tissue")))
    return out


def _sources(raw: Path) -> dict[str, Path]:
    from scview.core.marker_db import SOURCES

    return {n: raw / spec.filename for n, spec in SOURCES.items() if (raw / spec.filename).exists()}


def _stamp(paths: dict[str, Path]) -> list:
    out = []
    for name, p in sorted(paths.items()):
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        out.append([name, sum(1 for f in files if f.is_file()),
                    max((f.stat().st_mtime for f in files if f.is_file()), default=0)])
    return out


def build(dest: Path) -> dict[str, dict[str, dict[str, list[str]]]]:
    """part -> species -> cell type (as the tables name it) -> {organ: database rows}, from the
    raw files (a row naming two organs counts for each)."""
    paths = _sources(dest / "raw")
    parts: dict[str, Any] = {}
    readers = {"panglao": _panglao, "cellmarker2": _cellmarker2, "act": _act}
    for name, path in paths.items():
        try:
            if name == "cellmarker3":
                parts["cellmarker3"], parts["cellmarker3_curated"] = _cellmarker3(path)
            elif name in readers:
                parts[name] = readers[name](path)
        except Exception as exc:  # noqa: BLE001 - a source without tissues restricts nothing
            logger.warning("Marker database tissues of %s unreadable: %s", name, exc)
    return {p: {sp: {t: dict(o) for t, o in types.items() if t} for sp, types in by.items()}
            for p, by in parts.items()}


_cache: dict[str, tuple[list, dict]] = {}
_lock = threading.Lock()


def load(dest: Path) -> dict[str, dict[str, dict[str, list[str]]]]:
    """The tissue index of the databases in ``dest``: from ``tissues.json.gz`` while the raw files
    are the ones it was built from, else built (and saved) now. {} without raw files."""
    paths = _sources(dest / "raw")
    if not paths:
        return {}
    stamp = [VERSION, *_stamp(paths)]
    with _lock:
        hit = _cache.get(str(dest))
        if hit and hit[0] == stamp:
            return hit[1]
        f = dest / INDEX_FILE
        index = None
        if f.is_file():
            try:
                with gzip.open(f, "rt", encoding="utf-8") as fh:
                    saved = json.load(fh)
                if saved.get("stamp") == json.loads(json.dumps(stamp)):
                    index = saved["parts"]
            except (OSError, ValueError, KeyError):
                index = None
        if index is None:
            index = build(dest)
            try:
                tmp = f.with_suffix(".part")
                with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                    json.dump({"stamp": stamp, "parts": index}, fh)
                tmp.replace(f)
            except OSError as exc:  # read-only volume: kept in memory only
                logger.info("Marker database tissue index not saved: %s", exc)
        _cache[str(dest)] = (stamp, index)
        return index


# ---------------------------------------------------------------------------
# The dataset's side

TISSUE_COLUMNS = ("tissue", "tissue_general", "organ", "tissue_type_name")
MAX_TISSUE_VALUES = 20  # more distinct values than this: a column of something else


def dataset_tissue(adata, hint: str = "") -> tuple[set[str], str]:
    """(organs, where from): the user's hint, else an obs tissue column (CELLxGENE's ``tissue``),
    else the title in ``uns["about_title"]`` when it names one organ. (set(), why) when unknown."""
    if hint and hint.strip():
        o = organs(hint)
        return (o, f"the tissue given ('{hint.strip()}')") if o else (set(), f"'{hint.strip()}' names no organ known here")
    for col in TISSUE_COLUMNS:
        if col in adata.obs.columns:
            values = adata.obs[col].dropna().astype(str).unique()
            if 0 < len(values) <= MAX_TISSUE_VALUES:
                o = set().union(*(organs(v) for v in values))
                if o:
                    shown = ", ".join(sorted(values)[:4]) + (", ..." if len(values) > 4 else "")
                    return o, f"obs '{col}' ({shown})"
    title = adata.uns.get("about_title") if hasattr(adata, "uns") else None
    o = organs(title) if isinstance(title, str) else set()
    if len(o) == 1:
        return o, f"the dataset title ('{title}')"
    return set(), "no tissue known (give one to restrict the database types)"


PRESENT = 0.05  # a type is "of" an organ holding this share of its rows (not a stray study's)
HOME = 0.25  # a type with no organ holding this share is described across tissues: kept anywhere
EVERYWHERE_SHARE = 0.5  # ... as is a type mostly described in immune, blood, vessel or stroma rows


def type_organs(sdb_type: dict, index: dict, species: str) -> dict[str, float]:
    """A merged database type's organs as shares of its rows: each source type it was merged from
    counts equally (PanglaoDB names one organ; CellMarker rows put a brain type in blood now and then)."""
    shares: Counter = Counter()
    n = 0
    for part, name in sdb_type.get("members", ()):
        rows = index.get(part, {}).get(species, {}).get(name)
        total = sum(rows.values()) if rows else 0
        if total:
            n += 1
            for organ, k in rows.items():
                shares[organ] += k / total
    return {o: v / n for o, v in shares.items()} if n else {}


# Cell Ontology lineages every tissue has: leukocyte, hematopoietic cell, erythrocyte, platelet,
# endothelial cell, fibroblast, pericyte, smooth muscle cell, mesenchymal cell, stromal cell.
EVERYWHERE_CL = frozenset({"CL:0000738", "CL:0000988", "CL:0000232", "CL:0000233", "CL:0000115",
                           "CL:0000057", "CL:0000669", "CL:0000192", "CL:0008019", "CL:0000499"})


def everywhere_lineage(cl_id: str | None, ontology) -> bool:
    """Whether a type's Cell Ontology term is (under) a lineage every tissue has."""
    if not cl_id:
        return False
    if cl_id in EVERYWHERE_CL:
        return True
    return ontology is not None and bool(ontology.ancestors(cl_id) & EVERYWHERE_CL)


def allowed(type_tissues: dict[str, float], dataset: set[str], everywhere: bool = False) -> bool:
    """A type is kept when it has no tissue named, holds the dataset's organ among its own (at least
    ``PRESENT`` of its rows), is described across tissues (no organ holds ``HOME``: epithelial cells),
    or is of a lineage every tissue has (``everywhere``: by its Cell Ontology term, or mostly in
    immune, blood, vessel or stroma rows) unless half its rows are one other organ's (liver
    sinusoidal endothelial cells are not kidney's)."""
    if not type_tissues:
        return True
    if any(type_tissues.get(o, 0.0) >= PRESENT for o in dataset):
        return True
    top = max(type_tissues.values())
    if top < HOME:
        return True
    general = sum(v for o, v in type_tissues.items() if o in EVERYWHERE)
    if general >= EVERYWHERE_SHARE:
        return True
    return everywhere and not any(v >= EVERYWHERE_SHARE for o, v in type_tissues.items() if o not in EVERYWHERE)
