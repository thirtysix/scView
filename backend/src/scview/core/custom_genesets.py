"""User-uploaded gene-set collections ("custom collections").

A user uploads a .gmt file, or a two-column gene_set,gene table (.csv/.tsv), and
it becomes a named collection that sits next to the MSigDB ones: ORA, gene-set
scoring and the browser all accept its id wherever they accept an MSigDB id.

Each collection is stored on the data volume, so it survives restarts and needs
neither MSigDB nor a network connection:

    DATA_DIR/genesets/custom/<id>/meta.json   name, counts, source file name
    DATA_DIR/genesets/custom/<id>/sets.json   {"sets": {name: [genes]}, ...}

Ids look like ``custom-<32 hex>``. They are generated here, never reused after a
delete, and never contain a "." as MSigDB ids do, so a cached enrichment result
that names one can't be mistaken for an MSigDB collection, nor for a later
upload. The contents of a collection never change after upload (only its name
does), so the id alone identifies what a cached result was computed from.

Gene symbols are trimmed and upper-cased on the way in. MSigDB's human symbols
are upper-case, and gseapy matches them to a mouse dataset (Cd3e) by
upper-casing the dataset's genes; storing custom sets the same way means they
match human and mouse data exactly as MSigDB does. The few mixed-case human
symbols (C1orf112) are restored to the dataset's spelling at analysis time; see
``geneset_sources.match_query_case``.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import shutil
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from scview.core.dataset_manager import clean_dataset_name, is_safe_id

logger = logging.getLogger(__name__)

CUSTOM_PREFIX = "custom-"
CUSTOM_CATEGORY = "Custom"

# Limits. MSigDB's largest collection (GO:BP) has ~7,600 sets of at most 2,000
# genes, so these leave room for big custom libraries while bounding memory.
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_SETS = 25_000
MAX_GENES_PER_SET = 5_000
MAX_COLLECTIONS = 50
MAX_SET_NAME_CHARS = 200
MAX_GENE_CHARS = 100
MAX_DESCRIPTION_CHARS = 1_000
# Problems listed in a 400; the rest are summarised as "...and N more".
MAX_PROBLEMS_LISTED = 20

ALLOWED_EXTENSIONS = (".gmt", ".csv", ".tsv", ".txt")

_BAD_CATEGORIES = {"Cc", "Cf", "Cs", "Zl", "Zp"}
# Enrichr's GMT downloads append a weight to each gene ("TP53,1.0").
_WEIGHT_RE = re.compile(r"^(.+?),\s*[-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?$")
_HEADER_SET = {"gene_set", "geneset", "gene set", "set", "set_name", "term", "pathway",
               "name", "gs_name"}
_HEADER_GENE = {"gene", "genes", "gene_symbol", "symbol", "gene_name", "genesymbol"}


def is_custom_id(value: object) -> bool:
    """True for an id of the form this module generates."""
    return (
        isinstance(value, str)
        and value.startswith(CUSTOM_PREFIX)
        and is_safe_id(value)
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class GeneSetFileError(ValueError):
    """An upload that can't be used; ``problems`` lists what was wrong."""

    def __init__(self, message: str, problems: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.problems = problems or []


@dataclass
class ParsedGeneSets:
    format: str
    sets: dict[str, list[str]]
    descriptions: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _has_bad_chars(text: str) -> bool:
    return any(unicodedata.category(ch) in _BAD_CATEGORIES for ch in text)


def _short(text: str, n: int = 40) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


class _Collector:
    """Accumulates sets and problems while a file is read."""

    def __init__(self) -> None:
        self.sets: dict[str, list[str]] = {}
        self.seen_genes: dict[str, set[str]] = {}
        self.first_line: dict[str, int] = {}
        self.descriptions: dict[str, str] = {}
        self.problems: list[str] = []
        self.n_duplicate_genes = 0
        self.too_many_sets = False

    def problem(self, msg: str) -> None:
        self.problems.append(msg)

    def check_name(self, where: str, name: str) -> bool:
        if not name:
            self.problem(f"{where}: the gene set has no name.")
            return False
        if len(name) > MAX_SET_NAME_CHARS:
            self.problem(
                f"{where}: set name '{_short(name)}' is longer than {MAX_SET_NAME_CHARS} characters."
            )
            return False
        if _has_bad_chars(name):
            self.problem(f"{where}: set name '{_short(name)}' contains control or invisible characters.")
            return False
        return True

    def start_set(self, line: int, name: str) -> bool:
        if name not in self.sets:
            if len(self.sets) >= MAX_SETS:
                self.too_many_sets = True
                return False
            self.sets[name] = []
            self.seen_genes[name] = set()
            self.first_line[name] = line
        return True

    def add_gene(self, where: str, name: str, token: str) -> None:
        gene = clean_gene(token)
        if gene is None:
            return
        if len(gene) > MAX_GENE_CHARS:
            self.problem(f"{where}: gene '{_short(gene)}' is longer than {MAX_GENE_CHARS} characters.")
            return
        if any(ch.isspace() for ch in gene):
            self.problem(
                f"{where}: gene '{_short(gene)}' contains a space; put one gene per "
                "tab-separated field (GMT) or per row (CSV/TSV)."
            )
            return
        if _has_bad_chars(gene):
            self.problem(f"{where}: gene '{_short(gene)}' contains control or invisible characters.")
            return
        if gene in self.seen_genes[name]:
            self.n_duplicate_genes += 1
            return
        self.seen_genes[name].add(gene)
        self.sets[name].append(gene)

    def finish(self, fmt: str, filename: str, warnings: list[str]) -> ParsedGeneSets:
        for name, genes in self.sets.items():
            if len(genes) > MAX_GENES_PER_SET:
                self.problem(
                    f"set '{_short(name)}' (line {self.first_line[name]}) has {len(genes):,} genes; "
                    f"the limit is {MAX_GENES_PER_SET:,}."
                )
        if self.too_many_sets:
            self.problem(f"the file has more than {MAX_SETS:,} gene sets, the limit per collection.")
        if self.problems:
            listed = self.problems[:MAX_PROBLEMS_LISTED]
            extra = len(self.problems) - len(listed)
            if extra > 0:
                listed.append(f"...and {extra} more.")
            n = len(self.problems)
            raise GeneSetFileError(
                f"{filename}: {n} problem{'s' if n != 1 else ''} found, nothing was saved.", listed
            )
        if not self.sets:
            raise GeneSetFileError(f"{filename}: no gene sets found in the file.")
        if self.n_duplicate_genes:
            warnings.append(
                f"{self.n_duplicate_genes:,} repeated gene{'s' if self.n_duplicate_genes != 1 else ''} "
                "within a set were dropped (symbols are compared ignoring case)."
            )
        return ParsedGeneSets(fmt, self.sets, self.descriptions, warnings)


def clean_gene(token: str) -> str | None:
    """A gene token trimmed, stripped of an Enrichr weight and upper-cased."""
    gene = token.strip()
    if not gene:
        return None
    m = _WEIGHT_RE.match(gene)
    if m:
        gene = m.group(1).strip()
    return gene.upper() or None


def _decode(data: bytes, filename: str) -> str:
    if b"\x00" in data[:8192]:
        raise GeneSetFileError(
            f"{filename}: this looks like a binary file (compressed, or a spreadsheet?). "
            "Save it as plain-text .gmt, .csv or .tsv."
        )
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise GeneSetFileError(
            f"{filename}: the file is not UTF-8 text. Save it as UTF-8 .gmt, .csv or .tsv."
        )


def _parse_gmt(text: str, filename: str) -> ParsedGeneSets:
    c = _Collector()
    warnings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        where = f"line {lineno}"
        name = fields[0].strip()
        if not c.check_name(where, name):
            continue
        if len(fields) < 2:
            c.problem(
                f"{where}: '{_short(name)}' has no tab-separated fields after its name "
                "(a GMT line is name<TAB>description<TAB>gene<TAB>gene...)."
            )
            continue
        if name in c.sets:
            c.problem(f"{where}: set '{_short(name)}' appears again (first on line {c.first_line[name]}).")
            continue
        if not c.start_set(lineno, name):
            continue
        description = fields[1].strip()
        if description:
            c.descriptions[name] = description[:MAX_DESCRIPTION_CHARS]
        before = len(c.problems)
        for token in fields[2:]:
            c.add_gene(where, name, token)
        # Empty because every gene was rejected is already reported.
        if not c.sets[name] and len(c.problems) == before:
            c.problem(f"{where}: set '{_short(name)}' lists no genes.")
    return c.finish("gmt", filename, warnings)


def _parse_long(text: str, filename: str, delimiter: str, fmt: str) -> ParsedGeneSets:
    c = _Collector()
    warnings: list[str] = []
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    first = True
    sep = "tab" if delimiter == "\t" else "comma"
    try:
        for row in reader:
            cells = [cell.strip() for cell in row]
            if not any(cells) or cells[0].startswith("#"):
                continue
            where = f"line {reader.line_num}"
            if first:
                first = False
                if (len(cells) >= 2 and cells[0].lower() in _HEADER_SET
                        and cells[1].lower() in _HEADER_GENE):
                    continue
            filled = [cell for cell in cells if cell]
            if len(cells) < 2 or not cells[1] or len(filled) > 2:
                c.problem(
                    f"{where}: expected two {sep}-separated columns, gene_set and gene; "
                    f"found {len(filled)}."
                )
                continue
            name = cells[0]
            if not c.check_name(where, name):
                continue
            if not c.start_set(reader.line_num, name):
                continue
            c.add_gene(where, name, cells[1])
    except csv.Error as e:
        c.problem(f"line {reader.line_num}: could not read the table ({e}).")
    return c.finish(fmt, filename, warnings)


def _looks_like_gmt(text: str) -> bool:
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            if line.count("\t") >= 2:
                return True
    return False


def parse_gene_set_file(data: bytes, filename: str) -> ParsedGeneSets:
    """Parse an uploaded .gmt/.csv/.tsv/.txt file, or raise GeneSetFileError."""
    display = _short(Path(filename or "upload").name, 80)
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise GeneSetFileError(
            f"{display}: unsupported file type '{ext or '(none)'}'. "
            f"Upload one of: {', '.join(ALLOWED_EXTENSIONS)}."
        )
    if len(data) > MAX_FILE_BYTES:
        raise GeneSetFileError(
            f"{display}: the file is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB."
        )
    text = _decode(data, display)
    if ext == ".gmt":
        return _parse_gmt(text, display)
    if ext == ".csv":
        return _parse_long(text, display, ",", "csv")
    if ext == ".tsv":
        return _parse_long(text, display, "\t", "tsv")
    # .txt: a GMT has at least two tabs per line (name, description, genes).
    if _looks_like_gmt(text):
        return _parse_gmt(text, display)
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    if "\t" in first:
        return _parse_long(text, display, "\t", "tsv")
    return _parse_long(text, display, ",", "csv")


# ---------------------------------------------------------------------------
# Coverage against a dataset
# ---------------------------------------------------------------------------


def coverage(sets: dict[str, list[str]], gene_names: Iterable[str]) -> dict[str, int]:
    """How much of a collection a dataset measures, ignoring case.

    Counts ``n_genes`` (distinct genes in the collection), ``n_genes_present``
    (those the dataset has) and ``n_sets_with_genes`` (sets with at least one).
    Case is ignored because that is how ORA and scoring match them.
    """
    universe = {str(g).upper() for g in gene_names}
    all_genes: set[str] = set()
    n_sets_with = 0
    for genes in sets.values():
        all_genes.update(genes)
        if any(g in universe for g in genes):
            n_sets_with += 1
    return {
        "n_genes": len(all_genes),
        "n_genes_present": len(all_genes & universe),
        "n_sets_with_genes": n_sets_with,
    }


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class CustomCollectionError(ValueError):
    """A store operation that can't go ahead (bad name, limit reached)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def clean_source_filename(filename: str | None) -> str:
    """The uploaded file's base name, for display only (never used as a path)."""
    name = Path(filename or "").name
    name = "".join(ch for ch in name if unicodedata.category(ch) not in _BAD_CATEGORIES)
    return name[:120]


class CustomGeneSetStore:
    """Custom collections on disk, with their gene sets cached in memory."""

    def __init__(self, data_dir: str | Path) -> None:
        self.root = Path(data_dir) / "genesets" / "custom"
        self._sets: dict[str, dict[str, list[str]]] = {}

    # -- reading ----------------------------------------------------------

    def _dir(self, cid: str) -> Path | None:
        return self.root / cid if is_custom_id(cid) else None

    def list(self) -> list[dict]:
        """Every collection's metadata, oldest first."""
        if not self.root.is_dir():
            return []
        out = []
        for d in self.root.iterdir():
            if d.name.startswith(".") or not d.is_dir():
                continue
            meta = self.get_meta(d.name)
            if meta is not None:
                out.append(meta)
        out.sort(key=lambda m: (m.get("created_at", ""), m["id"]))
        return out

    def get_meta(self, cid: str) -> dict | None:
        d = self._dir(cid)
        if d is None:
            return None
        try:
            meta = json.loads((d / "meta.json").read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(meta, dict) or meta.get("id") != cid:
            return None
        return {**meta, "category": CUSTOM_CATEGORY, "available": True}

    def get_sets(self, cid: str) -> dict[str, list[str]] | None:
        """{set name: genes} for a collection, or None if it doesn't exist."""
        d = self._dir(cid)
        if d is None:
            return None
        if cid in self._sets:
            if d.is_dir():
                return self._sets[cid]
            self._sets.pop(cid, None)
        try:
            payload = json.loads((d / "sets.json").read_text())
            sets = {str(k): [str(g) for g in v] for k, v in payload["sets"].items()}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None
        self._sets[cid] = sets
        return sets

    def get_descriptions(self, cid: str) -> dict[str, str]:
        d = self._dir(cid)
        if d is None:
            return {}
        try:
            payload = json.loads((d / "sets.json").read_text())
            return {str(k): str(v) for k, v in (payload.get("descriptions") or {}).items()}
        except (OSError, ValueError, AttributeError):
            return {}

    def search(self, query: str, collection: str = "", limit: int = 50) -> list[dict]:
        """Sets whose name contains ``query`` (case-insensitive), like MSigDBLoader.search."""
        q = (query or "").upper().strip()
        ids = [collection] if collection else [m["id"] for m in self.list()]
        results: list[dict] = []
        for cid in ids:
            meta = self.get_meta(cid)
            sets = self.get_sets(cid)
            if meta is None or sets is None:
                continue
            for name, genes in sets.items():
                if not q or q in name.upper():
                    results.append(self._result(meta, name, genes))
                    if len(results) >= limit:
                        return results
        return results

    def get_gene_set(self, name: str, collection: str = "") -> dict | None:
        """One set by exact name, from ``collection`` or any custom collection."""
        ids = [collection] if collection else [m["id"] for m in self.list()]
        for cid in ids:
            meta = self.get_meta(cid)
            sets = self.get_sets(cid)
            if meta is not None and sets is not None and name in sets:
                return self._result(meta, name, sets[name])
        return None

    @staticmethod
    def _result(meta: dict, name: str, genes: list[str]) -> dict:
        return {
            "name": name,
            "collection": meta["id"],
            "collection_name": meta["name"],
            "n_genes": len(genes),
            "genes": genes,
        }

    # -- writing ----------------------------------------------------------

    def _check_name(self, name: object, exclude: str = "") -> str:
        try:
            clean = clean_dataset_name(name)
        except ValueError as e:
            raise CustomCollectionError(str(e))
        for meta in self.list():
            if meta["id"] != exclude and meta["name"].casefold() == clean.casefold():
                raise CustomCollectionError(
                    f"A custom collection named '{meta['name']}' already exists."
                )
        return clean

    def create(self, name: object, parsed: ParsedGeneSets, source_filename: str = "") -> dict:
        clean = self._check_name(name)
        if len(self.list()) >= MAX_COLLECTIONS:
            raise CustomCollectionError(
                f"There are already {MAX_COLLECTIONS} custom collections; delete one first."
            )
        cid = f"{CUSTOM_PREFIX}{uuid.uuid4().hex}"
        n_genes = len({g for genes in parsed.sets.values() for g in genes})
        meta = {
            "id": cid,
            "name": clean,
            "created_at": _now(),
            "source_filename": clean_source_filename(source_filename),
            "format": parsed.format,
            "n_sets": len(parsed.sets),
            "n_genes": n_genes,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        # Built under a hidden name and renamed into place, so a half-written
        # collection is never listed.
        tmp = self.root / f".tmp-{cid}"
        tmp.mkdir()
        try:
            (tmp / "sets.json").write_text(
                json.dumps({"sets": parsed.sets, "descriptions": parsed.descriptions})
            )
            (tmp / "meta.json").write_text(json.dumps(meta))
            os.replace(tmp, self.root / cid)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self._sets[cid] = parsed.sets
        logger.info("Saved custom gene-set collection %s (%s): %d sets", cid, clean, len(parsed.sets))
        return self.get_meta(cid) or meta

    def rename(self, cid: str, name: object) -> dict | None:
        meta = self.get_meta(cid)
        if meta is None:
            return None
        clean = self._check_name(name, exclude=cid)
        stored = {k: v for k, v in meta.items() if k not in ("category", "available")}
        stored["name"] = clean
        _write_json(self.root / cid / "meta.json", stored)
        return self.get_meta(cid)

    def delete(self, cid: str) -> bool:
        d = self._dir(cid)
        if d is None or not d.is_dir():
            return False
        # Move aside first so the collection disappears in one step.
        trash = self.root / f".trash-{cid}"
        os.replace(d, trash)
        shutil.rmtree(trash, ignore_errors=True)
        self._sets.pop(cid, None)
        logger.info("Deleted custom gene-set collection %s", cid)
        return True


_store: CustomGeneSetStore | None = None


def get_custom_store(data_dir: str) -> CustomGeneSetStore:
    """The store for ``data_dir`` (cached; tests switch data dirs freely)."""
    global _store
    root = Path(data_dir) / "genesets" / "custom"
    if _store is None or _store.root != root:
        _store = CustomGeneSetStore(data_dir)
    return _store
