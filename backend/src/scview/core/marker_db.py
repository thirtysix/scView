"""Marker databases: PanglaoDB + CellMarker 3.0 (or 2.0) + ACT, merged, checked against a label column.

Design and numbers: the marker-database assessment of 2026-10-19 (research/
marker-db-assessment/REPORT.md). In short:

- **Sources.** PanglaoDB (27 Mar 2020) and CellMarker 2.0 merged into one table
  keyed by normalised cell-type name, not ranked (a PanglaoDB-first cascade lost
  on Tabula Sapiens, 0.54 vs 0.71: its coarse "T cells" claims specific labels
  first). Native mouse rows for mouse data, symbols compared upper case.
- **Name matching** in ``core/marker_names.py`` (mode C: tokens, Cell Ontology,
  abbreviations). An unmatched label is "not in database", never "unsupported".
- **Scoring**: hypergeometric, background = the dataset's measured genes, label
  set = the label's 50 best-evidenced measured database genes, group side = the
  group's top 25 markers (``_markers_for``).
- **Verdict**: "no database support" when none of the label's database markers is
  among the group's top 25; "too few database markers" (abstain) when the matched
  type has < 5 measured database genes. On HPA clusters this flags 8% of correct
  labels and catches 77% of wrong same-tissue labels.
- **Suggestions**: the 3 database types scoring best against the FULL database
  set (top-1 right 0.42 on HPA, top-3 0.57), from PanglaoDB + CellMarker 2.0 only.
- **ACT** (Quan et al. 2023), the optional third source (addendum v2): in the
  verdict and the marker-score annotator (HPA lookup 0.67 -> 0.70, abstentions
  0.15 -> 0.11), not in the suggestions (top-3 0.57 -> 0.55). If it cannot be
  fetched, the others work alone.
- **CellMarker 3.0** (2026-07-22 release, unpublished; assessment v3) replaces 2.0
  when it can be fetched. Its 2.5M rows are 91% computational ("Method") markers,
  which help the verdict and wreck open-set ranking, so (``VIEWS``): the verdict
  uses PanglaoDB + CellMarker 3.0 (all rows) + ACT for human (HPA-other lookup /
  AUROC 0.73 / 0.93 vs 0.70 / 0.89, abstain 4% vs 11%; Kang 0.77) and CellMarker
  3.0 mouse rows alone for mouse (paul15 0.74 / 0.86; unions were lower); the
  suggestions and the marker-score annotator use PanglaoDB + CellMarker 3.0
  curated rows (parity with PanglaoDB + CellMarker 2.0 there). When 3.0 is
  missing, changed or failed, CellMarker 2.0 is fetched instead and the earlier
  PanglaoDB + CellMarker 2.0 (+ ACT) design applies; which one is in use is in the
  status, the manifest and every result.

scView does not ship the databases (none states a licence). Each is fetched
once, on first use, into ``DATA_DIR/marker_db`` (the data volume): the raw file
under ``raw/``, then one normalised table per source (species, cell_type, cl_id,
gene, weight) and ``manifest.json`` (URL, date, sha256, row counts). The two
marker files are frozen releases and their sha256 is pinned, so a wrong or
altered file is refused (CellMarker 2.0's download server is plain HTTP). The
Cell Ontology (``cl-basic.obo``, for name matching) is fetched the same way and
kept as a compact id -> name / synonyms / parents table; without it matching
falls back to names and abbreviations. A file placed in ``raw/`` by hand is used
without any download. ``MARKER_DB_AUTO_DOWNLOAD=false`` turns fetching off.
Fetch by hand: ``docker compose exec backend python -m scview.core.marker_db``.
"""

from __future__ import annotations

import csv
import fcntl
import gzip
import hashlib
import json
import logging
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from scview.core.marker_names import HOW_TEXT, Matcher, Ontology, norm, parse_obo
from scview.core.uns_keys import markers_key

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 128 * 1024 * 1024  # CellMarker 3.0's human zip is 49 MB
TIMEOUT = httpx.Timeout(90.0, connect=20.0)  # panglaodb.se can be very slow; a mirror follows
FAILURE_COOLDOWN = 600  # seconds before a failed source is tried again without an explicit retry

TOP_DB = 50  # a label's database set: its 50 best-evidenced measured genes
N_MARKERS = 25  # a group's markers
MIN_DB_GENES = 5  # fewer measured database genes than this: abstain
N_SUGGEST = 3
MAX_GROUPS = 100
# Mito, ribosomal (incl. RPLP0-2 and RPSA), immunoglobulin / TCR V segments and unnamed genes
SKIP_GENE = re.compile(r"^(MT-|RP[LS]\d|RPLP\d|RPSA$|IG[HKL]V|TR[ABDG]V|ENSG)", re.I)

VERDICT_TEXT = {
    "supported": "supported",
    "no_support": "no database support",
    "too_few": "too few database markers",
    "not_in_db": "not in database",
    "no_markers": "no marker genes",
}

NOTE = (
    "Each label is matched by name to the marker databases' cell types; its 50 "
    "best-evidenced database markers measured here are compared with the group's top 25 "
    "marker genes (detected in at least 20% of its cells, adjusted p < 0.05, sorted by "
    "log fold change, mitochondrial and ribosomal genes left out) with a hypergeometric "
    "test against the measured genes. \"No database support\" means none of them is among the top 25: a "
    "reason to look, not proof the label is wrong. Activation states, fine subtypes of one "
    "lineage (CD4 naive vs memory, CD8 T vs NK) and progenitors fail with every database."
)


class MarkerDBUnavailable(RuntimeError):
    """No marker database is on disk and none could be fetched (offline, disabled)."""


# ---------------------------------------------------------------------------
# Sources and parsers
# ---------------------------------------------------------------------------


def _parse_panglao(path: Path) -> list[tuple]:
    """PanglaoDB: weight = 2 x canonical + the species' sensitivity. Mouse rows get
    title-case symbols (PanglaoDB keeps one human-style symbol per row)."""
    out = []
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        need = {"species", "official gene symbol", "cell type", "canonical marker",
                "sensitivity_human", "sensitivity_mouse"}
        if not need <= set(reader.fieldnames or ()):
            raise ValueError(f"PanglaoDB file lacks columns {sorted(need - set(reader.fieldnames or ()))}")
        for r in reader:
            sp = (r["species"] or "").split()
            g = (r["official gene symbol"] or "").strip()
            for code, species, sens in (("Hs", "human", "sensitivity_human"), ("Mm", "mouse", "sensitivity_mouse")):
                if code not in sp or not g:
                    continue
                try:
                    s = float(r[sens])
                except (TypeError, ValueError):
                    s = 0.0
                s = 0.0 if s != s else s  # NaN
                w = 2.0 * ((r["canonical marker"] or "").strip() == "1") + s
                sym = g if species == "human" else g[:1].upper() + g[1:].lower()
                out.append((species, r["cell type"].strip(), "", sym, w))
    return out


def _cell(v) -> str:
    return "" if v is None else str(v).strip()


def _parse_cellmarker(path: Path) -> list[tuple]:
    """CellMarker 2.0 (Cell_marker_All.xlsx), normal cells of human and mouse: weight =
    the number of distinct papers citing (cell type, gene)."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = wb.active.iter_rows(values_only=True)
        head = [_cell(h) for h in next(rows)]
        need = ["species", "cell_type", "cell_name", "cellontology_id", "Symbol", "PMID"]
        missing = [c for c in need if c not in head]
        if missing:
            raise ValueError(f"CellMarker file lacks columns {missing}")
        i = {c: head.index(c) for c in need}
        cites: dict[tuple, set] = defaultdict(set)
        cl: dict[tuple, str] = {}
        for r in rows:
            if len(r) <= max(i.values()) or _cell(r[i["cell_type"]]) != "Normal cell":
                continue
            sp = _cell(r[i["species"]]).lower()
            if sp not in ("human", "mouse"):
                continue
            name = _cell(r[i["cell_name"]])
            if not name:
                continue
            cid = _cell(r[i["cellontology_id"]])
            if cid and cid not in ("NA", "NaN", "None"):
                cl.setdefault((sp, name), cid.replace("_", ":"))
            pmid = _cell(r[i["PMID"]]) or "?"
            for g in re.split(r"[,;]\s*", _cell(r[i["Symbol"]]).replace("[", "").replace("]", "")):
                g = g.strip()
                if g and g not in ("NA", "None"):
                    cites[(sp, name, g)].add(pmid)
    finally:
        wb.close()
    return [(sp, name, cl.get((sp, name), ""), g, float(len(p))) for (sp, name, g), p in cites.items()]


def _parse_act(path: Path) -> list[tuple]:
    """ACT per-tissue files (Species, Tissue, CellType, Marker = ranked comma list,
    Resource = PMIDs; Latin-1), pooled over tissues: weight = the tissue rows listing
    the gene plus a small bonus for ranking it early."""
    hit: Counter = Counter()
    rank: dict[tuple, float] = defaultdict(float)
    for f in sorted(path.glob("*/*.txt")):
        with open(f, encoding="latin-1", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            if not {"Species", "CellType", "Marker"} <= set(reader.fieldnames or ()):
                raise ValueError(f"ACT file {f.name} lacks the Species / CellType / Marker columns")
            for r in reader:
                sp = (r.get("Species") or "").strip().lower()
                ct = (r.get("CellType") or "").strip()
                genes = [g.strip() for g in (r.get("Marker") or "").split(",") if g.strip()]
                for i, g in enumerate(genes):
                    k = (sp, ct, g)
                    hit[k] += 1
                    rank[k] += (len(genes) - i) / len(genes)
    return [(sp, ct, "", g, n + rank[(sp, ct, g)] / n / 10)
            for (sp, ct, g), n in hit.items() if sp in ("human", "mouse") and ct]


def _pmid(x: str) -> str:
    x = (x or "").strip()
    return x[:-2] if x.endswith(".0") else x


def _parse_cellmarker3(root: Path) -> list[tuple]:
    """CellMarker 3.0 zips, read as a stream (the 830 MB of TSV is never written out):
    normal cells of human and mouse. Weight = distinct papers (the dataset id when a
    computational row has no PMID) plus a < 0.1 tie-breaker from distinct (paper,
    tissue, method) supports, for all rows and, separately, for the curated rows
    (marker_source other than "Method"); 0 when a gene has no curated row."""
    import io
    import zipfile

    # (species, name, gene) -> [papers, supports (<= 9), curated papers, curated supports (<= 9)]
    acc: dict[tuple, list] = {}
    cl: dict[tuple, str] = {}
    need = {"species", "disease_category", "cell_name", "cellontology_id", "symbol", "marker_source", "pmid",
            "series_id", "tissue_type", "method_details"}
    for zpath in sorted(root.glob("*.zip")):
        with zipfile.ZipFile(zpath) as z:
            for member in z.infolist():
                if not member.filename.endswith(".txt"):
                    continue
                with z.open(member) as raw:
                    reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=""),
                                            delimiter="\t")
                    if not need <= set(reader.fieldnames or ()):
                        raise ValueError(f"CellMarker 3.0 file lacks columns {sorted(need - set(reader.fieldnames or ()))}")
                    for r in reader:
                        sp = (r["species"] or "").strip().lower()
                        if sp not in ("human", "mouse") or r["disease_category"] != "Normal":
                            continue
                        name = (r["cell_name"] or "").strip()
                        if not name:
                            continue
                        cid = (r["cellontology_id"] or "").strip()
                        if cid.startswith("CL_"):
                            cl.setdefault((sp, name), cid.replace("_", ":"))
                        paper = _pmid(r["pmid"]) or (r["series_id"] or "").strip() or "?"
                        support = hash((paper, r["tissue_type"], r["method_details"]))
                        curated = r["marker_source"] != "Method"
                        for g in re.split(r"[,;]\s*", (r["symbol"] or "").replace("[", "").replace("]", "")):
                            g = g.strip()
                            if not g or g in ("NA", "None", "nan") or SKIP_GENE.match(g):
                                continue
                            key = (sp, name, g)
                            a = acc.get(key)
                            if a is None:
                                a = acc[key] = [None, None, None, None]
                            _add(a, 0, paper)
                            _add(a, 1, support, 9)
                            if curated:
                                _add(a, 2, paper)
                                _add(a, 3, support, 9)

    def w(papers, supports):
        return _count(papers) + 0.1 * _count(supports) / 9 if papers is not None else 0.0

    return [(sp, name, cl.get((sp, name), ""), g, w(a[0], a[1]), w(a[2], a[3]))
            for (sp, name, g), a in acc.items()]


# Most (type, gene) pairs have one paper: hold a single value until a second one comes
# (a set per pair would hold CellMarker 3.0's parse at ~0.9 GB).
def _add(a: list, i: int, v, cap: int | None = None) -> None:
    cur = a[i]
    if cur is None:
        a[i] = v
    elif isinstance(cur, set):
        if cap is None or len(cur) < cap:
            cur.add(v)
    elif cur != v:
        a[i] = {cur, v}


def _count(x) -> int:
    return 0 if x is None else len(x) if isinstance(x, set) else 1


@dataclass(frozen=True)
class SourceSpec:
    name: str
    label: str
    release: str
    filename: str
    urls: tuple[str, ...]  # tried in order
    sha256: str | None  # pinned for frozen releases
    home: str
    paper: str
    doi: str
    parse: Callable[[Path], list[tuple]]
    min_rows: int
    # Fetches a source made of many files into raw/<filename>/: (served_from, sha256, bytes)
    fetch: Callable[["SourceSpec", Path, httpx.Client | None], tuple[str, str, int]] | None = None


SOURCES: dict[str, SourceSpec] = {
    "panglao": SourceSpec(
        name="panglao",
        label="PanglaoDB",
        release="27 Mar 2020",
        filename="PanglaoDB_markers_27_Mar_2020.tsv.gz",
        urls=(
            "https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz",
            # The Internet Archive's copy of that file (byte-identical, sha256 pinned
            # below): panglaodb.se is unmaintained and often times out.
            "https://web.archive.org/web/2025id_/https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz",
        ),
        sha256="6779952ad40aa5a124de7bd0e18975c6630bd6006d6b3ef210a916caaa6b53c9",
        home="https://panglaodb.se/",
        paper="Franzén et al. 2019, Database (baz046)",
        doi="10.1093/database/baz046",
        parse=_parse_panglao,
        min_rows=10_000,
    ),
    "cellmarker3": SourceSpec(
        name="cellmarker3",
        label="CellMarker 3.0",
        release="2026-07-22 release, unpublished",
        filename="CellMarker3",
        urls=("https://bio-bigdata.hrbmu.edu.cn/CellMarker/file/",),
        sha256=None,  # one pinned sha256 per zip: CM3_FILES
        home="https://bio-bigdata.hrbmu.edu.cn/CellMarker/",
        # No 3.0 paper yet; the site asks for the 2.0 paper to be cited.
        paper="cite CellMarker 2.0: Hu et al. 2023, Nucleic Acids Res (PMID 36300619)",
        doi="10.1093/nar/gkac947",
        parse=_parse_cellmarker3,
        min_rows=200_000,
        fetch=lambda spec, raw_dir, client: _fetch_cm3(raw_dir / spec.filename, client),
    ),
    "cellmarker2": SourceSpec(
        name="cellmarker2",
        label="CellMarker 2.0",
        release="2.0 (files of 28 Sep 2022)",
        filename="Cell_marker_All.xlsx",
        # The CellMarker 2.0 server named in the paper; bio-bigdata.hrbmu.edu.cn/CellMarker
        # now serves a newer site without these files.
        urls=("http://117.50.127.228/CellMarker/CellMarker_download_files/file/Cell_marker_All.xlsx",),
        sha256="8c03c3ddd47aff8cad04e67ba8493a3becb8f75c229afae4cd75f3f08264dae2",
        home="http://117.50.127.228/CellMarker/",
        paper="Hu et al. 2023, Nucleic Acids Res (51:D870)",
        doi="10.1093/nar/gkac947",
        parse=_parse_cellmarker,
        min_rows=40_000,
    ),
    "act": SourceSpec(
        name="act",
        label="ACT",
        release="21 Feb 2024",
        filename="ACT",
        urls=("http://xteam.xbio.top/ACT/download.jsp",),
        sha256=None,  # one pinned sha256 per file: marker_db_act.sha256
        home="http://xteam.xbio.top/ACT/",
        paper="Quan et al. 2023, Genome Med (15:91)",
        doi="10.1186/s13073-023-01249-5",
        parse=_parse_act,
        min_rows=20_000,
        fetch=lambda spec, raw_dir, client: _fetch_act(raw_dir / spec.filename, client),
    ),
}
# Which parts each job uses (a part is a source, or "cellmarker3_curated": CellMarker
# 3.0 without its computational rows). With CellMarker 3.0 (assessment v3):
VIEWS_CM3 = {
    "verdict": {"human": ("panglao", "cellmarker3", "act"), "mouse": ("cellmarker3",)},
    "suggest": {"human": ("panglao", "cellmarker3_curated"), "mouse": ("panglao", "cellmarker3_curated")},
    "annotate": {"human": ("panglao", "cellmarker3_curated"), "mouse": ("panglao", "cellmarker3_curated")},
}
# Without it (CellMarker 2.0, addendum v2): suggestions on PanglaoDB + CellMarker 2.0
# only, since adding ACT raised lookup (HPA 0.67 -> 0.70) but lowered top-3 (0.57 -> 0.55).
VIEWS_CM2 = {
    "verdict": {"human": ("panglao", "cellmarker2", "act"), "mouse": ("panglao", "cellmarker2", "act")},
    "suggest": {"human": ("panglao", "cellmarker2"), "mouse": ("panglao", "cellmarker2")},
    "annotate": {"human": ("panglao", "cellmarker2", "act"), "mouse": ("panglao", "cellmarker2", "act")},
}
PART_LABELS = {"panglao": "PanglaoDB", "cellmarker3": "CellMarker 3.0",
               "cellmarker3_curated": "CellMarker 3.0 (curated)", "cellmarker2": "CellMarker 2.0", "act": "ACT"}

# CellMarker 3.0: the human and mouse zips (48.7 + 23.1 MB; together 32 KB smaller than
# all_cell_marker.zip, same rows), each pinned. A partial download is resumed.
CM3_BASE = "https://bio-bigdata.hrbmu.edu.cn/CellMarker/file/"
CM3_FILES = (
    ("human_cell_marker.zip", "22fa294165d7cda92d8ffb32c826acd74fa4656e568963aa3e1db3f94672c9fe", 48_740_702),
    ("mouse_cell_marker.zip", "93af58320bae8436ea811fbd0c7aeefb8218d5e3dd3b3bba60b0b7fb2cd2da4e", 23_071_482),
)

# ACT: 498 per-tissue files (277 human, 221 mouse). Its host serves https with an
# expired certificate, so they come over plain http and each is checked against the
# sha256 committed in marker_db_act.sha256; one mismatch and ACT is not loaded.
ACT_BASE = "http://xteam.xbio.top/ACT/download/"
ACT_SUMS = Path(__file__).with_name("marker_db_act.sha256")
ACT_WORKERS = 4
ACT_DELAY = 0.05  # seconds before each request: a small server

ONTOLOGY_URL = "http://purl.obolibrary.org/obo/cl/cl-basic.obo"
ONTOLOGY_FILE = "cl-basic.obo"
ONTOLOGY_MIN_TERMS = 2000


def marker_db_dir() -> Path:
    from scview.config import get_settings

    return Path(get_settings().DATA_DIR) / "marker_db"


def auto_download_enabled() -> bool:
    from scview.config import get_settings

    return bool(get_settings().MARKER_DB_AUTO_DOWNLOAD)


# ---------------------------------------------------------------------------
# Fetch and store
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(dest: Path) -> Path:
    return dest / "manifest.json"


def read_manifest(dest: Path) -> dict:
    try:
        m = json.loads(_manifest_path(dest).read_text())
        return m if isinstance(m, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_manifest(dest: Path, m: dict) -> None:
    tmp = dest / "manifest.json.part"
    tmp.write_text(json.dumps(m, indent=1))
    tmp.replace(_manifest_path(dest))


def _table(dest: Path, name: str) -> Path:
    return dest / f"{name}.tsv.gz"


def _ontology_table(dest: Path) -> Path:
    return dest / "cl.json.gz"


def _download(urls: tuple[str, ...], part: Path, sha256: str | None, client: httpx.Client) -> tuple[str, str, int]:
    """Fetch the first URL that answers into ``part``; (url, sha256, bytes). Raises
    when none does or the file is not the pinned release."""
    errors = []
    for url in urls:
        try:
            size = 0
            h = hashlib.sha256()
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(part, "wb") as fh:
                    for chunk in resp.iter_bytes():
                        size += len(chunk)
                        if size > MAX_DOWNLOAD_BYTES:
                            raise ValueError(f"download exceeds {MAX_DOWNLOAD_BYTES} bytes")
                        h.update(chunk)
                        fh.write(chunk)
            digest = h.hexdigest()
            if sha256 and digest != sha256:
                raise ValueError(f"sha256 {digest[:12]}... is not the release scView expects ({sha256[:12]}...)")
            return url, digest, size
        except Exception as e:  # network, HTTP, size or checksum: try the next URL
            part.unlink(missing_ok=True)
            errors.append(f"{url.split('/')[2]}: {e}")
    raise RuntimeError("; ".join(errors) or "no URL")


_progress: dict[str, tuple[int, int]] = {}  # source -> (bytes so far, bytes expected), while fetching


def _download_resumable(url: str, part: Path, sha256: str, expected: int, client: httpx.Client,
                        progress: tuple[str, int]) -> None:
    """Fetch ``url`` into ``part``, continuing a partial ``part`` with a Range request,
    then check the pinned sha256 (a mismatch deletes it). A network error keeps the
    partial file for the next try."""
    key, before = progress
    start = part.stat().st_size if part.is_file() else 0
    if start >= expected:
        start = 0
        part.unlink(missing_ok=True)
    headers = {"Range": f"bytes={start}-"} if start else {}
    with client.stream("GET", url, headers=headers) as resp:
        resp.raise_for_status()
        if resp.status_code != 206:
            start = 0  # the server sent the whole file
        size = start
        with open(part, "ab" if start else "wb") as fh:
            for chunk in resp.iter_bytes():
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"download exceeds {MAX_DOWNLOAD_BYTES} bytes")
                fh.write(chunk)
                _progress[key] = (before + size, _progress.get(key, (0, 0))[1])
    digest = _sha256(part)
    if digest != sha256:
        part.unlink(missing_ok=True)
        raise ValueError(f"{part.name.replace('.part', '')}: sha256 {digest[:12]}... is not the release scView "
                         f"expects ({sha256[:12]}...)")


def _fetch_cm3(root: Path, client: httpx.Client | None) -> tuple[str, str, int]:
    """Fetch the CellMarker 3.0 zips not yet under ``root``, each checked against its
    pinned sha256; fails closed (raises) on any mismatch or error."""
    root.mkdir(parents=True, exist_ok=True)
    total = sum(n for _, _, n in CM3_FILES)
    todo = []
    for name, digest, _n in CM3_FILES:
        p = root / name
        if not p.is_file():
            todo.append(name)
        elif _sha256(p) != digest:
            raise ValueError(f"CellMarker 3.0 file {name} is not the release scView expects")
    if todo:
        c, own = _client(client)
        _progress["cellmarker3"] = (0, total)
        try:
            done = 0
            for name, digest, n in CM3_FILES:
                p = root / name
                if p.is_file():
                    done += n
                    continue
                part = root / f"{Path(name).stem}.part.zip"
                _download_resumable(CM3_BASE + name, part, digest, n, c, ("cellmarker3", done))
                part.replace(p)
                done += n
        finally:
            _progress.pop("cellmarker3", None)
            if own:
                c.close()
    digest = hashlib.sha256("".join(f"{d}  {n}\n" for n, d, _ in CM3_FILES).encode()).hexdigest()
    return (CM3_BASE if todo else "local file"), digest, total


def _act_files() -> list[tuple[str, str]]:
    """(relative path, sha256) of every ACT file, from the committed list."""
    out = []
    for line in ACT_SUMS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            out.append((rel.strip(), digest.strip()))
    return out


def _fetch_act(root: Path, client: httpx.Client | None) -> tuple[str, str, int]:
    """Fetch the ACT files not yet under ``root`` (4 at a time, politely), each checked
    against its pinned sha256. Verified files stay, so a retry resumes; any failure
    or mismatch raises and ACT is not loaded."""
    from concurrent.futures import ThreadPoolExecutor
    from urllib.parse import quote

    files = _act_files()
    todo = []
    for rel, digest in files:
        p = root / rel
        if not p.is_file():
            todo.append((rel, digest))
        elif _sha256(p) != digest:
            raise ValueError(f"ACT file {rel} is not the release scView expects")
    if todo:
        c, own = _client(client)

        def one(item):
            rel, digest = item
            time.sleep(ACT_DELAY)
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            part = p.with_name(f"{p.stem}.part{p.suffix}")
            try:
                _download((ACT_BASE + quote(rel),), part, digest, c)
                part.replace(p)
                return None
            except Exception as e:
                part.unlink(missing_ok=True)
                return f"{rel}: {e}"

        try:
            with ThreadPoolExecutor(max_workers=ACT_WORKERS) as pool:
                errors = [e for e in pool.map(one, todo) if e]
        finally:
            if own:
                c.close()
        if errors:
            raise RuntimeError(f"{len(errors)} of {len(files)} ACT files failed ({errors[0]})")
    size = sum((root / rel).stat().st_size for rel, _ in files)
    return (ACT_BASE if todo else "local file"), hashlib.sha256(ACT_SUMS.read_bytes()).hexdigest(), size


def _client(client: httpx.Client | None) -> tuple[httpx.Client, bool]:
    if client is not None:
        return client, False
    return httpx.Client(timeout=TIMEOUT, follow_redirects=True,
                        headers={"User-Agent": "scView marker-db fetch"}), True


def _install_source(spec: SourceSpec, dest: Path, client: httpx.Client | None) -> dict:
    """Fetch (unless the raw file is already in raw/), check, parse and store one source."""
    raw_dir = dest / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / spec.filename
    entry: dict[str, Any] = {"label": spec.label, "release": spec.release}
    stem, suffix = Path(spec.filename).stem, Path(spec.filename).suffix
    part = raw_dir / f"{stem}.part{suffix}"  # openpyxl goes by the extension
    try:
        if spec.fetch is not None:
            served, digest, size = spec.fetch(spec, raw_dir, client)
            entry.update(url=spec.urls[0], served_from=served, sha256=digest, download_bytes=size)
            src = raw
        elif raw.is_file():
            digest = _sha256(raw)
            if spec.sha256 and digest != spec.sha256:
                raise ValueError(f"{raw} is not the release scView expects (sha256 {digest[:12]}...)")
            entry.update(url=spec.urls[0], served_from="local file", sha256=digest, download_bytes=raw.stat().st_size)
            src = raw
        else:
            c, own = _client(client)
            try:
                url, digest, size = _download(spec.urls, part, spec.sha256, c)
            finally:
                if own:
                    c.close()
            entry.update(url=spec.urls[0], served_from=url, sha256=digest, download_bytes=size)
            src = part
        rows = [r for r in spec.parse(src) if r[3] and not SKIP_GENE.match(r[3])]
        if len(rows) < spec.min_rows:
            raise ValueError(f"{spec.label} file gave {len(rows)} marker rows (expected at least {spec.min_rows})")
        tmp = dest / f"{spec.name}.tsv.gz.part"
        with gzip.open(tmp, "wt", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            # curated_weight: the weight from curated rows only (CellMarker 3.0; 0 = computational
            # rows only); the other sources are all curated
            w.writerow(["species", "cell_type", "cl_id", "gene", "weight", "curated_weight"])
            w.writerows(r if len(r) == 6 else (*r, r[4]) for r in rows)
        tmp.replace(_table(dest, spec.name))
        if src == part:
            part.replace(raw)
        entry.update(
            fetched_at=_now(),
            rows=len(rows),
            types={sp: len({r[1] for r in rows if r[0] == sp}) for sp in ("human", "mouse")},
        )
        if any(len(r) == 6 for r in rows[:1]):
            entry["curated_rows"] = sum(1 for r in rows if r[5] > 0)
        return entry
    finally:
        part.unlink(missing_ok=True)
        (dest / f"{spec.name}.tsv.gz.part").unlink(missing_ok=True)


def _install_ontology(dest: Path, client: httpx.Client | None) -> dict:
    raw_dir = dest / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / ONTOLOGY_FILE
    part = raw_dir / f"{ONTOLOGY_FILE}.part"
    entry: dict[str, Any] = {"label": "Cell Ontology"}
    try:
        if raw.is_file():
            src, entry["served_from"], entry["sha256"] = raw, "local file", _sha256(raw)
        else:
            c, own = _client(client)
            try:
                url, digest, size = _download((ONTOLOGY_URL,), part, None, c)
            finally:
                if own:
                    c.close()
            src = part
            entry.update(served_from=url, sha256=digest, download_bytes=size)
        text = src.read_text(encoding="utf-8", errors="replace")
        terms = parse_obo(text)
        if len(terms) < ONTOLOGY_MIN_TERMS or "CL:0000000" not in terms:
            raise ValueError(f"Cell Ontology file gave {len(terms)} terms (expected at least {ONTOLOGY_MIN_TERMS})")
        m = re.search(r"^data-version: (.+)$", text[:5000], re.M)
        tmp = dest / "cl.json.gz.part"
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump({k: list(v) for k, v in terms.items()}, f)
        tmp.replace(_ontology_table(dest))
        if src == part:
            part.replace(raw)
        entry.update(url=ONTOLOGY_URL, fetched_at=_now(), terms=len(terms),
                     version=m.group(1).strip() if m else None)
        return entry
    finally:
        part.unlink(missing_ok=True)
        (dest / "cl.json.gz.part").unlink(missing_ok=True)


_fetch_lock = threading.Lock()
_busy = threading.Event()


def _raw_complete(dest: Path, name: str) -> bool:
    """Every raw file of a source is in raw/ already (put there by hand): nothing to fetch."""
    raw = dest / "raw" / (ONTOLOGY_FILE if name == "ontology" else SOURCES[name].filename)
    if name == "act":
        return all((raw / rel).is_file() for rel, _ in _act_files())
    if name == "cellmarker3":
        return all((raw / f).is_file() for f, _, _ in CM3_FILES)
    return raw.is_file()


def _installed(dest: Path, name: str) -> bool:
    return _ontology_table(dest).is_file() if name == "ontology" else _table(dest, name).is_file()


def _needed(dest: Path, name: str) -> bool:
    """CellMarker 2.0 is the fallback: needed only while 3.0 is not installed."""
    return not (name == "cellmarker2" and _installed(dest, "cellmarker3"))


def ensure(
    dest: Path,
    *,
    auto_download: bool | None = None,
    retry: bool = False,
    client: httpx.Client | None = None,
) -> dict:
    """Fetch whatever is missing (call in a worker thread); return ``status``.

    A source that failed less than ``FAILURE_COOLDOWN`` seconds ago is not tried
    again unless ``retry``, so a dead site does not stall every check. One fetch at
    a time per process (a lock) and across processes (a lock file): a second
    caller waits, then finds the files in place."""
    auto = auto_download_enabled() if auto_download is None else auto_download
    names = [*SOURCES, "ontology"]  # CellMarker 3.0 is tried before its 2.0 fallback
    if all(_installed(dest, n) or not _needed(dest, n) for n in names):
        return status(dest, auto_download=auto)
    with _fetch_lock:
        try:
            dest.mkdir(parents=True, exist_ok=True)
            with open(dest / ".download.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                _busy.set()
                try:
                    _ensure_locked(dest, names, auto, retry, client)
                finally:
                    _busy.clear()
        except OSError as e:
            logger.warning("Could not prepare the marker database directory %s: %s", dest, e)
    return status(dest, auto_download=auto)


def _ensure_locked(dest: Path, names: list[str], auto: bool, retry: bool, client) -> None:
    m = read_manifest(dest)
    sources, failures = m.setdefault("sources", {}), m.setdefault("failures", {})
    for name in names:
        if _installed(dest, name) or not _needed(dest, name):
            continue
        if not auto and not _raw_complete(dest, name):
            continue
        last = failures.get(name, {}).get("at_epoch", 0)
        if not retry and time.time() - last < FAILURE_COOLDOWN:
            continue
        t0 = time.monotonic()
        try:
            entry = _install_ontology(dest, client) if name == "ontology" else _install_source(SOURCES[name], dest, client)
            entry["seconds"] = round(time.monotonic() - t0, 1)
            sources[name] = entry
            failures.pop(name, None)
            logger.info("Marker database %s ready (%s)", name, entry)
        except Exception as e:
            failures[name] = {"error": str(e)[:500], "at": _now(), "at_epoch": time.time()}
            logger.warning("Could not fetch marker database %s: %s", name, e)
        _write_manifest(dest, m)


def status(dest: Path, *, auto_download: bool | None = None) -> dict:
    """What is on disk, what failed, and whether fetching is on: for the UI."""
    auto = auto_download_enabled() if auto_download is None else auto_download
    m = read_manifest(dest)
    sources, failures = m.get("sources", {}), m.get("failures", {})
    out = []
    for name, spec in SOURCES.items():
        e = sources.get(name, {}) if _installed(dest, name) else {}
        f = failures.get(name)
        prog = _progress.get(name)
        out.append({
            "name": name, "label": spec.label, "release": spec.release, "home": spec.home,
            "paper": spec.paper, "doi": spec.doi, "url": spec.urls[0],
            "downloaded": _installed(dest, name),
            # False for CellMarker 2.0 while 3.0 is installed: not fetched, not missing
            "needed": _needed(dest, name),
            "progress": {"done": prog[0], "total": prog[1]} if prog else None,
            "fetched_at": e.get("fetched_at"), "served_from": e.get("served_from"),
            "sha256": e.get("sha256"), "rows": e.get("rows"), "types": e.get("types"),
            "download_bytes": e.get("download_bytes"),
            "error": f.get("error") if f and not _installed(dest, name) else None,
            "failed_at": f.get("at") if f and not _installed(dest, name) else None,
        })
    n = sum(s["downloaded"] for s in out)
    failed = any(s["error"] for s in out)
    have = {s["name"] for s in out if s["downloaded"]}
    cellmarker = "cellmarker3" if "cellmarker3" in have else "cellmarker2" if "cellmarker2" in have else None
    state = (
        "downloading" if _busy.is_set()
        else "ready" if {"panglao", "act"} <= have and cellmarker
        else "partial" if n
        else "failed" if failed
        else "disabled" if not auto
        else "missing"
    )
    onto = sources.get("ontology", {}) if _installed(dest, "ontology") else {}
    of = failures.get("ontology")
    return {
        "state": state,
        "auto_download": auto,
        # The CellMarker release in use: 3.0, or 2.0 when 3.0 could not be fetched
        "cellmarker": SOURCES[cellmarker].label if cellmarker else None,
        "sources": out,
        "ontology": {
            "downloaded": _installed(dest, "ontology"), "url": ONTOLOGY_URL,
            "version": onto.get("version"), "fetched_at": onto.get("fetched_at"),
            "error": of.get("error") if of and not _installed(dest, "ontology") else None,
        },
    }


# ---------------------------------------------------------------------------
# The merged database
# ---------------------------------------------------------------------------


def _read_types(path: Path, curated: bool = False) -> dict[str, dict[str, dict]]:
    """species -> cell type -> {"cl", "w": {gene: weight}, "genes": best first}; with
    ``curated``, from the curated rows' weights only."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            wt = float(r.get("curated_weight") or r["weight"]) if curated else float(r["weight"])
            if wt <= 0:
                continue
            t = out[r["species"]].setdefault(r["cell_type"], {"cl": r["cl_id"], "w": {}})
            g = r["gene"]
            t["w"][g] = max(wt, t["w"].get(g, 0.0))
            if r["cl_id"] and not t["cl"]:
                t["cl"] = r["cl_id"]
    for types in out.values():
        for t in types.values():
            t["genes"] = sorted(t["w"], key=lambda g: (-t["w"][g], g))
    return out


def merge(parts: list[tuple[str, dict[str, dict]]]) -> dict[str, dict]:
    """Types merged by normalised name across sources. A gene's merged weight is the
    sum over sources of its within-source rank score (1 at the top of that source's
    list, 0 from rank 50 on), so both sources' best genes lead."""
    merged: dict[str, dict] = {}
    for src, types in parts:
        for name, t in types.items():
            k = norm(name)
            if not k:
                continue
            m = merged.setdefault(" ".join(sorted(k)), {"cl": "", "w": defaultdict(float), "label": name, "sources": []})
            if t["cl"] and not m["cl"]:
                m["cl"] = t["cl"]
            if src not in m["sources"]:
                m["sources"].append(src)
            for i, g in enumerate(t["genes"]):
                m["w"][g] += max(0.0, (TOP_DB - i) / TOP_DB) + 1e-3
    out: dict[str, dict] = {}
    for key, m in merged.items():
        name = m["label"] if m["label"] not in out else key
        out[name] = {"cl": m["cl"], "sources": m["sources"],
                     "genes": sorted(m["w"], key=lambda g: (-m["w"][g], g))}
    return out


class SpeciesDB:
    """One species' merged database types with their name matcher."""

    def __init__(self, species: str, types: dict[str, dict], ontology: Ontology | None):
        self.species = species
        self.types = types
        self.names = sorted(types)
        self.matcher = Matcher({n: t["cl"] for n, t in types.items()}, ontology)


class MarkerDB:
    def __init__(self, dest: Path, sources: list[str], per_source: dict[str, dict], ontology: Ontology | None):
        self.dest = dest
        self.sources = sources
        self.per_source = per_source
        self.ontology = ontology
        self._species: dict[str, SpeciesDB] = {}  # "species|sources" -> merged types
        self._lock = threading.Lock()

    def species(self, species: str, sources: list[str] | tuple[str, ...] | None = None) -> SpeciesDB:
        """One species' merged types, from the parts named in ``sources`` (in that order),
        or from every source on disk."""
        use = [s for s in (sources if sources is not None else self.sources) if s in self.per_source]
        key = f"{species}|{','.join(use)}"
        with self._lock:
            if key not in self._species:
                parts = [(s, self.per_source[s].get(species, {})) for s in use]
                self._species[key] = SpeciesDB(species, merge(parts), self.ontology)
            return self._species[key]

    @property
    def cellmarker(self) -> str | None:
        """The CellMarker release in use."""
        return next((s for s in ("cellmarker3", "cellmarker2") if s in self.sources), None)

    def view(self, job: str, species: str) -> list[str]:
        """The parts on disk that ``job`` (verdict | suggest | annotate) uses for ``species``."""
        views = VIEWS_CM3 if self.cellmarker == "cellmarker3" else VIEWS_CM2
        parts = [p for p in views[job]["mouse" if species == "mouse" else "human"] if p in self.per_source]
        return parts or list(self.sources)

    def labels(self, parts) -> list[str]:
        return [PART_LABELS[p] for p in parts]

    def source_labels(self) -> list[str]:
        return self.labels(self.sources)


_loaded: dict[str, tuple[tuple, MarkerDB]] = {}
_load_lock = threading.Lock()


def load(dest: Path) -> MarkerDB | None:
    """The merged database from what is on disk (no fetching), or None when no source is."""
    files = [(n, _table(dest, n)) for n in SOURCES]
    present = [(n, p) for n, p in files if p.is_file()]
    if not present:
        return None
    onto = _ontology_table(dest)
    stamp = tuple((n, p.stat().st_mtime) for n, p in present) + ((onto.stat().st_mtime,) if onto.is_file() else ())
    with _load_lock:
        hit = _loaded.get(str(dest))
        if hit and hit[0] == stamp:
            return hit[1]
        ontology = None
        if onto.is_file():
            try:
                with gzip.open(onto, "rt", encoding="utf-8") as f:
                    ontology = Ontology({k: (v[0], list(v[1]), list(v[2])) for k, v in json.load(f).items()})
            except (OSError, ValueError, IndexError) as e:
                logger.warning("Cell Ontology table unreadable (%s); matching by name only", e)
        parts = {n: _read_types(p) for n, p in present}
        if "cellmarker3" in parts:
            parts["cellmarker3_curated"] = _read_types(_table(dest, "cellmarker3"), curated=True)
        db = MarkerDB(dest, [n for n, _ in present], parts, ontology)
        _loaded[str(dest)] = (stamp, db)
        return db


def get_db(
    dest: Path | None = None,
    *,
    auto_download: bool | None = None,
    retry: bool = False,
    client: httpx.Client | None = None,
) -> MarkerDB:
    """The database, fetched first where missing and allowed (worker thread). Raises
    ``MarkerDBUnavailable`` saying why when no source is available."""
    dest = dest or marker_db_dir()
    st = ensure(dest, auto_download=auto_download, retry=retry, client=client)
    db = load(dest)
    if db is None:
        if not st["auto_download"]:
            raise MarkerDBUnavailable(
                "The marker databases are not downloaded and MARKER_DB_AUTO_DOWNLOAD is off. "
                f"Put the files in {dest / 'raw'} or turn the download on.")
        errs = "; ".join(f"{s['label']}: {s['error']}" for s in st["sources"] if s["error"])
        raise MarkerDBUnavailable(
            "The marker databases could not be downloaded"
            + (f" ({errs})" if errs else "")
            + ". Check the internet connection and retry.")
    return db


# ---------------------------------------------------------------------------
# Group markers
# ---------------------------------------------------------------------------


def measured_genes(adata) -> pd.Index:
    """The genes a marker test could rank: ``.raw`` when present (the log-normalised
    values rank_genes_groups uses by default), else ``var``."""
    return adata.raw.var_names if adata.raw is not None else adata.var_names


def _group_names(adata, groupby: str) -> list[str]:
    col = adata.obs[groupby]
    present = set(col.dropna().astype(str))
    cats = [str(c) for c in col.cat.categories] if isinstance(col.dtype, pd.CategoricalDtype) else sorted(present)
    return [c for c in cats if c in present]


def _stored_key(adata, groupby: str, groups: list[str]) -> str | None:
    """The stored rank_genes_groups key for ``groupby`` when it covers the column's
    current groups (a column relabelled since its markers were computed does not)."""
    from scview.core.ora_background import resolve_markers_key

    key = resolve_markers_key(adata.uns, groupby)
    try:
        names = adata.uns[key]["names"].dtype.names if key else None
    except (KeyError, TypeError, AttributeError):
        names = None
    return key if names is not None and {str(n) for n in names} == set(groups) else None


def _computed(adata, groupby: str, groups: list[str]):
    """Wilcoxon markers (with pts) on a throwaway object sharing the matrix, so the
    dataset is not touched; None when fewer than two groups have 3 cells."""
    import anndata as ad
    import scanpy as sc

    src = adata.raw if adata.raw is not None else adata
    X = src.X
    if getattr(adata, "isbacked", False) and not isinstance(X, np.ndarray):
        X = X[:]
    counts = adata.obs[groupby].astype(str).value_counts()
    keep = [g for g in groups if counts.get(g, 0) >= 3]
    if len(keep) < 2:
        return None
    obs = pd.DataFrame({groupby: pd.Categorical(adata.obs[groupby].astype(str), categories=groups)},
                       index=adata.obs_names)
    tmp = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=src.var_names))
    sc.tl.rank_genes_groups(tmp, groupby=groupby, groups=keep, reference="rest", method="wilcoxon",
                            pts=True, use_raw=False, key_added=markers_key(groupby))
    return tmp


def _markers_for(adata, groupby: str) -> dict[str, list[str]]:
    """Each group's top 25 markers for the database check: ``core/marker_selection.py``
    (detected in >= 20% of the group, adjusted p < 0.05, by log fold change, no
    ribosomal / mito genes),
    from the column's stored rank_genes_groups, else computed without saving.
    Score order instead put FTL/CD63/ANXA5 ahead of CD14 in Kang's CD14 monocytes
    and ribosomal genes in the T-cell top 15 (Kang CellMarker lookup 0.46 -> 0.69).
    """
    from scview.core.marker_selection import group_markers

    groups = _group_names(adata, groupby)
    key = _stored_key(adata, groupby, groups)
    out = group_markers(adata, groupby, n=N_MARKERS, key=key) if key else {}
    if out:
        return out
    tmp = _computed(adata, groupby, groups)
    return group_markers(tmp, groupby, n=N_MARKERS) if tmp is not None else {}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class Scorer:
    """One species' database restricted to a dataset's measured genes (upper case)."""

    def __init__(self, sdb: SpeciesDB, measured):
        self.sdb = sdb
        self.case = {}
        for g in measured:
            self.case.setdefault(str(g).upper(), str(g))
        self.U = frozenset(self.case)
        self.top = {t: frozenset(g.upper() for g in sdb.types[t]["genes"][:TOP_DB]) & self.U for t in sdb.names}
        self.full = {t: frozenset(g.upper() for g in sdb.types[t]["genes"]) & self.U for t in sdb.names}
        self.open_names = [t for t in sdb.names if len(self.top[t]) >= MIN_DB_GENES]

    def query(self, genes: list[str]) -> frozenset:
        return frozenset(str(g).upper() for g in genes) & self.U

    def scores(self, Q: frozenset, tnames: list[str], full: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(-log10 p, overlap, measured set size) per type: hypergeometric over-representation of
        the type's top-50 (or full) measured set among ``Q``."""
        sets = self.full if full else self.top
        k = np.array([len(Q & sets[t]) for t in tnames], dtype=np.int64)
        n = np.array([len(sets[t]) for t in tnames], dtype=np.int64)
        if not len(tnames):
            return np.zeros(0), k, n
        p = hypergeom.sf(k - 1, len(self.U), n, len(Q))
        s = -np.log10(np.clip(p, 1e-300, 1.0))
        s[k == 0] = 0.0
        return s, k, n

    def genes(self, genes) -> list[str]:
        return sorted(self.case[g] for g in genes)

    def suggest(self, Q: frozenset, n: int = N_SUGGEST) -> list[dict]:
        if not Q:
            return []
        s, k, size = self.scores(Q, self.open_names, full=True)
        order = np.argsort(-s, kind="stable")[:n]
        return [self._type(self.open_names[i], Q, s[i], k[i], size[i], full=True) for i in order if k[i] > 0]

    def _type(self, t: str, Q, s, k, n, full=False) -> dict:
        info = self.sdb.types[t]
        sets = self.full if full else self.top
        return {
            "cell_type": t,
            "cl_id": info["cl"] or None,
            "sources": [PART_LABELS[x] for x in info["sources"]],
            "p": float(10 ** -s) if k else 1.0,
            "overlap": self.genes(Q & sets[t]),
            "n_db_genes": int(n),
        }


def _species(adata, species: str | None) -> tuple[str, str | None]:
    if species in ("human", "mouse"):
        return species, None
    from scview.core.species import detect_species

    call = detect_species(adata).species
    if call in ("human", "mouse"):
        return call, None
    return "human", "The species could not be told from the gene names; human database rows are used."


def _label_row(sc_: Scorer, sc_suggest: Scorer, label: str, markers: list[str]) -> dict:
    Q = sc_.query(markers)
    matched, how = sc_.sdb.matcher.match(label, "C")
    row: dict[str, Any] = {
        "label": label,
        "markers": markers,
        "match": {"types": matched, "how": how if matched else None,
                  "how_text": HOW_TEXT.get(how) if matched else None,
                  "via_parent": how == "cl-ancestor"},
        "db_type": None, "cl_id": None, "sources": [], "p": None, "overlap": [], "n_db_genes": 0,
        "suggestions": sc_suggest.suggest(Q),
    }
    if not Q:
        row["verdict"] = "no_markers"
    elif not matched:
        row["verdict"] = "not_in_db"
    else:
        s, k, n = sc_.scores(Q, matched)
        # best by score, then overlap; with no overlap anywhere, the type with the most measured genes
        i = max(range(len(matched)), key=lambda j: (s[j], k[j], n[j], -j))
        best = sc_._type(matched[i], Q, s[i], k[i], n[i])
        row.update(db_type=best["cell_type"], sources=best["sources"], p=best["p"], overlap=best["overlap"],
                   n_db_genes=best["n_db_genes"], cl_id=best["cl_id"])
        row["verdict"] = "too_few" if n[i] < MIN_DB_GENES else "supported" if k[i] > 0 else "no_support"
    row["verdict_text"] = VERDICT_TEXT[row["verdict"]]
    return row


def check(adata, column: str, species: str | None = None, db: MarkerDB | None = None) -> dict:
    """Database support for each label of ``column`` and the 3 best database types for
    its cells (call in a worker thread; reads the dataset, changes nothing)."""
    t0 = time.monotonic()
    if column not in adata.obs.columns:
        raise KeyError(column)
    groups = _group_names(adata, column)
    if not 2 <= len(groups) <= MAX_GROUPS:
        raise ValueError(f"'{column}' has {len(groups)} labels; the check needs 2 to {MAX_GROUPS}.")
    db = db or get_db()
    sp, sp_note = _species(adata, species)
    verdict_from, suggest_from = db.view("verdict", sp), db.view("suggest", sp)
    sdb = db.species(sp, verdict_from)
    if not sdb.types:
        raise ValueError(f"The marker databases have no {sp} rows.")
    measured = measured_genes(adata)
    sc_ = Scorer(sdb, measured)
    sc_suggest = sc_ if suggest_from == verdict_from else Scorer(db.species(sp, suggest_from), measured)
    markers = _markers_for(adata, column)
    labels = [_label_row(sc_, sc_suggest, g, markers.get(g, [])) for g in groups]
    counts = Counter(r["verdict"] for r in labels)
    return {
        "column": column,
        "species": sp,
        "species_note": sp_note,
        "labels": labels,
        "verdict_counts": {v: counts.get(v, 0) for v in VERDICT_TEXT},
        "n_measured": len(sc_.U),
        "n_db_types": len(sdb.types),
        "ontology": db.ontology is not None,
        "cellmarker": SOURCES[db.cellmarker].label if db.cellmarker else None,
        "verdict_sources": db.labels(verdict_from),
        "suggestion_sources": db.labels(suggest_from),
        "sources": [s for s in status(db.dest)["sources"] if s["downloaded"] and s["needed"]],
        "params": {"top_db_genes": TOP_DB, "group_markers": N_MARKERS, "min_db_genes": MIN_DB_GENES,
                   "suggestions": N_SUGGEST},
        "note": NOTE,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }


def annotate(adata, groupby: str, species: str | None = None, db: MarkerDB | None = None) -> tuple[dict[str, str], dict]:
    """Each group's best database type (the top suggestion), "Unknown" when no database
    type shares a marker with it; and what it rests on, for the provenance."""
    db = db or get_db()
    sp, _ = _species(adata, species)
    parts = db.view("annotate", sp)
    sc_ = Scorer(db.species(sp, parts), measured_genes(adata))
    markers = _markers_for(adata, groupby)
    mapping: dict[str, str] = {}
    for g in _group_names(adata, groupby):
        top = sc_.suggest(sc_.query(markers.get(g, [])), n=1)
        mapping[g] = top[0]["cell_type"] if top else "Unknown"
    return mapping, {
        "marker_source": "database",
        "databases": db.labels(parts),
        "cellmarker": SOURCES[db.cellmarker].label if db.cellmarker else None,
        "species": sp,
        "n_db_types": len(sc_.open_names),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(json.dumps(ensure(marker_db_dir(), auto_download=True, retry=True), indent=1))
