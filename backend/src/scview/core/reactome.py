"""Reactome pathway diagrams, laid out by Reactome, coloured by a DE result.

An enrichment row whose MSigDB set came from Reactome (``source_id`` R-HSA-n or
R-MMU-n, core/msigdb_loader.py) opens the pathway's own diagram. Reactome
publishes each diagram as two JSON files: the layout (``{stId}.json``: nodes,
reactions, compartments, shadows, with their coordinates) and the graph
(``{stId}.graph.json``: what each node is, its complex and set members, gene
names). They are fetched on first use from the release-pinned folder
``/download/<RELEASE>/diagram/`` into ``DATA_DIR/reactome/<RELEASE>/diagram`` and
kept for good (Reactome keeps only the current release online, so a cached
file is the only copy once the release moves on). There is no bulk JSON
archive, so a whole-release download is not an option; a diagram is ~200 KB.

Only 1,142 of Reactome's 2,889 human pathways have a diagram of their own; the
others are drawn inside an ancestor's, as a "shadow" outline around their
reactions. The hierarchy (``ReactomePathwaysRelation.txt``, parent -> child,
every species) and the list of pathways with diagrams
(``humanPathwaysWithDiagrams.txt``; there is no mouse list: an inferred mouse
pathway R-MMU-n has a diagram when R-HSA-n does) are fetched once, sha256
pinned, and a pathway without a diagram resolves to its nearest ancestor with
one; the model then carries the pathway's outline and its reactions to
highlight. A top-level overview whose diagram holds only sub-pathway boxes
(Interferon Signaling, Immune System) has nothing to colour: the model says
"too broad" and lists the sub-pathways to open instead.

Genes: a layout node is the graph node with its id among the graph node's
``diagramIds`` (or with ``dbId`` = the node's ``reactomeId``). A protein, gene
or RNA has ``geneNames`` (primary symbol first; mouse diagrams carry mouse
symbols); a complex or set has ``children``, followed down to them. No
UniProt mapping table is needed.

The model sent to the browser is simplified and renderer-ready: rounded
coordinates, base classes (a drug variant is drawn as its base class), every
string stripped of control characters and capped, no colours taken from the
files. Fetching sends Reactome nothing but the pathway id in the URL; Reactome's
analysis service, which takes uploaded data, is never used.

Licence: Reactome data CC0, diagrams CC BY 4.0 (attribution with a licence link
on every diagram). ``REACTOME_AUTO_DOWNLOAD=false`` turns fetching off; files
placed by hand in the folder above are used without any download. Fetch by
hand: ``docker compose exec backend python -m scview.core.reactome R-HSA-1059683``.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import logging
import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)

RELEASE = 98
BASE_URL = "https://reactome.org/download"
LICENCE = "CC BY 4.0"
LICENCE_URL = "https://reactome.org/license"
CITATION = "Milacic et al. 2024, Nucleic Acids Res (PMID 37941124)"
ATTRIBUTION = f"Reactome ({LICENCE}), release {RELEASE}"
BROWSER_URL = "https://reactome.org/PathwayBrowser/#/{st_id}"

# Frozen per release: pinned. A file that does not match is refused.
HIERARCHY_FILES: dict[str, str] = {
    "ReactomePathwaysRelation.txt": "aa3609ee852442899b400bf4817b36311d820883ecd75259ed77531d93d8f1dc",
    "humanPathwaysWithDiagrams.txt": "10f418a315816fe832320b7fdb945a648b94750cc0c352b0b92d7ac4b9d23e71",
}
MAX_DIAGRAM_BYTES = 40 * 1024 * 1024  # the largest layouts are a few MB
MAX_HIERARCHY_BYTES = 8 * 1024 * 1024
TIMEOUT = httpx.Timeout(60.0, connect=15.0)
PADJ_CUTOFF = 0.05
MAX_TEXT = 200
MAX_MODELS = 32  # built models kept in memory
# Fewer rows than this in the hierarchy files: truncated (release 98: 1,142 and ~5,000)
MIN_DIAGRAMS = 100
MIN_RELATIONS = 1000

ST_ID = re.compile(r"^R-(HSA|MMU)-\d{1,12}$")
SPECIES = {"HSA": "human", "MMU": "mouse"}
# Layout classes drawn as a pathway box, not an entity.
PATHWAY_CLASSES = frozenset({"ProcessNode", "EncapsulatedNode"})
BASE_CLASSES = frozenset({"Protein", "Complex", "Chemical", "EntitySet", "Gene", "RNA", "Entity",
                          "ProcessNode", "EncapsulatedNode"})
CONNECTOR_TYPES = frozenset({"INPUT", "OUTPUT", "CATALYST", "ACTIVATOR", "INHIBITOR"})
SHAPE_TYPES = frozenset({"ARROW", "CIRCLE", "DOUBLE_CIRCLE", "STOP", "BOX"})
LINK_CLASSES = frozenset({"EntitySetAndMemberLink", "Interaction", "FlowLine"})

_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏ -‮⁦-⁩]")


class ReactomeUnavailable(RuntimeError):
    """A file is not on disk and could not (or may not) be fetched."""


class PathwayNotFound(LookupError):
    """The release has no such pathway, or no diagram for it or any ancestor."""


class _NotThere(Exception):
    """The server answered 404."""


def check_st_id(st_id: str) -> str:
    """The id, or ValueError when it is not a human or mouse Reactome pathway id."""
    s = str(st_id or "").strip()
    if not ST_ID.match(s):
        raise ValueError(f"'{_text(st_id, 40)}' is not a human or mouse Reactome pathway id "
                         "(R-HSA-<number> or R-MMU-<number>).")
    return s


def species_of(st_id: str) -> str:
    return SPECIES[st_id.split("-")[1]]


def release_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "reactome" / str(RELEASE)


def _text(value: Any, limit: int = MAX_TEXT) -> str:
    """A string safe to show: control and direction characters out, whitespace
    collapsed, at most ``limit`` characters."""
    s = _CONTROL.sub(" ", str(value if value is not None else ""))
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _num(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(v, 1) if math.isfinite(v) else 0.0


def _pt(p: Any) -> list[float]:
    p = p if isinstance(p, dict) else {}
    return [_num(p.get("x")), _num(p.get("y"))]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _fetch(client: httpx.Client, url: str, dest: Path, *, max_bytes: int, sha256: str | None = None) -> bytes:
    """Download ``url`` to ``dest`` (atomically). _NotThere on 404; ValueError when
    the file is too large or its sha256 is not the pinned one."""
    buf = io.BytesIO()
    digest = hashlib.sha256()
    with client.stream("GET", url) as resp:
        if resp.status_code == 404:
            raise _NotThere(url)
        resp.raise_for_status()
        for chunk in resp.iter_bytes():
            if buf.tell() + len(chunk) > max_bytes:
                raise ValueError(f"{url.rsplit('/', 1)[-1]} exceeds {max_bytes} bytes")
            digest.update(chunk)
            buf.write(chunk)
    if sha256 and digest.hexdigest() != sha256:
        raise ValueError(f"{url.rsplit('/', 1)[-1]}: sha256 {digest.hexdigest()[:12]}... is not the "
                         f"release {RELEASE} file scView expects ({sha256[:12]}...)")
    raw = buf.getvalue()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        tmp.write_bytes(raw)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return raw


_fetch_lock = threading.Lock()


def _ensure(paths_urls: list[tuple[Path, str, str | None, int]], root: Path, *, auto_download: bool,
            client: httpx.Client | None, what: str) -> None:
    """Fetch every (path, url, sha256, max bytes) not yet on disk. _NotThere passes
    through; anything else becomes ReactomeUnavailable saying what to do."""
    missing = [t for t in paths_urls if not t[0].is_file()]
    if not missing:
        return
    if not auto_download:
        raise ReactomeUnavailable(
            f"The Reactome {what} is not downloaded and REACTOME_AUTO_DOWNLOAD is off. Fetch it once "
            "with: docker compose exec backend python -m scview.core.reactome <pathway id>, or place "
            f"the files in {root}."
        )
    own = client is None
    try:
        with _fetch_lock:
            root.mkdir(parents=True, exist_ok=True)
            with open(root / ".download.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                client = client or httpx.Client(timeout=TIMEOUT, follow_redirects=True)
                for path, url, sha, cap in missing:
                    if not path.is_file():
                        _fetch(client, url, path, max_bytes=cap, sha256=sha)
    except _NotThere:
        raise
    except Exception as e:
        logger.warning("Could not fetch the Reactome %s: %s", what, e)
        raise ReactomeUnavailable(
            f"The Reactome {what} is not downloaded yet and fetching it failed ({e}). scView fetches "
            f"each diagram once from reactome.org (release {RELEASE}); check the internet connection "
            "and try again."
        ) from e
    finally:
        if own and client is not None:
            client.close()


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


@dataclass
class Hierarchy:
    parents: dict[str, list[str]] = field(default_factory=dict)  # child -> parents (human, mouse)
    known: set[str] = field(default_factory=set)
    with_diagram: set[str] = field(default_factory=set)  # human ids

    def has_diagram(self, st_id: str) -> bool:
        """Human: listed; mouse: its human counterpart is (inferred layouts share the number)."""
        return f"R-HSA-{st_id.split('-')[2]}" in self.with_diagram

    def candidates(self, st_id: str) -> list[tuple[str, int]]:
        """(pathway, steps up) with a diagram, nearest first: the pathway itself, then
        its parents, then theirs (breadth first, ties by id)."""
        out: list[tuple[str, int]] = []
        seen = {st_id}
        level = [st_id]
        hops = 0
        while level:
            out.extend((p, hops) for p in sorted(level) if self.has_diagram(p))
            nxt = {q for p in level for q in self.parents.get(p, ()) if q not in seen}
            seen |= nxt
            level = sorted(nxt)
            hops += 1
        return out

    def ancestors(self, st_id: str) -> list[str]:
        """Every ancestor, nearest first."""
        out: list[str] = []
        seen = {st_id}
        level = [st_id]
        while level:
            nxt = sorted({q for p in level for q in self.parents.get(p, ()) if q not in seen})
            seen |= set(nxt)
            out.extend(nxt)
            level = nxt
        return out


def parse_hierarchy(relation: str, with_diagrams: str) -> Hierarchy:
    h = Hierarchy()
    for line in relation.splitlines():
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        parent, child = parts
        if not (ST_ID.match(parent) and ST_ID.match(child)):
            continue  # other species
        h.parents.setdefault(child, []).append(parent)
        h.known.update((parent, child))
    for line in with_diagrams.splitlines()[1:]:
        sid = line.split("\t", 1)[0].strip()
        if ST_ID.match(sid):
            h.with_diagram.add(sid)
            h.known.add(sid)
    if len(h.with_diagram) < MIN_DIAGRAMS or len(h.parents) < MIN_RELATIONS:
        raise ValueError("The Reactome hierarchy files look truncated.")
    return h


_hierarchy: dict[str, tuple[float, Hierarchy]] = {}


def load_hierarchy(data_dir: str | Path, *, auto_download: bool = True,
                   client: httpx.Client | None = None) -> Hierarchy:
    root = release_dir(data_dir)
    files = [(root / name, f"{BASE_URL}/{RELEASE}/{name}", sha, MAX_HIERARCHY_BYTES)
             for name, sha in HIERARCHY_FILES.items()]
    try:
        _ensure(files, root, auto_download=auto_download, client=client, what="pathway hierarchy")
    except _NotThere as e:
        raise ReactomeUnavailable(
            f"reactome.org no longer serves release {RELEASE} ({e}). Update scView for the current "
            "Reactome release, or place the release files by hand."
        ) from e
    rel, diag = (p for p, _u, _s, _c in files)
    key = str(root)
    mtime = rel.stat().st_mtime + diag.stat().st_mtime
    cached = _hierarchy.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    h = parse_hierarchy(rel.read_text(encoding="utf-8", errors="replace"),
                        diag.read_text(encoding="utf-8", errors="replace"))
    _write_manifest(root)
    _hierarchy[key] = (mtime, h)
    return h


def _write_manifest(root: Path) -> None:
    path = root / "manifest.json"
    if path.is_file():
        return
    meta = {
        "release": RELEASE, "licence": LICENCE, "licence_url": LICENCE_URL, "citation": CITATION,
        "base_url": f"{BASE_URL}/{RELEASE}",
        "hierarchy": {n: {"url": f"{BASE_URL}/{RELEASE}/{n}", "sha256": s} for n, s in HIERARCHY_FILES.items()},
        "diagrams": f"{BASE_URL}/{RELEASE}/diagram/<stId>.json and .graph.json, fetched on first use",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        path.write_text(json.dumps(meta, indent=1))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Diagram files
# ---------------------------------------------------------------------------


def load_files(st_id: str, data_dir: str | Path, *, auto_download: bool = True,
               client: httpx.Client | None = None) -> tuple[dict, dict]:
    """(layout, graph) of a pathway with its own diagram. _NotThere when the release
    has none; ValueError when a file is not a Reactome diagram of that pathway."""
    root = release_dir(data_dir)
    base = f"{BASE_URL}/{RELEASE}/diagram/{st_id}"
    layout_p = root / "diagram" / f"{st_id}.json"
    graph_p = root / "diagram" / f"{st_id}.graph.json"
    _ensure([(layout_p, f"{base}.json", None, MAX_DIAGRAM_BYTES),
             (graph_p, f"{base}.graph.json", None, MAX_DIAGRAM_BYTES)],
            root, auto_download=auto_download, client=client, what=f"diagram of {st_id}")
    try:
        layout = json.loads(layout_p.read_text(encoding="utf-8"))
        graph = json.loads(graph_p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        layout_p.unlink(missing_ok=True)
        graph_p.unlink(missing_ok=True)
        raise ValueError(f"The Reactome diagram files of {st_id} could not be read ({e}); "
                         "they were removed and will be fetched again.") from e
    if (not isinstance(layout, dict) or not isinstance(graph, dict)
            or not isinstance(layout.get("nodes"), list)
            or str(layout.get("stableId", st_id)) != st_id or str(graph.get("stId", st_id)) != st_id):
        layout_p.unlink(missing_ok=True)
        graph_p.unlink(missing_ok=True)
        raise ValueError(f"The files fetched for {st_id} are not its Reactome diagram.")
    return layout, graph


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def _shape(s: Any) -> dict | None:
    if not isinstance(s, dict) or s.get("type") not in SHAPE_TYPES:
        return None
    out: dict[str, Any] = {"type": s["type"], "empty": bool(s.get("empty", False))}
    for k in ("a", "b", "c"):
        if isinstance(s.get(k), dict):
            out[k] = _pt(s[k])
    for k in ("r", "r1"):
        if s.get(k) is not None:
            out[k] = _num(s[k])
    if s.get("s"):
        out["s"] = _text(s["s"], 4)
    return out


def _segments(segs: Any) -> list[list[float]]:
    out = []
    for seg in segs if isinstance(segs, list) else []:
        if isinstance(seg, dict):
            out.append([*_pt(seg.get("from")), *_pt(seg.get("to"))])
    return out


def _box(prop: Any) -> dict:
    p = prop if isinstance(prop, dict) else {}
    return {"x": _num(p.get("x")), "y": _num(p.get("y")), "w": _num(p.get("width")), "h": _num(p.get("height"))}


class _Genes:
    """Gene symbols of graph nodes, complexes and sets followed down to their members."""

    def __init__(self, graph: dict) -> None:
        self.by_db: dict[int, dict] = {}
        self.by_diagram: dict[int, dict] = {}
        for n in graph.get("nodes") or []:
            if not isinstance(n, dict):
                continue
            if isinstance(n.get("dbId"), int):
                self.by_db[n["dbId"]] = n
            for d in n.get("diagramIds") or []:
                if isinstance(d, int):
                    self.by_diagram.setdefault(d, n)
        self.aliases: dict[str, list[str]] = {}
        self._memo: dict[int, list[str]] = {}

    def node_for(self, layout_node: dict) -> dict | None:
        return self.by_diagram.get(layout_node.get("id")) or self.by_db.get(layout_node.get("reactomeId"))

    def genes(self, gnode: dict | None) -> list[str]:
        if not gnode:
            return []
        return self._genes(gnode, set())

    def _genes(self, gnode: dict, visiting: set[int]) -> list[str]:
        db = gnode.get("dbId")
        if db in self._memo:
            return self._memo[db]
        names = [n for n in (_text(x, 40) for x in gnode.get("geneNames") or []) if n]
        if names:
            out = [names[0]]
            if len(names) > 1:
                self.aliases.setdefault(names[0], names[1:6])
        else:
            out = []
            if db in visiting:
                return out
            visiting = visiting | {db}
            seen: set[str] = set()
            for c in gnode.get("children") or []:
                child = self.by_db.get(c)
                for g in self._genes(child, visiting) if child else []:
                    if g not in seen:
                        seen.add(g)
                        out.append(g)
        if isinstance(db, int):
            self._memo[db] = out
        return out


def _base_class(renderable: Any) -> tuple[str, bool]:
    r = str(renderable or "")
    drug = r.endswith("Drug")
    base = r[:-4] if drug else r
    return (base if base in BASE_CLASSES else "Entity"), drug


def build_model(layout: dict, graph: dict, *, requested: str | None = None,
                hierarchy: Hierarchy | None = None) -> dict:
    """The renderer's model of one diagram; ``requested``, a pathway drawn inside it,
    gets its outline and reactions as ``highlight``."""
    st_id = str(layout.get("stableId") or graph.get("stId") or "")
    genes = _Genes(graph)
    nodes: list[dict] = []
    connectors: list[dict] = []
    for n in layout.get("nodes") or []:
        if not isinstance(n, dict) or not isinstance(n.get("id"), int):
            continue
        cls, drug = _base_class(n.get("renderableClass"))
        gnode = genes.node_for(n)
        node = {
            "id": n["id"], "reactome_id": n.get("reactomeId") if isinstance(n.get("reactomeId"), int) else None,
            "class": cls, "label": _text(n.get("displayName")), **_box(n.get("prop")),
        }
        if drug:
            node["drug"] = True
        if n.get("isDisease"):
            node["disease"] = True
        if gnode and isinstance(gnode.get("stId"), str) and ST_ID.match(gnode["stId"]):
            node["st_id"] = gnode["stId"]
        if cls not in PATHWAY_CLASSES and cls != "Chemical":
            node["genes"] = genes.genes(gnode)
        end = _shape(n.get("endShape"))
        if end:
            node["end"] = end
        nodes.append(node)
        for c in n.get("connectors") or []:
            if not isinstance(c, dict) or c.get("type") not in CONNECTOR_TYPES:
                continue
            segs = _segments(c.get("segments"))
            if not segs:
                continue
            conn = {"node": n["id"], "edge": c.get("edgeId") if isinstance(c.get("edgeId"), int) else None,
                    "type": c["type"], "segments": segs}
            end = _shape(c.get("endShape"))
            if end:
                conn["end"] = end
            connectors.append(conn)

    reactions: list[dict] = []
    for e in layout.get("edges") or []:
        if not isinstance(e, dict) or not isinstance(e.get("id"), int):
            continue
        r = {"id": e["id"], "reactome_id": e.get("reactomeId") if isinstance(e.get("reactomeId"), int) else None,
             "label": _text(e.get("displayName")), "type": _text(e.get("reactionType"), 30),
             "segments": _segments(e.get("segments"))}
        shape = _shape(e.get("reactionShape"))
        if shape:
            r["shape"] = shape
        end = _shape(e.get("endShape"))
        if end:
            r["end"] = end
        reactions.append(r)

    links: list[dict] = []
    for link in layout.get("links") or []:
        if not isinstance(link, dict) or link.get("renderableClass") not in LINK_CLASSES:
            continue
        item = {"kind": link["renderableClass"], "segments": _segments(link.get("segments"))}
        end = _shape(link.get("endShape"))
        if end:
            item["end"] = end
        if item["segments"]:
            links.append(item)

    compartments = []
    for c in layout.get("compartments") or []:
        if not isinstance(c, dict):
            continue
        comp = {"id": c.get("id"), "label": _text(c.get("displayName"), 80), **_box(c.get("prop")),
                "text": _pt(c.get("textPosition"))}
        if isinstance(c.get("insets"), dict):
            comp["inset"] = _box(c["insets"])
        compartments.append(comp)
    compartments.sort(key=lambda c: -(c["w"] * c["h"]))

    bounds = {k: _num(layout.get(k)) for k in ("minX", "maxX", "minY", "maxY")}
    sp = species_of(st_id) if ST_ID.match(st_id) else ""
    model: dict[str, Any] = {
        "st_id": st_id, "name": _text(layout.get("displayName")), "species": sp,
        "requested": requested or st_id,
        "release": RELEASE, "attribution": ATTRIBUTION, "licence": LICENCE, "licence_url": LICENCE_URL,
        "citation": CITATION, "url": BROWSER_URL.format(st_id=requested or st_id),
        "bounds": bounds, "compartments": compartments, "nodes": nodes, "reactions": reactions,
        "connectors": connectors, "links": links, "aliases": genes.aliases,
        "n_genes": len({g for n in nodes for g in n.get("genes", [])}),
        "too_broad": False, "suggestions": [], "message": "", "highlight": None,
    }
    entities = [n for n in nodes if n["class"] not in PATHWAY_CLASSES]
    if not entities and not reactions:
        subs = [{"st_id": n["st_id"], "name": n["label"]} for n in nodes
                if n["class"] in PATHWAY_CLASSES and n.get("st_id")]
        model.update(
            too_broad=True, suggestions=subs,
            message=(f"{model['name'] or st_id} is an overview of {len(subs)} sub-pathways with no reactions "
                     "of its own to colour. Open one of them instead."),
        )
    if requested and requested != st_id and not model["too_broad"]:
        model["highlight"] = _highlight(layout, graph, requested, hierarchy)
        if model["highlight"] is None:
            model["message"] = (f"{requested} has no diagram of its own: it is drawn inside {model['name']} "
                                f"({st_id}), without an outline of its own steps.")
        else:
            h = model["highlight"]
            inner = "" if h["exact"] else f", within {h['name']}"
            model["message"] = (f"{requested} has no diagram of its own: it is drawn inside "
                                f"{model['name']} ({st_id}), outlined{inner}.")
    return model


def _highlight(layout: dict, graph: dict, requested: str, hierarchy: Hierarchy | None) -> dict | None:
    """The outline and reactions of ``requested`` (or of the nearest ancestor of it the
    diagram draws as a sub-pathway) inside an ancestor's diagram."""
    subs = {s.get("stId"): s for s in graph.get("subpathways") or [] if isinstance(s, dict)}
    order = [requested, *(hierarchy.ancestors(requested) if hierarchy else [])]
    match = next((subs[s] for s in order if s in subs), None)
    if match is None:
        return None
    events = {e for e in match.get("events") or [] if isinstance(e, int)}
    reaction_ids = [e["id"] for e in layout.get("edges") or []
                    if isinstance(e, dict) and e.get("reactomeId") in events and isinstance(e.get("id"), int)]
    outline: list[list[float]] = []
    for s in layout.get("shadows") or []:
        if isinstance(s, dict) and s.get("reactomeId") == match.get("dbId"):
            outline = [_pt(p) for p in s.get("points") or []]
            break
    return {"st_id": str(match.get("stId")), "name": _text(match.get("displayName")),
            "exact": match.get("stId") == requested, "outline": outline, "reactions": reaction_ids}


_models: OrderedDict[tuple, dict] = OrderedDict()
_models_lock = threading.Lock()


def get_diagram(st_id: str, data_dir: str | Path, *, auto_download: bool = True,
                client: httpx.Client | None = None) -> dict:
    """The model of ``st_id``'s diagram, or of the nearest ancestor's that draws it.
    ValueError (bad id), PathwayNotFound, ReactomeUnavailable."""
    st_id = check_st_id(st_id)
    hier = load_hierarchy(data_dir, auto_download=auto_download, client=client)
    if st_id not in hier.known:
        raise PathwayNotFound(
            f"{st_id} is not in Reactome release {RELEASE}: it may have been merged, renamed or "
            "removed since this gene set was made. Search reactome.org for its name."
        )
    # Top-level overviews (Interferon Signaling, Immune System) are not in the list but
    # have a diagram of sub-pathway boxes: tried as themselves, then found too broad.
    candidates = hier.candidates(st_id) or [(st_id, 0)]
    missing: list[str] = []
    for cand, _hops in candidates:
        path = release_dir(data_dir) / "diagram" / f"{cand}.json"
        key = (str(path), st_id)
        with _models_lock:
            cached = _models.get(key)
            if cached is not None and path.is_file():
                _models.move_to_end(key)
                return cached
        try:
            layout, graph = load_files(cand, data_dir, auto_download=auto_download, client=client)
        except _NotThere:
            missing.append(cand)
            continue
        model = build_model(layout, graph, requested=st_id, hierarchy=hier)
        with _models_lock:
            _models[key] = model
            while len(_models) > MAX_MODELS:
                _models.popitem(last=False)
        return model
    if not any(hier.has_diagram(m) for m in missing):
        raise PathwayNotFound(f"Reactome release {RELEASE} has no diagram of {st_id} or of any pathway above it.")
    raise PathwayNotFound(
        f"reactome.org has no diagram file for {', '.join(missing)} in release {RELEASE}, though the "
        "release lists one: the release may have moved on. Update scView, or place the files by hand."
    )


def clear_cache() -> None:
    with _models_lock:
        _models.clear()
    _hierarchy.clear()


# ---------------------------------------------------------------------------
# Colouring
# ---------------------------------------------------------------------------


@dataclass
class GeneTable:
    """Per-gene log fold change and adjusted p (NaN when unknown), keyed by upper-case symbol."""

    label: str
    rows: dict[str, tuple[str, float, float]]  # SYMBOL -> (gene, lfc, padj)
    has_padj: bool


def table_from_result(stored: Any) -> GeneTable:
    """A stored DE result (core/de_results.py)."""
    adj = stored.pvals_adj
    rows: dict[str, tuple[str, float, float]] = {}
    for i, g in enumerate(stored.genes):
        lfc = float(stored.logfoldchanges[i])
        p = float(adj[i]) if adj is not None else float("nan")
        rows.setdefault(str(g).upper(), (str(g), lfc, p))
    has = adj is not None and bool(np.isfinite(adj).any())
    return GeneTable(label=stored.label or "DE result", rows=rows, has_padj=has)


def table_from_markers(adata: Any, column: str, group: str) -> GeneTable:
    """One group's stored marker test (rank_genes_groups for ``column``): the group
    against the rest. LookupError when there is none."""
    from scview.core.ora_background import resolve_markers_key

    key = resolve_markers_key(adata.uns, column)
    if key is None or key not in adata.uns:
        raise LookupError(f"No marker genes for '{column}' yet: compute them in the Markers tab.")
    rgg = adata.uns[key]
    try:
        names = rgg["names"]
        groups = [str(g) for g in names.dtype.names]
    except (KeyError, TypeError, AttributeError):
        raise LookupError(f"The stored markers for '{column}' cannot be read.")
    if group not in groups:
        raise LookupError(f"'{group}' is not a group of '{column}' with stored markers.")
    genes = np.asarray(names[group]).astype(str)

    def field_of(name: str) -> np.ndarray:
        try:
            v = np.asarray(rgg[name][group], dtype=float)[: genes.size]
        except (KeyError, TypeError, ValueError):
            v = np.full(genes.size, np.nan)
        return v if v.size == genes.size else np.full(genes.size, np.nan)

    lfc, padj = field_of("logfoldchanges"), field_of("pvals_adj")
    rows: dict[str, tuple[str, float, float]] = {}
    for g, fc, p in zip(genes.tolist(), lfc.tolist(), padj.tolist()):
        rows.setdefault(g.upper(), (g, fc, p))
    return GeneTable(label=f"{column} {group} markers (vs rest)", rows=rows,
                     has_padj=bool(np.isfinite(padj).any()))


def _finite(v: float) -> float | None:
    return float(v) if v is not None and math.isfinite(v) else None


def diagram_values(model: dict, table: GeneTable, *, padj_cutoff: float = PADJ_CUTOFF) -> dict:
    """Per gene and per node of a diagram: the gene's log fold change (a complex or set:
    the mean of its measured genes) and how many of its genes changed (adjusted p <
    ``padj_cutoff``). A gene the result does not hold is "not measured"."""
    aliases = model.get("aliases") or {}
    genes: dict[str, dict] = {}
    for node in model.get("nodes", []):
        for g in node.get("genes", []):
            if g in genes:
                continue
            hit = table.rows.get(g.upper())
            if hit is None:
                hit = next((table.rows[a.upper()] for a in aliases.get(g, []) if a.upper() in table.rows), None)
            if hit is not None and _finite(hit[1]) is not None:
                genes[g] = {"gene": hit[0], "lfc": _finite(hit[1]), "padj": _finite(hit[2])}
    nodes: dict[str, dict] = {}
    for node in model.get("nodes", []):
        gl = node.get("genes")
        if not gl:
            continue
        measured = [genes[g] for g in gl if g in genes]
        changed = (sum(1 for v in measured if v["padj"] is not None and v["padj"] < padj_cutoff)
                   if table.has_padj else None)
        nodes[str(node["id"])] = {
            "value": float(np.mean([v["lfc"] for v in measured])) if measured else None,
            "n_genes": len(gl), "n_measured": len(measured), "n_changed": changed,
        }
    all_genes = {g for n in model.get("nodes", []) for g in n.get("genes", [])}
    return {
        "st_id": model.get("st_id"), "requested": model.get("requested"), "label": table.label,
        "genes": genes, "nodes": nodes, "padj_cutoff": padj_cutoff, "has_padj": table.has_padj,
        "n_diagram_genes": len(all_genes), "n_measured": len(genes),
        "n_changed": (sum(1 for v in genes.values() if v["padj"] is not None and v["padj"] < padj_cutoff)
                      if table.has_padj else None),
    }


def main(argv: list[str] | None = None) -> int:
    """Fetch the hierarchy and the given pathways' diagrams into DATA_DIR."""
    import sys

    from scview.config import get_settings

    ids = list(sys.argv[1:] if argv is None else argv)
    data_dir = get_settings().DATA_DIR
    h = load_hierarchy(data_dir, auto_download=True)
    print(f"Reactome release {RELEASE}: {len(h.known)} pathways, {len(h.with_diagram)} human diagrams")
    for sid in ids:
        m = get_diagram(sid, data_dir, auto_download=True)
        print(f"{sid}: drawn in {m['st_id']} ({len(m['nodes'])} nodes, {len(m['reactions'])} reactions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
