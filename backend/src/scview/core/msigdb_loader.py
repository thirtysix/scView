"""MSigDB gene-set loader — reads MSigDB subcategory JSON files.

Human collections (v2026.1.Hs, human symbols) are fetched on first start; the
mouse ones (v2026.2.Mm, mouse symbols: MH is the Hallmark sets mapped to mouse
orthologs by MSigDB) on first use by a mouse dataset (``core/msigdb_fetch.py``).
Every collection carries its ``species``; listings for a mouse dataset put the
mouse collections first and the human ones after them.

Provides hierarchical collection metadata, search, and gseapy-compatible
dict output for local enrichment analysis.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collection registry — one entry per subcategory file we ship in msigdb/
# ---------------------------------------------------------------------------

MSIGDB_COLLECTIONS: list[dict] = [
    # H: Hallmark
    {"id": "h.all", "category": "H", "subcategory": "", "name": "Hallmark",
     "description": "50 well-defined biological states and processes", "default": True},
    # C2: Curated Gene Sets
    {"id": "c2.cgp", "category": "C2", "subcategory": "CGP",
     "name": "Chemical & Genetic Perturbations",
     "description": "Gene sets from chemical and genetic perturbation experiments", "default": False},
    {"id": "c2.cp.biocarta", "category": "C2", "subcategory": "CP",
     "name": "BioCarta", "description": "BioCarta pathway gene sets", "default": False},
    {"id": "c2.cp.kegg_legacy", "category": "C2", "subcategory": "CP",
     "name": "KEGG Legacy", "description": "Legacy KEGG pathway gene sets", "default": False},
    {"id": "c2.cp.kegg_medicus", "category": "C2", "subcategory": "CP",
     "name": "KEGG Medicus", "description": "KEGG Medicus disease and drug pathway gene sets", "default": True},
    {"id": "c2.cp.pid", "category": "C2", "subcategory": "CP",
     "name": "PID", "description": "Pathway Interaction Database gene sets", "default": False},
    {"id": "c2.cp.reactome", "category": "C2", "subcategory": "CP",
     "name": "Reactome", "description": "Reactome pathway gene sets", "default": True},
    {"id": "c2.cp.wikipathways", "category": "C2", "subcategory": "CP",
     "name": "WikiPathways", "description": "WikiPathways community curated pathway gene sets", "default": True},
    # C3: Regulatory Target Gene Sets
    {"id": "c3.mir.mirdb", "category": "C3", "subcategory": "miR",
     "name": "miRDB", "description": "miRDB predicted microRNA targets", "default": False},
    {"id": "c3.mir.mir_legacy", "category": "C3", "subcategory": "miR",
     "name": "miR Legacy", "description": "Legacy microRNA target gene sets", "default": False},
    {"id": "c3.tft.gtrd", "category": "C3", "subcategory": "TFT",
     "name": "GTRD", "description": "GTRD transcription factor target gene sets", "default": False},
    {"id": "c3.tft.tft_legacy", "category": "C3", "subcategory": "TFT",
     "name": "TFT Legacy", "description": "Legacy transcription factor target gene sets", "default": False},
    # C4: Computational Gene Sets
    {"id": "c4.3ca", "category": "C4", "subcategory": "3CA",
     "name": "3CA", "description": "Cancer Cell Atlas computational gene sets", "default": False},
    {"id": "c4.cgn", "category": "C4", "subcategory": "CGN",
     "name": "CGN", "description": "Cancer Gene Neighborhoods", "default": False},
    {"id": "c4.cm", "category": "C4", "subcategory": "CM",
     "name": "CM", "description": "Cancer Modules", "default": False},
    # C5: Ontology Gene Sets
    {"id": "c5.go.bp", "category": "C5", "subcategory": "GO",
     "name": "GO Biological Process", "description": "Gene Ontology Biological Process terms", "default": True},
    {"id": "c5.go.cc", "category": "C5", "subcategory": "GO",
     "name": "GO Cellular Component", "description": "Gene Ontology Cellular Component terms", "default": True},
    {"id": "c5.go.mf", "category": "C5", "subcategory": "GO",
     "name": "GO Molecular Function", "description": "Gene Ontology Molecular Function terms", "default": True},
    {"id": "c5.hpo", "category": "C5", "subcategory": "HPO",
     "name": "HPO", "description": "Human Phenotype Ontology gene sets", "default": False},
    # C7: Immunologic Signature Gene Sets
    {"id": "c7.immunesigdb", "category": "C7", "subcategory": "ImmuneSigDB",
     "name": "ImmuneSigDB", "description": "Immunologic signature gene sets", "default": False},
    {"id": "c7.vax", "category": "C7", "subcategory": "VAX",
     "name": "VAX", "description": "Vaccine response gene sets", "default": False},
    # C8: Cell Type Signature Gene Sets
    {"id": "c8.all", "category": "C8", "subcategory": "",
     "name": "Cell Type Signatures", "description": "Cell type signature gene sets", "default": True},
]

for _c in MSIGDB_COLLECTIONS:
    _c["species"] = "human"

# Mouse collections (MSigDB's Mm resource): mouse gene symbols, ids as Broad's file
# names. Downloaded when a mouse dataset first asks for them, never on start.
MOUSE_COLLECTIONS: list[dict] = [
    {"id": "mh.all", "category": "MH", "subcategory": "", "name": "Hallmark (mouse)",
     "description": "The 50 Hallmark sets mapped to mouse orthologs", "default": True},
    {"id": "m1.all", "category": "M1", "subcategory": "", "name": "Positional",
     "description": "Gene sets by mouse chromosome cytogenetic band", "default": False},
    {"id": "m2.cgp", "category": "M2", "subcategory": "CGP",
     "name": "Chemical & Genetic Perturbations",
     "description": "Mouse gene sets from chemical and genetic perturbation experiments", "default": False},
    {"id": "m2.cp.biocarta", "category": "M2", "subcategory": "CP",
     "name": "BioCarta", "description": "BioCarta pathways, mouse orthologs", "default": False},
    {"id": "m2.cp.reactome", "category": "M2", "subcategory": "CP",
     "name": "Reactome", "description": "Reactome mouse pathway gene sets", "default": True},
    {"id": "m2.cp.wikipathways", "category": "M2", "subcategory": "CP",
     "name": "WikiPathways", "description": "WikiPathways mouse pathway gene sets", "default": True},
    {"id": "m3.gtrd", "category": "M3", "subcategory": "TFT",
     "name": "GTRD", "description": "GTRD transcription factor targets (mouse)", "default": False},
    {"id": "m3.mirdb", "category": "M3", "subcategory": "miR",
     "name": "miRDB", "description": "miRDB predicted microRNA targets (mouse)", "default": False},
    {"id": "m5.go.bp", "category": "M5", "subcategory": "GO",
     "name": "GO Biological Process", "description": "Gene Ontology Biological Process, mouse annotations",
     "default": True},
    {"id": "m5.go.cc", "category": "M5", "subcategory": "GO",
     "name": "GO Cellular Component", "description": "Gene Ontology Cellular Component, mouse annotations",
     "default": True},
    {"id": "m5.go.mf", "category": "M5", "subcategory": "GO",
     "name": "GO Molecular Function", "description": "Gene Ontology Molecular Function, mouse annotations",
     "default": True},
    {"id": "m5.mpo.mp", "category": "M5", "subcategory": "MPO",
     "name": "Mammalian Phenotype", "description": "Mammalian Phenotype Ontology gene sets (MGI)",
     "default": False},
    {"id": "m5.mpo.mpt", "category": "M5", "subcategory": "MPO",
     "name": "MP Tumor", "description": "Mammalian Phenotype tumor terms (MGI)", "default": False},
    {"id": "m7.all", "category": "M7", "subcategory": "",
     "name": "Immunologic (mouse)", "description": "Mouse immunologic signature gene sets", "default": False},
    {"id": "m8.all", "category": "M8", "subcategory": "",
     "name": "Cell Type Signatures (mouse)", "description": "Mouse cell type signature gene sets",
     "default": True},
]
for _c in MOUSE_COLLECTIONS:
    _c["species"] = "mouse"
del _c

ALL_COLLECTIONS: list[dict] = MSIGDB_COLLECTIONS + MOUSE_COLLECTIONS
_SPECIES_OF: dict[str, str] = {c["id"]: c["species"] for c in ALL_COLLECTIONS}

# Category display names
CATEGORY_NAMES: dict[str, str] = {
    "H": "Hallmark",
    "C2": "Curated Gene Sets",
    "C3": "Regulatory Targets",
    "C4": "Computational",
    "C5": "Ontology",
    "C7": "Immunologic",
    "C8": "Cell Type Signatures",
    "MH": "Hallmark (mouse)",
    "M1": "Positional (mouse)",
    "M2": "Curated Gene Sets (mouse)",
    "M3": "Regulatory Targets (mouse)",
    "M5": "Ontology (mouse)",
    "M7": "Immunologic (mouse)",
    "M8": "Cell Type Signatures (mouse)",
}

DEFAULT_COLLECTIONS: list[str] = [c["id"] for c in MSIGDB_COLLECTIONS if c["default"]]
MOUSE_DEFAULT_COLLECTIONS: list[str] = [c["id"] for c in MOUSE_COLLECTIONS if c["default"]]
HALLMARK = {"human": "h.all", "mouse": "mh.all"}


def collection_species(collection_id: str) -> str | None:
    """"human" or "mouse" for an MSigDB collection id; None for any other id (custom)."""
    return _SPECIES_OF.get(str(collection_id).lower())


def collections_for(species: str | None) -> list[dict]:
    """The registry entries a dataset of ``species`` lists, in order: a mouse dataset
    gets the mouse collections first, then the human ones; any other dataset (human,
    unknown) the human ones only, as before mouse collections existed."""
    return MOUSE_COLLECTIONS + MSIGDB_COLLECTIONS if species == "mouse" else MSIGDB_COLLECTIONS


def defaults_for(species: str | None) -> list[str]:
    """The default selection for a dataset of ``species``."""
    return list(MOUSE_DEFAULT_COLLECTIONS if species == "mouse" else DEFAULT_COLLECTIONS)


class MSigDBLoader:
    """Lazy loader for MSigDB JSON gene-set files.

    Loads each collection on first access and caches in memory.
    Supports subcategory-level files (e.g., c2.cp.reactome, c5.go.bp).
    """

    def __init__(self, msigdb_dir: str) -> None:
        self.dir = Path(msigdb_dir)
        self._cache: dict[str, dict[str, dict]] = {}  # collection_id -> {set_name: raw_data}

    def available_collections(self, species: str | None = None) -> list[dict]:
        """Collection metadata for a dataset of ``species`` (``collections_for``),
        each saying whether its file is on disk and how many sets it holds."""
        result = []
        for entry in collections_for(species):
            filepath = self._find_file(entry["id"])
            if filepath is not None:
                data = self._load_collection(entry["id"])
                n_sets = len(data) if data else 0
                result.append({**entry, "n_sets": n_sets, "available": True})
            else:
                result.append({**entry, "n_sets": 0, "available": False})
        return result

    def available_collections_hierarchical(self, species: str | None = None) -> list[dict]:
        """Return collections grouped by category and subcategory for tree display.
        Each category says its ``species``."""
        flat = self.available_collections(species)
        categories: dict[str, dict] = {}

        for col in flat:
            cat = col["category"]
            if cat not in categories:
                categories[cat] = {
                    "category": cat,
                    "name": CATEGORY_NAMES.get(cat, cat),
                    "species": col["species"],
                    "subcategories": {},
                    "collections": [],
                }

            subcat = col.get("subcategory", "")
            if subcat:
                if subcat not in categories[cat]["subcategories"]:
                    categories[cat]["subcategories"][subcat] = []
                categories[cat]["subcategories"][subcat].append(col)
            else:
                categories[cat]["collections"].append(col)

        # Convert subcategories dict to list format
        result = []
        for cat_data in categories.values():
            entry = {
                "category": cat_data["category"],
                "name": cat_data["name"],
                "species": cat_data["species"],
                "children": [],
            }
            # Add direct collections (no subcategory, e.g., h.all, c8.all)
            for col in cat_data["collections"]:
                entry["children"].append(col)
            # Add subcategory groups
            for subcat_name, subcat_cols in cat_data["subcategories"].items():
                entry["children"].append({
                    "subcategory": subcat_name,
                    "children": subcat_cols,
                })
            result.append(entry)

        return result

    def collection_size(self, collection_id: str) -> int:
        """Return the number of gene sets in a collection (loads lazily)."""
        data = self._load_collection(collection_id)
        return len(data) if data else 0

    def get_collection_as_dict(self, collection_id: str) -> dict[str, list[str]]:
        """Load a collection and return as {SET_NAME: [gene1, gene2, ...]} for gseapy."""
        data = self._load_collection(collection_id)
        if not data:
            return {}
        result: dict[str, list[str]] = {}
        for set_name, set_data in data.items():
            genes = set_data.get("geneSymbols", [])
            if genes:
                result[set_name] = genes
        return result

    def get_sources(self, collection_id: str) -> dict[str, tuple[str, str]]:
        """{SET_NAME: (source id, details URL)} from each set's ``exactSource`` and
        ``externalDetailsURL`` (R-HSA-/R-MMU- for Reactome, WP... for WikiPathways,
        N... for KEGG Medicus). Sets without them, or with a URL that is not http(s),
        get empty strings; the GMT form never has them."""
        data = self._load_collection(collection_id)
        if not data:
            return {}
        out: dict[str, tuple[str, str]] = {}
        for set_name, set_data in data.items():
            if not isinstance(set_data, dict):
                continue
            sid = _clean_source_id(set_data.get("exactSource"))
            url = _clean_url(set_data.get("externalDetailsURL"))
            if sid or url:
                out[set_name] = (sid, url)
        return out

    def get_multiple_collections_as_dict(
        self, collection_ids: list[str]
    ) -> dict[str, list[str]]:
        """Merge multiple collections into a single gseapy-compatible dict."""
        merged: dict[str, list[str]] = {}
        for cid in collection_ids:
            merged.update(self.get_collection_as_dict(cid))
        return merged

    def search(
        self,
        query: str,
        collection: str = "",
        limit: int = 50,
        species: str | None = None,
    ) -> list[dict]:
        """Search gene sets by name (case-insensitive substring match).

        Parameters
        ----------
        query : str
            Search substring to match against gene set names.
        collection : str
            If non-empty, restrict search to this collection ID.
        limit : int
            Max results to return.
        species : str
            Without ``collection``: search the collections a dataset of this
            species lists (``collections_for``), in that order.
        """
        q_upper = query.upper().strip() if query else ""
        results: list[dict] = []

        if collection:
            collections_to_search = [collection]
        else:
            collections_to_search = [c["id"] for c in collections_for(species)]

        for cid in collections_to_search:
            data = self._load_collection(cid)
            if data is None:
                continue
            for set_name, set_data in data.items():
                if not q_upper or q_upper in set_name.upper():
                    genes = set_data.get("geneSymbols", [])
                    results.append({
                        "name": set_name,
                        "collection": cid,
                        "n_genes": len(genes),
                        "genes": genes,
                    })
                    if len(results) >= limit:
                        return results

        return results

    def get_gene_set(self, name: str, collection: str = "", species: str | None = None) -> dict | None:
        """Retrieve a single gene set by exact name (MH and H share their set names, so
        ``species`` decides which comes first when no collection is given)."""
        if collection:
            collections_to_search = [collection]
        else:
            collections_to_search = [c["id"] for c in collections_for(species)]

        for cid in collections_to_search:
            data = self._load_collection(cid)
            if data and name in data:
                genes = data[name].get("geneSymbols", [])
                return {
                    "name": name,
                    "collection": cid,
                    "n_genes": len(genes),
                    "genes": genes,
                }
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_collection(self, collection_id: str) -> dict[str, dict] | None:
        cid = collection_id.lower()
        if cid in self._cache:
            return self._cache[cid]

        # Verify this is a known collection
        if cid not in _SPECIES_OF:
            return None

        filepath = self._find_file(cid)
        if filepath is None:
            logger.debug("MSigDB file not found for collection %s", cid)
            return None

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            self._cache[cid] = data
            logger.info(
                "Loaded MSigDB collection %s: %d gene sets from %s",
                cid, len(data), filepath.name,
            )
            return data
        except Exception as e:
            logger.error("Failed to load MSigDB file %s: %s", filepath, e)
            return None

    def _find_file(self, prefix: str) -> Path | None:
        """Find the JSON file matching the prefix (version-flexible glob)."""
        matches = list(self.dir.glob(f"{prefix}.*.json"))
        if matches:
            return matches[0]
        exact = self.dir / f"{prefix}.json"
        if exact.exists():
            return exact
        return None


_SOURCE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _clean_source_id(value) -> str:
    """A set's source id when it looks like one (letters, digits, - _ . :), else ""."""
    v = str(value or "").strip()
    return v if _SOURCE_ID.match(v) else ""


def _clean_url(value) -> str:
    """The first http(s) URL of a details field (MSigDB lists several comma-separated
    for a few sets), else ""."""
    for part in str(value or "").split(","):
        v = part.strip()
        if re.match(r"^https?://[^\s<>\"']{1,500}$", v):
            return v
    return ""


# Singleton accessor
_loader_instance: MSigDBLoader | None = None


def get_msigdb_loader(msigdb_dir: str) -> MSigDBLoader | None:
    """Return a cached MSigDBLoader, or None if directory is not configured."""
    global _loader_instance
    if not msigdb_dir:
        return None
    if _loader_instance is None or str(_loader_instance.dir) != msigdb_dir:
        path = Path(msigdb_dir)
        if not path.is_dir():
            logger.warning("MSIGDB_DIR does not exist: %s", msigdb_dir)
            return None
        _loader_instance = MSigDBLoader(msigdb_dir)
    return _loader_instance
