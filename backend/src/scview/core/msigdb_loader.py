"""MSigDB gene-set loader — reads MSigDB v2026.1 subcategory JSON files.

Provides hierarchical collection metadata, search, and gseapy-compatible
dict output for local enrichment analysis.
"""

from __future__ import annotations

import json
import logging
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

# Category display names
CATEGORY_NAMES: dict[str, str] = {
    "H": "Hallmark",
    "C2": "Curated Gene Sets",
    "C3": "Regulatory Targets",
    "C4": "Computational",
    "C5": "Ontology",
    "C7": "Immunologic",
    "C8": "Cell Type Signatures",
}

DEFAULT_COLLECTIONS: list[str] = [c["id"] for c in MSIGDB_COLLECTIONS if c["default"]]


class MSigDBLoader:
    """Lazy loader for MSigDB JSON gene-set files.

    Loads each collection on first access and caches in memory.
    Supports subcategory-level files (e.g., c2.cp.reactome, c5.go.bp).
    """

    def __init__(self, msigdb_dir: str) -> None:
        self.dir = Path(msigdb_dir)
        self._cache: dict[str, dict[str, dict]] = {}  # collection_id -> {set_name: raw_data}

    def available_collections(self) -> list[dict]:
        """Return collection metadata dicts for all files found on disk."""
        result = []
        for entry in MSIGDB_COLLECTIONS:
            filepath = self._find_file(entry["id"])
            if filepath is not None:
                data = self._load_collection(entry["id"])
                n_sets = len(data) if data else 0
                result.append({**entry, "n_sets": n_sets, "available": True})
            else:
                result.append({**entry, "n_sets": 0, "available": False})
        return result

    def available_collections_hierarchical(self) -> list[dict]:
        """Return collections grouped by category and subcategory for tree display."""
        flat = self.available_collections()
        categories: dict[str, dict] = {}

        for col in flat:
            cat = col["category"]
            if cat not in categories:
                categories[cat] = {
                    "category": cat,
                    "name": CATEGORY_NAMES.get(cat, cat),
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
        """
        q_upper = query.upper().strip() if query else ""
        results: list[dict] = []

        if collection:
            collections_to_search = [collection]
        else:
            collections_to_search = [c["id"] for c in MSIGDB_COLLECTIONS]

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

    def get_gene_set(self, name: str, collection: str = "") -> dict | None:
        """Retrieve a single gene set by exact name."""
        if collection:
            collections_to_search = [collection]
        else:
            collections_to_search = [c["id"] for c in MSIGDB_COLLECTIONS]

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
        known_ids = {c["id"] for c in MSIGDB_COLLECTIONS}
        if cid not in known_ids:
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
