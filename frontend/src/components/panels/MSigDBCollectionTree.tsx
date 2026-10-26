import { useState, useEffect, useCallback, useRef } from "react";
import { ChevronRight, ChevronDown, Loader2 } from "lucide-react";
import { apiFetch } from "@/api/client";
import { InfoTip } from "@/components/common/InfoTip";
import { CustomCollectionsGroup } from "./CustomCollectionsGroup";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

interface MSigDBCollection {
  id: string;
  category: string;
  subcategory: string;
  name: string;
  description: string;
  default: boolean;
  n_sets: number;
  available: boolean;
  /** Whose gene symbols the sets use. */
  species?: "human" | "mouse";
}

interface SubcategoryNode {
  subcategory: string;
  children: MSigDBCollection[];
}

interface CategoryNode {
  category: string;
  name: string;
  species?: "human" | "mouse";
  children: (MSigDBCollection | SubcategoryNode)[];
}

/** How the first-use fetch of the mouse collections is going (mouse datasets only). */
interface MouseDownload {
  state: "ready" | "downloading" | "failed" | "off" | "missing";
  release: string;
  message: string;
}

interface MSigDBCollectionsResponse {
  collections: MSigDBCollection[];
  hierarchy: CategoryNode[];
  /** The default selection for the dataset's species. */
  defaults: string[];
  species?: string;
  mouse_download?: MouseDownload;
}

const POLL_MS = 3000;

function sameSet(a: Set<string>, b: string[]): boolean {
  return a.size === b.length && b.every((x) => a.has(x));
}

/** The heading above each species' collections on a mouse dataset. */
function SpeciesHeading({ species }: { species: "human" | "mouse" }) {
  if (species === "mouse") {
    return <div className="mt-0.5 mb-0.5 text-[10px] font-semibold uppercase tracking-wide text-slate-400">Mouse gene sets</div>;
  }
  return (
    <div className="mt-1.5 mb-0.5 flex items-center gap-1 border-t border-slate-100 pt-1 text-[10px] font-semibold uppercase tracking-wide text-slate-400">
      Human gene sets (human symbols, case-matched)
      <InfoTip label="About human gene sets on mouse data">
        These sets list human gene symbols. On mouse data a gene is matched ignoring case, so Cd3e finds CD3E,
        but a gene whose mouse symbol differs from the human one is missed. Prefer the mouse collections above.
      </InfoTip>
    </div>
  );
}

function isSubcategory(
  node: MSigDBCollection | SubcategoryNode,
): node is SubcategoryNode {
  return "subcategory" in node && "children" in node && !("id" in node);
}

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

interface MSigDBCollectionTreeProps {
  selected: Set<string>;
  onChange: (selected: Set<string>) => void;
  compact?: boolean;
  /** Lets the Custom group say how many of each collection's genes the dataset has. */
  datasetId?: string | null;
}

export function MSigDBCollectionTree({
  selected,
  onChange,
  compact = false,
  datasetId,
}: MSigDBCollectionTreeProps) {
  const [hierarchy, setHierarchy] = useState<CategoryNode[]>([]);
  const [customIds, setCustomIds] = useState<string[]>([]);
  const [defaults, setDefaults] = useState<string[]>([]);
  const [species, setSpecies] = useState<string>("");
  const [download, setDownload] = useState<MouseDownload | null>(null);
  const [poll, setPoll] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  // The selection as it is now, read when the species' defaults arrive.
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const swappedFor = useRef<string | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(),
  );
  const [expandedSubcategories, setExpandedSubcategories] = useState<
    Set<string>
  >(new Set());

  // Fetch hierarchy (for the dataset's species: a mouse dataset lists mouse collections first)
  useEffect(() => {
    if (poll === 0) setIsLoading(true);
    const path = datasetId
      ? `/enrichment/msigdb-collections?dataset_id=${encodeURIComponent(datasetId)}`
      : "/enrichment/msigdb-collections";
    apiFetch<MSigDBCollectionsResponse>(path)
      .then((data) => {
        setHierarchy(data.hierarchy);
        setDefaults(data.defaults);
        setSpecies(data.species ?? "");
        setDownload(data.mouse_download ?? null);
        // Once per dataset: an untouched default selection becomes this species' defaults.
        const key = datasetId ?? "";
        if (swappedFor.current !== key) {
          swappedFor.current = key;
          const cur = selectedRef.current;
          const untouched = sameSet(cur, DEFAULT_MSIGDB_COLLECTIONS) || sameSet(cur, DEFAULT_MOUSE_MSIGDB_COLLECTIONS);
          if (untouched && !sameSet(cur, data.defaults)) onChangeRef.current(new Set(data.defaults));
        }
        // Auto-expand categories that have selected items
        const cats = new Set<string>();
        const subcats = new Set<string>();
        for (const cat of data.hierarchy) {
          for (const child of cat.children) {
            if (isSubcategory(child)) {
              for (const col of child.children) {
                if (data.defaults.includes(col.id)) {
                  cats.add(cat.category);
                  subcats.add(`${cat.category}:${child.subcategory}`);
                }
              }
            } else if (data.defaults.includes(child.id)) {
              cats.add(cat.category);
            }
          }
        }
        if (poll === 0) {
          setExpandedCategories(cats);
          setExpandedSubcategories(subcats);
        }
      })
      .catch(() => setHierarchy([]))
      .finally(() => setIsLoading(false));
  }, [datasetId, poll]);

  // While the mouse collections download, look again every few seconds.
  useEffect(() => {
    if (download?.state !== "downloading") return;
    const t = setTimeout(() => setPoll((n) => n + 1), POLL_MS);
    return () => clearTimeout(t);
  }, [download, poll]);

  const toggleCollection = useCallback(
    (id: string) => {
      const next = new Set(selected);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      onChange(next);
    },
    [selected, onChange],
  );

  const toggleCategory = useCallback(
    (cat: string) => {
      setExpandedCategories((prev) => {
        const next = new Set(prev);
        if (next.has(cat)) next.delete(cat);
        else next.add(cat);
        return next;
      });
    },
    [],
  );

  const toggleSubcategory = useCallback(
    (key: string) => {
      setExpandedSubcategories((prev) => {
        const next = new Set(prev);
        if (next.has(key)) next.delete(key);
        else next.add(key);
        return next;
      });
    },
    [],
  );

  const selectAll = useCallback(() => {
    const all = new Set<string>();
    for (const cat of hierarchy) {
      for (const child of cat.children) {
        if (isSubcategory(child)) {
          for (const col of child.children) {
            if (col.available) all.add(col.id);
          }
        } else if (child.available) {
          all.add(child.id);
        }
      }
    }
    for (const id of customIds) all.add(id);
    onChange(all);
  }, [hierarchy, customIds, onChange]);

  const selectDefaults = useCallback(() => {
    onChange(new Set(defaults));
  }, [defaults, onChange]);

  const selectNone = useCallback(() => {
    onChange(new Set());
  }, [onChange]);

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-2 text-xs text-slate-400">
        <Loader2 className="h-3 w-3 animate-spin" />
        Loading MSigDB collections...
      </div>
    );
  }

  const textSize = compact ? "text-[11px]" : "text-xs";
  const py = compact ? "py-0.5" : "py-1";

  return (
    <div className="space-y-1">
      {/* Quick actions */}
      <div className="flex items-center gap-2 text-[10px]">
        <span className={`font-medium text-slate-600 ${textSize}`}>
          Collections ({selected.size} selected)
        </span>
        <button
          type="button"
          onClick={selectDefaults}
          className="text-blue-600 hover:underline"
        >
          Defaults
        </button>
        <button
          type="button"
          onClick={selectAll}
          className="text-blue-600 hover:underline"
        >
          All
        </button>
        <button
          type="button"
          onClick={selectNone}
          className="text-slate-400 hover:text-slate-600 hover:underline"
        >
          None
        </button>
      </div>

      {/* Tree */}
      <div
        className={`max-h-64 overflow-auto rounded-lg border border-slate-200 bg-white ${compact ? "p-1.5" : "p-2"}`}
      >
        {/* The user's own collections, usable alongside MSigDB (and without it).
            First, so the upload is in view without scrolling past MSigDB. */}
        <div className="mb-1 border-b border-slate-100 pb-1">
          <CustomCollectionsGroup
            datasetId={datasetId}
            selected={selected}
            onChange={onChange}
            onLoaded={(cols) => setCustomIds(cols.map((c) => c.id))}
            compact={compact}
          />
        </div>
        {hierarchy.length === 0 && (
          <div className="py-1 text-xs text-slate-400">
            No MSigDB collections available. Check MSIGDB_DIR configuration.
          </div>
        )}
        {download && download.state !== "ready" && download.message && (
          <div
            role="status"
            className={`mb-1 flex items-start gap-1.5 text-[10px] ${download.state === "downloading" ? "text-slate-500" : "text-amber-700"}`}
          >
            {download.state === "downloading" && <Loader2 className="mt-0.5 h-3 w-3 flex-shrink-0 animate-spin" />}
            <span>{download.message}</span>
          </div>
        )}
        {hierarchy.map((cat, i) => {
          const heading =
            species === "mouse" && cat.species && cat.species !== hierarchy[i - 1]?.species ? (
              <SpeciesHeading species={cat.species} />
            ) : null;
          const isExpanded = expandedCategories.has(cat.category);
          const hasChildren =
            cat.children.length > 1 ||
            (cat.children.length === 1 && isSubcategory(cat.children[0]!));

          // For categories with a single direct collection (H, C8), render flat
          if (!hasChildren && cat.children.length === 1) {
            const col = cat.children[0] as MSigDBCollection;
            return (
              <div key={col.id}>
              {heading}
              <label
                className={`flex items-center gap-2 ${py} ${textSize} text-slate-700`}
                title={col.description}
              >
                <input
                  type="checkbox"
                  checked={selected.has(col.id)}
                  onChange={() => toggleCollection(col.id)}
                  disabled={!col.available}
                  className="h-3.5 w-3.5 rounded border-slate-300 accent-blue-600"
                />
                <span className="font-semibold text-slate-500">
                  {cat.category}:
                </span>
                <span>{col.name}</span>
                <span className="text-slate-400">({col.n_sets})</span>
              </label>
              </div>
            );
          }

          return (
            <div key={cat.category}>
              {heading}
              {/* Category header */}
              <button
                onClick={() => toggleCategory(cat.category)}
                className={`flex w-full items-center gap-1 ${py} ${textSize} font-semibold text-slate-600 hover:text-slate-800`}
              >
                {isExpanded ? (
                  <ChevronDown className="h-3 w-3 flex-shrink-0" />
                ) : (
                  <ChevronRight className="h-3 w-3 flex-shrink-0" />
                )}
                {cat.category}: {cat.name}
              </button>

              {isExpanded && (
                <div className="ml-4 border-l border-slate-100 pl-2">
                  {cat.children.map((child) => {
                    if (isSubcategory(child)) {
                      const subKey = `${cat.category}:${child.subcategory}`;
                      const isSubExpanded =
                        expandedSubcategories.has(subKey);

                      return (
                        <div key={subKey}>
                          <button
                            onClick={() => toggleSubcategory(subKey)}
                            className={`flex w-full items-center gap-1 ${py} ${textSize} font-medium text-slate-500 hover:text-slate-700`}
                          >
                            {isSubExpanded ? (
                              <ChevronDown className="h-3 w-3 flex-shrink-0" />
                            ) : (
                              <ChevronRight className="h-3 w-3 flex-shrink-0" />
                            )}
                            {child.subcategory}
                          </button>

                          {isSubExpanded && (
                            <div className="ml-4 border-l border-slate-100 pl-2">
                              {child.children.map((col) => (
                                <label
                                  key={col.id}
                                  className={`flex items-center gap-2 ${py} ${textSize} text-slate-700`}
                                  title={col.description}
                                >
                                  <input
                                    type="checkbox"
                                    checked={selected.has(col.id)}
                                    onChange={() => toggleCollection(col.id)}
                                    disabled={!col.available}
                                    className="h-3.5 w-3.5 rounded border-slate-300 accent-blue-600"
                                  />
                                  <span>{col.name}</span>
                                  <span className="text-slate-400">
                                    ({col.n_sets})
                                  </span>
                                </label>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    }

                    // Direct collection under category (e.g., CGP under C2)
                    const col = child;
                    return (
                      <label
                        key={col.id}
                        className={`flex items-center gap-2 ${py} ${textSize} text-slate-700`}
                        title={col.description}
                      >
                        <input
                          type="checkbox"
                          checked={selected.has(col.id)}
                          onChange={() => toggleCollection(col.id)}
                          disabled={!col.available}
                          className="h-3.5 w-3.5 rounded border-slate-300 accent-blue-600"
                        />
                        <span>{col.name}</span>
                        <span className="text-slate-400">({col.n_sets})</span>
                      </label>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export { type MSigDBCollection, type MSigDBCollectionsResponse };
export const DEFAULT_MSIGDB_COLLECTIONS = [
  "h.all",
  "c2.cp.kegg_medicus",
  "c2.cp.reactome",
  "c2.cp.wikipathways",
  "c5.go.bp",
  "c5.go.cc",
  "c5.go.mf",
  "c8.all",
];
/** A mouse dataset's defaults (backend msigdb_loader.MOUSE_DEFAULT_COLLECTIONS); the tree
 *  swaps an untouched selection to the species' defaults when it loads. */
export const DEFAULT_MOUSE_MSIGDB_COLLECTIONS = [
  "mh.all",
  "m2.cp.reactome",
  "m2.cp.wikipathways",
  "m5.go.bp",
  "m5.go.cc",
  "m5.go.mf",
  "m8.all",
];
