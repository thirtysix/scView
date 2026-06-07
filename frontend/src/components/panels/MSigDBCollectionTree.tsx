import { useState, useEffect, useCallback } from "react";
import { ChevronRight, ChevronDown, Loader2 } from "lucide-react";
import { apiFetch } from "@/api/client";

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
}

interface SubcategoryNode {
  subcategory: string;
  children: MSigDBCollection[];
}

interface CategoryNode {
  category: string;
  name: string;
  children: (MSigDBCollection | SubcategoryNode)[];
}

interface MSigDBCollectionsResponse {
  collections: MSigDBCollection[];
  hierarchy: CategoryNode[];
  defaults: string[];
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
}

export function MSigDBCollectionTree({
  selected,
  onChange,
  compact = false,
}: MSigDBCollectionTreeProps) {
  const [hierarchy, setHierarchy] = useState<CategoryNode[]>([]);
  const [defaults, setDefaults] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(),
  );
  const [expandedSubcategories, setExpandedSubcategories] = useState<
    Set<string>
  >(new Set());

  // Fetch hierarchy
  useEffect(() => {
    setIsLoading(true);
    apiFetch<MSigDBCollectionsResponse>("/enrichment/msigdb-collections")
      .then((data) => {
        setHierarchy(data.hierarchy);
        setDefaults(data.defaults);
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
        setExpandedCategories(cats);
        setExpandedSubcategories(subcats);
      })
      .catch(() => setHierarchy([]))
      .finally(() => setIsLoading(false));
  }, []);

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
    onChange(all);
  }, [hierarchy, onChange]);

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

  if (hierarchy.length === 0) {
    return (
      <div className="py-2 text-xs text-slate-400">
        No MSigDB collections available. Check MSIGDB_DIR configuration.
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
          MSigDB Collections ({selected.size} selected)
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
        {hierarchy.map((cat) => {
          const isExpanded = expandedCategories.has(cat.category);
          const hasChildren =
            cat.children.length > 1 ||
            (cat.children.length === 1 && isSubcategory(cat.children[0]!));

          // For categories with a single direct collection (H, C8), render flat
          if (!hasChildren && cat.children.length === 1) {
            const col = cat.children[0] as MSigDBCollection;
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
                <span className="font-semibold text-slate-500">
                  {cat.category}:
                </span>
                <span>{col.name}</span>
                <span className="text-slate-400">({col.n_sets})</span>
              </label>
            );
          }

          return (
            <div key={cat.category}>
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
