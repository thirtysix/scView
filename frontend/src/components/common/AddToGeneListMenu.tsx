import { useEffect, useRef, useState } from "react";
import { ListPlus } from "lucide-react";
import { useGeneListsStore } from "@/stores/geneListsStore";

interface Props {
  /** The genes to add (a function: read when the user picks a list, so a long table isn't copied on every render) */
  genes: string[] | (() => string[]);
  /** Name offered for a new list ("Markers: CD14 Mono") */
  defaultName: string;
  /** Where the genes came from, kept on a new list */
  source?: string;
  /** Button text; without it the button is an icon */
  label?: string;
  disabled?: boolean;
  title?: string;
  className?: string;
}

/**
 * "Add to gene list": the given genes into one of the user's gene lists
 * (stores/geneListsStore.ts), or into a new one. Used by the tables and
 * searches that produce genes (markers, DE, co-expression, gene search).
 */
export function AddToGeneListMenu({ genes, defaultName, source, label, disabled, title, className = "" }: Props) {
  const lists = useGeneListsStore((s) => s.lists);
  const [open, setOpen] = useState(false);
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState(defaultName);
  const [status, setStatus] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
  useEffect(() => {
    if (!status) return;
    const t = setTimeout(() => setStatus(null), 4000);
    return () => clearTimeout(t);
  }, [status]);

  const read = () => (typeof genes === "function" ? genes() : genes);
  const toggle = () => {
    setOpen(!open);
    setNaming(false);
    setName(defaultName);
  };
  const addTo = (id: string, listName: string) => {
    const n = useGeneListsStore.getState().addGenes(id, read());
    setStatus(n > 0 ? `Added ${n} gene${n === 1 ? "" : "s"} to “${listName}”` : `“${listName}” has them all`);
    setOpen(false);
  };
  const createNew = () => {
    const g = read();
    const id = useGeneListsStore.getState().create(name.trim() || defaultName, g, source);
    const made = id ? useGeneListsStore.getState().lists.find((l) => l.id === id) : null;
    setStatus(made ? `New list “${made.name}”: ${made.genes.length} genes` : "Too many lists; delete one first");
    setOpen(false);
  };

  return (
    <div ref={ref} className={`relative inline-flex items-center gap-1 ${className}`}>
      <button
        type="button"
        onClick={toggle}
        disabled={disabled}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={label ? undefined : "Add to gene list"}
        title={title ?? "Add these genes to a gene list (Gene lists tab)"}
        className={
          label
            ? "flex items-center gap-1 rounded border border-slate-300 bg-white px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-50"
            : "rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 disabled:opacity-50"
        }
      >
        <ListPlus className={label ? "h-3.5 w-3.5" : "h-3 w-3"} />
        {label}
      </button>
      {/* Over the content below, so a tight toolbar doesn't reflow */}
      {status && !open && (
        <span
          role="status"
          className="absolute right-0 top-full z-30 mt-1 whitespace-nowrap rounded border border-emerald-200 bg-white px-1.5 py-0.5 text-[10px] text-emerald-700 shadow-sm"
        >
          {status}
        </span>
      )}
      {open && (
        <div role="menu" className="absolute right-0 top-full z-30 mt-1 w-56 rounded-md border border-slate-200 bg-white p-1 text-xs shadow-lg">
          {lists.length > 0 && (
            <div className="max-h-48 overflow-y-auto">
              {lists.map((l) => (
                <button
                  key={l.id}
                  type="button"
                  role="menuitem"
                  onClick={() => addTo(l.id, l.name)}
                  className="flex w-full items-center justify-between gap-2 rounded px-2 py-1 text-left text-slate-700 hover:bg-slate-100"
                >
                  <span className="truncate">{l.name}</span>
                  <span className="text-[10px] text-slate-400">{l.genes.length}</span>
                </button>
              ))}
            </div>
          )}
          {naming ? (
            <form
              className="mt-1 flex items-center gap-1 border-t border-slate-100 px-1 pt-1"
              onSubmit={(e) => {
                e.preventDefault();
                createNew();
              }}
            >
              <input
                autoFocus
                value={name}
                onChange={(e) => setName(e.target.value)}
                aria-label="New list name"
                maxLength={200}
                className="min-w-0 flex-1 rounded border border-slate-200 px-1.5 py-0.5 focus:border-primary focus:outline-none"
              />
              <button type="submit" className="rounded bg-primary px-2 py-0.5 font-medium text-white hover:bg-primary/90">
                Add
              </button>
            </form>
          ) : (
            <button
              type="button"
              role="menuitem"
              onClick={() => setNaming(true)}
              className={`w-full rounded px-2 py-1 text-left font-medium text-primary hover:bg-slate-100 ${lists.length > 0 ? "mt-1 border-t border-slate-100" : ""}`}
            >
              New list…
            </button>
          )}
        </div>
      )}
    </div>
  );
}
