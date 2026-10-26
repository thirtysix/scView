import { useEffect, useId, useRef, useState } from "react";
import { X } from "lucide-react";
import { olsUrl, searchCellTypes, UNKNOWN_TERM, type CellTypeTerm } from "@/api/ontology";

/** A Cell Ontology term attached to a label. */
export interface PickedTerm {
  id: string;
  name: string;
}

type Option =
  | { kind: "label"; key: string; label: string }
  | { kind: "term"; key: string; term: CellTypeTerm };

const MAX_KNOWN = 5;
const DEBOUNCE_MS = 200;

interface CellTypeComboboxProps {
  /** The visible field name */
  label: string;
  value: string;
  onChange: (text: string) => void;
  term: PickedTerm | null;
  onTermChange: (term: PickedTerm | null) => void;
  /** The column's labels, offered before the ontology's terms */
  knownLabels?: string[];
  placeholder?: string;
  autoFocus?: boolean;
  /** Picking a term leaves the text alone (a term field, not a name field) */
  keepText?: boolean;
}

/**
 * A text field that also searches the Cell Ontology (WAI-ARIA combobox: arrow keys move
 * through the list, Enter picks, Escape closes it). Free text stays allowed; picking a
 * term fills the field with the term's name and attaches its id, shown as a chip below
 * that the user can remove. Typing after picking keeps the term: the name is the user's.
 */
export function CellTypeCombobox({
  label,
  value,
  onChange,
  term,
  onTermChange,
  knownLabels = [],
  placeholder,
  autoFocus,
  keepText = false,
}: CellTypeComboboxProps) {
  const base = useId();
  const inputId = `${base}-input`;
  const listId = `${base}-list`;
  const inputRef = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const [terms, setTerms] = useState<CellTypeTerm[]>([]);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [typed, setTyped] = useState(false); // search only for what the user typed

  const query = value.trim();
  useEffect(() => {
    if (!typed || !query) {
      setTerms([]);
      return;
    }
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => {
      setSearching(true);
      searchCellTypes(query, 10, ctrl.signal)
        .then((r) => {
          setTerms(r.terms);
          setError(null);
        })
        .catch((e: unknown) => {
          if (ctrl.signal.aborted) return;
          setTerms([]);
          setError(e instanceof Error ? e.message : "The Cell Ontology search failed.");
        })
        .finally(() => {
          if (!ctrl.signal.aborted) setSearching(false);
        });
    }, DEBOUNCE_MS);
    return () => {
      ctrl.abort();
      window.clearTimeout(timer);
    };
  }, [query, typed]);

  const q = query.toLowerCase();
  const known = typed && q ? knownLabels.filter((l) => l.toLowerCase().includes(q) && l !== value).slice(0, MAX_KNOWN) : [];
  const options: Option[] = [
    ...known.map((l) => ({ kind: "label" as const, key: `l:${l}`, label: l })),
    ...terms.map((t) => ({ kind: "term" as const, key: `t:${t.id}`, term: t })),
  ];
  const expanded = open && (options.length > 0 || !!error);
  const optionId = (i: number) => `${base}-opt-${i}`;

  useEffect(() => {
    setActive(-1);
  }, [query, terms.length]);

  // Escape closes the list before the surrounding dialog sees it (a native listener
  // runs before the dialog's, which React's own handlers would not).
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && expanded) {
        e.preventDefault();
        e.stopPropagation();
        setOpen(false);
      }
    };
    el.addEventListener("keydown", onKey);
    return () => el.removeEventListener("keydown", onKey);
  }, [expanded]);

  const pick = (opt: Option) => {
    if (opt.kind === "label") {
      onChange(opt.label);
    } else {
      if (!keepText) onChange(opt.term.name);
      onTermChange({ id: opt.term.id, name: opt.term.name });
    }
    setTyped(false);
    setOpen(false);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (!options.length) return;
      e.preventDefault();
      setOpen(true);
      const step = e.key === "ArrowDown" ? 1 : -1;
      setActive((a) => (a < 0 ? (step > 0 ? 0 : options.length - 1) : (a + step + options.length) % options.length));
    } else if (e.key === "Enter" && expanded && active >= 0 && options[active]) {
      e.preventDefault(); // pick, don't submit the form
      pick(options[active]);
    } else if (e.key === "Tab") {
      setOpen(false);
    }
  };

  const status = searching
    ? "Searching the Cell Ontology…"
    : expanded && !error
      ? `${terms.length} Cell Ontology term${terms.length === 1 ? "" : "s"}${known.length ? `, ${known.length} label${known.length === 1 ? "" : "s"} in this column` : ""}`
      : "";

  return (
    <div className="relative space-y-1">
      <label htmlFor={inputId} className="block text-[11px] font-medium text-slate-500">
        {label}
      </label>
      <input
        ref={inputRef}
        id={inputId}
        autoFocus={autoFocus}
        role="combobox"
        aria-expanded={expanded}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-activedescendant={expanded && active >= 0 ? optionId(active) : undefined}
        autoComplete="off"
        value={value}
        placeholder={placeholder}
        onChange={(e) => {
          onChange(e.target.value);
          setTyped(true);
          setOpen(true);
        }}
        onKeyDown={onKeyDown}
        onBlur={() => setOpen(false)}
        className="w-full rounded border border-slate-300 px-2 py-1 text-xs text-slate-700 focus:border-blue-400 focus:outline-none"
      />
      <ul
        id={listId}
        role="listbox"
        aria-label={`${label}: suggestions`}
        hidden={!expanded}
        className="absolute left-0 right-0 z-50 max-h-56 overflow-y-auto rounded border border-slate-200 bg-white py-0.5 text-xs shadow-lg"
      >
        {error && (
          <li role="presentation" className="px-2 py-1 text-[11px] text-amber-800">
            Cell Ontology search unavailable: {error}
          </li>
        )}
        {options.map((opt, i) => (
          <li
            key={opt.key}
            id={optionId(i)}
            role="option"
            aria-selected={i === active}
            onMouseDown={(e) => e.preventDefault()} // keep focus in the field
            onClick={() => pick(opt)}
            onMouseEnter={() => setActive(i)}
            title={opt.kind === "term" ? (opt.term.definition ?? undefined) : undefined}
            className={`cursor-pointer px-2 py-1 ${i === active ? "bg-blue-50" : ""}`}
          >
            {opt.kind === "label" ? (
              <span className="text-slate-700">
                {opt.label} <span className="text-[10px] text-slate-400">in this column</span>
              </span>
            ) : (
              <span className="flex items-baseline justify-between gap-2">
                <span className="text-slate-700">
                  {opt.term.name}
                  {opt.term.matched && (
                    <span className="text-[10px] text-slate-400"> ({opt.term.matched})</span>
                  )}
                </span>
                <span className="shrink-0 font-mono text-[10px] text-slate-400">{opt.term.id}</span>
              </span>
            )}
          </li>
        ))}
      </ul>
      <span role="status" aria-live="polite" className="sr-only">
        {status}
      </span>
      {term && (
        <div className="flex items-center gap-1 text-[11px] text-slate-600">
          <span>Cell Ontology:</span>
          <TermChip id={term.id} name={term.name} />
          <button
            type="button"
            onClick={() => onTermChange(null)}
            aria-label={`Remove the Cell Ontology term ${term.name}`}
            title="Remove the term (the label stays)"
            className="rounded p-0.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      )}
    </div>
  );
}

/** A term's name and id, the id linking to its page on the EBI Ontology Lookup Service. */
export function TermChip({ id, name }: { id: string; name?: string | null }) {
  if (id === UNKNOWN_TERM) {
    return (
      <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[11px] text-slate-600">unknown cell type</span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded bg-emerald-50 px-1.5 py-0.5 text-[11px] text-emerald-800">
      {name && <span>{name}</span>}
      <a
        href={olsUrl(id)}
        target="_blank"
        rel="noopener noreferrer"
        className="font-mono text-[10px] underline decoration-dotted"
        title="Open the term in the EBI Ontology Lookup Service"
      >
        {id}
      </a>
    </span>
  );
}
