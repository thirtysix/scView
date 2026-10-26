import { useCallback, useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { Check, Loader2, Sparkles, X } from "lucide-react";
import {
  clearLabelTerms,
  getLabelTerms,
  setLabelTerms,
  suggestLabelTerms,
  UNKNOWN_TERM,
  type LabelTerms,
  type TermSuggestion,
} from "@/api/ontology";
import { Caption } from "@/components/common/Caption";
import { CellTypeCombobox, TermChip } from "@/components/common/CellTypeCombobox";
import { DialogBox } from "@/components/common/DialogBox";
import { InfoTip } from "@/components/common/InfoTip";
import { prettyObsLabel } from "@/lib/formatting";

interface CellOntologyCardProps {
  datasetId: string;
  column: string;
  onClose: () => void;
}

/**
 * The Cell Ontology terms of a label column's labels: each label's term (a chip linking
 * to the term's page), change one by searching, mark it unknown, clear it, or ask for
 * suggestions from the label names and accept or reject each. Only terms the user
 * accepts are saved; a complete mapping lets the CELLxGENE export write
 * cell_type_ontology_term_id.
 */
export function CellOntologyCard({ datasetId, column, onClose }: CellOntologyCardProps) {
  const [data, setData] = useState<LabelTerms | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null); // label being saved, or "*"
  const [editing, setEditing] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<Record<string, TermSuggestion[]> | null>(null);
  const [suggesting, setSuggesting] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setSuggestions(null);
    getLabelTerms(datasetId, column)
      .then((r) => !cancelled && setData(r))
      .catch((e: unknown) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [datasetId, column]);

  const save = useCallback(
    async (label: string, id: string | null) => {
      setBusy(label);
      setError(null);
      try {
        setData(await setLabelTerms(datasetId, column, { [label]: id }));
        setEditing(null);
        setSuggestions((s) => {
          if (!s) return s;
          const next = { ...s };
          delete next[label];
          return next;
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(null);
      }
    },
    [datasetId, column],
  );

  const suggest = async () => {
    setSuggesting(true);
    setError(null);
    try {
      const r = await suggestLabelTerms(datasetId, column);
      const out: Record<string, TermSuggestion[]> = {};
      for (const row of r.labels) {
        const fresh = row.suggestions.filter((s) => s.id !== row.current);
        if (fresh.length && !(row.current && row.suggestions[0]?.id === row.current)) out[row.label] = fresh;
      }
      setSuggestions(out);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSuggesting(false);
    }
  };

  const clearAll = async () => {
    setBusy("*");
    setError(null);
    try {
      setData(await clearLabelTerms(datasetId, column));
      setConfirmClear(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const nSuggested = suggestions ? Object.keys(suggestions).length : 0;

  return createPortal(
    <div className="fixed inset-0 z-[60] flex items-start justify-end bg-slate-900/10 p-4" onMouseDown={(e) => {
      if (e.target === e.currentTarget) onClose();
    }}>
      <DialogBox
        label={`Cell Ontology terms of ${prettyObsLabel(column)}`}
        onClose={onClose}
        className="flex max-h-[85vh] w-[420px] max-w-full flex-col rounded-lg border border-slate-200 bg-white text-xs shadow-xl"
      >
        <div className="flex items-center justify-between gap-2 border-b border-slate-100 px-3 py-2">
          <h2 className="flex items-center gap-1 text-sm font-semibold text-slate-700">
            Cell Ontology: {prettyObsLabel(column)}
            <InfoTip label="About Cell Ontology terms" width={300}>
              The Cell Ontology (CL) is the shared vocabulary of cell types that atlases and CELLxGENE use.
              Giving each label a term makes the labels comparable across datasets. Suggestions come from the
              label names only; nothing is saved until you accept it. A term for every label lets the
              CELLxGENE export write <code>cell_type_ontology_term_id</code>.
            </InfoTip>
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-3 py-2">
          {!data && !error && (
            <p role="status" className="flex items-center gap-1 text-slate-500">
              <Loader2 className="h-3 w-3 animate-spin" /> Loading…
            </p>
          )}
          {data && (
            <>
              <Caption className="mb-2">
                {data.n_mapped} of {data.n_labels} labels have a term.
                {data.complete
                  ? " Complete: the CELLxGENE export can write the cell type terms."
                  : data.n_cells_unlabelled > 0
                    ? ` ${data.n_cells_unlabelled.toLocaleString()} cells have no label.`
                    : ""}
              </Caption>
              <ul className="space-y-1.5" aria-label="Labels and their terms">
                {data.labels.map((row) => {
                  const sug = suggestions?.[row.label];
                  return (
                    <li key={row.label} className="rounded border border-slate-100 px-2 py-1.5">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="truncate font-medium text-slate-700" title={row.label}>
                            {row.label}{" "}
                            <span className="font-normal text-slate-400">({row.n_cells.toLocaleString()} cells)</span>
                          </div>
                          <div className="mt-0.5">
                            {row.id ? (
                              <TermChip id={row.id} name={row.name} />
                            ) : (
                              <span className="text-[11px] text-slate-400">No term</span>
                            )}
                          </div>
                        </div>
                        <div className="flex shrink-0 items-center gap-1">
                          {busy === row.label && <Loader2 className="h-3 w-3 animate-spin text-slate-400" aria-label="Saving" />}
                          <button
                            type="button"
                            onClick={() => {
                              setEditing(editing === row.label ? null : row.label);
                              setQuery("");
                            }}
                            aria-expanded={editing === row.label}
                            aria-label={`${row.id ? "Change" : "Choose"} the term of ${row.label}`}
                            className="rounded px-1.5 py-0.5 text-[11px] text-blue-600 hover:bg-blue-50"
                          >
                            {row.id ? "Change" : "Choose"}
                          </button>
                          {row.id !== UNKNOWN_TERM && (
                            <button
                              type="button"
                              onClick={() => void save(row.label, UNKNOWN_TERM)}
                              disabled={busy !== null}
                              title="No known cell type (CELLxGENE's 'unknown')"
                              aria-label={`Unknown cell type for ${row.label}`}
                              className="rounded px-1.5 py-0.5 text-[11px] text-slate-500 hover:bg-slate-100 disabled:opacity-50"
                            >
                              Unknown
                            </button>
                          )}
                          {row.id && (
                            <button
                              type="button"
                              onClick={() => void save(row.label, null)}
                              disabled={busy !== null}
                              aria-label={`Clear the term of ${row.label}`}
                              title="Clear the term"
                              className="rounded p-0.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600 disabled:opacity-50"
                            >
                              <X className="h-3 w-3" />
                            </button>
                          )}
                        </div>
                      </div>
                      {editing === row.label && (
                        <div className="mt-1.5">
                          <CellTypeCombobox
                            label={`Search a term for ${row.label}`}
                            autoFocus
                            keepText
                            value={query}
                            onChange={setQuery}
                            term={null}
                            onTermChange={(t) => t && void save(row.label, t.id)}
                            placeholder="Name, synonym or CL id"
                          />
                        </div>
                      )}
                      {sug && sug.length > 0 && (
                        <div className="mt-1.5 rounded bg-amber-50 px-2 py-1">
                          <div className="mb-0.5 text-[10px] font-medium uppercase tracking-wide text-amber-800">
                            Suggested
                          </div>
                          <ul className="space-y-0.5">
                            {sug.map((s) => (
                              <li key={s.id} className="flex items-center justify-between gap-2">
                                <span className="min-w-0 truncate text-slate-700" title={s.definition ?? undefined}>
                                  {s.name} <span className="font-mono text-[10px] text-slate-400">{s.id}</span>{" "}
                                  <span className="text-[10px] text-slate-400">({s.how_text})</span>
                                </span>
                                <button
                                  type="button"
                                  onClick={() => void save(row.label, s.id)}
                                  disabled={busy !== null}
                                  aria-label={`Accept ${s.name} for ${row.label}`}
                                  className="flex shrink-0 items-center gap-0.5 rounded px-1.5 py-0.5 text-[11px] text-emerald-700 hover:bg-emerald-100 disabled:opacity-50"
                                >
                                  <Check className="h-3 w-3" /> Accept
                                </button>
                              </li>
                            ))}
                          </ul>
                          <button
                            type="button"
                            onClick={() =>
                              setSuggestions((cur) => {
                                const next = { ...(cur ?? {}) };
                                delete next[row.label];
                                return next;
                              })
                            }
                            aria-label={`Reject the suggestions for ${row.label}`}
                            className="mt-0.5 rounded px-1 text-[11px] text-slate-500 hover:bg-amber-100"
                          >
                            Reject
                          </button>
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            </>
          )}
          {error && (
            <p role="alert" className="mt-2 rounded bg-red-50 px-2 py-1 text-[11px] text-red-700">
              {error}
            </p>
          )}
          {suggestions && (
            <p role="status" className="mt-2 text-[11px] text-slate-500">
              {nSuggested
                ? `Suggestions for ${nSuggested} label${nSuggested === 1 ? "" : "s"}; accept or reject each.`
                : "No new suggestions from the label names."}
            </p>
          )}
        </div>

        <div className="flex items-center justify-between gap-2 border-t border-slate-100 px-3 py-2">
          <button
            type="button"
            onClick={() => void suggest()}
            disabled={!data || suggesting}
            className="flex items-center gap-1 rounded bg-blue-500 px-2.5 py-1 text-xs font-medium text-white hover:bg-blue-600 disabled:opacity-60"
          >
            {suggesting ? <Loader2 className="h-3 w-3 animate-spin" /> : <Sparkles className="h-3 w-3" />}
            Suggest terms
          </button>
          {data && data.n_mapped > 0 && (
            confirmClear ? (
              <span className="flex items-center gap-1">
                <span className="text-[11px] text-slate-500">Clear all {data.n_mapped} terms?</span>
                <button
                  type="button"
                  onClick={() => void clearAll()}
                  disabled={busy !== null}
                  className="rounded px-1.5 py-0.5 text-[11px] font-medium text-red-700 hover:bg-red-50"
                >
                  Clear
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmClear(false)}
                  className="rounded px-1.5 py-0.5 text-[11px] text-slate-500 hover:bg-slate-100"
                >
                  Keep
                </button>
              </span>
            ) : (
              <button
                type="button"
                onClick={() => setConfirmClear(true)}
                className="rounded px-1.5 py-0.5 text-[11px] text-slate-500 hover:bg-slate-100"
              >
                Clear all terms
              </button>
            )
          )}
        </div>
      </DialogBox>
    </div>,
    document.body,
  );
}
