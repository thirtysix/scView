import { useEffect, useMemo, useRef, useState } from "react";
import { useDialogFocus } from "@/hooks/useDialogFocus";
import { Filter, Loader2, Plus, Trash2, X } from "lucide-react";
import { cellsMatching, filterCells } from "@/api/cells";
import type { DatasetInfo, ObsColumnInfo } from "@/api/types";
import { GeneSearch } from "@/components/common/GeneSearch";
import {
  describeFilter,
  filterConditions,
  GENE_OPS,
  pruneFilter,
  type CellFilter,
  type Condition,
  type FilterOp,
  type GeneOp,
} from "@/lib/cellFilter";
import { formatNumber, prettyObsLabel } from "@/lib/formatting";
import { isCategoricalDtype } from "@/lib/obsColumns";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";

// The filter being edited: conditions carry a uid so rows keep their own input
// state when one above them is removed.
interface DraftCondition {
  uid: number;
  cond: Condition;
}
interface DraftGroup {
  uid: number;
  op: FilterOp;
  conditions: DraftCondition[];
}
type DraftItem = DraftCondition | DraftGroup;
interface Draft {
  op: FilterOp;
  items: DraftItem[];
}

let nextUid = 1;
const uid = () => nextUid++;
const isGroup = (i: DraftItem): i is DraftGroup => "conditions" in i;
const newGene = (): DraftCondition => ({ uid: uid(), cond: { kind: "gene", gene: "", op: ">", value: null } });
const emptyDraft = (): Draft => ({ op: "and", items: [newGene()] });

function fromFilter(f: CellFilter): Draft {
  return {
    op: f.op,
    items: f.items.map((i) =>
      i.kind === "group"
        ? { uid: uid(), op: i.op, conditions: i.conditions.map((c) => ({ uid: uid(), cond: c })) }
        : { uid: uid(), cond: i },
    ),
  };
}

function toFilter(d: Draft, layer: string): CellFilter {
  const items = d.items.map((i) =>
    isGroup(i) ? { kind: "group" as const, op: i.op, conditions: i.conditions.map((c) => c.cond) } : i.cond,
  );
  const f: CellFilter = { op: d.op, items, layer: "" };
  // The layer only matters to gene conditions; leave it out otherwise.
  return filterConditions(f).some((c) => c.kind === "gene") ? { ...f, layer } : f;
}

// The last filter edited per dataset, so closing the panel (or a tab switch,
// which remounts the toolbar) does not lose it.
const drafts = new Map<string, Draft>();
export const clearCellFilterDrafts = () => drafts.clear();

const GENE_FIELD = "\u0000gene";
const isNumericDtype = (dtype: string) => /^(u?int|float)/i.test(dtype);

function fieldColumns(dataset: DatasetInfo): { categories: ObsColumnInfo[]; numbers: ObsColumnInfo[] } {
  const cols = dataset.obs_columns ?? [];
  return {
    categories: cols.filter((c) => isCategoricalDtype(c.dtype) && (c.values?.length ?? 0) > 0),
    numbers: cols.filter((c) => isNumericDtype(c.dtype)),
  };
}

/**
 * Toolbar "Filter": build AND/OR conditions on genes (expression on the current
 * layer) and metadata (values of a categorical column, a range of a numeric one),
 * with a live count of the matching cells. "Select" makes them the selection, as
 * a lasso would, so DE, "Label selection…" and highlighting work on it.
 */
export function CellFilterControl() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  if (!dataset) return null;
  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="dialog"
        title="Select cells by conditions on genes and metadata"
        className={`flex items-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors ${
          open ? "bg-orange-50 text-orange-700 ring-1 ring-orange-300" : "text-slate-600 hover:bg-slate-100"
        }`}
      >
        <Filter className="h-3.5 w-3.5" />
        Filter
      </button>
      {open && <CellFilterPanel key={dataset.id} dataset={dataset} onClose={() => setOpen(false)} />}
    </div>
  );
}

interface CellFilterPanelProps {
  dataset: DatasetInfo;
  onClose: () => void;
}

export function CellFilterPanel({ dataset, onClose }: CellFilterPanelProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  useDialogFocus(dialogRef, onClose);
  const layer = useSettingsStore((s) => s.expressionLayer);
  const [draft, setDraftState] = useState<Draft>(() => {
    const current = useSelectionStore.getState().selectionFilter;
    const last = drafts.get(dataset.id);
    return current ? fromFilter(current) : last?.items.length ? last : emptyDraft();
  });
  const setDraft = (next: Draft) => {
    drafts.set(dataset.id, next);
    setDraftState(next);
  };
  const filter = useMemo(() => pruneFilter(toFilter(draft, layer)), [draft, layer]);
  const filterKey = filter ? JSON.stringify(filter) : "";
  const fields = useMemo(() => fieldColumns(dataset), [dataset]);

  // Live count, a moment after the last edit; an older answer is dropped.
  const [count, setCount] = useState<{ n: number; total: number } | null>(null);
  const [counting, setCounting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setError(null);
    if (!filter) {
      setCount(null);
      setCounting(false);
      return;
    }
    setCounting(true);
    const ctrl = new AbortController();
    const t = setTimeout(() => {
      filterCells(dataset.id, filter, false, ctrl.signal)
        .then((r) => setCount({ n: r.n_matching, total: r.n_cells }))
        .catch((e: unknown) => {
          if (ctrl.signal.aborted) return;
          setCount(null);
          setError(e instanceof Error ? e.message : String(e));
        })
        .finally(() => {
          if (!ctrl.signal.aborted) setCounting(false);
        });
    }, 300);
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
    // filterKey stands for `filter` (a new object on every render of the same filter)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset.id, filterKey]);

  const [selecting, setSelecting] = useState(false);
  const select = async () => {
    if (!filter || selecting) return;
    setSelecting(true);
    setError(null);
    try {
      const cells = await cellsMatching(dataset.id, filter);
      const store = useSelectionStore.getState();
      store.setHighlight(null); // a filter is not a group, as with a lasso
      if (cells.size === 0) store.clearSelection();
      else store.setSelection(cells, filter);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSelecting(false);
    }
  };

  const setItem = (uidOf: number, next: DraftItem | null) =>
    setDraft({
      ...draft,
      items: draft.items.flatMap((i) => (i.uid === uidOf ? (next ? [next] : []) : [i])),
    });

  const hasGene = !!filter && filterConditions(filter).some((c) => c.kind === "gene");
  const layerLabel = dataset.expression_layers?.find((l) => l.key === layer)?.label ?? layer;

  return (
    <div
      ref={dialogRef}
      role="dialog"
      aria-label="Filter cells"
      className="absolute left-0 top-full z-50 mt-1 w-[30rem] max-w-[90vw] space-y-2.5 rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg"
    >
      <div className="flex items-center justify-between">
        <span className="font-semibold text-slate-700">Filter cells</span>
        <button onClick={onClose} aria-label="Close" className="rounded p-0.5 text-slate-400 hover:bg-slate-100">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      <OpToggle label="Cells matching" op={draft.op} onChange={(op) => setDraft({ ...draft, op })} />

      <div className="max-h-[50vh] space-y-1.5 overflow-y-auto pr-1">
        {draft.items.map((item) =>
          isGroup(item) ? (
            <div key={item.uid} className="space-y-1.5 rounded-md border border-slate-200 bg-slate-50 p-2">
              <div className="flex items-center justify-between">
                <OpToggle label="Group: cells matching" op={item.op} onChange={(op) => setItem(item.uid, { ...item, op })} />
                <button
                  onClick={() => setItem(item.uid, null)}
                  aria-label="Remove group"
                  title="Remove group"
                  className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-red-600"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
              {item.conditions.map((c) => (
                <ConditionRow
                  key={c.uid}
                  datasetId={dataset.id}
                  fields={fields}
                  cond={c.cond}
                  onChange={(cond) =>
                    setItem(item.uid, {
                      ...item,
                      conditions: item.conditions.map((x) => (x.uid === c.uid ? { ...x, cond } : x)),
                    })
                  }
                  onRemove={() => {
                    const rest = item.conditions.filter((x) => x.uid !== c.uid);
                    setItem(item.uid, rest.length ? { ...item, conditions: rest } : null);
                  }}
                />
              ))}
              <AddButton onClick={() => setItem(item.uid, { ...item, conditions: [...item.conditions, newGene()] })}>
                Condition
              </AddButton>
            </div>
          ) : (
            <ConditionRow
              key={item.uid}
              datasetId={dataset.id}
              fields={fields}
              cond={item.cond}
              onChange={(cond) => setItem(item.uid, { ...item, cond })}
              onRemove={() => setItem(item.uid, null)}
            />
          ),
        )}
      </div>

      <div className="flex gap-1">
        <AddButton onClick={() => setDraft({ ...draft, items: [...draft.items, newGene()] })}>Condition</AddButton>
        <AddButton
          onClick={() =>
            setDraft({ ...draft, items: [...draft.items, { uid: uid(), op: draft.op === "and" ? "or" : "and", conditions: [newGene()] }] })
          }
        >
          Group
        </AddButton>
        <button
          onClick={() => setDraft(emptyDraft())}
          className="ml-auto rounded px-2 py-1 text-[11px] text-slate-500 hover:bg-slate-100"
        >
          Reset
        </button>
      </div>

      <div className="rounded bg-slate-50 px-2 py-1.5 text-[11px] text-slate-600" aria-label="Filter as text">
        {filter ? describeFilter(filter) : "Fill in a condition to count the cells."}
        {hasGene && <span className="block text-[10px] text-slate-400">Gene values: {layerLabel}</span>}
      </div>

      {error && (
        <p role="alert" className="rounded bg-red-50 px-2 py-1 text-[11px] text-red-700">
          {error}
        </p>
      )}

      <div className="flex items-center justify-between gap-2">
        <span role="status" className="text-[11px] text-slate-500">
          {counting ? (
            <Loader2 className="inline h-3 w-3 animate-spin" />
          ) : count ? (
            <>
              <span className="font-semibold text-slate-700">{formatNumber(count.n)}</span> of{" "}
              {formatNumber(count.total)} cells match
            </>
          ) : null}
        </span>
        <button
          onClick={() => void select()}
          disabled={!filter || selecting || counting || !!error || count?.n === 0}
          className="flex items-center gap-1 rounded bg-blue-500 px-3 py-1 text-xs font-medium text-white hover:bg-blue-600 disabled:opacity-50"
        >
          {selecting && <Loader2 className="h-3 w-3 animate-spin" />}
          Select
        </button>
      </div>
    </div>
  );
}

function OpToggle({ label, op, onChange }: { label: string; op: FilterOp; onChange: (op: FilterOp) => void }) {
  return (
    <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
      <span>{label}</span>
      <div className="flex overflow-hidden rounded border border-slate-300" role="radiogroup" aria-label={label}>
        {(["and", "or"] as const).map((o) => (
          <button
            key={o}
            role="radio"
            aria-checked={op === o}
            onClick={() => onChange(o)}
            title={o === "and" ? "Every condition (AND)" : "Any condition (OR)"}
            className={`px-1.5 py-0.5 text-[10px] font-semibold ${
              op === o ? "bg-blue-500 text-white" : "bg-white text-slate-600 hover:bg-slate-100"
            }`}
          >
            {o === "and" ? "ALL (AND)" : "ANY (OR)"}
          </button>
        ))}
      </div>
    </div>
  );
}

function AddButton({ onClick, children }: { onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[11px] font-medium text-blue-600 hover:bg-blue-50"
    >
      <Plus className="h-3 w-3" />
      {children}
    </button>
  );
}

interface ConditionRowProps {
  datasetId: string;
  fields: { categories: ObsColumnInfo[]; numbers: ObsColumnInfo[] };
  cond: Condition;
  onChange: (cond: Condition) => void;
  onRemove: () => void;
}

function ConditionRow({ datasetId, fields, cond, onChange, onRemove }: ConditionRowProps) {
  const field = cond.kind === "gene" ? GENE_FIELD : cond.column;
  const pickField = (f: string) => {
    if (f === GENE_FIELD) onChange({ kind: "gene", gene: "", op: ">", value: null });
    else if (fields.categories.some((c) => c.name === f)) onChange({ kind: "category", column: f, values: [] });
    else onChange({ kind: "range", column: f, min: null, max: null });
  };
  const selectCls =
    "rounded border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-700 focus:border-blue-400 focus:outline-none";
  const categoryValues =
    cond.kind === "category" ? fields.categories.find((c) => c.name === cond.column)?.values ?? [] : [];

  return (
    <div className="flex items-start gap-1.5" data-testid="filter-condition">
      <select
        aria-label="Condition on"
        value={field}
        onChange={(e) => pickField(e.target.value)}
        className={`${selectCls} w-32 flex-shrink-0`}
      >
        <option value={GENE_FIELD}>Gene expression</option>
        {fields.categories.length > 0 && (
          <optgroup label="Categories">
            {fields.categories.map((c) => (
              <option key={c.name} value={c.name}>
                {prettyObsLabel(c.name)}
              </option>
            ))}
          </optgroup>
        )}
        {fields.numbers.length > 0 && (
          <optgroup label="Numbers">
            {fields.numbers.map((c) => (
              <option key={c.name} value={c.name}>
                {prettyObsLabel(c.name)}
              </option>
            ))}
          </optgroup>
        )}
      </select>

      {/* Keyed by the field: another column starts with empty boxes */}
      <div key={field} className="flex min-w-0 flex-1 flex-wrap items-center gap-1.5">
        {cond.kind === "gene" && (
          <>
            {cond.gene ? (
              <button
                onClick={() => onChange({ ...cond, gene: "" })}
                title="Pick another gene"
                className="rounded border border-slate-300 bg-white px-2 py-1 font-mono text-xs text-slate-700 hover:bg-slate-50"
              >
                {cond.gene}
              </button>
            ) : (
              <GeneSearch
                datasetId={datasetId}
                onSelect={(gene) => onChange({ ...cond, gene: gene.trim() })}
                placeholder="Gene"
                className="w-32"
              />
            )}
            <select
              aria-label="Comparison"
              value={cond.op}
              onChange={(e) => onChange({ ...cond, op: e.target.value as GeneOp })}
              className={selectCls}
            >
              {GENE_OPS.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
            <NumberField label="Value" value={cond.value} onChange={(value) => onChange({ ...cond, value })} />
          </>
        )}
        {cond.kind === "category" && (
          <div className="max-h-24 w-full overflow-y-auto rounded border border-slate-200 bg-white px-1.5 py-1">
            {categoryValues.map((v) => (
              <label key={v} className="flex items-center gap-1.5 text-[11px] text-slate-700">
                <input
                  type="checkbox"
                  checked={cond.values.includes(v)}
                  onChange={(e) =>
                    onChange({
                      ...cond,
                      values: e.target.checked
                        ? categoryValues.filter((x) => x === v || cond.values.includes(x))
                        : cond.values.filter((x) => x !== v),
                    })
                  }
                />
                <span className="truncate">{v}</span>
              </label>
            ))}
          </div>
        )}
        {cond.kind === "range" && (
          <>
            <NumberField label="Minimum" value={cond.min} onChange={(min) => onChange({ ...cond, min })} />
            <span className="text-slate-400">to</span>
            <NumberField label="Maximum" value={cond.max} onChange={(max) => onChange({ ...cond, max })} />
            {cond.min != null && cond.max != null && cond.min > cond.max && (
              <span className="text-[10px] text-amber-700">min is above max</span>
            )}
          </>
        )}
      </div>

      <button
        onClick={onRemove}
        aria-label="Remove condition"
        title="Remove condition"
        className="flex-shrink-0 rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-red-600"
      >
        <Trash2 className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}

/** A number box that keeps what is typed ("1.", "-") while the value is null or a number. */
function NumberField({ label, value, onChange }: { label: string; value: number | null; onChange: (v: number | null) => void }) {
  const [text, setText] = useState(value == null ? "" : String(value));
  return (
    <input
      aria-label={label}
      inputMode="decimal"
      value={text}
      placeholder={label === "Value" ? "value" : label === "Minimum" ? "min" : "max"}
      onChange={(e) => {
        setText(e.target.value);
        const t = e.target.value.trim();
        const n = t === "" ? null : Number(t);
        onChange(n != null && Number.isFinite(n) ? n : null);
      }}
      className="w-20 rounded border border-slate-300 px-1.5 py-1 text-xs text-slate-700 focus:border-blue-400 focus:outline-none"
    />
  );
}
