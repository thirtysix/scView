import { useMemo, useState } from "react";
import { InfoTip } from "@/components/common/InfoTip";
import {
  CUT_LABELS,
  type CutAt,
  type LabelCount,
  type LabelEdits,
  NO_EDITS,
  SEPARATORS,
  hasEdits,
  likelySeparator,
  planLabels,
  planProblem,
  prefixGroups,
  separatorLabel,
  withGroups,
} from "@/lib/spotLabels";

/**
 * Reference labels of a spot composition run: every label with its cells, "out" and "merge into"; catch-all names
 * marked (never ticked for you); a live type count; a helper that merges labels by a shared name, previewed first.
 * Scrolls inside itself, so a column of 59 labels does not grow the page.
 */
export function SpotReferenceLabels({
  labels,
  suggested,
  edits,
  onChange,
  maxTypes,
  minCells,
  disabled,
}: {
  labels: LabelCount[];
  suggested: string[];
  edits: LabelEdits;
  onChange: (e: LabelEdits) => void;
  maxTypes: number;
  minCells: number;
  disabled?: boolean;
}) {
  const plan = useMemo(() => planLabels(labels, edits, minCells), [labels, edits, minCells]);
  const problem = planProblem(plan, edits, maxTypes, minCells);
  const edited = hasEdits(edits);
  const [opened, setOpened] = useState<boolean | null>(null);
  const open = opened ?? (edited || plan.types.length > maxTypes);
  const [sep, setSep] = useState<string | null>(null);
  const [cut, setCut] = useState<CutAt>("number");
  const [previewing, setPreviewing] = useState(false);
  const [unticked, setUnticked] = useState<string[]>([]);
  const separator = sep ?? likelySeparator(labels);
  const suggestedSet = useMemo(() => new Set(suggested), [suggested]);
  const groups = useMemo(() => prefixGroups(labels, edits.exclude, separator, cut), [labels, edits.exclude, separator, cut]);
  const ticked = groups.filter((g) => !unticked.includes(g.name));
  const afterPreview = planLabels(labels, { exclude: edits.exclude, merge: withGroups(edits.merge, ticked) }, minCells);
  const untickedSuggested = suggested.filter((s) => !edits.exclude.includes(s));

  const setOut = (label: string, out: boolean) =>
    onChange({ ...edits, exclude: out ? [...edits.exclude, label] : edits.exclude.filter((x) => x !== label) });
  const setMerge = (label: string, name: string) => {
    const merge = { ...edits.merge };
    if (name) merge[label] = name;
    else delete merge[label];
    onChange({ ...edits, merge });
  };
  const apply = () => {
    onChange({ ...edits, merge: withGroups(edits.merge, ticked) });
    setPreviewing(false);
  };

  return (
    <div className="rounded border border-slate-200 p-1.5" data-testid="spot-labels">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <button
          type="button"
          className="font-medium text-slate-700 hover:underline"
          aria-expanded={open}
          onClick={() => setOpened(!open)}
        >
          Reference labels
        </button>
        <InfoTip label="About reference labels" width={320}>
          Left out: those cells are not used, as if they had no label, so no {"spot's"} RNA can go to them. Merged: their
          cells make one type under the name you give (an existing label&apos;s name joins it); merging comes before the
          minimum of {minCells} cells and the limit of {maxTypes} types. A label both left out and merged is left out. A
          catch-all label can take signal from the real types, so look at those marked; nothing is left out unless you
          tick it.
        </InfoTip>
        <span className={problem ? "text-amber-700" : "text-slate-500"} data-testid="spot-labels-count">
          {plan.types.length} {plan.types.length === 1 ? "type" : "types"}, at most {maxTypes}
        </span>
        {plan.nOut > 0 && <span className="text-slate-500">{plan.nOut} out</span>}
        {plan.nMerged > 0 && <span className="text-slate-500">{plan.nMerged} merged</span>}
        {!open && untickedSuggested.length > 0 && (
          <span className="text-slate-500">{untickedSuggested.length} look like catch-alls</span>
        )}
        {open && untickedSuggested.length > 0 && (
          <button
            type="button"
            className="ml-auto rounded border px-1.5 py-0.5 hover:bg-slate-50 disabled:opacity-50"
            disabled={disabled}
            onClick={() => onChange({ ...edits, exclude: [...edits.exclude, ...untickedSuggested] })}
          >
            Leave out the {untickedSuggested.length} marked
          </button>
        )}
        {open && edited && (
          <button
            type="button"
            className={`${untickedSuggested.length > 0 ? "" : "ml-auto "}rounded border px-1.5 py-0.5 hover:bg-slate-50 disabled:opacity-50`}
            disabled={disabled}
            onClick={() => onChange(NO_EDITS)}
          >
            Clear
          </button>
        )}
      </div>
      {open && (
        <div className="mt-1 space-y-1.5">
          <div className="max-h-56 overflow-y-auto rounded border border-slate-100" data-testid="spot-labels-table">
            <table className="w-full">
              <thead className="sticky top-0 bg-white">
                <tr className="text-left text-slate-500">
                  <th className="px-1 font-normal" title="The reference's label">Label</th>
                  <th className="px-1 text-right font-normal" title="Reference cells with this label">Cells</th>
                  <th className="px-1 text-center font-normal" title="Leave these cells out of the reference">Out</th>
                  <th className="px-1 font-normal" title="A broader name; an existing label's name joins it">Merge into</th>
                </tr>
              </thead>
              <tbody>
                {labels.map((l) => {
                  const out = edits.exclude.includes(l.type);
                  return (
                    <tr key={l.type} className={out ? "text-slate-400" : "text-slate-700"}>
                      <td className="max-w-[12rem] truncate px-1" title={l.type}>
                        <span className={out ? "line-through" : ""}>{l.type}</span>
                        {suggestedSet.has(l.type) && (
                          <span
                            className="ml-1 rounded bg-amber-50 px-1 text-[10px] text-amber-700"
                            title="Its name looks like a catch-all label; it is used unless you tick Out."
                            data-testid="spot-label-suggested"
                          >
                            catch-all?
                          </span>
                        )}
                      </td>
                      <td className={`px-1 text-right ${l.n_cells < minCells ? "text-amber-700" : ""}`} title={l.n_cells < minCells ? `Under ${minCells} cells: not used unless merged` : undefined}>
                        {l.n_cells.toLocaleString()}
                      </td>
                      <td className="px-1 text-center">
                        <input
                          type="checkbox"
                          checked={out}
                          disabled={disabled}
                          aria-label={`Leave out ${l.type}`}
                          onChange={(e) => setOut(l.type, e.target.checked)}
                        />
                      </td>
                      <td className="px-1 py-0.5">
                        <input
                          type="text"
                          list="spot-label-names"
                          className="w-full min-w-[6rem] rounded border px-1 py-0 disabled:bg-slate-50"
                          value={edits.merge[l.type] ?? ""}
                          disabled={disabled || out}
                          aria-label={`Merge ${l.type} into`}
                          onChange={(e) => setMerge(l.type, e.target.value)}
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <datalist id="spot-label-names">
              {labels.map((l) => (
                <option key={l.type} value={l.type} />
              ))}
            </datalist>
          </div>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-slate-600" data-testid="spot-labels-coarsen">
            <span>Merge by shared name: cut at</span>
            <select
              className="rounded border px-1 py-0.5"
              value={cut}
              disabled={disabled}
              aria-label="Where to cut a label's name"
              onChange={(e) => setCut(e.target.value as CutAt)}
            >
              {(Object.keys(CUT_LABELS) as CutAt[]).map((c) => (
                <option key={c} value={c}>
                  {CUT_LABELS[c]}
                </option>
              ))}
            </select>
            <label className="flex items-center gap-1">
              separator
              <select
                className="rounded border px-1 py-0.5"
                value={separator}
                disabled={disabled}
                aria-label="Separator in the label's name"
                onChange={(e) => setSep(e.target.value)}
              >
                {SEPARATORS.map((s) => (
                  <option key={s} value={s}>
                    {separatorLabel(s)}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className="rounded border px-1.5 py-0.5 hover:bg-slate-50 disabled:opacity-50"
              disabled={disabled}
              onClick={() => {
                setUnticked([]);
                setPreviewing(!previewing);
              }}
            >
              {previewing ? "Hide preview" : "Preview"}
            </button>
          </div>
          {previewing && (
            <div className="space-y-1 rounded bg-slate-50 p-1.5" data-testid="spot-labels-preview">
              {groups.length === 0 ? (
                <p className="text-slate-500">No two labels share a name this way.</p>
              ) : (
                <>
                  <ul className="max-h-32 space-y-0.5 overflow-y-auto">
                    {groups.map((g) => (
                      <li key={g.name}>
                        <label className="flex items-start gap-1">
                          <input
                            type="checkbox"
                            className="mt-0.5"
                            checked={!unticked.includes(g.name)}
                            aria-label={`Merge into ${g.name}`}
                            onChange={(e) =>
                              setUnticked(e.target.checked ? unticked.filter((n) => n !== g.name) : [...unticked, g.name])
                            }
                          />
                          <span>
                            <b className="font-medium">{g.name}</b>: {g.labels.join(", ")} ({g.n_cells.toLocaleString()} cells)
                          </span>
                        </label>
                      </li>
                    ))}
                  </ul>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-slate-600" data-testid="spot-labels-preview-count">
                      {plan.types.length} types would become {afterPreview.types.length}.
                    </span>
                    <button
                      type="button"
                      className="rounded border px-1.5 py-0.5 hover:bg-white disabled:opacity-50"
                      disabled={disabled || ticked.length === 0}
                      onClick={apply}
                    >
                      Apply {ticked.length} {ticked.length === 1 ? "merge" : "merges"}
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
