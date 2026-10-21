import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Loader2, Tag } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { Panel } from "@/components/common/Panel";
import { InfoTip } from "@/components/common/InfoTip";
import { Caption } from "@/components/common/Caption";
import { getQcStates, saveQcStateLabel, type QcStateGroup, type QcStateLabelSaved, type QcStates as Result } from "@/api/qcStates";
import { useCategoryColors, useCategoryOrder } from "@/hooks/useCategoryColors";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { qcGroupOptions, QC_MAX_GROUPS } from "@/lib/qcGroups";
import {
  columnReason,
  errorText,
  levelColumns,
  pctText,
  shareRange,
  stateTraces,
  stripAxis,
  summaryText,
  validStateName,
} from "@/lib/qcStates";
import { plotConfig } from "@/lib/plotConfig";
import { useDatasetStore } from "@/stores/datasetStore";

const FILL_STYLE = { width: "100%", height: "100%" } as const;
const TH = "whitespace-nowrap px-1.5 py-1 text-left font-medium";
const NUM = "whitespace-nowrap px-1.5 py-1 text-right tabular-nums";

const HEAD_TIPS = {
  group: "The group of the chosen column (cell type or cluster).",
  n: "Cells in the group. Groups under 100 cells are not fitted.",
  states:
    "1 or 2: a Gaussian mixture with 1 and with 2 components is fitted to each cell's logit RP share (and logit mito fraction when measured). Two states need all three rules: BIC improves by more than 10, the components are apart (D >= 2) and the smaller holds at least 5% of the cells.",
  bic: "BIC of the 1-component fit minus BIC of the 2-component fit. Positive favours two components; over 10 is strong evidence, but with thousands of cells a skewed or tailed distribution also passes.",
  d: "Ashman's D: distance between the two components' means in units of their average spread (Mahalanobis form in 2-D). Two Gaussians are cleanly apart from D = 2; under 2 the 'second component' is the shape of one broad state.",
  share: "Share of the group's cells assigned to the flagged state: low-RP high-mito (or low-RP when mito is not measured, or 'second state' when the low-RP component has less mito).",
  rp: "Mean ribosomal-protein (RPL*/RPS*) share of non-mitochondrial counts, %, of the cells in each state: main / flagged.",
  mt: "Mean % mitochondrial counts of the cells in each state: main / flagged.",
};

function Th({ children, tip, right = false }: { children: ReactNode; tip: string; right?: boolean }) {
  return (
    <th className={right ? `${TH} text-right` : TH} title={tip}>
      {children}
    </th>
  );
}

function meanPair(g: QcStateGroup, key: "mean_rp_share" | "mean_pct_mt"): string {
  const vals = g.components.map((c) => c[key]);
  if (vals.every((v) => v == null)) return "—";
  return vals.map((v) => (v == null ? "—" : v.toFixed(1))).join(" / ");
}

/**
 * QC states (batch 11, proposal item 2): per cell type, is the high-mito part a
 * tail or a second state? Ribosomal-protein share of non-mito counts against
 * % mito per cell, 1 vs 2 Gaussian components per group, and the flagged state
 * offered as an obs label rather than a threshold (core/qc_states.py).
 */
export function QcStates({ datasetId, refreshKey = 0 }: { datasetId: string; refreshKey?: number }) {
  const [open, setOpen] = useState(false);
  const [column, setColumn] = useState<string>(""); // "" = the backend's default
  const [data, setData] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [focus, setFocus] = useState<string>(""); // "" = all groups
  const [name, setName] = useState("qc_state");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<QcStateLabelSaved | null>(null);
  const obsColumns = useDatasetStore((s) => s.currentDataset?.obs_columns);
  const refresh = useDatasetRefresh();

  const shown = data?.column ?? null;
  const colorOf = useCategoryColors(shown);
  const order = useCategoryOrder(shown);
  const options = useMemo(() => qcGroupOptions(obsColumns, QC_MAX_GROUPS), [obsColumns]);

  // A new dataset or a pipeline run makes the shown fit stale.
  useEffect(() => {
    setData(null);
    setError(null);
    setSaved(null);
    setFocus("");
  }, [datasetId, refreshKey]);
  useEffect(() => {
    setColumn("");
  }, [datasetId]);

  useEffect(() => {
    if (!open || !datasetId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    getQcStates(datasetId, column || undefined)
      .then((d) => {
        if (cancelled) return;
        setData(d);
        setFocus("");
      })
      .catch((e) => {
        if (!cancelled) setError(errorText(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, datasetId, column, refreshKey]);

  const traces = useMemo(
    () => (data?.scatter ? stateTraces(data, order, colorOf, focus || null) : []),
    [data, order, colorOf, focus],
  );
  const twoD = data?.scatter?.y != null;
  const strip = useMemo(() => (data && !twoD ? stripAxis(data, order) : null), [data, twoD, order]);
  const levels = data ? levelColumns(data) : { levels: [], asRange: false };
  const fittedCells = data?.overall?.n_fitted ?? 0;
  const canSave = !!data?.column && data.two_state_groups.length > 0;

  const save = async () => {
    if (!data?.column || !validStateName(name)) return;
    setSaving(true);
    setError(null);
    setSaved(null);
    try {
      const s = await saveQcStateLabel(datasetId, data.column, name);
      await refresh(datasetId, s.name); // the column shows in colour-by straight away
      setSaved(s);
    } catch (e) {
      setError(`Could not save the label: ${errorText(e)}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-2 rounded-xl border border-slate-200 bg-white p-3" data-testid="qc-states">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <h3 className="text-sm font-semibold text-slate-700">QC states: RP share vs mito</h3>
          <InfoTip label="About the QC state check" width={300}>
            A fixed mito cutoff can remove most of one cell type when that type holds a second state:
            cells with a low ribosomal-protein (RP) share, high % mito and more ambient RNA. For each
            group with 100 or more cells, a Gaussian mixture with 1 and with 2 components is fitted to
            each cell&apos;s RP share of non-mito counts and its % mito (both on the logit scale). Two
            states are called only when the BIC improves by more than 10, the components are apart
            (Ashman&apos;s D of 2 or more) and the smaller holds at least 5% of cells. The state can be
            saved as a label, to compare or exclude it, instead of a threshold.
          </InfoTip>
        </div>
        <span className="flex items-center gap-2">
          {loading && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            className="rounded border border-slate-200 px-2 py-0.5 text-[11px] font-medium text-slate-600 hover:bg-slate-100"
          >
            {open ? "Hide" : "Show"}
          </button>
        </span>
      </div>
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {error}
        </p>
      )}
      {open && (
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
          <label htmlFor="qc-states-column">By</label>
          <select
            id="qc-states-column"
            value={column || data?.column || ""}
            onChange={(e) => setColumn(e.target.value)}
            disabled={loading}
            title="The label column whose groups are fitted one by one: a cell-type column by default, else the clustering."
            className="min-w-0 max-w-[45%] rounded border border-slate-200 bg-white px-1.5 py-0.5 text-xs text-slate-600 disabled:opacity-50"
          >
            {!data?.column && <option value="">Default</option>}
            {data?.column && !options.some((o) => o.name === data.column) && (
              <option value={data.column}>{data.column}</option>
            )}
            {options.map((o) => (
              <option key={o.name} value={o.name} disabled={o.disabled}>
                {o.name} ({o.n}
                {o.disabled ? ` groups: over ${QC_MAX_GROUPS}` : ""})
              </option>
            ))}
          </select>
          {data?.scatter && (
            <>
              <label htmlFor="qc-states-focus">Show</label>
              <select
                id="qc-states-focus"
                value={focus}
                onChange={(e) => setFocus(e.target.value)}
                title="All groups (colour = group, crosses = flagged state), or one group (red = flagged state)."
                className="min-w-0 max-w-[35%] rounded border border-slate-200 bg-white px-1.5 py-0.5 text-xs text-slate-600"
              >
                <option value="">All groups</option>
                {data.groups
                  .filter((g) => g.n_fitted > 0)
                  .map((g) => (
                    <option key={g.name} value={g.name}>
                      {g.name}
                      {g.n_states === 2 ? " (2 states)" : ""}
                    </option>
                  ))}
              </select>
            </>
          )}
        </div>
      )}
      {open && data && !data.column && <p className="text-xs text-slate-500">{data.note}</p>}
      {open && data?.column && data.scatter && (
        <>
          {data.note && (
            <p className="rounded-md bg-slate-50 px-2 py-1.5 text-[11px] text-slate-600" data-testid="qc-states-note">
              {data.note}
            </p>
          )}
          <Panel
            title={
              twoD
                ? `RP share vs % mito${focus ? `: ${focus}` : ` by ${data.column}`}`
                : `RP share by ${data.column}${focus ? `: ${focus}` : ""}`
            }
            className="h-[360px]"
          >
            <Plot
              data={traces}
              layout={{
                autosize: true,
                margin: { l: 54, r: 10, t: 28, b: twoD ? 40 : 70 },
                xaxis: twoD
                  ? { title: { text: data.scatter.x_label, font: { size: 10 } }, tickfont: { size: 9 } }
                  : focus
                    ? { showticklabels: false, zeroline: false }
                    : { tickvals: strip?.tickvals, ticktext: strip?.ticktext, tickfont: { size: 9 }, tickangle: -35 },
                yaxis: {
                  title: { text: twoD ? (data.scatter.y_label ?? "") : data.scatter.x_label, font: { size: 10 } },
                  tickfont: { size: 9 },
                },
                showlegend: true,
                legend: { font: { size: 9 }, itemsizing: "constant" },
              }}
              config={plotConfig(`qc_states_${data.column}`)}
              style={FILL_STYLE}
              useResizeHandler
            />
          </Panel>
          <Caption
            n={{ value: data.scatter.n_shown, unit: "cells shown", of: fittedCells }}
            testId="qc-states-caption"
          >
            {twoD
              ? "Each point is a cell: its ribosomal-protein share of non-mito counts against its % mito; "
              : "Each point is a cell: its ribosomal-protein share of all counts (no % mito: the fit is 1-D on this share); "}
            up to {data.scatter.max_per_group.toLocaleString()} cells per group, sampled with a fixed seed.
            {focus ? " Red: the group's flagged state." : twoD ? " Crosses: each group's flagged state." : " Red: the flagged state."}
          </Caption>
          <Caption>
            {summaryText(data)}
            {columnReason(data.reason) ? ` Grouped by ${columnReason(data.reason)}.` : ""}
          </Caption>
          <div className="overflow-x-auto">
            <table className="w-full text-[11px] text-slate-600" data-testid="qc-states-table">
              <thead className="border-b border-slate-200 text-slate-500">
                <tr>
                  <Th tip={HEAD_TIPS.group}>{data.column}</Th>
                  <Th tip={HEAD_TIPS.n} right>Cells</Th>
                  <Th tip={HEAD_TIPS.states} right>States</Th>
                  <Th tip={HEAD_TIPS.bic} right>ΔBIC</Th>
                  <Th tip={HEAD_TIPS.d} right>D</Th>
                  <Th tip={HEAD_TIPS.share} right>Flagged</Th>
                  <Th tip={HEAD_TIPS.rp} right>RP %</Th>
                  {data.has_mt && <Th tip={HEAD_TIPS.mt} right>Mito %</Th>}
                  {data.condition &&
                    (levels.asRange ? (
                      <Th
                        tip={`Lowest and highest flagged share across the levels of ${data.condition.column} (levels with 20+ of the group's cells).`}
                      >
                        By {data.condition.column}
                      </Th>
                    ) : (
                      levels.levels.map((lv) => (
                        <Th
                          key={lv}
                          right
                          tip={`Flagged share among the group's cells with ${data.condition!.column} = ${lv} (and how many cells that is). The share moving with the ${data.condition!.kind} is what a fixed filter turns into a composition shift.`}
                        >
                          {lv}
                        </Th>
                      ))
                    ))}
                </tr>
              </thead>
              <tbody>
                {data.groups.map((g) => (
                  <tr
                    key={g.name}
                    className={`border-b border-slate-100 ${g.n_states === 2 ? "bg-red-50/60" : ""}`}
                    title={g.reason}
                  >
                    <td className="px-1.5 py-1 font-medium text-slate-700">{g.name}</td>
                    <td className={NUM}>{g.n_cells.toLocaleString()}</td>
                    <td className={NUM}>{g.fitted ? g.n_states : "not fitted"}</td>
                    <td className={NUM}>{g.delta_bic == null ? "—" : g.delta_bic.toFixed(0)}</td>
                    <td className={NUM}>{g.separation == null ? "—" : g.separation.toFixed(1)}</td>
                    <td className={NUM}>{g.n_states === 2 ? pctText(g.flagged_share) : "—"}</td>
                    <td className={NUM}>{meanPair(g, "mean_rp_share")}</td>
                    {data.has_mt && <td className={NUM}>{meanPair(g, "mean_pct_mt")}</td>}
                    {data.condition &&
                      (levels.asRange ? (
                        <td className="whitespace-nowrap px-1.5 py-1">{g.n_states === 2 ? (shareRange(g) ?? "—") : "—"}</td>
                      ) : (
                        levels.levels.map((lv) => {
                          const c = g.per_condition[lv];
                          return (
                            <td key={lv} className={NUM} title={c ? `${c.n.toLocaleString()} cells` : "no cells"}>
                              {c?.share != null ? pctText(c.share) : "—"}
                            </td>
                          );
                        })
                      ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Caption>
            Hover a row for the rule that decided it. Two states need ΔBIC over {data.rules?.min_delta_bic ?? 10}, D of{" "}
            {data.rules?.min_separation ?? 2} or more and at least {pctText(data.rules?.min_weight ?? 0.05)} of cells in
            the smaller state.
            {data.condition
              ? ` Shares per ${data.condition.kind} use ${data.condition.column}.`
              : " No condition or sample column: no per-condition shares."}
          </Caption>
          <div className="flex flex-wrap items-center gap-2 border-t border-slate-100 pt-2 text-[11px]">
            <label htmlFor="qc-states-name" className="text-slate-600">
              Label column
            </label>
            <input
              id="qc-states-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-28 rounded border border-slate-200 px-1.5 py-0.5 text-xs"
              title="Name of the obs column to write (letters, digits, _ . -; a letter first)."
            />
            <button
              type="button"
              onClick={save}
              disabled={!canSave || saving || !validStateName(name)}
              title={
                canSave
                  ? "Write each cell's state (main, or the group's flagged state) as a categorical column, to colour by, compare or exclude."
                  : "No group has two states: the label would be 'main' for every cell."
              }
              className="inline-flex items-center gap-1 rounded bg-slate-700 px-2 py-0.5 font-medium text-white hover:bg-slate-800 disabled:opacity-40"
            >
              {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Tag className="h-3 w-3" />}
              Save as label
            </button>
            {saved && (
              <span className="text-slate-500" data-testid="qc-states-saved">
                Saved {saved.name}:{" "}
                {Object.entries(saved.counts)
                  .map(([k, v]) => `${v.toLocaleString()} ${k}`)
                  .join(", ")}
                . Colour the scatter by it in Unified View.
              </span>
            )}
          </div>
        </>
      )}
    </div>
  );
}
