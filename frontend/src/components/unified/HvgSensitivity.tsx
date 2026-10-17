import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Dna, Loader2, Palette, X } from "lucide-react";
import {
  getHvgSensitivityInfo,
  keepHvgSetting,
  runHvgSensitivity,
  type HvgFlavor,
  type HvgRow,
  type HvgSensitivityInfo,
  type HvgSetting,
  type HvgStart,
} from "@/api/clustering";
import { fetchEmbeddingBinary } from "@/api/embeddings";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { decodeArrowBuffer } from "@/lib/arrow";
import { prettyObsLabel } from "@/lib/formatting";
import {
  clusterStability,
  describeMatch,
  formatDuration,
  jaccardTone,
  otherSettings,
  perCellValues,
  secondsLeft,
  shortSettingLabel,
  verdictLabel,
  type ClusterStability,
} from "@/lib/hvgSensitivity";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

/**
 * "Variable genes" in the Unified View toolbar: a floating panel that re-clusters
 * the dataset under other variable-gene settings without changing it, compares
 * each with the current clustering, and keeps a chosen one as a column.
 */
export function HvgSensitivityControl() {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const [open, setOpen] = useState(false);
  if (!datasetId) return null;
  return (
    <>
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        title="Would the clusters survive a different choice of variable genes?"
        className={`flex items-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors ${
          open ? "bg-blue-50 text-blue-700 ring-1 ring-blue-300" : "text-slate-600 hover:bg-slate-100"
        }`}
      >
        <Dna className="h-3.5 w-3.5" />
        Variable genes
      </button>
      {open &&
        createPortal(
          <HvgSensitivityPanel key={datasetId} datasetId={datasetId} onClose={() => setOpen(false)} />,
          document.body,
        )}
    </>
  );
}

const METHOD_LABEL = { leiden: "Leiden", louvain: "Louvain" } as const;

interface Choice {
  setting: HvgSetting;
  checked: boolean;
}

function settingOf(flavor: HvgFlavor, n: number, label: string): HvgSetting {
  return { flavor, n_top_genes: n, key: `${flavor}_${n}`, label: `${label}, ${n.toLocaleString("en-US")} genes` };
}

export function HvgSensitivityPanel({ datasetId, onClose }: { datasetId: string; onClose: () => void }) {
  const refresh = useDatasetRefresh();
  const [info, setInfo] = useState<HvgSensitivityInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [choices, setChoices] = useState<Choice[]>([]);
  const [addFlavor, setAddFlavor] = useState<HvgFlavor>("seurat");
  const [addGenes, setAddGenes] = useState("5000");
  const [addError, setAddError] = useState<string | null>(null);
  const [start, setStart] = useState<HvgStart | null>(null);
  const [rows, setRows] = useState<HvgRow[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [keeping, setKeeping] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    getHvgSensitivityInfo(datasetId)
      .then((i) => {
        if (cancelled) return;
        setInfo(i);
        setChoices((i.default_settings ?? []).map((setting) => ({ setting, checked: true })));
      })
      .catch((e) => {
        if (!cancelled) setInfoError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
      abortRef.current?.abort(); // closing the panel stops a running check
    };
  }, [datasetId]);

  const selected = choices.filter((c) => c.checked).map((c) => c.setting);
  const tooMany = info ? selected.length > info.max_settings : false;
  const currentKey = info?.current?.key;

  const addSetting = () => {
    if (!info) return;
    const n = Number(addGenes);
    const flavor = info.flavors?.find((f) => f.flavor === addFlavor);
    if (!flavor || flavor.unavailable) return setAddError(flavor?.unavailable ?? "Pick a flavour.");
    if (!Number.isInteger(n) || n < info.min_genes || n > Math.min(info.max_genes, info.n_genes)) {
      return setAddError(
        `Variable genes must be a whole number between ${info.min_genes.toLocaleString()} and ${Math.min(info.max_genes, info.n_genes).toLocaleString()}.`,
      );
    }
    const setting = settingOf(addFlavor, n, flavor.label);
    setAddError(null);
    setChoices((prev) =>
      prev.some((c) => c.setting.key === setting.key)
        ? prev.map((c) => (c.setting.key === setting.key ? { ...c, checked: true } : c))
        : [...prev, { setting, checked: true }],
    );
  };

  const run = useCallback(async () => {
    if (!info || running || selected.length === 0 || tooMany) return;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setRunning(true);
    setError(null);
    setRows([]);
    setStart(null);
    try {
      await runHvgSensitivity(
        datasetId,
        { settings: selected.map((s) => ({ flavor: s.flavor, n_top_genes: s.n_top_genes })) },
        { onStart: setStart, onResult: (r) => setRows((prev) => [...prev, r]), signal: ctrl.signal },
      );
    } catch (e) {
      if (!ctrl.signal.aborted) setError(e instanceof Error ? e.message : "The check failed.");
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null;
      setRunning(false);
    }
  }, [datasetId, info, running, selected, tooMany]);

  const reference = start?.reference ?? info?.recorded?.reference ?? "";
  const stability = useMemo(() => clusterStability(rows), [rows]);
  const others = useMemo(() => otherSettings(rows), [rows]);
  const control = rows.find((r) => r.is_current);

  const showColumn = (column: string) => useSettingsStore.getState().setColorBy(column);

  const keep = async (row: HvgRow) => {
    setKeeping(row.key);
    setError(null);
    try {
      const res = await keepHvgSetting(datasetId, row);
      if (res.created) await refresh(datasetId);
      setRows((prev) => prev.map((r) => (r.column === res.column ? { ...r, exists: true } : r)));
      showColumn(res.column);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not keep that setting.");
    } finally {
      setKeeping(null);
    }
  };

  // Colour each cell by its current cluster's Jaccard: averaged over the other
  // settings, or under one of them.
  const colour = async (values: Map<string, number>, label: string) => {
    setError(null);
    const dataset = useDatasetStore.getState().currentDataset;
    const categories = dataset?.obs_columns.find((c) => c.name === reference)?.values;
    if (!dataset || dataset.id !== datasetId || !categories) {
      setError(`Can't colour by ${prettyObsLabel(reference)}: it has too many groups to list.`);
      return;
    }
    try {
      const buffer = await fetchEmbeddingBinary(datasetId, useSettingsStore.getState().embedding, reference);
      const codes = decodeArrowBuffer(buffer)[reference];
      if (!codes) throw new Error("no cluster labels came back");
      const store = useUnifiedViewStore.getState();
      store.setOverlayValues(perCellValues(codes, categories, values));
      store.setOverlayLabel(label);
      store.setScatterOverlay({ type: "score", name: label });
      store.setOverlayDatasetId(datasetId);
    } catch (e) {
      setError(`Could not colour the plot: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const colourByMean = () =>
    void colour(
      new Map(stability.filter((s) => s.mean != null).map((s) => [s.cluster, s.mean!])),
      "Cluster stability (mean Jaccard)",
    );
  const colourByRow = (row: HvgRow) =>
    void colour(new Map(row.clusters.map((c) => [c.cluster, c.jaccard])), `Jaccard under ${row.label}`);

  // A highlight selects the group's cells once the plot is coloured by its column.
  const highlight = (cluster: string) => {
    if (useSettingsStore.getState().colorBy !== reference) showColumn(reference);
    useSelectionStore.getState().setHighlight({ column: reference, value: cluster });
  };

  const total = start?.settings.length ?? selected.length;
  const left = running
    ? secondsLeft(rows.map((r) => r.seconds), total, info?.seconds_per_setting ?? null)
    : null;
  const rec = start?.recorded ?? info?.recorded;
  const estimate = info?.seconds_per_setting != null ? info.seconds_per_setting * selected.length : null;

  return (
    <div
      role="dialog"
      aria-label="Variable-gene sensitivity check"
      className="fixed right-4 top-20 z-50 flex max-h-[calc(100vh-6rem)] w-[40rem] max-w-[calc(100vw-2rem)] flex-col gap-2 overflow-y-auto rounded-xl border border-slate-200 bg-white p-4 text-xs shadow-xl"
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-sm font-semibold text-slate-800">Would the clusters survive other variable genes?</div>
          <p className="mt-0.5 text-[11px] text-slate-500">
            Each setting picks its own variable genes, then re-runs PCA, the neighbour graph and the clustering
            {rec
              ? ` (${METHOD_LABEL[rec.method]} at ${rec.resolution}, ${rec.n_comps} PCs, ${rec.n_neighbors} neighbours${rec.batch_key ? `, Harmony on ${rec.batch_key}` : ""})`
              : ""}
            {reference ? <> and compares it with <span className="font-medium">{prettyObsLabel(reference)}</span></> : null}.
            Nothing is saved until you keep one.
          </p>
        </div>
        <button onClick={onClose} aria-label="Close" className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600">
          <X className="h-4 w-4" />
        </button>
      </div>

      {infoError && <p role="alert" className="rounded bg-red-50 px-2 py-1.5 text-red-700">{infoError}</p>}
      {info && !info.available && (
        <p role="alert" className="rounded bg-amber-50 px-2 py-1.5 text-amber-800">{info.reason}</p>
      )}
      {!info && !infoError && (
        <div className="flex items-center gap-1.5 text-slate-400">
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Checking the dataset…
        </div>
      )}
      {rec && rec.notes.length > 0 && (
        <ul className="list-disc pl-4 text-[11px] text-slate-500">
          {rec.notes.map((n) => <li key={n}>{n}</li>)}
        </ul>
      )}

      {info?.available && (
        <fieldset disabled={running} className="space-y-1.5">
          <legend className="text-[11px] font-medium text-slate-500">
            Settings (up to {info.max_settings})
          </legend>
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
            {choices.map((c) => (
              <label key={c.setting.key} className="flex items-center gap-1.5 text-slate-700">
                <input
                  type="checkbox"
                  checked={c.checked}
                  onChange={(e) =>
                    setChoices((prev) =>
                      prev.map((p) => (p.setting.key === c.setting.key ? { ...p, checked: e.target.checked } : p)),
                    )
                  }
                />
                {c.setting.label}
                {c.setting.key === currentKey && <span className="text-[10px] text-slate-400">(current: control)</span>}
              </label>
            ))}
          </div>
          <div className="flex items-center gap-1.5">
            <select
              aria-label="Flavour to add"
              value={addFlavor}
              onChange={(e) => setAddFlavor(e.target.value as HvgFlavor)}
              className="rounded border border-slate-300 px-1 py-0.5 text-xs text-slate-700"
            >
              {info.flavors?.map((f) => (
                <option key={f.flavor} value={f.flavor} disabled={!!f.unavailable} title={f.unavailable ?? undefined}>
                  {f.label}
                  {f.unavailable ? " (unavailable)" : ""}
                </option>
              ))}
            </select>
            <input
              aria-label="Genes to add"
              value={addGenes}
              onChange={(e) => setAddGenes(e.target.value)}
              className="w-20 rounded border border-slate-300 px-1.5 py-0.5 text-xs text-slate-700"
            />
            <span className="text-slate-500">genes</span>
            <button
              type="button"
              onClick={addSetting}
              className="rounded border border-slate-300 px-2 py-0.5 text-slate-700 hover:bg-slate-50"
            >
              Add
            </button>
          </div>
          {addError && <p className="text-[11px] text-amber-700">{addError}</p>}
          {info.flavors?.some((f) => f.unavailable) && (
            <p className="text-[10px] text-slate-400">
              {info.flavors
                .filter((f) => f.unavailable)
                .map((f) => `${f.label}: ${f.unavailable}`)
                .join(" ")}
            </p>
          )}
        </fieldset>
      )}

      {info?.available && (
        <div className="flex items-center gap-2">
          {running ? (
            <button
              type="button"
              onClick={() => abortRef.current?.abort()}
              className="rounded border border-slate-300 px-3 py-1 text-xs text-slate-600 hover:bg-slate-50"
            >
              Stop
            </button>
          ) : (
            <button
              type="button"
              onClick={() => void run()}
              disabled={selected.length === 0 || tooMany}
              className="rounded bg-blue-500 px-3 py-1 text-xs font-medium text-white hover:bg-blue-600 disabled:opacity-50"
            >
              Run
            </button>
          )}
          {running ? (
            <span className="flex items-center gap-1.5 text-slate-500">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Setting {Math.min(rows.length + 1, total)} of {total}
              {left != null && left > 0 ? `, about ${formatDuration(left)} left` : ""}…
            </span>
          ) : (
            <span className="text-[11px] text-slate-500">
              {tooMany
                ? `Pick at most ${info.max_settings} settings.`
                : estimate != null && selected.length > 0
                  ? `About ${formatDuration(estimate)} for ${selected.length} setting${selected.length === 1 ? "" : "s"} (the first run after the server starts adds ~${formatDuration(info.first_run_seconds)}).`
                  : "Pick at least one setting."}
            </span>
          )}
        </div>
      )}

      {rows.length > 0 && (
        <table className="w-full border-collapse text-[11px]">
          <thead>
            <tr className="border-b border-slate-200 text-left text-slate-500">
              <th className="py-1 pr-2 font-medium">Setting</th>
              <th className="py-1 pr-2 font-medium" title="Variable genes shared with the current ones / this setting's own">
                Shared genes
              </th>
              <th className="py-1 pr-2 font-medium">Clusters</th>
              <th className="py-1 pr-2 font-medium" title="Adjusted Rand index with the current clustering: 1 means the same partition">
                ARI
              </th>
              <th className="py-1 pr-2 font-medium" title="Adjusted mutual information with the current clustering: 1 means the same partition">
                AMI
              </th>
              <th className="py-1" />
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key} className="border-b border-slate-100">
                <td className="py-1 pr-2 font-medium text-slate-700">
                  {r.label}
                  {r.is_current && <span className="ml-1 font-normal text-slate-400">(current, re-run)</span>}
                </td>
                <td className="py-1 pr-2 tabular-nums text-slate-600">
                  {r.shared_hvgs == null ? "—" : `${r.shared_hvgs.toLocaleString()} / ${r.n_hvgs.toLocaleString()}`}
                </td>
                <td className="py-1 pr-2 tabular-nums text-slate-700">{r.n_clusters}</td>
                <td className={`py-1 pr-2 tabular-nums ${r.ari >= 0.9 ? "text-slate-700" : "font-semibold text-amber-700"}`}>
                  {r.ari.toFixed(2)}
                </td>
                <td className="py-1 pr-2 tabular-nums text-slate-600">{r.ami.toFixed(2)}</td>
                <td className="py-1 text-right">
                  <div className="flex items-center justify-end gap-1">
                    <button
                      onClick={() => colourByRow(r)}
                      title={`Colour the plot by each cluster's Jaccard under ${r.label}`}
                      aria-label={`Colour by Jaccard under ${r.label}`}
                      className="rounded p-0.5 text-slate-500 hover:bg-slate-100"
                    >
                      <Palette className="h-3.5 w-3.5" />
                    </button>
                    {r.is_current ? (
                      <span className="px-1.5 text-slate-400">control</span>
                    ) : r.exists ? (
                      <button
                        onClick={() => showColumn(r.column)}
                        title={`Saved as ${r.column}: colour the plot by it`}
                        className="rounded px-1.5 py-0.5 text-blue-600 hover:bg-blue-50"
                      >
                        Show
                      </button>
                    ) : (
                      <button
                        onClick={() => void keep(r)}
                        disabled={keeping !== null}
                        title={`Save these clusters as ${r.column}`}
                        className="flex items-center gap-1 rounded border border-slate-300 px-1.5 py-0.5 text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                      >
                        {keeping === r.key && <Loader2 className="h-3 w-3 animate-spin" />}
                        Keep this one
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {control && control.ari < 0.99 && (
        <p className="rounded bg-amber-50 px-2 py-1.5 text-[11px] text-amber-800">
          Re-running the current setting gives ARI {control.ari.toFixed(2)}, not 1: the stored clusters can&rsquo;t be
          reproduced exactly (the data or the graph changed after clustering), so read the other rows against this one.
        </p>
      )}

      {others.length > 0 && stability.length > 0 && (
        <StabilityTable
          stability={stability}
          others={others}
          reference={reference}
          onHighlight={highlight}
          onColour={colourByMean}
        />
      )}

      {error && <p role="alert" className="rounded bg-red-50 px-2 py-1.5 text-red-700">{error}</p>}

      {rows.length > 0 && (
        <p className="text-[10px] leading-snug text-slate-400">
          A cluster is found again when its best match under a setting has a Jaccard index of 0.75 or more (Hennig
          2007); at 0.5 or less it has dissolved. If not found again, it splits (its cells go to clusters of their own),
          merges (it stays together inside a bigger cluster) or dissolves (both). &ldquo;Keep this one&rdquo; adds a
          clustering column; the dataset&rsquo;s active clustering stays as it is.
        </p>
      )}
    </div>
  );
}

const VERDICT_TONE = {
  robust: "text-emerald-700",
  depends: "text-amber-700",
  unstable: "text-red-700",
} as const;

/** Per current cluster: its Jaccard under each other setting, least stable first. */
function StabilityTable({
  stability,
  others,
  reference,
  onHighlight,
  onColour,
}: {
  stability: ClusterStability[];
  others: HvgRow[];
  reference: string;
  onHighlight: (cluster: string) => void;
  onColour: () => void;
}) {
  const counts = { robust: 0, depends: 0, unstable: 0 };
  for (const s of stability) if (s.verdict) counts[s.verdict] += 1;
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between gap-2">
        <div className="text-[11px] font-medium text-slate-600">
          Clusters of {prettyObsLabel(reference)}: {counts.robust} robust, {counts.depends} depend on the genes,{" "}
          {counts.unstable} unstable
        </div>
        <button
          onClick={onColour}
          className="flex items-center gap-1 rounded border border-slate-300 px-1.5 py-0.5 text-slate-700 hover:bg-slate-50"
        >
          <Palette className="h-3 w-3" /> Colour plot by stability
        </button>
      </div>
      <table className="w-full border-collapse text-[11px]">
        <thead>
          <tr className="border-b border-slate-200 text-left text-slate-500">
            <th className="py-1 pr-2 font-medium">Cluster</th>
            <th className="py-1 pr-2 text-right font-medium">Cells</th>
            {others.map((r) => (
              <th key={r.key} className="px-0.5 py-1 text-center font-medium" title={r.label}>
                {shortSettingLabel(r)}
              </th>
            ))}
            <th className="px-1 py-1 text-right font-medium" title="Mean best-match Jaccard over these settings">
              Mean
            </th>
            <th className="py-1 pl-1 font-medium">Verdict</th>
          </tr>
        </thead>
        <tbody>
          {stability.map((s) => (
            <tr
              key={s.cluster}
              onClick={() => onHighlight(s.cluster)}
              title="Highlight this cluster on the plot"
              className="cursor-pointer border-b border-slate-100 hover:bg-slate-50"
            >
              <td className="max-w-[9rem] truncate py-0.5 pr-2 font-medium text-slate-700">{s.cluster}</td>
              <td className="py-0.5 pr-2 text-right tabular-nums text-slate-500">{s.n_cells.toLocaleString()}</td>
              {s.matches.map((m, i) => (
                <td key={others[i]!.key} className="px-0.5 py-0.5">
                  <div
                    title={m ? `${others[i]!.label}. ${describeMatch(m)}` : undefined}
                    className={`rounded px-1 text-center tabular-nums ${jaccardTone(m?.jaccard ?? null)}`}
                  >
                    {m ? m.jaccard.toFixed(2) : "—"}
                    {m && m.status !== "stable" && <span className="ml-0.5 text-[9px]">{m.status[0]}</span>}
                  </div>
                </td>
              ))}
              <td className="px-1 py-0.5 text-right tabular-nums font-medium text-slate-700">
                {s.mean == null ? "—" : s.mean.toFixed(2)}
              </td>
              <td className={`py-0.5 pl-1 ${s.verdict ? VERDICT_TONE[s.verdict] : "text-slate-400"}`}>
                {verdictLabel(s)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-[10px] text-slate-400">
        Letters: s splits, m merges, d dissolves. Click a cluster to highlight it on the plot.
      </p>
    </div>
  );
}
