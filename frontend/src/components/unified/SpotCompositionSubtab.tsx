import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import {
  checkSpotReference,
  getSpotDeconvolution,
  getSpotShare,
  runSpotDeconvolution,
  spotsBelow,
  type ShareKind,
  type SpotInfo,
  type SpotMethod,
  type SpotReferenceCheck,
  type SpotSummary,
} from "@/api/spotDeconvolution";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { unitWord } from "@/lib/spatial";
import { SpotCompositionCheck } from "@/components/unified/SpotCompositionCheck";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { errorMessage } from "@/lib/errorMessage";
import { aboutGb, referenceChoices } from "@/lib/labelTransfer";
import {
  fmtShare,
  heatmapLayout,
  shareFill,
  shareLabel,
  shareText,
  binCaveat,
  SPOT_METHOD_LABELS,
  toOverlayValues,
  typeRows,
} from "@/lib/spotComposition";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

/** The types listed before "show all". */
const TYPE_LIMIT = 15;

/**
 * Spot composition (C047): each Visium spot split into the cell types of another
 * library dataset (backend core/spot_deconvolution.py, an RCTD-style fit). A type's
 * share colours the tissue on a fixed 0-1 scale, so types compare.
 */
export function SpotCompositionSubtab() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const datasets = useDatasetStore((s) => s.availableDatasets);
  const refresh = useDatasetRefresh();
  const overlay = useUnifiedViewStore((s) => s.scatterOverlay);
  const [info, setInfo] = useState<SpotInfo | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [refId, setRefId] = useState("");
  const [column, setColumn] = useState("");
  const [method, setMethod] = useState<SpotMethod>("rctd");
  // The count floor: spots with fewer counts are not fitted (the unit's default from the info).
  const [minUmi, setMinUmi] = useState<number | null>(null);
  const [check, setCheck] = useState<SpotReferenceCheck | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SpotSummary | null>(null);
  const [kind, setKind] = useState<ShareKind>("rna");
  const [showAll, setShowAll] = useState(false);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!datasetId) return;
    setInfo(null);
    setResult(null);
    setLoadError(null);
    setRefId("");
    let live = true;
    getSpotDeconvolution(datasetId)
      .then((i) => {
        if (!live) return;
        setInfo(i);
        setResult(i.saved);
        setMinUmi(i.saved?.min_umi ?? i.min_umi_default ?? null);
        if (i.saved) {
          setRefId(i.saved.reference.id);
          setColumn(i.saved.celltype_column);
          setMethod(i.saved.method);
        }
      })
      .catch((e) => live && setLoadError(errorMessage(e)));
    return () => {
      live = false;
      abort.current?.abort();
    };
  }, [datasetId]);

  // The reference check comes first: columns, types under the minimum, species, genes, estimate.
  useEffect(() => {
    setCheck(null);
    if (!datasetId || !refId) return;
    let live = true;
    checkSpotReference(datasetId, refId, column || null, method)
      .then((c) => {
        if (!live) return;
        setCheck(c);
        if (!column && c.celltype_column) setColumn(c.celltype_column);
      })
      .catch((e) => live && setCheck({ ok: false, reason: errorMessage(e) }));
    return () => {
      live = false;
    };
  }, [datasetId, refId, column, method]);

  const rows = useMemo(() => (result ? typeRows(result, kind) : []), [result, kind]);
  const heat = useMemo(() => (result ? heatmapLayout(result, kind) : null), [result, kind]);
  const active = overlay?.type === "score" && overlay.spotShare ? overlay.spotShare : null;

  if (!dataset || !datasetId) return <div className="text-xs text-slate-400">No dataset loaded.</div>;
  if (loadError) return <p className="text-xs text-red-600">{loadError}</p>;
  if (!info) return <div className="text-xs text-slate-400">Loading...</div>;

  const choices = referenceChoices(datasets, datasetId);

  const showShare = async (type: string, k: ShareKind = kind) => {
    try {
      const res = await getSpotShare(datasetId, type, k);
      const store = useUnifiedViewStore.getState();
      const name = shareLabel(type, k);
      store.setOverlayValues(toOverlayValues(res.values));
      store.setOverlayLabel(name);
      store.setScatterOverlay({ type: "score", name, spotShare: { type, kind: k } });
      store.setOverlayDatasetId(datasetId);
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const changeKind = (k: ShareKind) => {
    setKind(k);
    if (active) void showShare(active.type, k);
  };

  const colourByTop = () => {
    if (!result) return;
    useUnifiedViewStore.getState().clearOverlayState();
    useSettingsStore.getState().setColorBy(result.columns.top);
  };

  const run = async () => {
    setRunning(true);
    setError(null);
    setProgress("Starting");
    abort.current = new AbortController();
    try {
      const r = await runSpotDeconvolution(
        datasetId,
        { reference_id: refId, celltype_column: column, method, ...(minUmi != null ? { min_umi: minUmi } : {}) },
        {
          signal: abort.current.signal,
          onProgress: (m, done, total) => setProgress(done != null && total ? `${m} (${done}/${total})` : m),
        },
      );
      setResult(r);
      await refresh(datasetId, r.columns.top);
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) setError(errorMessage(e));
    } finally {
      setRunning(false);
      setProgress("");
    }
  };

  const small = check?.types?.filter((t) => t.small) ?? [];
  const nTypes = (check?.types?.length ?? 0) - small.length;
  const shownRows = showAll ? rows : rows.slice(0, TYPE_LIMIT);
  const spatial = dataset?.spatial;
  const unit = unitWord(spatial, false);
  const units = unitWord(spatial);
  const bins = spatial?.unit === "bin";
  const [floorLo, floorHi] = info.min_umi_range ?? [1, 10000];
  const floorOk = minUmi != null && Number.isInteger(minUmi) && minUmi >= floorLo && minUmi <= floorHi;
  const below = floorOk ? spotsBelow(info.counts, minUmi) : null;

  return (
    <div className="space-y-3 text-xs" data-testid="spot-composition">
      <div className="flex items-center gap-1.5 font-medium text-slate-700">
        Spot composition
        <InfoTip label="About spot composition" width={320}>
          {bins ? "A Visium HD bin is a square of the capture grid, which can hold parts of several cells." : "Each Visium spot holds several cells."}{" "}
          This estimates which cell types make up each {unit}, from the profiles of an annotated single-cell dataset in
          your library (the reference): a fit of the {unit}&apos;s counts as a mix of those profiles, as RCTD does.
        </InfoTip>
      </div>
      {!info.available ? (
        <p className="text-slate-500">{info.reason}</p>
      ) : (
        <div className="space-y-2 rounded border border-slate-200 p-2">
          <div className="flex flex-wrap items-center gap-2">
            <label className="flex items-center gap-1">
              Reference
              <select
                className="rounded border px-1 py-0.5"
                value={refId}
                disabled={running}
                onChange={(e) => {
                  setRefId(e.target.value);
                  setColumn("");
                }}
                aria-label="Reference dataset"
              >
                <option value="">choose a dataset</option>
                {choices.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </label>
            {check?.label_columns && check.label_columns.length > 0 && (
              <label className="flex items-center gap-1">
                Cell types
                <select
                  className="rounded border px-1 py-0.5"
                  value={column}
                  disabled={running}
                  onChange={(e) => setColumn(e.target.value)}
                  aria-label="Reference cell-type column"
                >
                  {check.label_columns.map((c) => (
                    <option key={c.column} value={c.column}>
                      {c.column} ({c.n_labels})
                    </option>
                  ))}
                </select>
              </label>
            )}
            <label className="flex items-center gap-1">
              Method
              <select
                className="rounded border px-1 py-0.5"
                value={method}
                disabled={running}
                onChange={(e) => setMethod(e.target.value as SpotMethod)}
                aria-label="Fitting method"
              >
                {info.methods.map((m) => (
                  <option key={m} value={m}>
                    {SPOT_METHOD_LABELS[m]}
                  </option>
                ))}
              </select>
              <InfoTip label="About the methods" width={340}>
                RCTD-style: each spot&apos;s counts are modelled as Poisson with log-normal noise around a mix of the
                reference&apos;s profiles, after a correction for the platform (how the spots measure genes differently
                from the reference). Among the best methods in independent benchmarks. NNLS: a plain least-squares fit of
                the same profiles; quicker, and a useful baseline.
              </InfoTip>
            </label>
          </div>
          {check && !check.ok && <p className="text-amber-700">{check.reason}</p>}
          {check?.ok && (
            <div className="space-y-0.5 text-slate-500" data-testid="spot-reference-check">
              <p>
                {check.n_cells?.toLocaleString()} reference cells, {nTypes} types with {info.min_type_cells}+ cells;{" "}
                {check.shared_genes_by_name?.toLocaleString()} genes shared by name; reference {check.species}, this
                dataset {check.dataset_species}.
                {check.estimate &&
                  ` Needs ${aboutGb(check.estimate.peak_bytes)}, ${check.estimate.note} (${check.estimate.seconds_range[0]}-${check.estimate.seconds_range[1]} s).`}
              </p>
              {check.estimate?.large_reference && (
                <p data-testid="spot-large-reference">
                  The reference is a large dataset: its counts are read from disk in blocks of cells, one pass (two for
                  the self-check), never all at once.
                </p>
              )}
              {small.length > 0 && (
                <p className="text-amber-700">
                  Left out (fewer than {info.min_type_cells} cells): {small.map((t) => `${t.type} (${t.n_cells})`).join(", ")}.
                </p>
              )}
              {check.shared_genes_by_name != null &&
                check.min_shared_genes != null &&
                check.shared_genes_by_name < check.min_shared_genes && (
                  <p className="text-amber-700">
                    Few genes match by name; Ensembl ids are matched to symbols when a gene-id column exists.
                  </p>
                )}
            </div>
          )}
          <div className="flex flex-wrap items-center gap-2">
            <button
              className="rounded bg-primary px-3 py-1 text-white disabled:opacity-50"
              disabled={running || !refId || !column || !check?.ok || !floorOk}
              onClick={run}
            >
              {running ? <Loader2 className="inline h-3 w-3 animate-spin" /> : null} Fit {units}
            </button>
            <label className="flex items-center gap-1">
              Count floor
              <input
                type="number"
                className="w-20 rounded border px-1 py-0.5"
                value={minUmi ?? ""}
                min={floorLo}
                max={floorHi}
                step={1}
                disabled={running}
                aria-label="Count floor"
                onChange={(e) => setMinUmi(e.target.value === "" ? null : Number(e.target.value))}
              />
              <InfoTip label="About the count floor" width={320}>
                A {unit} with fewer counts than this is not fitted (&quot;low counts&quot;). RCTD&apos;s default for a spot is{" "}
                {info.min_spot_umi}; a smaller Visium HD bin holds fewer counts, so its default scales with the bin&apos;s area
                (25 at 8 um): on a mouse brain, 8 um bins of 20-100 counts agreed with their 16 um parent bin as bins of 100+
                did.
              </InfoTip>
            </label>
            {below && (
              <Caption testId="spot-floor-caption">
                {below.atLeast ? "At least " : ""}
                {below.below.toLocaleString()} of {info.counts!.n.toLocaleString()} {units} (
                {((100 * below.below) / Math.max(1, info.counts!.n)).toFixed(1)}%) are below it.
              </Caption>
            )}
            {!floorOk && minUmi != null && (
              <Caption testId="spot-floor-caption">
                The count floor is a whole number from {floorLo} to {floorHi.toLocaleString()}.
              </Caption>
            )}
            {running && (
              <>
                <span className="text-slate-600" data-testid="spot-progress">
                  {progress}
                </span>
                <button className="text-slate-500 hover:underline" onClick={() => abort.current?.abort()}>
                  Cancel
                </button>
              </>
            )}
          </div>
          {error && <p className="text-red-600">{error}</p>}
        </div>
      )}

      {result && !running && (
        <div className="space-y-2" data-testid="spot-composition-result">
          <p className="text-slate-700">
            {result.types.length} types from <b>{result.reference.name}</b> ({result.celltype_column}),{" "}
            {result.method_name}; {result.n_fitted.toLocaleString()} of {result.n_spots.toLocaleString()} {units} fitted on{" "}
            {result.n_genes.toLocaleString()} genes
            {result.s_e != null && `; spot noise ${result.s_e.toFixed(1)}`}
            {result.libraries.some((l) => l.s_g != null) &&
              `; platform effect ${result.libraries
                .filter((l) => l.s_g != null)
                .map((l) => (result.libraries.length > 1 ? `${l.id} ${l.s_g!.toFixed(2)}` : l.s_g!.toFixed(2)))
                .join(", ")}`}
            .
            <InfoTip label="About the numbers" width={320}>
              Spot noise: the spread (log scale) of a gene&apos;s count around the mix in one spot. Platform effect: the
              spread of the genes&apos; differences between the reference and the spots, corrected for before the fit
              (above about 1, the two measure genes very differently).
            </InfoTip>
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-slate-500">Shares of</span>
            {(["rna", "cell"] as const).map((k) => (
              <label key={k} className="flex items-center gap-1">
                <input type="radio" name="spot-share-kind" checked={kind === k} onChange={() => changeKind(k)} />
                {k === "rna" ? "RNA" : "cells (estimated)"}
              </label>
            ))}
            <InfoTip label="RNA or cell shares" width={300}>
              The fit estimates each type&apos;s share of a spot&apos;s RNA. Cell shares divide that by each type&apos;s
              RNA per cell in the reference, so they assume cell sizes carry over from the reference to the tissue.
            </InfoTip>
            <button className="ml-auto rounded border px-2 py-0.5 hover:bg-slate-50" onClick={colourByTop}>
              Colour by dominant type
            </button>
          </div>
          <table className="w-full">
            <thead>
              <tr className="text-left text-slate-500">
                <th title="Reference cell type; click to colour the tissue by its share">Type</th>
                <th className="text-right" title="Mean share over the fitted spots">Mean</th>
                <th className="text-right" title={`Largest share in one ${unit}`}>Max</th>
                <th className="text-right" title={`${units} where this type has the largest RNA share`}>Top in</th>
              </tr>
            </thead>
            <tbody>
              {shownRows.map((r) => (
                <tr key={r.type} className={active?.type === r.type ? "bg-slate-100" : ""}>
                  <td>
                    <button
                      className="text-left text-primary hover:underline"
                      onClick={() => showShare(r.type)}
                      title={`Colour the plot by ${r.type}'s share (fixed 0-100% scale)`}
                    >
                      {r.type}
                    </button>
                  </td>
                  <td className="text-right">{fmtShare(r.mean)}</td>
                  <td className="text-right">{fmtShare(r.max)}</td>
                  <td className="text-right">{r.nTop.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Caption n={{ value: result.n_fitted, unit: units }}>
            {kind === "rna" ? `Share of each ${unit}'s RNA` : `Estimated share of each ${unit}'s cells`} per reference type.
            {rows.length > TYPE_LIMIT && !showAll && ` Top ${TYPE_LIMIT} of ${rows.length} by mean share.`}
          </Caption>
          {rows.length > TYPE_LIMIT && (
            <button className="text-slate-500 hover:underline" onClick={() => setShowAll((v) => !v)}>
              {showAll ? "Show fewer" : `Show all ${rows.length}`}
            </button>
          )}

          {heat && heat.clusters.length > 0 && (
            <div className="overflow-x-auto" data-testid="spot-heatmap">
              <table className="border-collapse">
                <thead>
                  <tr>
                    <th className="pr-1 text-left font-normal text-slate-500" title={`Clusters of ${result.cluster_column}`}>
                      {result.cluster_column}
                    </th>
                    {heat.types.map((t) => (
                      <th key={t.type} className="max-w-[3rem] truncate px-0.5 font-normal text-slate-500" title={t.type}>
                        {t.type}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {heat.clusters.map((c) => (
                    <tr key={c}>
                      <td className="pr-1 text-slate-600">{c}</td>
                      {heat.types.map((t) => {
                        const v = heat.values[c]?.[t.index];
                        return (
                          <td
                            key={t.type}
                            className="h-5 w-8 text-center text-[10px] text-slate-700"
                            style={{ backgroundColor: shareFill(v), color: shareText(v) || undefined }}
                            title={`${c}, ${t.type}: ${fmtShare(v)}`}
                          >
                            {v != null ? Math.round(v * 100) : ""}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              <Caption n={{ value: heat.clusters.length, unit: "clusters" }}>
                Mean {kind === "rna" ? "RNA" : "cell"} share (%) of each type in each cluster&apos;s spots
                {heat.types.length < result.types.length ? `; the ${heat.types.length} largest types` : ""}.
              </Caption>
            </div>
          )}

          {result.notes.length > 0 && (
            <ul className="list-disc pl-4 text-slate-500">
              {result.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          )}
          <div className="rounded bg-amber-50 p-2 text-amber-800" data-testid="spot-caveats">
            <p>
              Shares are estimates of each {unit}&apos;s RNA (or cells), not counts. A cell type missing from the
              reference is given to its nearest type, so its {units} look like that type: use a reference that covers the
              tissue. The platform correction
              assumes one effect for all types, so a single-nucleus reference or a probe-based (FFPE) section fits
              least well. The reference must be the same species.
            </p>
            {bins && <p className="mt-1" data-testid="spot-bin-caveat">{binCaveat(spatial?.bin_size_um)}</p>}
          </div>
        </div>
      )}

      {info.available && refId && column && check?.ok && (
        <SpotCompositionCheck datasetId={datasetId} referenceId={refId} celltypeColumn={column} disabled={running} />
      )}

      <p className="text-[11px] text-slate-400">
        Method after RCTD (Cable et al. 2022, Nat Biotech 40:517), written from the paper; the self-check follows the
        Spotless benchmark&apos;s synthetic spots (Sang-aram et al. 2024, eLife 12:RP88431).
      </p>
    </div>
  );
}
