import { useEffect, useMemo, useState } from "react";
import { AlertCircle, Loader2 } from "lucide-react";
import type { ProteinInfo } from "@/api/types";
import { getDsbStatus, runDsb, type DsbStatus } from "@/api/dsb";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { dropletsInWindow, dsbNote, type ProteinNorm } from "@/lib/dsb";
import { plotConfig } from "@/lib/plotConfig";
import { InfoTip } from "@/components/common/InfoTip";
import { Caption } from "@/components/common/Caption";
import Plot from "@/components/plots/Plot";

const CLR_HELP =
  "CLR: each protein's count relative to the cell's other proteins (centred log-ratio). Needs no empty " +
  "droplets; the antibody background is not removed.";
const DSB_HELP =
  "DSB: each protein relative to its background in empty droplets (how many background SDs above them " +
  "a cell is), with each cell's technical noise taken out using the isotype controls. Around 0 is " +
  "background; clearly positive is staining (Mulè et al. 2022).";
const WINDOW_HELP =
  "The empty droplets used as background: droplets that are not cells, with RNA and protein UMI totals " +
  "inside these log10 ranges. The defaults are the dsb vignette's (protein 1.5 to 3, RNA up to 2.5).";
const NEG_HELP =
  "Without empty droplets the background is estimated from the cells (dsb's ModelNegativeADTnorm): the " +
  "lower of two groups of each protein's values. A protein every cell expresses can then sit near 0.";

/** The error's own sentence: the backend's detail, without "API error 400:". */
function message(err: unknown): string {
  const raw = err instanceof Error ? err.message.replace(/^API error \d+: /, "") : String(err);
  try {
    return String(JSON.parse(raw).detail ?? raw);
  } catch {
    return raw;
  }
}

interface Props {
  datasetId: string;
  protein: ProteinInfo;
  norm: ProteinNorm;
  onNormChange: (norm: ProteinNorm) => void;
}

/**
 * CLR | DSB choice for the Protein tab, with what the chosen values are, and the DSB
 * computation: its empty-droplet window (and the droplets it takes), isotype
 * controls, and the reason when there are no empty droplets.
 */
export function ProteinNormalisation({ datasetId, protein, norm, onNormChange }: Props) {
  const hasDsb = Boolean(protein.dsb_layer);
  const shown: ProteinNorm = hasDsb ? norm : "clr";
  const [open, setOpen] = useState(false);

  return (
    <div className="space-y-1.5 rounded-lg border border-slate-200 p-2">
      <fieldset className="flex flex-wrap items-center gap-3 text-xs text-slate-700">
        <legend className="sr-only">Protein normalisation</legend>
        <span className="text-[11px] font-medium text-slate-500">Normalisation</span>
        <label className="flex items-center gap-1">
          <input type="radio" name="protein-norm" checked={shown === "clr"} onChange={() => onNormChange("clr")} />
          CLR
          <InfoTip label="About CLR">{CLR_HELP}</InfoTip>
        </label>
        <label className={`flex items-center gap-1 ${hasDsb ? "" : "text-slate-400"}`}>
          <input
            type="radio"
            name="protein-norm"
            checked={shown === "dsb"}
            disabled={!hasDsb}
            onChange={() => onNormChange("dsb")}
          />
          DSB
          <InfoTip label="About DSB">{DSB_HELP}</InfoTip>
        </label>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="ml-auto rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-600 hover:border-primary hover:text-primary"
        >
          {hasDsb ? "Recompute DSB…" : "Compute DSB…"}
        </button>
      </fieldset>
      <p className="text-[11px] leading-snug text-slate-500" data-testid="protein-norm-note">
        {shown === "dsb" && protein.dsb
          ? dsbNote(protein.dsb)
          : hasDsb
            ? "CLR per cell (centred log-ratio). DSB values are computed: choose DSB to use them."
            : "CLR per cell (centred log-ratio). DSB is not computed yet."}
      </p>
      {open && (
        <DsbSettings
          datasetId={datasetId}
          onDone={() => {
            setOpen(false);
            onNormChange("dsb");
          }}
        />
      )}
    </div>
  );
}

function DsbSettings({ datasetId, onDone }: { datasetId: string; onDone: () => void }) {
  const refresh = useDatasetRefresh();
  const [status, setStatus] = useState<DsbStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rna, setRna] = useState<[number, number]>([0.3, 2.5]);
  const [prot, setProt] = useState<[number, number]>([1.5, 3]);
  const [isotypes, setIsotypes] = useState<string[]>([]);
  const [denoise, setDenoise] = useState(true);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getDsbStatus(datasetId)
      .then((s) => {
        if (cancelled) return;
        setStatus(s);
        const last = s.record;
        setRna(last?.rna_window ?? s.defaults?.rna_window ?? [0.3, 2.5]);
        setProt(last?.prot_window ?? s.defaults?.prot_window ?? [1.5, 3]);
        setIsotypes(last?.isotypes ?? s.isotypes ?? []);
      })
      .catch((e) => !cancelled && setError(message(e)));
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  const counts = useMemo(
    () => (status?.samples ?? []).map((s) => ({ sample: s.sample, n: dropletsInWindow(s.density, rna, prot), s })),
    [status, rna, prot],
  );

  if (error && !status) {
    return (
      <p className="flex items-center gap-1.5 text-xs text-amber-700">
        <AlertCircle className="h-3.5 w-3.5 flex-shrink-0" /> {error}
      </p>
    );
  }
  if (!status) {
    return (
      <div className="flex items-center gap-2 py-2 text-xs text-slate-400">
        <Loader2 className="h-4 w-4 animate-spin" /> Reading the empty droplets…
      </div>
    );
  }
  if (!status.available) {
    return <p className="text-xs text-amber-700">{status.reason}</p>;
  }

  const empty = Boolean(status.empty_droplets);
  const minDrops = status.defaults?.min_droplets ?? 50;
  const tooFew = empty && counts.some((c) => c.n < minDrops);
  const missing = counts.flatMap((c) => c.s.missing_antibodies);
  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      await runDsb(datasetId, {
        mode: empty ? "empty_droplets" : "model_negative",
        rna_window: rna,
        prot_window: prot,
        isotypes,
        denoise,
      });
      await refresh(datasetId);
      onDone();
    } catch (e) {
      setError(message(e));
    } finally {
      setRunning(false);
    }
  };

  const bound = (label: string, value: number, max: number, set: (v: number) => void) => (
    <label className="flex items-center gap-1">
      <span className="text-slate-500">{label}</span>
      <input
        type="number"
        step={0.1}
        min={0}
        max={max}
        value={value}
        onChange={(e) => set(Math.round(Number(e.target.value) * 10) / 10)}
        className="w-14 rounded border border-slate-300 px-1 py-0.5 text-xs"
      />
    </label>
  );

  return (
    <div className="space-y-2 border-t border-slate-100 pt-2 text-xs text-slate-700" data-testid="dsb-settings">
      {empty ? (
        <>
          <div className="flex items-center gap-1 text-[11px] font-medium text-slate-500">
            Empty droplets (log10 UMIs) <InfoTip label="About the empty-droplet window">{WINDOW_HELP}</InfoTip>
          </div>
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            {bound("RNA from", rna[0], status.defaults?.rna_max ?? 4, (v) => setRna([v, rna[1]]))}
            {bound("to", rna[1], status.defaults?.rna_max ?? 4, (v) => setRna([rna[0], v]))}
            {bound("Protein from", prot[0], status.defaults?.prot_max ?? 5, (v) => setProt([v, prot[1]]))}
            {bound("to", prot[1], status.defaults?.prot_max ?? 5, (v) => setProt([prot[0], v]))}
          </div>
          {counts[0] && <WindowPlot density={counts[0].s.density} rna={rna} prot={prot} sample={counts[0].sample} />}
          <ul className="space-y-0.5" data-testid="dsb-droplets">
            {counts.map((c) => (
              <li key={c.sample} className={c.n < minDrops ? "text-amber-700" : "text-slate-600"}>
                {c.sample}: {c.n.toLocaleString()} empty droplets in the window ({c.s.n_cells.toLocaleString()} cells)
                {c.n < minDrops ? `: at least ${minDrops} are needed` : ""}
              </li>
            ))}
          </ul>
          {missing.length > 0 && (
            <p className="text-amber-700">The raw matrix lacks {missing.join(", ")}: attach the raw matrix of the same run.</p>
          )}
        </>
      ) : (
        <p className="text-slate-600" data-testid="dsb-no-empty">
          {status.empty_droplets_reason} <InfoTip label="About DSB without empty droplets">{NEG_HELP}</InfoTip>
        </p>
      )}
      <fieldset className="space-y-0.5">
        <legend className="text-[11px] font-medium text-slate-500">Isotype controls (per-cell noise)</legend>
        {(status.proteins ?? []).length === 0 && <p className="text-slate-400">No proteins.</p>}
        <div className="flex flex-wrap gap-x-3">
          {(status.proteins ?? []).map((p) => (
            <label key={p} className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={isotypes.includes(p)}
                onChange={(e) => setIsotypes(e.target.checked ? [...isotypes, p] : isotypes.filter((x) => x !== p))}
              />
              {p}
            </label>
          ))}
        </div>
        {isotypes.length === 0 && (
          <p className="text-[11px] text-slate-500">
            No isotypes ticked: the per-cell noise is each cell's background mean alone.
          </p>
        )}
      </fieldset>
      <label className="flex items-center gap-1">
        <input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} />
        Remove each cell's technical noise (dsb step II)
      </label>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={run}
          disabled={running || tooFew || missing.length > 0}
          className="rounded bg-primary px-2.5 py-1 text-xs font-medium text-white disabled:opacity-50"
        >
          {running ? "Computing…" : empty ? "Compute DSB" : "Compute DSB without empty droplets"}
        </button>
        {running && <Loader2 className="h-4 w-4 animate-spin text-slate-400" />}
      </div>
      {error && (
        <p className="flex items-center gap-1.5 text-amber-700" role="alert">
          <AlertCircle className="h-3.5 w-3.5 flex-shrink-0" /> {error}
        </p>
      )}
    </div>
  );
}

/** The droplets that are not cells by log10 RNA and protein UMIs, the window drawn on them. */
function WindowPlot({
  density,
  rna,
  prot,
  sample,
}: {
  density: { rna_bin: number[]; prot_bin: number[]; n: number[]; bin_width: number };
  rna: [number, number];
  prot: [number, number];
  sample: string;
}) {
  const w = density.bin_width;
  const total = density.n.reduce((a, b) => a + b, 0);
  return (
    <div>
      <Plot
        data={[
          {
            type: "scattergl" as const,
            mode: "markers" as const,
            x: density.rna_bin.map((b) => (b + 0.5) * w),
            y: density.prot_bin.map((b) => (b + 0.5) * w),
            marker: {
              size: 5,
              symbol: "square",
              color: density.n.map((n) => Math.log10(n + 1)),
              colorscale: "Viridis",
              showscale: false,
            },
            text: density.n.map((n) => `${n.toLocaleString()} droplets`),
            hovertemplate: "RNA %{x:.2f}, protein %{y:.2f}<br>%{text}<extra></extra>",
          },
        ]}
        layout={{
          height: 200,
          margin: { t: 8, r: 8, b: 36, l: 42 },
          xaxis: { title: { text: "log10 RNA UMIs", font: { size: 10 } }, tickfont: { size: 9 }, zeroline: false },
          yaxis: { title: { text: "log10 protein UMIs", font: { size: 10 } }, tickfont: { size: 9 }, zeroline: false },
          shapes: [
            {
              type: "rect",
              x0: rna[0],
              x1: rna[1],
              y0: prot[0],
              y1: prot[1],
              line: { color: "#dc2626", width: 1.5 },
            },
          ],
          showlegend: false,
        }}
        config={plotConfig(`dsb_empty_droplets_${sample}`)}
        useResizeHandler
        style={{ width: "100%" }}
      />
      <Caption n={{ value: total, unit: "droplets that are not cells" }}>
        Droplets of {sample} by RNA and protein UMIs (colour: droplets per bin, log scale); the box is the window.
      </Caption>
    </div>
  );
}
