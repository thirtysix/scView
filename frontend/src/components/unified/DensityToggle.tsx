import { useState } from "react";
import { Layers, Loader2 } from "lucide-react";
import { InfoTip } from "@/components/common/InfoTip";
import { DENSITY_MAX_GENES, DENSITY_TIP, showDensity } from "@/lib/density";
import { errorText } from "@/lib/embeddingReliability";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

/**
 * "Density" next to the gene colour (Expression subtab): while on, the gene
 * picked (here or anywhere) colours the plot by its expression-weighted
 * density instead of its expression (lib/density.ts). Off: the expression again.
 */
export function DensityToggle({
  datasetId,
  gene,
  onGeneSelect,
}: {
  datasetId: string;
  gene: string;
  /** Show a gene's expression (and its violin) */
  onGeneSelect: (gene: string) => void;
}) {
  const on = useUnifiedViewStore((s) => s.densityMode);
  const setOn = useUnifiedViewStore((s) => s.setDensityMode);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggle = async (next: boolean) => {
    setOn(next);
    setError(null);
    if (!next) {
      onGeneSelect(gene);
      return;
    }
    setBusy(true);
    try {
      await showDensity(datasetId, [gene]);
    } catch (e) {
      setOn(false);
      setError(`Density: ${errorText(e)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mt-1.5">
      <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
        <input
          type="checkbox"
          checked={on}
          disabled={busy}
          onChange={(e) => void toggle(e.target.checked)}
          className="h-3 w-3"
        />
        Density
        <InfoTip label="What the density shows">{DENSITY_TIP}</InfoTip>
        {busy && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
      </label>
      {error && (
        <p role="alert" className="mt-1 text-[11px] text-red-600">
          {error}
        </p>
      )}
    </div>
  );
}

/**
 * "Joint density" of the picked genes (MultiGenePanel), on the main plot: the
 * product of each gene's density over its peak, high where all are expressed.
 * Closes a feature grid or blend, which would cover the plot.
 */
export function JointDensityButton({
  datasetId,
  genes,
  onShow,
}: {
  datasetId: string;
  genes: string[];
  /** Called before the density is drawn (close what covers the plot) */
  onShow?: () => void;
}) {
  const overlay = useUnifiedViewStore((s) => s.scatterOverlay);
  const overlayDatasetId = useUnifiedViewStore((s) => s.overlayDatasetId);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const shown =
    overlay?.type === "density" &&
    overlayDatasetId === datasetId &&
    overlay.genes.length > 1 &&
    overlay.genes.every((g) => genes.includes(g));
  const usable = genes.length >= 2 && genes.length <= DENSITY_MAX_GENES;

  const click = async () => {
    setError(null);
    const store = useUnifiedViewStore.getState();
    if (shown) {
      store.setScatterOverlay(null);
      store.setOverlayValues(null);
      store.setOverlayLabel("");
      return;
    }
    onShow?.();
    setBusy(true);
    try {
      await showDensity(datasetId, genes);
    } catch (e) {
      setError(`Joint density: ${errorText(e)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <span className="flex items-center gap-1">
        <button
          type="button"
          disabled={!usable || busy}
          aria-pressed={shown}
          onClick={() => void click()}
          title={usable ? "Colour the plot by where all these genes are expressed together (weighted density)" : "Pick 2 or more genes"}
          className={`flex items-center gap-1 rounded border px-2 py-1 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-50 ${
            shown ? "border-blue-300 bg-blue-50 text-blue-700" : "border-slate-300 bg-white text-slate-600 hover:bg-slate-100"
          }`}
        >
          {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Layers className="h-3.5 w-3.5" />}
          {shown ? "Hide joint density" : "Joint density"}
        </button>
        <InfoTip label="What the joint density shows">{DENSITY_TIP}</InfoTip>
      </span>
      {error && (
        <p role="alert" className="w-full text-[11px] text-red-600">
          {error}
        </p>
      )}
    </>
  );
}
