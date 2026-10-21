import { ListTree } from "lucide-react";
import { GatesPanel } from "@/components/unified/GatesPanel";
import { useDatasetStore } from "@/stores/datasetStore";
import { staleReason, useGatesStore } from "@/stores/gatesStore";

/**
 * Toolbar "Gates (n)": opens the gate tree as a strip under the toolbar's
 * controls (it stays open while lassoing or filtering, so a gate can be drawn
 * inside the one just clicked).
 */
export function GatesControl() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const open = useGatesStore((s) => s.panelOpen);
  const setOpen = useGatesStore((s) => s.setPanelOpen);
  const record = useGatesStore((s) => (dataset ? s.records[dataset.id] : undefined));
  if (!dataset) return null;
  const stale = record ? staleReason(record, dataset) : null;
  const n = record?.gates.length ?? 0;
  return (
    <>
      <button
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-controls="gates-panel"
        title={
          stale
            ? `${n} saved gate${n === 1 ? "" : "s"}, not applied: the dataset's cells changed since they were drawn`
            : "Saved cell gates: name a selection, nest gates, combine them, see n and % of parent"
        }
        className={`flex items-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors ${
          open ? "bg-orange-50 text-orange-700 ring-1 ring-orange-300" : "text-slate-600 hover:bg-slate-100"
        }`}
      >
        <ListTree className="h-3.5 w-3.5" />
        Gates ({n})
        {stale && <span className="h-1.5 w-1.5 rounded-full bg-amber-500" aria-label="stale" />}
      </button>
      {open && (
        <div id="gates-panel" className="order-last basis-full border-t border-slate-100 pt-1.5">
          <GatesPanel key={dataset.id} dataset={dataset} onClose={() => setOpen(false)} />
        </div>
      )}
    </>
  );
}
