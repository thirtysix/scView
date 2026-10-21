import { useDatasetStore } from "@/stores/datasetStore";
import { formatNumber } from "@/lib/formatting";
import { DatasetDownloadMenu } from "@/components/common/DatasetDownloadMenu";
import { SessionMenu } from "@/components/common/SessionMenu";
import { ShareLinkButton } from "@/components/common/ShareLinkButton";

export function Header() {
  const currentDataset = useDatasetStore((s) => s.currentDataset);

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-slate-200 bg-white px-6">
      <div className="flex items-center gap-4">
        {currentDataset ? (
          <>
            <h1 className="text-sm font-semibold text-slate-900">
              {currentDataset.name}
            </h1>
            <div className="flex gap-3 text-xs text-slate-500">
              {currentDataset.n_cells != null && (
                <span title="Cells (rows of obs) in the dataset as stored">{formatNumber(currentDataset.n_cells)} cells</span>
              )}
              {currentDataset.n_genes != null && (
                <span title="Genes (rows of var) in the dataset's expression matrix">{formatNumber(currentDataset.n_genes)} genes</span>
              )}
              <span
                className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                  currentDataset.status === "ready"
                    ? "bg-emerald-100 text-emerald-700"
                    : currentDataset.status === "converting"
                      ? "bg-amber-100 text-amber-700"
                      : currentDataset.status === "error"
                        ? "bg-red-100 text-red-700"
                        : "bg-slate-100 text-slate-600"
                }`}
              >
                {currentDataset.status}
              </span>
            </div>
          </>
        ) : (
          <h1 className="text-sm text-slate-400">No dataset loaded</h1>
        )}
      </div>
      <div className="flex items-center gap-2">
        <SessionMenu />
        <ShareLinkButton />
        {currentDataset?.status === "ready" && <DatasetDownloadMenu datasetId={currentDataset.id} />}
      </div>
    </header>
  );
}
