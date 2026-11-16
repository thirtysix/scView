import { CopyPlus } from "lucide-react";

import { getDataset } from "@/api/datasets";
import type { SubsetOf } from "@/api/types";
import { Help } from "@/components/common/Help";
import { useDatasetStore } from "@/stores/datasetStore";

/**
 * Where a subset came from (backend core/subset.py, metadata `subset_of`): its parent, how
 * many of the parent's cells and how they were chosen, and what came along. Shown on Data
 * Assessment, where the new dataset is processed.
 */
export function SubsetSourceCard({ source }: { source: SubsetOf }) {
  const parentListed = useDatasetStore((s) => s.availableDatasets.some((d) => d.id === source.id));
  const openParent = async () => {
    const ds = await getDataset(source.id).catch(() => null);
    if (ds) useDatasetStore.getState().setCurrentDataset(ds);
  };
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm" data-testid="subset-source">
      <div className="flex items-start gap-2">
        <CopyPlus className="mt-0.5 h-4 w-4 flex-shrink-0 text-slate-400" />
        <div className="min-w-0 flex-1 space-y-1">
          <p className="text-slate-700">
            <Help id="subset.source">Subset of</Help>{" "}
            {parentListed ? (
              <button
                type="button"
                onClick={() => void openParent()}
                className="font-medium text-blue-600 hover:underline"
              >
                {source.name ?? source.id}
              </button>
            ) : (
              <span className="font-medium">{source.name ?? source.id} (no longer in the list)</span>
            )}
            : {source.n_cells.toLocaleString("en-US")} of {source.n_parent_cells.toLocaleString("en-US")} cells, chosen by{" "}
            {source.described}.
          </p>
          {!!(source.carried?.length || source.not_carried?.length) && (
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs text-slate-500">
              {source.carried?.length ? (
                <>
                  <dt>Carried</dt>
                  <dd>{["the cells' counts", ...source.carried].join("; ")}</dd>
                </>
              ) : null}
              {source.not_carried?.length ? (
                <>
                  <dt>Not carried</dt>
                  <dd>{source.not_carried.join("; ")}</dd>
                </>
              ) : null}
            </dl>
          )}
        </div>
      </div>
    </div>
  );
}
