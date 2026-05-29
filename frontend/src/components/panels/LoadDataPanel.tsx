import { useCallback, useEffect, useState } from "react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { uploadDataset, listDatasets, getDataset } from "@/api/datasets";
import {
  Upload,
  FileUp,
  Trash2,
  Loader2,
  FileCode2,
  FlaskConical,
  Layers,
  ArrowRight,
} from "lucide-react";
import { formatNumber } from "@/lib/formatting";

export function LoadDataPanel() {
  const {
    availableDatasets,
    setAvailableDatasets,
    setCurrentDataset,
    isUploading,
    setUploading,
  } = useDatasetStore();
  const setPanel = useViewStore((s) => s.setPanel);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Populate previously-loaded datasets on mount so they can be reopened
  // without re-uploading.
  useEffect(() => {
    listDatasets()
      .then(setAvailableDatasets)
      .catch(() => {
        /* non-fatal: the list just stays empty */
      });
  }, [setAvailableDatasets]);

  const handleFile = useCallback(
    async (file: File) => {
      const ext = file.name.split(".").pop()?.toLowerCase();
      if (!ext || !["h5ad", "rds", "rdata"].includes(ext)) {
        setError("Unsupported file type. Please upload .h5ad, .rds, or .Rdata files.");
        return;
      }

      setError(null);
      setUploading(true);

      try {
        const result = await uploadDataset(file);
        // Poll for dataset readiness
        let dataset = await getDataset(result.id);
        let attempts = 0;
        while (dataset.status === "converting" && attempts < 60) {
          await new Promise((r) => setTimeout(r, 2000));
          dataset = await getDataset(result.id);
          attempts++;
        }

        setCurrentDataset(dataset);
        const datasets = await listDatasets();
        setAvailableDatasets(datasets);

        if (dataset.status === "ready") {
          setPanel("assessment");
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed");
      } finally {
        setUploading(false);
      }
    },
    [setUploading, setCurrentDataset, setAvailableDatasets, setPanel]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Load Data</h2>
        <p className="mt-1 text-sm text-slate-500">
          Upload a single-cell RNA-seq dataset to get started.
        </p>
      </div>

      {/* Upload zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        className={`relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-12 transition-colors ${
          dragActive
            ? "border-primary bg-primary/5"
            : "border-slate-300 bg-white hover:border-primary/50"
        }`}
      >
        {isUploading ? (
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="h-10 w-10 animate-spin text-primary" />
            <p className="text-sm font-medium text-slate-600">
              Uploading and processing...
            </p>
          </div>
        ) : (
          <>
            <FileUp className="mb-4 h-12 w-12 text-slate-400" />
            <p className="text-base font-medium text-slate-700">
              Drag and drop your dataset here
            </p>
            <p className="mt-1 text-sm text-slate-400">
              Supports .h5ad (Scanpy) and .rds / .Rdata (Seurat)
            </p>
            <label className="mt-4 cursor-pointer rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primary/90">
              <Upload className="mr-2 inline-block h-4 w-4" />
              Browse Files
              <input
                type="file"
                accept=".h5ad,.rds,.rdata,.RData"
                onChange={handleInputChange}
                className="hidden"
              />
            </label>
          </>
        )}
      </div>

      {error && (
        <div className="rounded-lg bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Accepted formats */}
      <div>
        <h3 className="mb-3 text-sm font-semibold text-slate-700">
          What can I upload here?
        </h3>
        <div className="grid gap-3 sm:grid-cols-3">
          <FormatCard icon={FileCode2} title="AnnData (.h5ad)">
            Scanpy's native single-cell format. Export it from Python with{" "}
            <code className="rounded bg-slate-100 px-1">adata.write_h5ad()</code>.
            Loads directly — no conversion needed.
          </FormatCard>

          <FormatCard icon={FlaskConical} title="Seurat (.rds / .RData)">
            An R Seurat object saved with{" "}
            <code className="rounded bg-slate-100 px-1">
              saveRDS(obj, "data.rds")
            </code>
            . scView converts it to AnnData for you (this can take a minute).
          </FormatCard>

          <button
            type="button"
            onClick={() => setPanel("ingest")}
            className="group flex flex-col rounded-xl border border-primary/30 bg-primary/5 p-4 text-left transition-colors hover:border-primary hover:bg-primary/10"
          >
            <div className="flex items-center justify-between">
              <Layers className="h-5 w-5 text-primary" />
              <ArrowRight className="h-4 w-4 text-primary opacity-0 transition-opacity group-hover:opacity-100" />
            </div>
            <span className="mt-2 text-sm font-semibold text-slate-800">
              10x, CSV or multiple files
            </span>
            <span className="mt-1 text-xs leading-relaxed text-slate-500">
              Raw CellRanger output (matrix + barcodes + features), a{" "}
              <code className="rounded bg-slate-100 px-1">.h5</code>, an
              expression table, or several samples to merge → use the{" "}
              <span className="font-medium text-primary">Add Data</span> tab.
            </span>
          </button>
        </div>
      </div>

      {/* Previous datasets */}
      {availableDatasets.length > 0 && (
        <div>
          <h3 className="mb-3 text-sm font-semibold text-slate-700">
            Previous Datasets
          </h3>
          <div className="space-y-2">
            {availableDatasets.map((ds) => (
              <div
                key={ds.id}
                className="flex items-center justify-between rounded-lg border border-slate-200 bg-white p-3"
              >
                <button
                  onClick={async () => {
                    const dataset = await getDataset(ds.id);
                    setCurrentDataset(dataset);
                    if (dataset.status === "ready") setPanel("overview");
                  }}
                  className="flex items-center gap-3 text-left"
                >
                  <div>
                    <p className="text-sm font-medium text-slate-900">
                      {ds.name}
                    </p>
                    <p className="text-xs text-slate-500">
                      {ds.n_cells != null && `${formatNumber(ds.n_cells)} cells`}
                      {ds.n_genes != null &&
                        ` · ${formatNumber(ds.n_genes)} genes`}
                    </p>
                  </div>
                </button>
                <button className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-red-500">
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function FormatCard({
  icon: Icon,
  title,
  children,
}: {
  icon: React.ElementType;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="flex items-center gap-2">
        <Icon className="h-5 w-5 text-primary" />
        <span className="text-sm font-semibold text-slate-800">{title}</span>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-slate-500">{children}</p>
    </div>
  );
}
