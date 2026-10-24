import { useCallback, useEffect, useRef, useState } from "react";
import { Download, ExternalLink, Loader2, Search, X } from "lucide-react";

import { getDataset, listDatasets } from "@/api/datasets";
import {
  cancelPublicImport,
  formatBytes,
  getDiscoverDataset,
  importDiscoverDataset,
  searchDiscover,
  type DiscoverDataset,
  type DiscoverDetail,
  type DiscoverSearch,
} from "@/api/publicImport";
import { Caption, capText } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";

const PAGE = 50;

interface Progress {
  stage: string;
  message: string;
  done?: number;
  total?: number;
}

/**
 * Search CELLxGENE Discover's public datasets and import one into the library,
 * pinned to its dataset version (backend core/public_import.py). The imported
 * dataset opens like any upload.
 */
export function PublicImport({ onClose }: { onClose: () => void }) {
  const setCurrentDataset = useDatasetStore((s) => s.setCurrentDataset);
  const setAvailableDatasets = useDatasetStore((s) => s.setAvailableDatasets);
  const setPanel = useViewStore((s) => s.setPanel);

  const [q, setQ] = useState("");
  const [organism, setOrganism] = useState("");
  const [tissue, setTissue] = useState("");
  const [result, setResult] = useState<DiscoverSearch | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<DiscoverDetail | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const jobId = useRef<string | null>(null);
  const query = useRef(q);
  query.current = q;

  const run = useCallback(async (text: string, org: string, tis: string) => {
    setLoading(true);
    setError(null);
    try {
      setResult(await searchDiscover({ q: text, organism: org, tissue: tis, limit: PAGE }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "The search failed.");
    } finally {
      setLoading(false);
    }
  }, []);

  // Filters search at once; the text box on Enter or Search.
  useEffect(() => {
    void run(query.current, organism, tissue);
  }, [organism, tissue, run]);

  const pick = useCallback(async (d: DiscoverDataset) => {
    setError(null);
    try {
      setDetail(await getDiscoverDataset(d.dataset_version_id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not read that dataset.");
    }
  }, []);

  const startImport = useCallback(async () => {
    if (!detail) return;
    setError(null);
    setProgress({ stage: "start", message: "Starting…" });
    try {
      const id = await importDiscoverDataset(detail.dataset_version_id, ({ event, data }) => {
        if (event === "job") jobId.current = String(data.job_id);
        else if (event === "stage")
          setProgress((p) => ({ ...p, stage: String(data.stage), message: String(data.message) }));
        else if (event === "progress")
          setProgress((p) => ({
            stage: p?.stage ?? "download",
            message: p?.message ?? "Downloading",
            done: Number(data.done),
            total: Number(data.total),
          }));
        else if (event === "memory")
          setProgress((p) => ({ stage: p?.stage ?? "verify", message: String(data.message) }));
      });
      const ds = await getDataset(id);
      setAvailableDatasets(await listDatasets());
      setCurrentDataset(ds);
      setPanel("assessment");
    } catch (e) {
      setError(e instanceof Error ? e.message : "The import failed.");
    } finally {
      jobId.current = null;
      setProgress(null);
    }
  }, [detail, setAvailableDatasets, setCurrentDataset, setPanel]);

  const cancel = useCallback(async () => {
    if (jobId.current) {
      try {
        await cancelPublicImport(jobId.current);
      } catch {
        /* finished meanwhile */
      }
    }
  }, []);

  const results = result?.results ?? [];
  const cap = result ? capText({ shown: results.length, total: result.total, by: "cell count" }) : null;
  const pct = progress?.total ? Math.min(100, (100 * (progress.done ?? 0)) / progress.total) : null;

  return (
    <div className="space-y-3 rounded-xl border border-slate-200 bg-white p-4" data-testid="public-import">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="flex items-center gap-1.5 text-sm font-semibold text-slate-700">
            Import a public dataset from CELLxGENE Discover
            <InfoTip label="About public datasets" width={320}>
              CELLxGENE Discover (cellxgene.cziscience.com) publishes curated single-cell datasets with
              their cell types and the study they come from. scView downloads the chosen dataset's .h5ad
              from datasets.cellxgene.cziscience.com, pinned to its version id, so the exact file can be
              fetched again. The data is licensed CC BY 4.0: cite it (the citation is kept with the
              dataset and shown in its info, the methods and the report).
            </InfoTip>
          </h3>
          <p className="mt-0.5 text-xs text-slate-500">
            The list comes from Discover's public API and is refreshed once a day. Nothing is downloaded
            until you press Import.
          </p>
        </div>
        <button onClick={onClose} aria-label="Close" className="rounded p-1 text-slate-400 hover:bg-slate-100">
          <X className="h-4 w-4" />
        </button>
      </div>

      <form
        className="flex flex-wrap items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void run(q, organism, tissue);
        }}
      >
        <input
          type="search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search title, tissue, disease, assay…"
          aria-label="Search public datasets"
          className="min-w-[14rem] flex-1 rounded-lg border border-slate-300 px-3 py-1.5 text-sm focus:border-primary focus:outline-none"
        />
        <select
          value={organism}
          onChange={(e) => {
            setOrganism(e.target.value);
            setTissue("");
          }}
          aria-label="Organism"
          className="rounded-lg border border-slate-300 px-2 py-1.5 text-sm"
        >
          <option value="">Any organism</option>
          {(result?.facets.organism ?? []).map(([o, n]) => (
            <option key={o} value={o}>
              {o} ({n})
            </option>
          ))}
        </select>
        <select
          value={tissue}
          onChange={(e) => setTissue(e.target.value)}
          aria-label="Tissue"
          className="max-w-[14rem] rounded-lg border border-slate-300 px-2 py-1.5 text-sm"
        >
          <option value="">Any tissue</option>
          {(result?.facets.tissue ?? []).map(([t, n]) => (
            <option key={t} value={t}>
              {t} ({n})
            </option>
          ))}
        </select>
        <button
          type="submit"
          disabled={loading}
          className="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-sm text-white hover:bg-primary/90 disabled:opacity-50"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />} Search
        </button>
      </form>

      {error && <div className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</div>}

      {result && (
        <div className="max-h-80 overflow-auto rounded-lg border border-slate-100">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 bg-slate-50 text-slate-500">
              <tr>
                <th className="px-2 py-1.5 font-medium">Dataset</th>
                <th className="px-2 py-1.5 font-medium">Organism</th>
                <th className="px-2 py-1.5 font-medium">Tissue</th>
                <th className="px-2 py-1.5 text-right font-medium" title="Cells in the dataset">
                  Cells
                </th>
                <th className="px-2 py-1.5 text-right font-medium" title="Size of the .h5ad download">
                  Size
                </th>
                <th className="px-2 py-1.5 font-medium">Licence</th>
              </tr>
            </thead>
            <tbody>
              {results.map((d) => (
                <tr
                  key={d.dataset_version_id}
                  onClick={() => void pick(d)}
                  className={`cursor-pointer border-t border-slate-100 hover:bg-slate-50 ${
                    detail?.dataset_version_id === d.dataset_version_id ? "bg-primary/5" : ""
                  }`}
                >
                  <td className="px-2 py-1.5">
                    <div className="font-medium text-slate-700">{d.title}</div>
                    <div className="text-slate-400">{d.collection_doi_label ?? d.collection_name}</div>
                  </td>
                  <td className="px-2 py-1.5 text-slate-600">{d.organism.join(", ")}</td>
                  <td className="px-2 py-1.5 text-slate-600">
                    {d.tissue.slice(0, 2).join(", ")}
                    {d.tissue.length > 2 ? ` +${d.tissue.length - 2}` : ""}
                  </td>
                  <td className="px-2 py-1.5 text-right tabular-nums">{d.cell_count.toLocaleString()}</td>
                  <td className="px-2 py-1.5 text-right tabular-nums">{formatBytes(d.size_bytes)}</td>
                  <td className="px-2 py-1.5 text-slate-600">{d.license}</td>
                </tr>
              ))}
              {results.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-2 py-3 text-center text-slate-500">
                    No dataset matches.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
      {result && (
        <Caption n={{ value: result.total, unit: "matching datasets", of: result.n_datasets }}>
          Public datasets on CELLxGENE Discover, most cells first. {cap ?? ""} Listing fetched{" "}
          {new Date(result.fetched_at).toLocaleString()}.
        </Caption>
      )}

      {detail && (
        <div className="space-y-2 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm" data-testid="public-detail">
          <p className="font-medium text-slate-800">{detail.title}</p>
          <p className="text-xs text-slate-600">
            {detail.collection_name}
            {detail.collection_doi && (
              <>
                {" · "}
                <a
                  href={`https://doi.org/${detail.collection_doi}`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-0.5 text-primary hover:underline"
                >
                  {detail.collection_doi_label ?? detail.collection_doi} <ExternalLink className="h-3 w-3" />
                </a>
              </>
            )}
          </p>
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs text-slate-600">
            <dt className="text-slate-400">Cells</dt>
            <dd>
              {detail.cell_count.toLocaleString()} ({detail.n_cell_types} cell types)
            </dd>
            <dt className="text-slate-400">Tissue</dt>
            <dd>{detail.tissue.join(", ")}</dd>
            <dt className="text-slate-400">Disease</dt>
            <dd>{detail.disease.join(", ")}</dd>
            <dt className="text-slate-400">Assay</dt>
            <dd>{detail.assay.join(", ")}</dd>
            <dt className="text-slate-400">Version</dt>
            <dd className="break-all font-mono">{detail.dataset_version_id}</dd>
          </dl>
          <p className="text-xs text-slate-600" data-testid="public-citation">
            <span className="font-medium">Cite: </span>
            {detail.citation}
          </p>
          <p className="text-xs text-slate-500">
            Downloads <span className="break-all font-mono">{detail.download_url}</span> (
            {formatBytes(detail.size_bytes)}), licensed{" "}
            <a href={detail.license_url} target="_blank" rel="noreferrer" className="text-primary hover:underline">
              {detail.license}
            </a>
            . scView keeps the file as downloaded plus a working copy with gene symbols:{" "}
            {formatBytes(detail.disk.need_bytes)} of disk with a margin
            {detail.disk.free_bytes !== null ? `, ${formatBytes(detail.disk.free_bytes)} free` : ""}.
          </p>
          {!detail.disk.ok && (
            <p className="rounded bg-amber-50 p-2 text-xs text-amber-700">Not enough free disk for this dataset.</p>
          )}
          {progress ? (
            <div className="space-y-1" data-testid="public-progress">
              <div className="flex items-center gap-2 text-xs text-slate-600">
                <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
                <span className="min-w-0 flex-1 truncate">{progress.message}</span>
                {pct !== null && (
                  <span className="tabular-nums">
                    {formatBytes(progress.done ?? 0)} of {formatBytes(progress.total ?? 0)}
                  </span>
                )}
                <button
                  onClick={() => void cancel()}
                  className="rounded border border-slate-300 px-2 py-0.5 text-xs text-slate-600 hover:bg-white"
                >
                  Cancel
                </button>
              </div>
              {pct !== null && (
                <div className="h-1.5 overflow-hidden rounded bg-slate-200">
                  <div className="h-full bg-primary" style={{ width: `${pct}%` }} />
                </div>
              )}
            </div>
          ) : (
            <button
              onClick={() => void startImport()}
              disabled={!detail.disk.ok}
              className="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-sm text-white hover:bg-primary/90 disabled:opacity-50"
            >
              <Download className="h-4 w-4" /> Import
            </button>
          )}
        </div>
      )}
    </div>
  );
}
