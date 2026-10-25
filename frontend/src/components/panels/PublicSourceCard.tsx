import { Globe } from "lucide-react";

import { formatBytes, type PublicSource } from "@/api/publicImport";
import { InfoTip } from "@/components/common/InfoTip";

/**
 * Where a public dataset came from (a CELLxGENE Discover import, backend
 * core/public_import.py): the citation to use, the pinned version and the file's
 * hash. Shown on Data Assessment; the same record is in the dataset (uns about_*),
 * the methods text and the report.
 */
export function PublicSourceCard({ source }: { source: PublicSource }) {
  const size = Number(source.about_size_bytes);
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm" data-testid="public-source">
      <div className="flex items-start gap-2">
        <Globe className="mt-0.5 h-4 w-4 flex-shrink-0 text-slate-400" />
        <div className="min-w-0 flex-1 space-y-1">
          <p className="flex items-center gap-1.5 text-slate-700">
            <span>
              From {source.about_source}
              {source.about_collection ? `: ${source.about_collection}` : ""}
            </span>
            <InfoTip label="About this source" width={300}>
              Downloaded from CELLxGENE Discover, pinned to the dataset version below; that version never
              changes, so the same file can be fetched again from its address. Discover's values in X are
              normalised; scView uses the raw counts in .raw and shows gene symbols (Ensembl ids are kept
              in the gene table as gene_ids). Discover's cell_type column is the author annotation.
            </InfoTip>
          </p>
          <p className="text-xs text-slate-600" data-testid="public-source-citation">
            <span className="font-medium">Cite: </span>
            {source.citation_text}
          </p>
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs text-slate-500">
            <dt>Version</dt>
            <dd className="break-all font-mono">{source.about_dataset_version_id}</dd>
            <dt>Schema</dt>
            <dd>{source.about_schema_version}</dd>
            <dt>Licence</dt>
            <dd>{source.about_license}</dd>
            <dt>File</dt>
            <dd className="break-all">
              {source.about_url}
              {Number.isFinite(size) && size > 0 ? ` (${formatBytes(size)})` : ""}, downloaded{" "}
              {source.about_downloaded_at.slice(0, 10)}
            </dd>
            <dt>SHA-256</dt>
            <dd className="break-all font-mono">{source.about_sha256}</dd>
          </dl>
        </div>
      </div>
    </div>
  );
}
