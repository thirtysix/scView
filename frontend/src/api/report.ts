import { readError } from "./client";
import { API_BASE } from "@/lib/constants";

/** ok: the section has its numbers; not_computed: its input was never computed (the
 *  note says where to compute it); not_applicable: it cannot apply to this dataset. */
export type SectionStatus = "ok" | "not_computed" | "not_applicable";

/** One section of a cluster or gene report (backend core/report.py). `data`'s shape
 *  depends on `id`; the section components in components/report/ read it. */
export interface ReportSection<D = Record<string, unknown>> {
  id: string;
  title: string;
  /** A plain sentence saying what is shown, with the n. */
  caption: string;
  /** One sentence on how the numbers were computed. */
  method: string;
  status: SectionStatus;
  note: string | null;
  data: D;
}

/** A sub-part with its own status (the marker databases, doublets, QC states, ...). */
export interface ReportPart {
  status: SectionStatus;
  note?: string | null;
  [key: string]: unknown;
}

interface ReportBase {
  title: string;
  dataset: { name: string; n_cells: number; n_genes: number };
  design: { condition: string | null; donor: string | null };
  scview_version: string;
  generated_at: string;
  sections: ReportSection[];
  elapsed_seconds: number;
}

export interface ClusterReport extends ReportBase {
  kind: "cluster";
  column: string;
  value: string;
  n_cells: number;
}

export interface GeneReport extends ReportBase {
  kind: "gene";
  gene: string;
  groupby: string;
}

export type Report = ClusterReport | GeneReport;

/** A failed report request; `suggestion` is a close gene name for an unknown gene. */
export class ReportError extends Error {
  status: number;
  suggestion: string | null;

  constructor(message: string, status: number, suggestion: string | null = null) {
    super(message);
    this.name = "ReportError";
    this.status = status;
    this.suggestion = suggestion;
  }
}

async function getReport<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { signal });
  if (!res.ok) {
    const text = await readError(res);
    let message = text;
    let suggestion: string | null = null;
    try {
      const detail = (JSON.parse(text) as { detail?: unknown }).detail;
      if (typeof detail === "string") message = detail;
      else if (detail && typeof detail === "object") {
        const d = detail as { message?: string; suggestion?: string | null };
        message = d.message ?? text;
        suggestion = d.suggestion ?? null;
      }
    } catch {
      // not JSON: readError's text (e.g. the backend-down sentence) is the message
    }
    throw new ReportError(message || `Report failed (${res.status}).`, res.status, suggestion);
  }
  return res.json() as Promise<T>;
}

function clusterQuery(column: string, value: string): string {
  return new URLSearchParams({ column, value }).toString();
}

function geneQuery(gene: string, groupby?: string): string {
  const q = new URLSearchParams({ gene });
  if (groupby) q.set("groupby", groupby);
  return q.toString();
}

export function fetchClusterReport(
  datasetId: string, column: string, value: string, signal?: AbortSignal,
): Promise<ClusterReport> {
  return getReport<ClusterReport>(`/datasets/${datasetId}/report/cluster?${clusterQuery(column, value)}`, signal);
}

export function fetchGeneReport(
  datasetId: string, gene: string, groupby?: string, signal?: AbortSignal,
): Promise<GeneReport> {
  return getReport<GeneReport>(`/datasets/${datasetId}/report/gene?${geneQuery(gene, groupby)}`, signal);
}

/** The self-contained HTML download of a cluster report (relative: works through nginx). */
export function clusterReportHtmlUrl(datasetId: string, column: string, value: string): string {
  return `${API_BASE}/datasets/${datasetId}/report/cluster.html?${clusterQuery(column, value)}`;
}

export function geneReportHtmlUrl(datasetId: string, gene: string, groupby?: string): string {
  return `${API_BASE}/datasets/${datasetId}/report/gene.html?${geneQuery(gene, groupby)}`;
}
