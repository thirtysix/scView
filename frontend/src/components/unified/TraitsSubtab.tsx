import { useEffect, useMemo, useState } from "react";
import { Download, Loader2 } from "lucide-react";
import { Caption, CapNote } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import { SortableTh, ariaSort } from "@/components/common/SortableTh";
import { GeneLinkOuts } from "@/components/unified/GeneLinkOuts";
import {
  downloadGwas,
  efoUrl,
  getGeneTraits,
  getGwasClusters,
  getGwasStatus,
  pubmedUrl,
  runGwasClusters,
  type GeneTraits,
  type GwasGroup,
  type GwasProgress,
  type GwasResult,
  type GwasStatus,
  type GwasTraitRow,
} from "@/api/gwas";
import { downloadCsv } from "@/lib/csv";
import { errorMessage } from "@/lib/errorMessage";
import { formatPValue, prettyObsLabel } from "@/lib/formatting";
import { useDatasetStore } from "@/stores/datasetStore";

const SECONDS_PER_GROUP = 1; // Kang: 13 cell types in 10-13 s on 2 cores
const LONG_SECONDS = 4;

type SortKey = "p" | "p_empirical" | "p_adj" | "n_loci" | "odds_ratio";
const COLUMNS: { key: SortKey; label: string; title: string }[] = [
  { key: "n_loci", label: "Loci", title: "Distinct loci the overlap genes sit in (lead SNPs within 1 Mb merged); a trait is listed from 3" },
  { key: "odds_ratio", label: "Odds ratio", title: "How much more often the markers are trait genes than the other background genes" },
  { key: "p", label: "p", title: "Hypergeometric p of the gene overlap, against the background genes" },
  { key: "p_empirical", label: "Empirical p", title: "Share of random gene sets, matched for how many traits each gene is linked to, that cover as many loci" },
  { key: "p_adj", label: "Adjusted p", title: "Benjamini-Hochberg adjusted p (hypergeometric) across the traits tested for this group" },
];

/** Rows in the chosen order (largest first for loci and odds ratio, smallest for p). */
export function sortTraitRows(rows: GwasTraitRow[], key: SortKey, dir: "asc" | "desc"): GwasTraitRow[] {
  const sign = dir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => sign * (a[key] - b[key]) || a.p - b.p);
}

function seconds(s: number): string {
  return s < 60 ? `${Math.max(1, Math.round(s))} s` : `${Math.round(s / 60)} min`;
}

function progressText(p: GwasProgress | null): string {
  if (!p) return "Starting…";
  if (p.stage === "download" && p.bytes != null) {
    const mb = (p.bytes / 1e6).toFixed(0);
    return p.total ? `Downloading ${p.file ?? ""}: ${mb} of ${(p.total / 1e6).toFixed(0)} MB` : `Downloading: ${mb} MB`;
  }
  if (p.stage === "check") return `Checking ${p.file ?? "the files"}…`;
  return p.rows ? `Preparing: ${p.rows.toLocaleString()} associations read…` : "Preparing the catalog…";
}

function csvRows(result: GwasResult): (string | number)[][] {
  return result.groups.flatMap((g) =>
    g.rows.map((r) => [
      g.group, r.trait, r.uri, r.parent, r.n_trait_genes, r.n_overlap, r.n_loci, r.genes.join(";"),
      r.odds_ratio, r.p, r.p_empirical, r.n_draws, r.p_adj, r.lead ? "yes" : "no",
    ]),
  );
}

const CSV_HEADERS = [
  "group", "trait", "trait_uri", "parent_term", "trait_genes_in_background", "overlap_genes", "overlap_loci",
  "genes", "odds_ratio", "p", "p_empirical", "n_random_sets", "p_adj", "lead",
];

function Citation({ release, attribution }: { release: string; attribution?: string }) {
  return (
    <p className="text-[11px] text-slate-500" data-testid="gwas-citation">
      Data: {attribution ?? `NHGRI-EBI GWAS Catalog, release ${release}`}; Sollis et al. 2023, Nucleic Acids Res
      (PMID{" "}
      <a href={pubmedUrl(36350656)} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
        36350656
      </a>
      ). Associations with p ≤ 5e-8; the extended MHC is left out.
    </p>
  );
}

/** One gene's traits in the catalog, with PubMed ids and link-outs. */
function GeneTraitsCard({ gene, data, loading }: { gene: string; data: GeneTraits | null; loading: boolean }) {
  return (
    <div className="space-y-1.5 rounded-lg border border-slate-200 p-2 dark:border-slate-700" data-testid="gene-traits">
      <div className="text-xs font-medium text-slate-700 dark:text-slate-200">
        GWAS traits mapped to {gene}
        {data?.via_ortholog && data.query !== data.gene ? ` (human ${data.gene})` : ""}
      </div>
      <GeneLinkOuts gene={data?.gene ?? gene} ensembl={data?.ensembl} />
      {loading && <Loader2 className="h-3.5 w-3.5 animate-spin text-slate-400" />}
      {!loading && !data && <p className="text-[11px] text-slate-500">The catalog maps no genome-wide significant association to {gene}.</p>}
      {data && (
        <>
          <CapNote shown={data.shown} total={data.n_traits} by="p" />
          <div className="max-h-56 overflow-auto">
            <table className="w-full text-[11px]">
              <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800">
                <tr className="text-left text-slate-600 dark:text-slate-300">
                  <th className="px-1.5 py-1" title="EFO trait the catalog maps the association to">Trait</th>
                  <th className="px-1.5 py-1" title="The trait's parent term in the catalog's grouping">Parent term</th>
                  <th className="px-1.5 py-1 text-right" title="Smallest p of an association mapped to this gene">p</th>
                  <th className="px-1.5 py-1 text-right" title="Distinct loci of this gene for the trait">Loci</th>
                  <th className="px-1.5 py-1" title="Studies (PubMed), best p first">PubMed</th>
                </tr>
              </thead>
              <tbody>
                {data.traits.map((t) => (
                  <tr key={t.uri} className="border-t border-slate-100 dark:border-slate-800">
                    <td className="px-1.5 py-0.5 text-slate-700 dark:text-slate-200">
                      <a href={efoUrl(t.uri)} target="_blank" rel="noopener noreferrer" className="hover:underline">
                        {t.trait}
                      </a>
                    </td>
                    <td className="px-1.5 py-0.5 text-slate-500">{t.parent || "–"}</td>
                    <td className="px-1.5 py-0.5 text-right tabular-nums">{t.p > 0 ? formatPValue(t.p) : `1e-${Math.round(t.mlog)}`}</td>
                    <td className="px-1.5 py-0.5 text-right tabular-nums">{t.n_loci}</td>
                    <td className="px-1.5 py-0.5">
                      {t.pmids.map((id, i) => (
                        <span key={id}>
                          {i > 0 && ", "}
                          <a href={pubmedUrl(id)} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
                            {id}
                          </a>
                        </span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

/**
 * GWAS trait links (C088): traits whose GWAS Catalog genes overlap each group's
 * marker genes more than degree-matched random genes do (backend
 * core/gwas_catalog.py). Clicking a trait lists its overlap genes; clicking a gene
 * colours the plot by it (as the Expression tab does) and shows its own traits.
 */
export function TraitsSubtab({
  onGeneClick,
  activeGene,
  groupByColumn,
  setGroupByColumn,
}: {
  onGeneClick: (gene: string) => void;
  activeGene?: string | null;
  groupByColumn: string;
  setGroupByColumn: (col: string) => void;
}) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const [status, setStatus] = useState<GwasStatus | null>(null);
  const [result, setResult] = useState<GwasResult | null>(null);
  const [includeMeasurements, setIncludeMeasurements] = useState(false);
  const [running, setRunning] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [progress, setProgress] = useState<GwasProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [group, setGroup] = useState("");
  const [trait, setTrait] = useState<string | null>(null);
  const [gene, setGene] = useState<string | null>(null);
  const [geneData, setGeneData] = useState<GeneTraits | null>(null);
  const [geneLoading, setGeneLoading] = useState(false);
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "p", dir: "asc" });

  const groupColumns = useMemo(() => {
    if (!dataset) return [];
    const nCells = dataset.n_cells ?? 0;
    return dataset.obs_columns.filter(
      (c) =>
        (c.dtype === "category" || c.dtype === "object") &&
        c.n_unique >= 2 &&
        c.n_unique <= 100 &&
        !(nCells > 0 && c.n_unique / nCells >= 0.9),
    );
  }, [dataset]);
  const column = groupColumns.some((c) => c.name === groupByColumn) ? groupByColumn : (groupColumns[0]?.name ?? "");

  useEffect(() => {
    let cancelled = false;
    getGwasStatus()
      .then((s) => !cancelled && setStatus(s))
      .catch((e) => !cancelled && setError(errorMessage(e)));
    return () => {
      cancelled = true;
    };
  }, []);

  // The last run, restored on opening the tab.
  useEffect(() => {
    if (!datasetId) return;
    let cancelled = false;
    setResult(null);
    getGwasClusters(datasetId)
      .then((r) => !cancelled && setResult(r))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  const current: GwasGroup | null = useMemo(() => {
    if (!result) return null;
    return (
      result.groups.find((g) => g.group === group) ??
      result.groups.find((g) => g.n_leads > 0) ??
      result.groups[0] ??
      null
    );
  }, [result, group]);
  const rows = useMemo(() => (current ? sortTraitRows(current.rows, sort.key, sort.dir) : []), [current, sort]);
  const selected = rows.find((r) => r.uri === trait) ?? null;

  if (!dataset || !datasetId) return <div className="text-xs text-slate-400">No dataset loaded.</div>;

  const download = () => {
    setDownloading(true);
    setError(null);
    setProgress(null);
    downloadGwas(setProgress)
      .then(setStatus)
      .catch((e) => setError(errorMessage(e)))
      .finally(() => setDownloading(false));
  };

  const run = () => {
    if (!column) return;
    setRunning(true);
    setError(null);
    setTrait(null);
    runGwasClusters(datasetId, { column, include_measurements: includeMeasurements })
      .then((r) => {
        setResult(r);
        setGroup("");
      })
      .catch((e) => setError(errorMessage(e)))
      .finally(() => setRunning(false));
  };

  const showGene = (humanGene: string, datasetGene: string) => {
    onGeneClick(datasetGene);
    setGene(humanGene);
    setGeneData(null);
    setGeneLoading(true);
    getGeneTraits(humanGene)
      .then(setGeneData)
      .catch((e) => setError(errorMessage(e)))
      .finally(() => setGeneLoading(false));
  };

  const onSort = (key: SortKey) =>
    setSort((s) =>
      s.key === key
        ? { key, dir: s.dir === "asc" ? "desc" : "asc" }
        : { key, dir: key === "n_loci" || key === "odds_ratio" ? "desc" : "asc" },
    );

  const nGroups = groupColumns.find((c) => c.name === column)?.n_unique ?? 0;
  const estimate = nGroups * SECONDS_PER_GROUP;
  const caveat = result?.caveat ?? status?.caveat;
  const release = result?.release ?? status?.release ?? "";

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1.5 text-xs font-medium text-slate-700 dark:text-slate-200">
        GWAS trait links
        <InfoTip label="About GWAS trait links" width={320}>
          For each group, its marker genes are compared with the genes the GWAS Catalog maps near variants associated
          with each trait. A trait is listed when the overlap covers at least 3 separate loci. The empirical p compares
          with random gene sets that match the markers on how many traits each gene is linked to.
        </InfoTip>
      </div>

      {caveat && (
        <p
          className="rounded border border-amber-200 bg-amber-50 px-2.5 py-2 text-[11px] leading-snug text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200"
          data-testid="gwas-caveat"
        >
          {caveat}
        </p>
      )}
      {release && <Citation release={release} attribution={result?.attribution ?? status?.attribution} />}

      {status && !status.downloaded && (
        <div className="space-y-1.5 rounded-lg border border-slate-200 p-2.5 text-xs dark:border-slate-700">
          <p className="text-slate-600 dark:text-slate-300">
            The GWAS Catalog ({status.release} release) is not downloaded yet: about {status.download_mb} MB from
            ftp.ebi.ac.uk, once. Preparing it takes about a minute.
            {!status.auto_download && !status.raw_present &&
              " Downloading is turned off on this server (GWAS_AUTO_DOWNLOAD): place the two files in the data folder's gwas/ folder, then press the button."}
          </p>
          <button
            type="button"
            onClick={download}
            disabled={downloading}
            className="flex items-center gap-1 rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:opacity-90 disabled:opacity-50"
          >
            {downloading && <Loader2 className="h-3 w-3 animate-spin" />}
            {downloading ? "Downloading…" : status.raw_present ? "Prepare GWAS Catalog" : "Download GWAS Catalog"}
          </button>
          {downloading && (
            <p className="text-[11px] text-slate-500" data-testid="gwas-progress">
              {progressText(progress)}
            </p>
          )}
        </div>
      )}

      {status?.downloaded && (
        <div className="space-y-1.5">
          <div className="flex flex-wrap items-end gap-2 text-xs">
            <label className="flex flex-col gap-0.5 text-[11px] text-slate-500">
              Group by
              <select
                value={column}
                onChange={(e) => setGroupByColumn(e.target.value)}
                className="rounded border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-700 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
              >
                {groupColumns.map((c) => (
                  <option key={c.name} value={c.name}>
                    {prettyObsLabel(c.name)} ({c.n_unique} groups)
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1 pb-1 text-[11px] text-slate-600 dark:text-slate-300">
              <input
                type="checkbox"
                checked={includeMeasurements}
                onChange={(e) => setIncludeMeasurements(e.target.checked)}
              />
              Include measurement traits
              <InfoTip label="About measurement traits" width={260}>
                Traits under &ldquo;Other measurement&rdquo; are mostly protein and metabolite levels, where a gene is
                often found next to its own product. They are left out unless you include them.
              </InfoTip>
            </label>
            <button
              type="button"
              onClick={run}
              disabled={running || !column}
              className="flex items-center gap-1 rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:opacity-90 disabled:opacity-50"
            >
              {running && <Loader2 className="h-3 w-3 animate-spin" />}
              {running ? "Computing…" : result?.column === column ? "Run again" : "Find trait links"}
            </button>
          </div>
          {estimate >= LONG_SECONDS && (
            <p className="text-[11px] text-slate-500" data-testid="time-note">
              {nGroups} groups: about {seconds(estimate)}. Markers come from the Markers tab (top 100 per group).
            </p>
          )}
        </div>
      )}

      <RequestError message={error} />

      {result && current && (
        <div className="space-y-2">
          <Caption n={[{ value: result.groups.length, unit: "groups" }, { value: result.n_background, unit: "background genes" }]}>
            Traits overlapping the top {result.n_markers} markers of each {prettyObsLabel(result.column)} group
            {result.include_measurements ? ", measurement traits included" : ""}; background = genes measured here that
            the catalog maps to any trait.
          </Caption>
          {result.mapping && (
            <p className="text-[11px] text-slate-500" data-testid="gwas-mapping">
              Mouse data: {result.mapping}; traits and genes are shown with their human names.
            </p>
          )}
          <label className="flex items-center gap-1.5 text-[11px] text-slate-500">
            Group
            <select
              value={current.group}
              onChange={(e) => {
                setGroup(e.target.value);
                setTrait(null);
              }}
              className="rounded border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-700 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
            >
              {result.groups.map((g) => (
                <option key={g.group} value={g.group}>
                  {g.group} ({g.n_leads} lead{g.n_leads === 1 ? "" : "s"})
                </option>
              ))}
            </select>
          </label>
          <p className="text-[11px] text-slate-500">
            {current.n_query.toLocaleString()} of {current.n_markers.toLocaleString()} markers are in the background;{" "}
            {current.n_tested.toLocaleString()} traits tested, {current.n_with_loci.toLocaleString()} with overlap in{" "}
            {result.min_loci}+ loci, {current.n_leads} lead{current.n_leads === 1 ? "" : "s"} (adjusted p &lt; 0.05 and
            empirical p &lt; 0.05).
          </p>
          <div className="flex items-center justify-between text-[11px] text-slate-500">
            <CapNote shown={current.rows.length} total={current.n_with_loci} by="p" rest="the rest are weaker" />
            <button
              type="button"
              onClick={() => downloadCsv(`gwas_traits_${result.column}`, CSV_HEADERS, csvRows(result))}
              className="ml-auto flex items-center gap-1 text-primary hover:underline"
              title="Every listed trait of every group, with its overlap genes and p-values"
            >
              <Download className="h-3 w-3" /> CSV
            </button>
          </div>
          {rows.length === 0 ? (
            <p className="text-[11px] text-slate-500">No trait&apos;s genes overlap this group&apos;s markers in {result.min_loci} or more loci.</p>
          ) : (
            <div className="max-h-96 overflow-auto rounded-lg border border-slate-200 dark:border-slate-700">
              <table className="w-full text-[11px]">
                <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800">
                  <tr className="border-b border-slate-200 text-slate-600 dark:border-slate-700 dark:text-slate-300">
                    <th className="px-2 py-1 text-left" title="EFO trait; click to list the overlap genes">Trait</th>
                    <th className="px-2 py-1 text-left" title="The trait's parent term in the catalog's grouping">Parent term</th>
                    <th className="px-2 py-1 text-right" title="Overlap genes / trait genes in the background">Genes</th>
                    {COLUMNS.map((c) => (
                      <SortableTh
                        key={c.key}
                        onSort={() => onSort(c.key)}
                        sort={ariaSort(c.key, sort.key, sort.dir)}
                        title={c.title}
                        align="right"
                        className="px-2 py-1"
                      >
                        {c.label}
                      </SortableTh>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr
                      key={r.uri}
                      className={`border-b border-slate-100 dark:border-slate-800 ${r.uri === trait ? "bg-primary/10" : ""}`}
                    >
                      <td className="px-2 py-0.5">
                        <button
                          type="button"
                          onClick={() => setTrait(r.uri === trait ? null : r.uri)}
                          aria-expanded={r.uri === trait}
                          className={`text-left text-primary hover:underline ${r.lead ? "font-semibold" : ""}`}
                          title={r.lead ? "Lead: adjusted p < 0.05 and empirical p < 0.05" : undefined}
                        >
                          {r.trait}
                        </button>
                      </td>
                      <td className="px-2 py-0.5 text-slate-500">{r.parent || "–"}</td>
                      <td className="px-2 py-0.5 text-right tabular-nums text-slate-700 dark:text-slate-200">
                        {r.n_overlap} / {r.n_trait_genes}
                        <span className="block max-w-[9rem] truncate text-[10px] text-slate-500" title={r.dataset_genes.join(", ")}>
                          {r.dataset_genes.slice(0, 4).join(", ")}
                          {r.dataset_genes.length > 4 ? ", …" : ""}
                        </span>
                      </td>
                      <td className="px-2 py-0.5 text-right tabular-nums text-slate-700 dark:text-slate-200">{r.n_loci}</td>
                      <td className="px-2 py-0.5 text-right tabular-nums text-slate-700 dark:text-slate-200">{r.odds_ratio.toFixed(1)}</td>
                      <td className="px-2 py-0.5 text-right tabular-nums text-slate-700 dark:text-slate-200">{formatPValue(r.p)}</td>
                      <td
                        className="px-2 py-0.5 text-right tabular-nums text-slate-700 dark:text-slate-200"
                        title={`${r.n_draws.toLocaleString()} random gene sets`}
                      >
                        {r.p_empirical <= 1 / (r.n_draws + 1) ? `< ${(1 / r.n_draws).toPrecision(1)}` : formatPValue(r.p_empirical)}
                      </td>
                      <td className="px-2 py-0.5 text-right tabular-nums text-slate-700 dark:text-slate-200">{formatPValue(r.p_adj)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {selected && (
            <div className="space-y-1 rounded-lg border border-slate-200 p-2 dark:border-slate-700" data-testid="trait-genes">
              <div className="text-xs font-medium text-slate-700 dark:text-slate-200">
                {selected.trait}: {selected.n_overlap} overlap gene{selected.n_overlap === 1 ? "" : "s"} in {selected.n_loci} loci
              </div>
              <div className="flex flex-wrap gap-1">
                {selected.genes.map((g, i) => {
                  const dg = selected.dataset_genes[i] ?? g;
                  return (
                    <button
                      key={g}
                      type="button"
                      onClick={() => showGene(g, dg)}
                      aria-pressed={dg === activeGene}
                      className={`rounded border px-1.5 py-0.5 text-[11px] ${
                        dg === activeGene
                          ? "border-primary bg-primary/10 text-primary"
                          : "border-slate-300 text-slate-700 hover:border-primary dark:border-slate-600 dark:text-slate-200"
                      }`}
                      title={`Colour the plot by ${dg} and list its GWAS traits`}
                    >
                      {dg !== g ? `${dg} (${g})` : g}
                    </button>
                  );
                })}
              </div>
              <a href={efoUrl(selected.uri)} target="_blank" rel="noopener noreferrer" className="text-[11px] text-primary hover:underline">
                About this trait (EFO)
              </a>
            </div>
          )}
        </div>
      )}

      {gene && <GeneTraitsCard gene={gene} data={geneData} loading={geneLoading} />}
    </div>
  );
}
