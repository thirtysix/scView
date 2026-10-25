import { useEffect, useMemo, useState } from "react";
import { Download, Loader2, Share2 } from "lucide-react";
import { getDatasetPpi, type PpiDirection, type PpiResult, type PpiScore } from "@/api/ppi";
import { Caption, capText } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import { SortableTh, ariaSort } from "@/components/common/SortableTh";
import { PpiNetworkCanvas } from "@/components/unified/PpiNetworkCanvas";
import { downloadCsv } from "@/lib/csv";
import { errorText } from "@/lib/programs";
import {
  COMMUNITY_SERIES,
  PPI_CSV_HEADERS,
  communityColour,
  ppiCsvRow,
  ppiInputNote,
  ppiSummary,
  sortNodes,
  type HubSortField,
  type PpiColourBy,
} from "@/lib/ppi";

export interface PpiSourceOption {
  /** `de:<result id>`, `markers:<column>/<group>` or `program:<k>`. */
  value: string;
  label: string;
}

interface Props {
  datasetId: string;
  /** The lists to choose from; the first is chosen to begin with. */
  sources: PpiSourceOption[];
  /** DE results: up, down or both directions. */
  showDirection?: boolean;
  onGeneClick?: (gene: string) => void;
  title?: string;
}

const SCORES: { value: PpiScore; label: string }[] = [
  { value: 400, label: "400 (medium)" },
  { value: 700, label: "700 (high)" },
  { value: 900, label: "900 (highest)" },
];
const SIZES = [50, 100, 200, 500];
const TABLE_ROWS = 50;

/**
 * "Protein network": how the proteins of a gene list interact in STRING
 * (physical links), with the hubs, communities and whether the list is more
 * connected than chance. Built on request; STRING is fetched once on first use.
 */
export function ProteinNetwork({ datasetId, sources, showDirection = false, onGeneClick, title = "Protein network" }: Props) {
  const [source, setSource] = useState(sources[0]?.value ?? "");
  const [score, setScore] = useState<PpiScore>(400);
  const [direction, setDirection] = useState<PpiDirection>("up");
  const [n, setN] = useState(100);
  const [colourBy, setColourBy] = useState<PpiColourBy>("community");
  const [result, setResult] = useState<PpiResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortField, setSortField] = useState<HubSortField>("degree");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  // A different list (new DE result, other group) makes the old network stale.
  const sourceKey = sources.map((s) => s.value).join("|");
  useEffect(() => {
    setSource((cur) => (sources.some((s) => s.value === cur) ? cur : sources[0]?.value ?? ""));
    setResult(null);
    setError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey]);

  const run = async () => {
    if (!source) return;
    setLoading(true);
    setError(null);
    try {
      const res = await getDatasetPpi(datasetId, source, { n, direction, score });
      setResult(res);
      setColourBy(res.nodes.some((x) => x.logfc !== null) && showDirection ? "logfc" : "community");
    } catch (e) {
      setResult(null);
      setError(errorText(e));
    } finally {
      setLoading(false);
    }
  };

  const rows = useMemo(() => (result ? sortNodes(result.nodes, sortField, sortDir) : []), [result, sortField, sortDir]);
  const hasLfc = !!result?.nodes.some((x) => x.logfc !== null);
  const sortBy = (f: HubSortField) => {
    if (f === sortField) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortField(f);
      setSortDir(f === "gene" || f === "community" ? "asc" : "desc");
    }
  };
  const exportCsv = () => {
    if (!result) return;
    downloadCsv(`ppi_${result.label ?? "list"}_${result.score}_${datasetId}.csv`, PPI_CSV_HEADERS, result.nodes.map(ppiCsvRow));
  };
  const note = result ? ppiInputNote(result) : null;
  const cap = result ? capText({ shown: Math.min(TABLE_ROWS, rows.length), total: rows.length, by: "the chosen column", rest: "the CSV has all" }) : null;

  if (sources.length === 0) return null;

  return (
    <section aria-label={title} className="flex flex-col gap-2 rounded-lg border border-slate-200 p-2.5 text-xs">
      <div className="flex items-center gap-2">
        <Share2 className="h-3.5 w-3.5 flex-shrink-0 text-slate-400" aria-hidden />
        <h3 className="font-semibold text-slate-700">{title}</h3>
        <InfoTip width={320}>
          Which proteins of the list physically interact in STRING, the hubs (most partners) and communities
          (groups more linked to each other than to the rest, found with Leiden). &ldquo;Expected&rdquo; is the
          mean number of interactions among 1,000 random protein sets of the same size, each protein matched on
          how many partners it has in the whole network; p is the share of those sets with at least as many
          interactions. Score: STRING&apos;s confidence that a link is real (0-1000). Your gene list never
          leaves this machine.
        </InfoTip>
        <span className="flex-1" />
        <button
          type="button"
          onClick={run}
          disabled={loading || !source}
          className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 font-medium text-white transition hover:bg-primary/90 disabled:opacity-50"
        >
          {loading && <Loader2 className="h-3 w-3 animate-spin" />}
          {result ? "Rebuild" : "Build network"}
        </button>
      </div>

      <div className="flex flex-wrap items-end gap-2">
        {sources.length > 1 && (
          <label className="flex flex-col gap-0.5 text-[11px] text-slate-500">
            List
            <select value={source} onChange={(e) => setSource(e.target.value)}
              className="rounded border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-700">
              {sources.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
            </select>
          </label>
        )}
        {showDirection && (
          <label className="flex flex-col gap-0.5 text-[11px] text-slate-500">
            Genes
            <select value={direction} onChange={(e) => setDirection(e.target.value as PpiDirection)}
              className="rounded border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-700">
              <option value="up">up</option>
              <option value="down">down</option>
              <option value="both">up and down</option>
            </select>
          </label>
        )}
        <label className="flex flex-col gap-0.5 text-[11px] text-slate-500">
          Top genes
          <select value={n} onChange={(e) => setN(Number(e.target.value))}
            className="rounded border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-700">
            {SIZES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-0.5 text-[11px] text-slate-500">
          Minimum score
          <select value={score} onChange={(e) => setScore(Number(e.target.value) as PpiScore)}
            className="rounded border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-700">
            {SCORES.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
          </select>
        </label>
        {result && (
          <fieldset className="flex items-center gap-2 text-[11px] text-slate-500">
            <legend className="sr-only">Colour proteins by</legend>
            <label className="flex items-center gap-1">
              <input type="radio" name={`ppi-colour-${title}`} checked={colourBy === "community"} onChange={() => setColourBy("community")} />
              community
            </label>
            <label className={`flex items-center gap-1 ${hasLfc ? "" : "opacity-50"}`}>
              <input type="radio" name={`ppi-colour-${title}`} checked={colourBy === "logfc"} disabled={!hasLfc} onChange={() => setColourBy("logfc")} />
              log2FC
            </label>
          </fieldset>
        )}
      </div>

      {showDirection && !result && (
        <p className="text-[11px] text-slate-500">
          Uses the DE genes with adjusted p &lt; 0.05 and |log2FC| &ge; 0.25, smallest p first; at most 500.
        </p>
      )}

      <RequestError message={error} />

      {result && (
        <>
          <p className="text-[12px] font-medium text-slate-700" data-testid="ppi-summary">
            {result.label ? `${result.label}: ` : ""}{ppiSummary(result)}
            {result.n_communities > 0 ? `; ${result.n_communities} communit${result.n_communities === 1 ? "y" : "ies"}` : ""}
          </p>
          {note && <p className="text-[11px] text-amber-700">{note}</p>}

          {result.n_links > 0 ? (
            <PpiNetworkCanvas nodes={result.nodes} links={result.links} colourBy={colourBy} onNodeClick={onGeneClick ? (x) => onGeneClick(x.gene) : undefined} />
          ) : (
            <p className="text-[11px] text-slate-500">No two proteins of this list interact at score &ge; {result.score}.</p>
          )}
          {result.n_links > 0 && (
            <p className="flex flex-wrap items-center gap-2 text-[10px] text-slate-500">
              {colourBy === "community" ? (
                <>
                  {COMMUNITY_SERIES.slice(0, Math.min(result.n_communities, COMMUNITY_SERIES.length)).map((_, i) => (
                    <span key={i} className="flex items-center gap-1">
                      <span className="inline-block h-2 w-2 rounded-full" style={{ background: communityColour(i + 1) }} />
                      community {i + 1}
                    </span>
                  ))}
                  {result.n_communities > COMMUNITY_SERIES.length && <span>grey: smaller communities</span>}
                </>
              ) : (
                <span>blue: lower, red: higher (log2FC, symmetric about 0); grey: no fold change</span>
              )}
              <span>Size: number of partners. {result.n_isolated > 0 ? `${result.n_isolated} proteins without a partner are not drawn.` : ""}</span>
            </p>
          )}

          <div className="max-h-64 overflow-auto rounded border border-slate-200">
            <table className="w-full text-[11px]">
              <caption className="sr-only">Proteins of the network, sorted by {sortField}</caption>
              <thead className="sticky top-0 bg-slate-50 text-slate-500">
                <tr>
                  <SortableTh onSort={() => sortBy("gene")} sort={ariaSort("gene", sortField, sortDir)} className="px-2 py-1">Gene</SortableTh>
                  <SortableTh onSort={() => sortBy("degree")} sort={ariaSort("degree", sortField, sortDir)} title="Partners within the list" align="right" className="px-2 py-1">Partners</SortableTh>
                  <SortableTh onSort={() => sortBy("betweenness")} sort={ariaSort("betweenness", sortField, sortDir)} title="Share of shortest paths through this protein (0-1): bridges between communities score high" align="right" className="px-2 py-1">Betweenness</SortableTh>
                  <SortableTh onSort={() => sortBy("community")} sort={ariaSort("community", sortField, sortDir)} title="Leiden community; 1 is the largest. Blank: no partner" align="right" className="px-2 py-1">Community</SortableTh>
                  {hasLfc && (
                    <SortableTh onSort={() => sortBy("logfc")} sort={ariaSort("logfc", sortField, sortDir)} title="log2 fold change in the source list" align="right" className="px-2 py-1">log2FC</SortableTh>
                  )}
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, TABLE_ROWS).map((x) => (
                  <tr key={x.string_id} className="border-t border-slate-100">
                    <td className="px-2 py-0.5">
                      {onGeneClick ? (
                        <button type="button" onClick={() => onGeneClick(x.gene)} className="font-medium text-primary hover:underline" title="Show its expression on the scatter plot">
                          {x.gene}
                        </button>
                      ) : (
                        <span className="font-medium">{x.gene}</span>
                      )}
                      {x.hub && <span className="ml-1 text-[9px] uppercase text-slate-400">hub</span>}
                    </td>
                    <td className="px-2 py-0.5 text-right tabular-nums">{x.degree}</td>
                    <td className="px-2 py-0.5 text-right tabular-nums">{x.betweenness.toFixed(3)}</td>
                    <td className="px-2 py-0.5 text-right tabular-nums">{x.community ?? ""}</td>
                    {hasLfc && <td className="px-2 py-0.5 text-right tabular-nums">{x.logfc === null ? "" : x.logfc.toFixed(2)}</td>}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-start gap-2">
            <Caption n={[{ value: result.n_proteins, unit: "proteins" }, { value: result.n_links, unit: "interactions" }]} className="flex-1">
              {cap ? `${cap} ` : ""}
              {result.attribution}; {result.citation}. Score &ge; {result.score}; {result.n_draws.toLocaleString()} degree-matched random sets (seed {result.seed}).
            </Caption>
            <button type="button" onClick={exportCsv} className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-700">
              <Download className="h-3 w-3" aria-hidden /> CSV
            </button>
          </div>
        </>
      )}
    </section>
  );
}
