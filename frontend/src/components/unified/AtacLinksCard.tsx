import { useEffect, useMemo, useState } from "react";
import {
  getLatestLinks,
  getLinksEstimate,
  getLinksLocus,
  getLinksPage,
  linksCsvUrl,
  type LinkRow,
  type LinksEstimate,
  type LinksGenes,
  type LinksLocus,
  type LinksRun,
} from "@/api/atacLinks";
import type { AtacState } from "@/api/atac";
import { CapNote, Caption } from "@/components/common/Caption";
import { Help } from "@/components/common/Help";
import { HoverTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import { formatDistance, formatP } from "@/lib/atac";
import { cellsText, countsText, estimateText, linksBody, parseGeneList } from "@/lib/atacLinks";
import { errorMessage } from "@/lib/errorMessage";
import { prettyObsLabel } from "@/lib/formatting";
import { useAtacRunStore } from "@/stores/atacRunStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { AtacRunStatus } from "./AtacRunStatus";

const PAGE = 100;
const MAX_PAGE = 1000;
const SELECT = "rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700";

type SortKey = "gene" | "peak" | "abs_distance" | "abs_score" | "z" | "p" | "padj";

const HEADERS: { key: SortKey; label: string; title: string; right: boolean }[] = [
  { key: "gene", label: "Gene", title: "The gene whose expression this peak tracks", right: false },
  { key: "peak", label: "Peak", title: "The linked peak", right: false },
  { key: "abs_distance", label: "To TSS", title: "From the peak to the gene's TSS: + downstream, - upstream on its strand. Sorts by distance, either way.", right: true },
  { key: "abs_score", label: "Score", title: "Pearson r of the peak's counts with the gene's log-normalised expression across these cells. Sorts by |r|.", right: true },
  { key: "z", label: "z", title: "How far the score is from 200 matched peaks on other chromosomes, in their standard deviations", right: true },
  { key: "p", label: "p", title: "One-sided, from |z|: what Signac reports and keeps at p < 0.05", right: true },
  { key: "padj", label: "BH", title: "Benjamini-Hochberg across every tested pair, added here; Signac reports raw p", right: true },
];

/**
 * Peak-gene links (backend core/atac_links.py, Signac's LinkPeaks): the choice of genes and cells, what it costs, a
 * run with progress and Cancel, and the sortable table. A row opens that gene's locus with arcs under the coverage
 * tracks (AtacCoverage), or on a peak track alone when no fragments file is attached.
 */
export function AtacLinksCard({
  datasetId,
  state,
  columns,
  onLocus,
}: {
  datasetId: string;
  state: AtacState;
  columns: string[];
  onLocus: (locus: LinksLocus, selectedPeak: string) => void;
}) {
  const run = useAtacRunStore();
  const setStore = useAtacRunStore.setState;
  const dataset = useDatasetStore((s) => s.currentDataset);
  const live = run.datasetId === datasetId && run.running;
  const elsewhere = !live && !!state.running;
  const [genes, setGenes] = useState<LinksGenes>("variable");
  const [listText, setListText] = useState("");
  const [column, setColumn] = useState("");
  const [values, setValues] = useState<string[]>([]);
  const [estimate, setEstimate] = useState<LinksEstimate | null>(null);
  const [estimateError, setEstimateError] = useState<string | null>(null);
  const [runs, setRuns] = useState<LinksRun[]>([]);
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<SortKey>("p");
  const [descending, setDescending] = useState(false);
  const [limit, setLimit] = useState(PAGE);
  const [rows, setRows] = useState<LinkRow[] | null>(null);
  const [total, setTotal] = useState(0);
  const [stale, setStale] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const links = run.links;

  const groupValues = useMemo(() => dataset?.obs_columns.find((c) => c.name === column)?.values ?? [], [dataset, column]);
  const listed = genes === "list" ? parseGeneList(listText) : [];
  const listEmpty = genes === "list" && listed.length === 0;
  const noCells = !!column && values.length === 0;

  // The last result on these cells and every kept run, when this tab opens or a run ends.
  const version = run.version;
  useEffect(() => {
    let cancelled = false;
    getLatestLinks(datasetId)
      .then((d) => {
        if (cancelled) return;
        setRuns(d.runs);
        setStore((s) => (s.links?.id && d.runs.some((r) => r.id === s.links?.id) ? s : { links: d.result }));
      })
      .catch(() => {
        if (!cancelled) setStore({ links: null });
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, setStore, version]);

  // What this choice would need, before it runs.
  const estimateKey = JSON.stringify([genes, listed, column, values]);
  useEffect(() => {
    let cancelled = false;
    setEstimate(null);
    setEstimateError(null);
    if (listEmpty || noCells) return;
    const handle = setTimeout(() => {
      getLinksEstimate(datasetId, linksBody(genes, listText, column, values))
        .then((e) => {
          if (!cancelled) setEstimate(e);
        })
        .catch((e) => {
          if (!cancelled) setEstimateError(errorMessage(e));
        });
    }, 300);
    return () => {
      cancelled = true;
      clearTimeout(handle);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- estimateKey stands for the choices
  }, [datasetId, estimateKey, listEmpty, noCells]);

  // The page in view: the table's search and sort are the backend's, over every link.
  const linksId = links?.id;
  useEffect(() => {
    setRows(null);
    setLimit(PAGE);
    setSelected(null);
  }, [linksId]);
  useEffect(() => {
    if (!linksId) return;
    let cancelled = false;
    const handle = setTimeout(() => {
      getLinksPage(datasetId, linksId, { search, sort, descending, limit })
        .then((p) => {
          if (cancelled) return;
          setRows(p.rows);
          setTotal(p.total);
          setStale(p.stale);
        })
        .catch((e) => {
          if (!cancelled) setError(errorMessage(e));
        });
    }, search ? 250 : 0);
    return () => {
      cancelled = true;
      clearTimeout(handle);
    };
  }, [datasetId, linksId, search, sort, descending, limit]);

  const start = () => void run.runLinks(datasetId, linksBody(genes, listText, column, values));

  const open = async (row: LinkRow) => {
    if (!linksId) return;
    setSelected(`${row.gene}|${row.peak}`);
    setError(null);
    try {
      onLocus(await getLinksLocus(datasetId, linksId, row.gene), row.peak);
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const openRun = async (id: string) => {
    try {
      setStore({ links: await getLinksPage(datasetId, id, { limit: PAGE }) });
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const sortBy = (key: SortKey) => {
    if (key === sort) setDescending((d) => !d);
    else {
      setSort(key);
      setDescending(key === "abs_score" || key === "z");
    }
  };

  const table = rows ?? links?.rows ?? [];
  const shownTotal = rows ? total : links?.total ?? 0;

  return (
    <section className="space-y-2 rounded-lg border border-slate-200 p-2" aria-label="Peak-gene links">
      <h4 className="text-xs font-semibold text-slate-600">
        <HoverTip
          width={340}
          tip={<>
            Signac&rsquo;s LinkPeaks. A peak within 500 kb of a gene&rsquo;s TSS whose counts rise and fall with the gene&rsquo;s
            expression across the same cells is a candidate regulatory element for it. Score: Pearson r of the two. z: how far
            that r is from the r of 200 peaks on other chromosomes with the same GC, count and width, which cannot be regulating
            this gene in cis, in their standard deviations. Kept at one-sided p &lt; 0.05 as Signac does; BH across every tested
            pair is added. A link is a correlation: across all cells it mostly says which cell types open the peak and express
            the gene.
          </>}
        >
          Peak-gene links
        </HoverTip>
      </h4>
      <div className="flex flex-wrap items-center gap-1.5">
        <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
          <Help id="atac.links-genes">Genes</Help>
          <select aria-label="Genes" value={genes} onChange={(e) => setGenes(e.target.value as LinksGenes)} className={SELECT}>
            <option value="variable">Variable genes</option>
            <option value="all">All genes</option>
            <option value="list">A list</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
          <Help id="atac.links-cells">Cells</Help>
          <select
            aria-label="Cells from"
            value={column}
            onChange={(e) => {
              setColumn(e.target.value);
              setValues([]);
            }}
            className={SELECT}
          >
            <option value="">All cells</option>
            {columns.map((c) => <option key={c} value={c}>{prettyObsLabel(c)}</option>)}
          </select>
        </label>
        {!live && (
          <button type="button" disabled={run.running || elsewhere || listEmpty || noCells || !!estimateError} onClick={start}
                  className="rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-50">
            Find links
          </button>
        )}
      </div>
      {genes === "list" && (
        <textarea
          aria-label="Gene list"
          rows={2}
          value={listText}
          onChange={(e) => setListText(e.target.value)}
          placeholder="LYZ, MS4A1, CD3D"
          className="w-full rounded border border-slate-300 bg-white px-2 py-1 text-[11px] text-slate-700"
        />
      )}
      {column && (
        <div className="flex max-h-24 flex-wrap gap-x-2 gap-y-0.5 overflow-auto" role="group" aria-label="Groups to use">
          {groupValues.map((v) => (
            <label key={v} className="flex items-center gap-1 text-[11px] text-slate-600">
              <input
                type="checkbox"
                checked={values.includes(v)}
                onChange={(e) => setValues((cur) => (e.target.checked ? [...cur, v] : cur.filter((x) => x !== v)))}
              />
              {v}
            </label>
          ))}
        </div>
      )}
      {!live && estimate && (
        <p className="text-[11px] leading-snug text-slate-500" data-testid="atac-links-estimate">
          {estimateText(estimate)}
          {estimate.missing.length > 0 && ` Not in this dataset or without a position: ${estimate.missing.join(", ")}.`}
        </p>
      )}
      <RequestError message={estimateError} />
      <AtacRunStatus datasetId={datasetId} kind="links" />
      <RequestError message={error} />
      {runs.length > 1 && (
        <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
          <Help id="atac.links-runs">Kept runs</Help>
          <select aria-label="Kept runs" value={linksId ?? ""} onChange={(e) => void openRun(e.target.value)} className={SELECT}>
            {!linksId && <option value="">Choose a run</option>}
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.label}{r.stale ? " (older data)" : ""}
              </option>
            ))}
          </select>
        </label>
      )}
      {links && (
        <div className="space-y-1.5" data-testid="atac-links">
          {stale && (
            <p className="text-[11px] leading-snug text-amber-700">
              The cells or the RNA changed since this ran (a pipeline run, a filter or a relabel), so these links describe the
              data as it was. Run it again for current links.
            </p>
          )}
          <div className="flex flex-wrap items-center gap-1.5">
            <input
              aria-label="Search gene or peak"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setLimit(PAGE);
              }}
              placeholder="Search gene or peak"
              className="w-40 rounded border border-slate-300 bg-white px-2 py-0.5 text-[11px] text-slate-700"
            />
            <a href={linksCsvUrl(datasetId, links.id)} download className="text-[11px] text-primary hover:underline">CSV</a>
            <span className="text-[11px] text-slate-500">{cellsText(links)}</span>
          </div>
          {links.missing.length > 0 && (
            <p className="text-[11px] leading-snug text-amber-700">Not in this dataset or without a position: {links.missing.join(", ")}.</p>
          )}
          <div className="max-h-80 overflow-auto">
            <table className="w-full min-w-max border-collapse text-[11px] text-slate-600 [&_td]:px-1 [&_th]:px-1">
              <thead className="sticky top-0 bg-white text-left text-slate-500">
                <tr>
                  {HEADERS.map((h) => (
                    <th key={h.key} className={`font-medium ${h.right ? "text-right" : ""}`} title={h.title}
                        aria-sort={sort === h.key ? (descending ? "descending" : "ascending") : "none"}>
                      <button type="button" onClick={() => sortBy(h.key)} className="hover:text-primary">
                        {h.label}{sort === h.key ? (descending ? " ↓" : " ↑") : ""}
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.map((r) => (
                  <tr
                    key={`${r.gene}|${r.peak}`}
                    onClick={() => void open(r)}
                    title={`Open ${r.gene}'s links at its locus`}
                    className={`cursor-pointer hover:bg-slate-100 ${selected === `${r.gene}|${r.peak}` ? "bg-slate-100" : ""}`}
                  >
                    <td className="whitespace-nowrap">{r.gene}</td>
                    <td className="whitespace-nowrap font-mono text-[10px]">{r.peak}</td>
                    <td className="whitespace-nowrap text-right tabular-nums">{formatDistance(r.distance_to_tss)}</td>
                    <td className="text-right tabular-nums">{r.score.toFixed(3)}</td>
                    <td className="text-right tabular-nums">{r.z.toFixed(1)}</td>
                    <td className="whitespace-nowrap text-right tabular-nums">{formatP(r.p)}</td>
                    <td className="whitespace-nowrap text-right tabular-nums">{formatP(r.padj)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <CapNote shown={table.length} total={shownTotal} by="the sort above" rest="search for a gene, or the CSV has them all" />
          {table.length < shownTotal && limit < MAX_PAGE && (
            <button type="button" onClick={() => setLimit(Math.min(MAX_PAGE, limit + PAGE * 4))}
                    className="rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-700 hover:border-primary hover:text-primary">
              Show more
            </button>
          )}
          <Caption n={{ value: links.counts.n_cells, unit: "cells" }}>
            {countsText(links)} {links.caption}
          </Caption>
        </div>
      )}
    </section>
  );
}
