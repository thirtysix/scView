import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Copy, Loader2 } from "lucide-react";
import {
  computeDEOverlap,
  listDEResults,
  type DEOverlapGene,
  type DEOverlapResponse,
  type DEResultsList,
  type OverlapDirection,
} from "@/api/deOverlap";
import { AddToGeneListMenu } from "@/components/common/AddToGeneListMenu";
import { Caption, CapNote } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { UpSetPlot } from "@/components/plots/UpSetPlot";
import { rgbCss } from "@/lib/categoryColors";
import { getPalette } from "@/lib/colors";
import {
  MAX_COLUMNS,
  MAX_EXTRA_SETS,
  columnTitle,
  defaultChecked,
  directionText,
  genesText,
  lfcText,
  overlapSummary,
  runOrder,
  setTitle,
} from "@/lib/deOverlap";
import { errorText } from "@/lib/qcStates";
import { intersectionName } from "@/lib/upset";
import { useGeneListsStore } from "@/stores/geneListsStore";
import { useSettingsStore } from "@/stores/settingsStore";

interface Props {
  datasetId: string;
  /** The DE subtab's newest result id: a new one refreshes the list. */
  latestResultId?: string | null;
  /** The DE subtab's cut-offs, used until changed here. */
  pThreshold: number;
  fcThreshold: number;
  onGeneClick: (gene: string) => void;
}

const DIRECTIONS: [OverlapDirection, string, string][] = [
  ["up", "Up", "Genes higher in group A (the first group, or the selection) of each comparison"],
  ["down", "Down", "Genes higher in group B (the second group, or the rest) of each comparison"],
  [
    "both",
    "Both",
    "Genes significant either way. The sign is kept: a gene up in one result and down in another is discordant, counted apart, never as shared.",
  ],
];

const inputCls = "w-16 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs text-slate-700";

/**
 * "Compare DE results": the significant genes of several DE results the backend
 * still keeps, as an UpSet plot of their exclusive intersections (backend
 * core/de_overlap.py). Click a bar for its genes, with each result's log2FC, to
 * copy or save as a gene list.
 */
export function DeOverlap({ datasetId, latestResultId, pThreshold, fcThreshold, onGeneClick }: Props) {
  const [open, setOpen] = useState(false);
  const [list, setList] = useState<DEResultsList | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const touched = useRef(false);
  // Result ids already listed: one run since is ticked too, while there is room.
  const known = useRef<Set<string>>(new Set());
  const [direction, setDirection] = useState<OverlapDirection>("both");
  const [padjOverride, setPadj] = useState<number | null>(null);
  const [lfcOverride, setLfc] = useState<number | null>(null);
  const padj = padjOverride ?? pThreshold;
  const lfc = lfcOverride ?? fcThreshold;
  const [restrict, setRestrict] = useState(false);
  const [extraIds, setExtraIds] = useState<Set<string>>(new Set());
  const [res, setRes] = useState<DEOverlapResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const geneLists = useGeneListsStore((s) => s.lists);
  const palette = getPalette(useSettingsStore((s) => s.categoricalPalette));

  // A new dataset starts over.
  useEffect(() => {
    setList(null);
    setChecked(new Set());
    touched.current = false;
    known.current = new Set();
    setRes(null);
    setSelectedKey(null);
  }, [datasetId]);

  // The kept results, with their significant counts at the cut-offs in use.
  const [listTick, setListTick] = useState(0);
  useEffect(() => {
    let cancelled = false;
    const t = setTimeout(() => {
      listDEResults(datasetId, padj, lfc)
      .then((l) => {
        if (cancelled) return;
        setList(l);
        setListError(null);
        const ids = new Set(l.results.map((r) => r.result_id));
        const added = runOrder(l.results)
          .map((r) => r.result_id)
          .filter((id) => !known.current.has(id));
        known.current = ids;
        setChecked((prev) => {
          if (!touched.current) return new Set(defaultChecked(l.results));
          const next = [...prev].filter((id) => ids.has(id));
          for (const id of added) if (next.length < l.keep && !next.includes(id)) next.push(id);
          return new Set(next);
        });
      })
      .catch((e) => !cancelled && setListError(errorText(e)));
    }, 200);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [datasetId, latestResultId, padj, lfc, listTick]);

  const ordered = useMemo(() => runOrder(list?.results ?? []), [list]);
  // Keys, so a refreshed list or an edit to another gene list does not refetch the overlap.
  const idsKey = ordered
    .filter((r) => checked.has(r.result_id))
    .map((r) => r.result_id)
    .join(",");
  const extrasKey = useMemo(
    () => JSON.stringify(geneLists.filter((l) => extraIds.has(l.id)).map((l) => ({ name: l.name, genes: l.genes }))),
    [geneLists, extraIds],
  );
  const nPicked = idsKey ? idsKey.split(",").length : 0;

  // The overlap, whenever the section is open and two or more results are picked.
  useEffect(() => {
    const ids = idsKey ? idsKey.split(",") : [];
    if (!open || ids.length < 2) {
      setRes(null);
      return;
    }
    const extras = JSON.parse(extrasKey) as { name: string; genes: string[] }[];
    let cancelled = false;
    setLoading(true);
    const t = setTimeout(() => {
      computeDEOverlap(datasetId, {
        result_ids: ids,
        direction,
        padj_max: padj,
        lfc_min: lfc,
        restrict_to_common_tested: restrict,
        ...(extras.length ? { extra_sets: extras } : {}),
      })
        .then((r) => {
          if (cancelled) return;
          setRes(r);
          setError(null);
        })
        .catch((e) => {
          if (cancelled) return;
          setRes(null);
          setError(errorText(e));
          setListTick((n) => n + 1); // a result may have dropped out
        })
        .finally(() => !cancelled && setLoading(false));
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [open, datasetId, idsKey, direction, padj, lfc, restrict, extrasKey]);

  const toggle = (id: string) => {
    touched.current = true;
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  const toggleExtra = (id: string) =>
    setExtraIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else if (next.size < MAX_EXTRA_SETS) next.add(id);
      return next;
    });

  const sets = useMemo(() => res?.sets ?? [], [res]);
  const deSets = sets.filter((s) => s.kind === "de");
  const color = (i: number) => rgbCss(palette[i % palette.length]!);
  const shown = useMemo(() => (res ? res.intersections.slice(0, MAX_COLUMNS) : []), [res]);
  const plotSets = useMemo(
    () =>
      res
        ? res.sets.map((s, i) => ({
            name: s.name,
            n: s.n,
            color: rgbCss(palette[i % palette.length]!),
            title: setTitle(s, res),
          }))
        : [],
    [res, palette],
  );
  const plotColumns = useMemo(
    () => shown.map((x) => ({ sets: x.sets, n: x.n, title: columnTitle(x, sets) })),
    [shown, sets],
  );
  const keyOf = (s: number[]) => s.join(",");
  const selectedIndex = shown.findIndex((x) => keyOf(x.sets) === selectedKey);
  const picked: { title: string; members: number[]; n: number; genes: DEOverlapGene[]; name: string } | null =
    res && selectedKey === "discordant"
      ? {
          title: `${res.discordant.n.toLocaleString()} discordant genes: up in one result, down in another`,
          members: [],
          n: res.discordant.n,
          genes: res.discordant.genes,
          name: "discordant",
        }
      : selectedIndex >= 0
        ? {
            title: columnTitle(shown[selectedIndex]!, sets),
            members: shown[selectedIndex]!.sets,
            n: shown[selectedIndex]!.n,
            genes: shown[selectedIndex]!.genes,
            name: intersectionName(
              sets.map((s) => s.name),
              shown[selectedIndex]!.sets,
            ),
          }
        : null;

  const copy = async (genes: DEOverlapGene[]) => {
    try {
      await navigator.clipboard.writeText(genesText(genes));
      setCopied(`Copied ${genes.length.toLocaleString()} genes`);
    } catch {
      setCopied("Copy failed: the browser blocked the clipboard");
    }
    setTimeout(() => setCopied(null), 3000);
  };

  const nResults = list?.results.length ?? 0;
  const keep = list?.keep ?? 8;

  return (
    <div className="rounded-lg border border-slate-200 text-xs" data-testid="de-overlap">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full items-center gap-1.5 px-2.5 py-2 text-left font-medium text-slate-700 hover:bg-slate-50"
        title="Compare the significant genes of several DE results the backend still keeps (UpSet plot)"
      >
        {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
        Compare DE results
        <span className="font-normal text-slate-400">
          {list ? `${nResults} kept for this dataset` : ""}
        </span>
      </button>
      {open && (
        <div className="flex flex-col gap-2 border-t border-slate-100 p-2.5">
          {listError && <div className="rounded border border-red-200 bg-red-50 p-2 text-red-700">{listError}</div>}
          {list && nResults < 2 ? (
            <p className="text-slate-500" data-testid="de-overlap-empty">
              {nResults === 0
                ? "No DE results kept yet. Run a comparison above (Compute DE) at least twice, for example STIM vs CTRL within two cell types, then compare their significant genes here."
                : "One DE result kept so far. Run another comparison above (Compute DE), then compare their significant genes here."}
            </p>
          ) : (
            list && (
              <>
                <fieldset className="flex flex-col gap-0.5">
                  <legend className="mb-0.5 font-medium text-slate-500">
                    Results to compare <span className="font-normal text-slate-400">(2 to {keep})</span>
                  </legend>
                  {ordered.map((r) => (
                    <label key={r.result_id} className="flex items-center gap-1.5 text-slate-700">
                      <input
                        type="checkbox"
                        checked={checked.has(r.result_id)}
                        onChange={() => toggle(r.result_id)}
                        disabled={!checked.has(r.result_id) && checked.size >= keep}
                      />
                      <span className="min-w-0 flex-1 truncate" title={r.label}>
                        {r.label}
                      </span>
                      <span
                        className="whitespace-nowrap tabular-nums text-slate-500"
                        title={
                          r.has_pvalues
                            ? `Significant at adjusted p ≤ ${padj} and |log2FC| ≥ ${lfc}, higher in group A (up) and in group B (down), of ${r.n_tested.toLocaleString()} genes tested`
                            : `Logistic regression has no p-values: the top ${list.logreg_top_n} genes per direction by |coefficient| past |log2FC| ${lfc}, of ${r.n_tested.toLocaleString()} genes tested`
                        }
                      >
                        {r.n_sig_up.toLocaleString()} up · {r.n_sig_down.toLocaleString()} down
                        {!r.has_pvalues && " (top by coef.)"}
                      </span>
                    </label>
                  ))}
                </fieldset>

                <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
                  <div className="flex rounded border border-slate-300 bg-white p-0.5" role="group" aria-label="Direction">
                    {DIRECTIONS.map(([d, label, help]) => (
                      <button
                        key={d}
                        type="button"
                        onClick={() => setDirection(d)}
                        aria-pressed={direction === d}
                        title={help}
                        className={`rounded px-2 py-0.5 transition ${
                          direction === d ? "bg-slate-200 font-medium text-slate-800" : "text-slate-500 hover:text-slate-700"
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                  <label className="flex items-center gap-1 text-slate-500" title="Benjamini-Hochberg adjusted p-value cut-off (each result's own adjustment; DESeq2's padj for pseudobulk)">
                    adj. p ≤
                    <input
                      type="number"
                      aria-label="Adjusted p cut-off"
                      min={0.0001}
                      max={1}
                      step={0.01}
                      value={padj}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (v > 0 && v <= 1) setPadj(v);
                      }}
                      className={inputCls}
                    />
                  </label>
                  <label className="flex items-center gap-1 text-slate-500" title="Minimum absolute log2 fold change">
                    |log2FC| ≥
                    <input
                      type="number"
                      aria-label="log2FC cut-off"
                      min={0}
                      max={20}
                      step={0.25}
                      value={lfc}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (v >= 0 && v <= 20) setLfc(v);
                      }}
                      className={inputCls}
                    />
                  </label>
                  <label
                    className="flex items-center gap-1 text-slate-600"
                    title="Results can test different genes (the detection prefilter keeps genes seen in enough cells of the compared groups). On: count only genes every picked result tested, so a gene missing from a set was tested there and not significant."
                  >
                    <input type="checkbox" checked={restrict} onChange={(e) => setRestrict(e.target.checked)} />
                    only genes tested in all
                  </label>
                </div>

                {geneLists.length > 0 && (
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-slate-600">
                    <span className="text-slate-500" title={`Your gene lists (Gene lists tab), compared beside the DE results as sets without direction; up to ${MAX_EXTRA_SETS}`}>
                      Also compare:
                    </span>
                    {geneLists.map((l) => (
                      <label key={l.id} className="flex items-center gap-1">
                        <input
                          type="checkbox"
                          checked={extraIds.has(l.id)}
                          onChange={() => toggleExtra(l.id)}
                          disabled={!extraIds.has(l.id) && extraIds.size >= MAX_EXTRA_SETS}
                        />
                        <span className="max-w-[10rem] truncate" title={`${l.name} (${l.genes.length} genes)`}>
                          {l.name}
                        </span>
                      </label>
                    ))}
                  </div>
                )}

                {nPicked < 2 && <p className="text-slate-500">Tick two or more results to compare them.</p>}
                {error && <div className="rounded border border-red-200 bg-red-50 p-2 text-red-700">{error}</div>}
                {loading && !res && nPicked >= 2 && (
                  <p className="flex items-center gap-1 text-slate-400">
                    <Loader2 className="h-3 w-3 animate-spin" /> Comparing…
                  </p>
                )}

                {res && (
                  <>
                    {res.n_union === 0 ? (
                      <p className="text-slate-500">No gene passes the cut-offs in any picked result.</p>
                    ) : (
                      <div className="overflow-x-auto">
                        <UpSetPlot
                          sets={plotSets}
                          columns={plotColumns}
                          selected={selectedIndex >= 0 ? selectedIndex : null}
                          onSelect={(i) => setSelectedKey(keyOf(shown[i]!.sets))}
                          label={`UpSet plot of ${sets.length} gene sets: ${res.n_union} genes in ${res.n_intersections} exclusive intersections`}
                        />
                      </div>
                    )}
                    {res.hidden_intersections > 0 || res.n_intersections > shown.length ? (
                      <p className="text-[11px] text-slate-500">
                        The {shown.length} largest of {res.n_intersections.toLocaleString()} intersections are drawn; the
                        other {(res.n_intersections - shown.length).toLocaleString()} hold{" "}
                        {(res.n_union - res.discordant.n - shown.reduce((a, x) => a + x.n, 0)).toLocaleString()} genes.
                      </p>
                    ) : null}
                    {res.direction === "both" && res.discordant.n > 0 && (
                      <button
                        type="button"
                        onClick={() => setSelectedKey("discordant")}
                        aria-pressed={selectedKey === "discordant"}
                        className={`self-start rounded border px-2 py-0.5 ${
                          selectedKey === "discordant"
                            ? "border-amber-300 bg-amber-50 text-amber-800"
                            : "border-slate-300 bg-white text-slate-600 hover:bg-slate-50"
                        }`}
                        title="Genes significant in two or more results with opposite signs: not counted in any bar"
                      >
                        Discordant: {res.discordant.n.toLocaleString()} gene{res.discordant.n === 1 ? "" : "s"} up in
                        one result, down in another
                      </button>
                    )}
                    <Caption testId="de-overlap-caption">
                      Comparisons run since the backend started (last {keep} kept, over all datasets).{" "}
                      {overlapSummary(res)}{" "}
                      {res.restrict_to_common_tested
                        ? `Only the ${res.n_common_tested.toLocaleString()} genes every picked result tested are counted.`
                        : "A gene missing from a set may not have been tested in that result (each result tests the genes detected in enough of its cells)."}
                    </Caption>
                    {res.notes.length > 0 && (
                      <ul className="list-disc pl-4 text-[11px] leading-snug text-slate-500" data-testid="de-overlap-notes">
                        {res.notes.map((n) => (
                          <li key={n}>{n}</li>
                        ))}
                      </ul>
                    )}
                    {res.pairs.length > 0 && (
                      <details className="text-[11px] text-slate-500">
                        <summary className="cursor-pointer select-none">
                          Pairwise overlap (Jaccard){" "}
                          <InfoTip>
                            Shared genes over the genes in either set. With direction &ldquo;Both&rdquo;, two DE results
                            share a gene only when it moves the same way in both; opposite signs are counted as
                            discordant.
                          </InfoTip>
                        </summary>
                        <table className="mt-1">
                          <tbody>
                            {res.pairs.map((p) => (
                              <tr key={`${p.a}-${p.b}`}>
                                <td className="pr-2">
                                  <span className="mr-1 inline-block h-2 w-2 rounded-[2px]" style={{ backgroundColor: color(p.a) }} />
                                  {sets[p.a]?.name}
                                </td>
                                <td className="pr-2">
                                  <span className="mr-1 inline-block h-2 w-2 rounded-[2px]" style={{ backgroundColor: color(p.b) }} />
                                  {sets[p.b]?.name}
                                </td>
                                <td className="pr-2 tabular-nums" title={`${p.shared} shared of ${p.union} in either`}>
                                  {p.jaccard == null ? "–" : p.jaccard.toFixed(2)}
                                </td>
                                <td className="tabular-nums">
                                  {p.shared.toLocaleString()} shared
                                  {p.discordant > 0 ? `, ${p.discordant.toLocaleString()} discordant` : ""}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </details>
                    )}

                    {picked ? (
                      <div className="flex flex-col gap-1 rounded border border-slate-200 p-2" data-testid="de-overlap-genes">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="min-w-0 flex-1 font-medium text-slate-700">{picked.title}</span>
                          <button
                            type="button"
                            onClick={() => copy(picked.genes)}
                            className="flex items-center gap-1 rounded border border-slate-300 bg-white px-2 py-1 text-slate-600 hover:bg-slate-100"
                            title="Copy the listed genes, one per line"
                          >
                            <Copy className="h-3 w-3" /> Copy
                          </button>
                          <AddToGeneListMenu
                            label="Save as gene list"
                            genes={() => picked.genes.map((g) => g.gene)}
                            defaultName={`DE overlap: ${picked.name}`.slice(0, 200)}
                            source={`DE overlap (${directionText(res.direction)}; adjusted p ≤ ${res.padj_max}, |log2FC| ≥ ${res.lfc_min}): ${picked.name}`}
                            title="Save the listed genes as a gene list (Gene lists tab, where it can also be tested for enrichment)"
                          />
                          {copied && (
                            <span role="status" className="text-[10px] text-emerald-700">
                              {copied}
                            </span>
                          )}
                        </div>
                        <CapNote shown={picked.genes.length} total={picked.n} by="mean |log2FC|" />
                        <div className="max-h-64 overflow-auto">
                          <table className="w-full text-xs">
                            <thead className="sticky top-0 bg-slate-50">
                              <tr className="border-b border-slate-200">
                                <th className="px-2 py-1 text-left font-medium text-slate-600" title="Gene symbol; click a row to colour the scatter by its expression">
                                  Gene
                                </th>
                                {deSets.map((s) => (
                                  <th
                                    key={s.index}
                                    className="max-w-[7rem] truncate px-2 py-1 text-right font-medium text-slate-600"
                                    title={`log2 fold change in ${s.name} (group A over group B; bold where the gene is in this set; – where it was not tested)`}
                                  >
                                    <span className="mr-1 inline-block h-2 w-2 rounded-[2px]" style={{ backgroundColor: color(s.index) }} />
                                    {s.name}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {picked.genes.map((g) => (
                                <tr
                                  key={g.gene}
                                  onClick={() => onGeneClick(g.gene)}
                                  className="cursor-pointer border-b border-slate-50 hover:bg-blue-50"
                                >
                                  <td className="px-2 py-1 font-mono font-medium text-slate-800">{g.gene}</td>
                                  {deSets.map((s, k) => {
                                    const v = g.lfc[k];
                                    const member = (g.sets ?? picked.members).includes(s.index);
                                    return (
                                      <td
                                        key={s.index}
                                        className={`px-2 py-1 text-right tabular-nums ${member ? "font-semibold" : "text-slate-400"} ${
                                          member && v != null ? (v > 0 ? "text-red-600" : "text-blue-600") : ""
                                        }`}
                                        title={v == null ? "Not tested in this result" : undefined}
                                      >
                                        {lfcText(v)}
                                      </td>
                                    );
                                  })}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    ) : (
                      res.n_union > 0 && <p className="text-[11px] text-slate-400">Click a bar for its genes.</p>
                    )}
                  </>
                )}
              </>
            )
          )}
        </div>
      )}
    </div>
  );
}
