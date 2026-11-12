import { useMemo, useState, type ReactNode } from "react";
import { AlertTriangle } from "lucide-react";
import type { Data, Layout } from "plotly.js";
import { HoverTip } from "@/components/common/InfoTip";
import Plot from "@/components/plots/Plot";
import {
  donorBoxFigure,
  donorCaption,
  donorSlopeFigure,
  donorsPerCondition,
  fewDonorsCaution,
  formatChange,
  pairedCaption,
  type DonorPoint,
  type PairedDesign,
  type PairedTest,
  type UnpairedTest,
} from "@/lib/donorPlot";
import { formatPValue } from "@/lib/formatting";
import { plotConfig } from "@/lib/plotConfig";

interface Props {
  points: readonly DonorPoint[];
  groups: readonly string[];
  /** [""] when not split by a condition. */
  conditions: readonly string[];
  sampleKey: string;
  colorOf: (category: string, kind: "condition" | "group") => string;
  /** Y axis title. */
  valueLabel: string;
  format: (v: number) => string;
  title?: string;
  /** Plot name for the PNG/SVG download. */
  plotName: string;
  datasetName?: string;
  /** Height of the one-axis plot. */
  height?: number;
  /** Start with one panel per group. */
  defaultFacet?: boolean;
  /** Text after the n-donors caption. */
  note?: string;
  /** How to read the plot: shown on hovering the caption. */
  captionTip?: ReactNode;
  /** Whether the donors recur across conditions; paired designs get a slope chart. */
  design?: PairedDesign | null;
  /** Group -> paired test (two conditions); printed under the slope chart. */
  pairedTests?: Readonly<Record<string, PairedTest | null | undefined>>;
  /** Group -> unpaired donor-level test (different donors in the two conditions); under the boxes. */
  unpairedTests?: Readonly<Record<string, UnpairedTest | null | undefined>>;
  /** Unit of a change between conditions, e.g. "log2 CPM" or "percentage points". */
  unit?: string;
  /** Formats a change (default: `format`; a per-cent formatter would double the unit). */
  changeFormat?: (v: number) => string;
  /** Says what the pseudobulk value is, for the mean-change tooltip. */
  unitHelp?: string;
}

const TEST_HELP = {
  wilcoxon:
    "Wilcoxon signed-rank test, two-sided: ranks each donor's change between the two conditions by " +
    "size and asks whether the positive and negative changes balance. Makes no assumption about the " +
    "shape of the changes; exact for small n without ties (scipy). With n donors it cannot go below " +
    "p = 2 / 2^n, so 8 donors all moving the same way give p = 0.0078 at best.",
  ttest:
    "Paired t-test, two-sided: tests whether the mean of the per-donor changes is zero, assuming the " +
    "changes are roughly normal. More powerful than the signed-rank test when they are, misleading " +
    "with an outlying donor.",
  change: "Mean over the paired donors of (second condition minus first), in the plotted units.",
  pairs: "Donors with a value in both conditions in this group (a donor with too few cells in one is left out).",
  ranksum:
    "Wilcoxon rank-sum (Mann-Whitney U) test, two-sided, on one value per donor: do the donors of one " +
    "condition rank above the other's? Exact for small n without ties (scipy). With n and m donors it " +
    "cannot go below p = 2 / C(n + m, n): 7 vs 6 donors give 0.0012 at best, however far apart.",
  welch:
    "Welch t-test, two-sided, on one value per donor: compares the two conditions' means without " +
    "assuming equal variances. Rests on roughly normal donor values; one outlying donor moves it.",
  diff: "Second condition's mean over its donors minus the first's, in the plotted units.",
};

/**
 * One point per donor (or sample) in each group, split by condition. Unpaired:
 * box + strip on one axis, or one panel per group. Paired (the same donors in
 * more than one condition): a slope chart per group with a line per donor and
 * the mean per condition, and with two conditions the signed-rank and paired
 * t-test per group; boxes stay a toggle away. The caption gives the donors
 * per condition and warns below three.
 */
export function DonorBoxPlot({
  points, groups, conditions, sampleKey, colorOf, valueLabel, format, title, plotName,
  datasetName, height = 380, defaultFacet = false, note, captionTip, design, pairedTests, unpairedTests, unit = "", unitHelp,
  changeFormat = format,
}: Props) {
  const [facet, setFacet] = useState(defaultFacet);
  const [lines, setLines] = useState(true);
  const paired = Boolean(design?.paired) && conditions.length > 1;
  const slope = paired && lines;
  const two = conditions.length === 2;
  const nDonors = useMemo(() => donorsPerCondition(points, conditions), [points, conditions]);
  const caution = fewDonorsCaution(nDonors, conditions);
  const fig = useMemo(
    () =>
      slope
        ? donorSlopeFigure(points, { groups, conditions, colorOf, format, tests: two ? pairedTests : undefined })
        : donorBoxFigure(points, { groups, conditions, colorOf, facet, format }),
    [slope, points, groups, conditions, colorOf, facet, format, two, pairedTests],
  );
  const tested = useMemo(
    () => (slope && two ? groups.filter((g) => pairedTests?.[g] && points.some((p) => p.group === g)) : []),
    [slope, two, groups, pairedTests, points],
  );
  const faceted = slope || facet;
  const unpairedTested = useMemo(
    () => (!paired && two ? groups.filter((g) => unpairedTests?.[g] && points.some((p) => p.group === g)) : []),
    [paired, two, groups, unpairedTests, points],
  );

  return (
    <div>
      <div className="mb-1 flex items-center gap-2 text-xs">
        {paired && (
          <div className="flex overflow-hidden rounded-md border border-slate-300" role="group" aria-label="Plot mode">
            {([true, false] as const).map((on) => (
              <button
                key={String(on)}
                aria-pressed={lines === on}
                onClick={() => setLines(on)}
                title={on ? "A line per donor joining its values across the conditions" : "A box per condition, every donor a point"}
                className={`px-2 py-0.5 ${lines === on ? "bg-primary/10 font-medium text-primary" : "bg-white text-slate-600 hover:bg-slate-50"}`}
              >
                {on ? "Paired lines" : "Boxes"}
              </button>
            ))}
          </div>
        )}
        {!slope && (
          <div className="ml-auto flex overflow-hidden rounded-md border border-slate-300" role="group" aria-label="Layout">
            {([false, true] as const).map((on) => (
              <button
                key={String(on)}
                aria-pressed={facet === on}
                onClick={() => setFacet(on)}
                title={on ? "One small plot per group, each on its own scale" : "Every group on one axis"}
                className={`px-2 py-0.5 ${facet === on ? "bg-primary/10 font-medium text-primary" : "bg-white text-slate-600 hover:bg-slate-50"}`}
              >
                {on ? "Panel per group" : "One axis"}
              </button>
            ))}
          </div>
        )}
      </div>
      <Plot
        data={fig.data as unknown as Data[]}
        layout={{
          autosize: true,
          height: faceted ? fig.height : height,
          margin: faceted ? { t: 40, r: 16, b: 30, l: 48 } : { t: title ? 30 : 10, r: 16, b: 110, l: 56 },
          ...(title ? { title: { text: title, font: { size: 13, color: "#334155" } } } : {}),
          yaxis: {
            title: faceted ? undefined : { text: valueLabel, font: { size: 11, color: "#64748b" } },
            gridcolor: "#f1f5f9",
            tickfont: { size: 10 },
          },
          legend: { orientation: "h", y: faceted ? 1.08 : 1.12, font: { size: 11 } },
          paper_bgcolor: "white",
          plot_bgcolor: "white",
          ...fig.layout,
        } as Partial<Layout>}
        config={plotConfig(plotName, datasetName)}
        useResizeHandler
        style={{ width: "100%" }}
      />
      <p className="mt-1 text-xs text-slate-500" data-testid="donor-caption">
        <HoverTip tip={captionTip} disabled={!captionTip}>
        {slope ? (
          <>
            One line per {sampleKey} ({pairedCaption(points, conditions)}); the heavier marker is the mean per condition.
            {two
              ? " Each group's title carries its Wilcoxon signed-rank p; the table below has both tests."
              : ` ${conditions.length} conditions: the lines follow each ${sampleKey} across them, and there is no test (a paired test compares two conditions).`}
          </>
        ) : (
          <>One point per {sampleKey} ({donorCaption(nDonors, conditions)}).</>
        )}
        {faceted && ` Y axis: ${valueLabel}.`}
        </HoverTip>
        {note ? ` ${note}` : ""}
      </p>
      {slope && two && tested.length > 0 && (
        <table className="mt-1.5 text-[11px] text-slate-600" data-testid="paired-tests">
          <thead>
            <tr className="text-left text-slate-500">
              <th className="pr-3 font-medium">Group</th>
              <th className="pr-3 text-right font-medium" title={TEST_HELP.pairs}>
                n pairs
              </th>
              <th className="pr-3 text-right font-medium" title={unitHelp ? `${TEST_HELP.change} ${unitHelp}` : TEST_HELP.change}>
                Mean change
              </th>
              <th className="pr-3 text-right font-medium">
                <HoverTip tip={TEST_HELP.wilcoxon} align="right">Wilcoxon signed-rank p</HoverTip>
              </th>
              <th className="text-right font-medium">
                <HoverTip tip={TEST_HELP.ttest} align="right">Paired t p</HoverTip>
              </th>
            </tr>
          </thead>
          <tbody>
            {tested.map((g) => {
              const t = pairedTests![g]!;
              return (
                <tr key={g} className="border-t border-slate-100">
                  <td className="pr-3 py-0.5 text-slate-800">{g}</td>
                  <td className="pr-3 text-right tabular-nums">{t.n_pairs}</td>
                  <td className="pr-3 text-right tabular-nums">{formatChange(t.mean_diff, changeFormat, unit)}</td>
                  <td className="pr-3 text-right tabular-nums" title={t.caution ?? undefined}>
                    {t.wilcoxon_p == null ? "–" : formatPValue(t.wilcoxon_p)}
                    {t.caution && t.wilcoxon_p != null ? "*" : ""}
                  </td>
                  <td className="text-right tabular-nums" title={t.caution ?? undefined}>
                    {t.ttest_p == null ? "–" : formatPValue(t.ttest_p)}
                    {t.caution && t.ttest_p != null ? "*" : ""}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {slope && two && tested.some((g) => pairedTests![g]!.caution) && (
        <p className="mt-1 text-xs text-slate-500" data-testid="paired-caution">
          * Under 5 pairs: read as descriptive. An exact signed-rank test on n pairs cannot go below
          p = 2 / 2^n two-sided, and a paired t on so few differences rests on their normality. A dash
          is a group with under 2 pairs or no change.
        </p>
      )}
      {unpairedTested.length > 0 && (
        <table className="mt-1.5 text-[11px] text-slate-600" data-testid="unpaired-tests">
          <thead>
            <tr className="text-left text-slate-500">
              <th className="pr-3 font-medium">Group</th>
              <th className="pr-3 text-right font-medium" title="Donors with a value in each condition here">
                n ({conditions[0]} vs {conditions[1]})
              </th>
              <th className="pr-3 text-right font-medium" title={unitHelp ? `${TEST_HELP.diff} ${unitHelp}` : TEST_HELP.diff}>
                Difference ({conditions[1]} − {conditions[0]})
              </th>
              <th className="pr-3 text-right font-medium">
                <HoverTip tip={TEST_HELP.ranksum} align="right">Rank-sum p</HoverTip>
              </th>
              <th className="text-right font-medium">
                <HoverTip tip={TEST_HELP.welch} align="right">Welch t p</HoverTip>
              </th>
            </tr>
          </thead>
          <tbody>
            {unpairedTested.map((g) => {
              const t = unpairedTests![g]!;
              return (
                <tr key={g} className="border-t border-slate-100">
                  <td className="pr-3 py-0.5 text-slate-800">{g}</td>
                  <td className="pr-3 text-right tabular-nums">{t.n_a} vs {t.n_b}</td>
                  <td className="pr-3 text-right tabular-nums">{formatChange(t.mean_diff, changeFormat, unit)}</td>
                  <td className="pr-3 text-right tabular-nums" title={t.caution ?? t.method ?? undefined}>
                    {t.mannwhitney_p == null ? "–" : formatPValue(t.mannwhitney_p)}
                    {t.caution && t.mannwhitney_p != null ? "*" : ""}
                  </td>
                  <td className="text-right tabular-nums" title={t.caution ?? undefined}>
                    {t.welch_p == null ? "–" : formatPValue(t.welch_p)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {unpairedTested.length > 0 && (
        <p className="mt-1 text-xs text-slate-500" data-testid="unpaired-caption">
          Different {sampleKey} values in {conditions[0]} and {conditions[1]}: each group compares one value per{" "}
          {sampleKey} ({conditions[1]} minus {conditions[0]}). A dash is a group with under 3 a side.
          {unpairedTested.some((g) => unpairedTests![g]!.caution && unpairedTests![g]!.mannwhitney_p != null) &&
            " * At the floor of the exact rank-sum test for these numbers of donors: a smaller p needs more donors."}
        </p>
      )}
      {caution && (
        <p role="alert" className="mt-1 flex gap-1.5 text-xs text-amber-800">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
          {caution}
        </p>
      )}
    </div>
  );
}
