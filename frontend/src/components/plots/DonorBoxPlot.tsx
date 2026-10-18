import { useMemo, useState } from "react";
import { AlertTriangle } from "lucide-react";
import type { Data, Layout } from "plotly.js";
import Plot from "@/components/plots/Plot";
import {
  donorBoxFigure,
  donorCaption,
  donorsPerCondition,
  fewDonorsCaution,
  type DonorPoint,
} from "@/lib/donorPlot";
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
}

/**
 * Box + strip with one point per donor (or sample) in each group, split by
 * condition: on one axis, or one panel per group with its own scale. The
 * caption gives the donors per condition and warns below three.
 */
export function DonorBoxPlot({
  points, groups, conditions, sampleKey, colorOf, valueLabel, format, title, plotName,
  datasetName, height = 380, defaultFacet = false, note,
}: Props) {
  const [facet, setFacet] = useState(defaultFacet);
  const nDonors = useMemo(() => donorsPerCondition(points, conditions), [points, conditions]);
  const caution = fewDonorsCaution(nDonors, conditions);
  const fig = useMemo(
    () => donorBoxFigure(points, { groups, conditions, colorOf, facet, format }),
    [points, groups, conditions, colorOf, facet, format],
  );

  return (
    <div>
      <div className="mb-1 flex items-center gap-2 text-xs">
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
      </div>
      <Plot
        data={fig.data as unknown as Data[]}
        layout={{
          autosize: true,
          height: facet ? fig.height : height,
          margin: facet ? { t: 40, r: 16, b: 30, l: 48 } : { t: title ? 30 : 10, r: 16, b: 110, l: 56 },
          ...(title ? { title: { text: title, font: { size: 13, color: "#334155" } } } : {}),
          yaxis: {
            title: facet ? undefined : { text: valueLabel, font: { size: 11, color: "#64748b" } },
            gridcolor: "#f1f5f9",
            tickfont: { size: 10 },
          },
          legend: { orientation: "h", y: facet ? 1.08 : 1.12, font: { size: 11 } },
          paper_bgcolor: "white",
          plot_bgcolor: "white",
          ...fig.layout,
        } as Partial<Layout>}
        config={plotConfig(plotName, datasetName)}
        useResizeHandler
        style={{ width: "100%" }}
      />
      <p className="mt-1 text-xs text-slate-500" data-testid="donor-caption">
        One point per {sampleKey} ({donorCaption(nDonors, conditions)}).
        {facet && ` Y axis: ${valueLabel}.`}
        {note ? ` ${note}` : ""}
      </p>
      {caution && (
        <p role="alert" className="mt-1 flex gap-1.5 text-xs text-amber-800">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
          {caution}
        </p>
      )}
    </div>
  );
}
