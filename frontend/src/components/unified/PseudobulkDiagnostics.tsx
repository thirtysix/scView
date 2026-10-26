import Plot from "@/components/plots/Plot";
import { InfoTip } from "@/components/common/InfoTip";
import { Caption } from "@/components/common/Caption";
import { plotConfig } from "@/lib/plotConfig";
import { formatR, PB_METHODS, RUV_CHOICES } from "@/lib/pseudobulk";
import type { PseudobulkHetero, PseudobulkMethod, PseudobulkRuv } from "@/api/de";

const FILL_STYLE = { width: "100%", height: "100%" };

/** The pseudobulk model and RUV factor pickers (DE subtab, between groups). */
export function PseudobulkModelRows({
  pbMethod,
  ruvK,
  onChange,
  selectCls,
}: {
  pbMethod: PseudobulkMethod;
  ruvK: number;
  onChange: (patch: { pbMethod?: PseudobulkMethod; ruvK?: number }) => void;
  selectCls: string;
}) {
  return (
    <>
      <div className="flex items-center gap-1.5">
        <label htmlFor="pb-model" className="w-12 font-medium text-slate-500">
          Model
        </label>
        <select
          id="pb-model"
          value={pbMethod}
          onChange={(e) => onChange({ pbMethod: e.target.value as PseudobulkMethod })}
          className={selectCls}
        >
          {PB_METHODS.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
        <InfoTip label="Which model to use" width={300} align="right">
          <span className="block space-y-1">
            {PB_METHODS.map((m) => (
              <span key={m.value} className="block">
                <strong>{m.label}</strong>: {m.when}
              </span>
            ))}
          </span>
        </InfoTip>
      </div>
      <div className="flex items-center gap-1.5">
        <label htmlFor="pb-ruv" className="w-12 font-medium text-slate-500">
          RUV
        </label>
        <select
          id="pb-ruv"
          value={ruvK}
          onChange={(e) => onChange({ ruvK: Number(e.target.value) })}
          className={selectCls}
        >
          {RUV_CHOICES.map((k) => (
            <option key={k} value={k}>
              {k === 0 ? "None" : `${k} factor${k === 1 ? "" : "s"}`}
            </option>
          ))}
        </select>
        <InfoTip label="About RUV factors" width={300} align="right">
          Remove Unwanted Variation: adds k factors, learned from genes that do not respond to the
          condition, to the design. Use 1 when samples differ for technical reasons (depth, batch,
          handling); each factor costs a degree of freedom, so keep k small with few donors.
        </InfoTip>
      </div>
    </>
  );
}

/** Each group's voom trend and the variance check (flag: try voomByGroup). */
function VarianceCheck({ h }: { h: PseudobulkHetero }) {
  const traces = h.groups.map((g) => ({
    type: "scatter" as const,
    mode: "lines" as const,
    name: `${g.group} (${g.n_samples})`,
    x: g.trend.map((p) => p[0]),
    y: g.trend.map((p) => p[1]),
    hovertemplate: `${g.group}<br>mean log2 count %{x:.2f}<br>sqrt SD %{y:.3f}<extra></extra>`,
  }));
  return (
    <div className="space-y-1" data-testid="pb-variance-check">
      <p
        className={`flex items-start gap-1 text-[11px] leading-snug ${
          h.flag ? "rounded border border-amber-200 bg-amber-50 px-1.5 py-1 text-amber-800" : "text-slate-600"
        }`}
      >
        <span>{h.message}</span>
        <InfoTip label="About the variance check" width={280}>
          voom fits how a gene&apos;s variability falls as its count rises (the trend), and weights each
          value by it. Here each group is fitted on its own samples. When one group&apos;s donors vary
          about {h.threshold}x as much as the other&apos;s or more, one shared trend understates that
          group&apos;s noise and overstates the other&apos;s; voomByGroup gives each group its own.
        </InfoTip>
      </p>
      <div className="h-[170px]">
        <Plot
          data={traces}
          layout={{
            autosize: true,
            margin: { l: 44, r: 8, t: 6, b: 34 },
            xaxis: { title: { text: "mean log2 count", font: { size: 10 } }, tickfont: { size: 9 } },
            yaxis: { title: { text: "sqrt(SD)", font: { size: 10 } }, tickfont: { size: 9 } },
            showlegend: true,
            legend: { font: { size: 9 }, orientation: "h", y: 1.0, yanchor: "bottom" },
          }}
          config={plotConfig("pseudobulk_voom_trends")}
          style={FILL_STYLE}
          useResizeHandler
        />
      </div>
      <Caption n={h.groups.map((g) => ({ value: g.n_samples, unit: `${g.group} samples` }))}>
        voom trend per group: square root of each gene&apos;s residual SD against its mean log2 count,
        each group fitted on its own samples.
      </Caption>
    </div>
  );
}

/** The RUV factors and how they line up with the samples' covariates. */
function RuvFactors({ ruv }: { ruv: PseudobulkRuv }) {
  return (
    <div className="space-y-1" data-testid="pb-ruv">
      <p className="flex items-start gap-1 text-[11px] leading-snug text-slate-600">
        <span>
          RUV: {ruv.k} unwanted factor{ruv.k === 1 ? "" : "s"} from {ruv.control_source}, added to the
          design.
        </span>
        <InfoTip label="About RUV" width={280}>
          Remove Unwanted Variation (RUVg): genes that should not respond to the condition still vary
          between samples; their shared pattern (a factor per sample) is treated as technical and
          adjusted for. A factor that tracks depth or cell count is the usual, harmless case; one that
          tracks the condition removes real signal.
        </InfoTip>
      </p>
      {ruv.warning && (
        <p className="rounded border border-amber-200 bg-amber-50 px-1.5 py-1 text-[11px] text-amber-800">
          {ruv.warning}
        </p>
      )}
      <table className="w-full text-[11px]">
        <thead>
          <tr className="text-left text-slate-500">
            <th className="font-medium">Factor</th>
            <th className="font-medium" title="Pearson r of the factor with the condition (A = 1, B = 0)">
              r condition
            </th>
            <th className="font-medium" title="Pearson r with log10 of each sample's summed counts">
              r depth
            </th>
            <th className="font-medium" title="Pearson r with each sample's number of cells">
              r cells
            </th>
            <th
              className="font-medium"
              title="Share of the factor's variance between donors (paired designs; the donor effect is taken out first, so near 0)"
            >
              donor R²
            </th>
          </tr>
        </thead>
        <tbody>
          {ruv.factors.map((f) => (
            <tr key={f.name} className="text-slate-700">
              <td>{f.name}</td>
              <td>{formatR(f.r_condition)}</td>
              <td>{formatR(f.r_log_depth)}</td>
              <td>{formatR(f.r_n_cells)}</td>
              <td>{formatR(f.r2_donor)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <details className="text-[11px] text-slate-600">
        <summary className="cursor-pointer text-slate-500">Factor values per sample</summary>
        <table className="mt-1 w-full">
          <thead>
            <tr className="text-left text-slate-500">
              <th className="font-medium">Donor</th>
              <th className="font-medium">Group</th>
              <th className="font-medium">Cells</th>
              <th className="font-medium" title="log10 of the sample's summed counts">log10 depth</th>
              {ruv.factors.map((f) => (
                <th key={f.name} className="font-medium">{f.name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ruv.samples.map((s) => (
              <tr key={`${s.donor}\u001f${s.group}`}>
                <td>{s.donor}</td>
                <td>{s.group}</td>
                <td>{s.n_cells.toLocaleString()}</td>
                <td>{s.log10_depth.toFixed(2)}</td>
                {s.W.map((w, j) => (
                  <td key={j}>{w.toFixed(3)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </details>
      <Caption
        n={[
          { value: ruv.samples.length, unit: "samples" },
          { value: ruv.n_controls, unit: "control genes" },
        ]}
      />
    </div>
  );
}

/** Below the pseudobulk model choice: the per-group variance check of the last
 *  run and its RUV factors (backend core/pseudobulk.py). */
export function PseudobulkDiagnostics({
  hetero,
  ruv,
}: {
  hetero?: PseudobulkHetero | null;
  ruv?: PseudobulkRuv | null;
}) {
  if (!hetero && !ruv) return null;
  return (
    <div className="space-y-2 rounded border border-slate-200 bg-slate-50/50 p-1.5">
      {hetero && <VarianceCheck h={hetero} />}
      {ruv && <RuvFactors ruv={ruv} />}
    </div>
  );
}
