import type { PseudobulkQc } from "@/api/sampleQc";

type Rgb = readonly [number, number, number];

/** "PC1 (75%)": the PC and its share of the variance of the profiles. */
export function pcLabel(pc: number, explained: readonly number[]): string {
  const v = explained[pc - 1];
  return v === undefined ? `PC${pc}` : `PC${pc} (${v >= 0.1 ? Math.round(v * 100) : (v * 100).toFixed(1)}%)`;
}

/** A p-value in two significant figures, small ones in e-notation. */
export function fmtP(p: number | null | undefined): string {
  if (p === null || p === undefined || !isFinite(p)) return "–";
  if (p < 1e-3) return p.toExponential(0);
  return p < 0.01 ? p.toFixed(3) : p.toFixed(2);
}

/** Cell shade for an R^2 (0..1): one sequential hue, untested cells unshaded. */
export function r2Shade(r2: number | null | undefined): string | undefined {
  if (r2 === null || r2 === undefined || !isFinite(r2)) return undefined;
  const a = 0.04 + 0.5 * Math.min(1, Math.max(0, r2));
  return `rgba(37, 99, 235, ${a.toFixed(3)})`;
}

/** The covariate to colour by first: what drives PC1 when it is categorical, else
 *  the condition the profiles are split by, else the first categorical one. */
export function defaultColourBy(data: PseudobulkQc): string | null {
  const cat = data.covariates.filter((c) => c.kind === "categorical").map((c) => c.name);
  const pc1 = data.drivers.PC1?.covariate;
  if (pc1 && cat.includes(pc1)) return pc1;
  if (data.split_by && cat.includes(data.split_by)) return data.split_by;
  return cat[0] ?? data.covariates[0]?.name ?? null;
}

export interface PcaTrace {
  type: "scatter";
  mode: "text+markers";
  name?: string;
  x: number[];
  y: number[];
  text: string[];
  textposition?: "top center";
  hovertext: string[];
  hoverinfo: "text";
  showlegend: boolean;
  marker: Record<string, unknown>;
}

/** PC-a vs PC-b of the profiles, coloured by one covariate: a trace per value of a
 *  categorical one (legend can hide it), one coloured trace for a numeric one.
 *  Outliers get a red ring and their name. */
export function pcaTraces(
  data: PseudobulkQc,
  colourBy: string | null,
  colorOf: (v: string) => Rgb,
  colorscale: unknown,
  pcs: readonly [number, number] = [1, 2],
): PcaTrace[] {
  const cov = data.covariates.find((c) => c.name === colourBy);
  const [a, b] = pcs;
  const point = (i: number) => {
    const p = data.profiles[i]!;
    const v = cov ? cov.values[i] : undefined;
    return {
      x: p.pcs[a - 1] ?? 0,
      y: p.pcs[b - 1] ?? 0,
      label: p.outlier ? p.id : "",
      hover:
        `${p.id}<br>${p.n_cells.toLocaleString()} cells` +
        (cov ? `<br>${cov.name}: ${typeof v === "number" ? +v.toFixed(3) : v}` : "") +
        (p.distance !== undefined ? `<br>distance ${p.distance.toFixed(1)}${p.outlier ? " (outlier)" : ""}` : ""),
      ring: p.outlier ? { color: "#dc2626", width: 2 } : { color: "#ffffff", width: 0.5 },
    };
  };
  const all = data.profiles.map((_, i) => point(i));
  if (!cov || cov.kind === "numeric") {
    return [
      {
        type: "scatter",
        mode: "text+markers",
        x: all.map((p) => p.x),
        y: all.map((p) => p.y),
        text: all.map((p) => p.label),
        textposition: "top center",
        hovertext: all.map((p) => p.hover),
        hoverinfo: "text",
        showlegend: false,
        marker: {
          size: 10,
          color: cov ? (cov.values as number[]) : "#64748b",
          colorscale: cov ? colorscale : undefined,
          showscale: !!cov,
          colorbar: cov ? { title: { text: cov.name, font: { size: 9 } }, thickness: 8, len: 0.8 } : undefined,
          line: { color: all.map((p) => p.ring.color), width: all.map((p) => p.ring.width) },
        },
      },
    ];
  }
  const levels: string[] = [];
  cov.values.forEach((v) => {
    if (!levels.includes(String(v))) levels.push(String(v));
  });
  return levels.map((level) => {
    const idx = cov.values.map((v, i) => (String(v) === level ? i : -1)).filter((i) => i >= 0);
    const pts = idx.map((i) => all[i]!);
    const [r, g, bl] = colorOf(level);
    return {
      type: "scatter",
      mode: "text+markers",
      name: `${level} (${idx.length})`,
      x: pts.map((p) => p.x),
      y: pts.map((p) => p.y),
      text: pts.map((p) => p.label),
      textposition: "top center",
      hovertext: pts.map((p) => p.hover),
      hoverinfo: "text",
      showlegend: true,
      marker: {
        size: 10,
        color: `rgb(${r},${g},${bl})`,
        line: { color: pts.map((p) => p.ring.color), width: pts.map((p) => p.ring.width) },
      },
    };
  });
}
