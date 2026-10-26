import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { Data, Layout } from "plotly.js";

const { seen } = vi.hoisted(() => ({ seen: [] as Record<string, unknown>[] }));
// Plotly needs a real canvas: capture what the plot hands react-plotly.
vi.mock("react-plotly.js/factory", () => ({
  default: () => (props: Record<string, unknown>) => {
    seen.push(props);
    return <div data-testid="plotly" />;
  },
}));
vi.mock("@/lib/plotly", () => ({ default: {} }));
vi.mock("@/lib/plotConfig", () => ({ plotConfig: () => ({}) }));

import { labelSpots, VolcanoPlot, volcanoLabelGenes } from "./VolcanoPlot";
import { useThemeStore } from "@/lib/theme";
import type { DEGene } from "@/api/de";

const gene = (name: string, lfc: number, padj: number): DEGene =>
  ({ gene: name, logfoldchange: lfc, pval_adj: padj }) as DEGene;
const genes = [
  gene("UP1", 3, 1e-50),
  gene("UP2", 2, 1e-10),
  gene("DN1", -4, 1e-30),
  gene("FLAT", 0.1, 0.9),
  ...Array.from({ length: 30 }, (_, i) => gene(`UPX${i}`, 1.5, 1e-3)),
];
const last = () => seen[seen.length - 1]!;
const names = () => ((last().layout as Partial<Layout>).annotations ?? []).map((a) => a.text);

beforeEach(() => {
  seen.length = 0;
  useThemeStore.setState({ preference: "light", systemDark: false });
});

describe("volcanoLabelGenes", () => {
  const yOf = (g: DEGene) => -Math.log10(g.pval_adj ?? 1);
  it("takes the most significant up and down genes together", () => {
    const up = [genes[0]!, genes[1]!];
    const down = [genes[2]!];
    expect(volcanoLabelGenes(up, down, 2, yOf).map((g) => g.gene)).toEqual(["UP1", "DN1"]);
    expect(volcanoLabelGenes(up, down, 0, yOf)).toEqual([]);
  });
  it("splits the names between the two sides, the rest from the longer side", () => {
    const up = Array.from({ length: 8 }, (_, i) => gene(`U${i}`, 5, 1e-300));
    const down = Array.from({ length: 8 }, (_, i) => gene(`D${i}`, -1.5, 1e-20 * (i + 1)));
    const picked = volcanoLabelGenes(up, down, 10, yOf).map((g) => g.gene);
    expect(picked.filter((n) => n.startsWith("U"))).toHaveLength(5);
    expect(picked.filter((n) => n.startsWith("D"))).toHaveLength(5);
    expect(volcanoLabelGenes(up, down.slice(0, 2), 10, yOf)).toHaveLength(10);
    expect(volcanoLabelGenes(up.slice(0, 1), [], 10, yOf)).toHaveLength(1);
  });

  it("breaks ties by the size of the fold change", () => {
    const a = gene("A", 1, 1e-5);
    const b = gene("B", 3, 1e-5);
    expect(volcanoLabelGenes([a, b], [], 1, yOf)[0]!.gene).toBe("B");
  });
});

describe("labelSpots", () => {
  it("stacks names that share one point at least a line apart, on the inner side near the edge", () => {
    const pts = [{ x: 5, y: 300 }, { x: 6, y: 300 }, { x: 7, y: 300 }, { x: -3, y: 100 }, { x: 19, y: 10 }];
    const s = labelSpots(pts, [-20, 20], [0, 300], 296);
    const right = [0, 1, 2].map((i) => s[i]!.ay);
    expect(right[1]! - right[0]!).toBeGreaterThanOrEqual(13);
    expect(right[2]! - right[1]!).toBeGreaterThanOrEqual(13);
    expect(s[0]!.xanchor).toBe("left");
    expect(s[3]!.xanchor).toBe("right");
    expect(s[4]!.xanchor).toBe("right"); // x = 19 of 20: inwards
  });
});

describe("VolcanoPlot", () => {
  it("names the top 10 significant genes by default, never a non-significant one", () => {
    render(<VolcanoPlot genes={genes} fcThreshold={1} pThreshold={0.05} />);
    expect(names()).toHaveLength(10);
    expect(names().slice(0, 3)).toEqual(["UP1", "DN1", "UP2"]);
    expect(names()).not.toContain("FLAT");
  });

  it("follows the Name genes choice", () => {
    render(<VolcanoPlot genes={genes} fcThreshold={1} pThreshold={0.05} />);
    const select = screen.getByLabelText(/name genes/i);
    fireEvent.change(select, { target: { value: "0" } });
    expect(names()).toHaveLength(0);
    fireEvent.change(select, { target: { value: "20" } });
    expect(names()).toHaveLength(20);
  });

  it("uses a darker grey for non-significant genes in dark mode", () => {
    const grey = () => ((last().data as Data[])[0] as { marker: { color: string } }).marker.color;
    render(<VolcanoPlot genes={genes} fcThreshold={1} pThreshold={0.05} />);
    const light = grey();
    seen.length = 0;
    useThemeStore.setState({ preference: "dark" });
    render(<VolcanoPlot genes={genes} fcThreshold={1} pThreshold={0.05} />);
    expect(grey()).not.toBe(light);
  });
});
