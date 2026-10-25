import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { PpiResult } from "@/api/ppi";

const { getDatasetPpi } = vi.hoisted(() => ({ getDatasetPpi: vi.fn() }));
vi.mock("@/api/ppi", async (orig) => ({
  ...(await orig<typeof import("@/api/ppi")>()),
  getDatasetPpi,
}));

import { ProteinNetwork } from "./ProteinNetwork";

globalThis.ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} } as unknown as typeof ResizeObserver;

const node = (gene: string, degree: number, community: number | null, logfc: number | null, hub = false) => ({
  gene, name: gene, string_id: `9606.${gene}`, degree, betweenness: degree / 10, community, logfc,
  x: degree ? degree : null, y: degree ? 1 : null, hub,
});

const RESULT: PpiResult = {
  species: "human", score: 400, version: "12.0", attribution: "STRING v12.0 physical links (string-db.org, CC BY 4.0)",
  citation: "Szklarczyk et al. 2023", n_given: 4, n_used: 4, max_genes: 500, truncated: false, n_mapped: 3,
  unmapped: ["NOTAGENE"], n_unmapped: 1, n_proteins: 3, n_links: 1, expected_links: 0.12, p_value: 1 / 1001,
  n_draws: 1000, seed: 0, n_isolated: 1, n_communities: 1,
  nodes: [node("STAT1", 1, 1, 2.5, true), node("IRF7", 1, 1, 3.1, true), node("CD14", 0, null, -1)],
  links: [{ source: 0, target: 1, score: 990 }],
  hubs: ["STAT1", "IRF7"], method: "", label: "CD14 Mono: STIM vs CTRL (Wilcoxon), up genes",
  source: "de:r1", direction: "up",
};

describe("ProteinNetwork", () => {
  beforeEach(() => {
    getDatasetPpi.mockReset().mockResolvedValue(RESULT);
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  });

  it("builds the network on request and lists every protein in a keyboard table", async () => {
    const onGene = vi.fn();
    render(<ProteinNetwork datasetId="d1" sources={[{ value: "de:r1", label: "STIM" }]} showDirection onGeneClick={onGene} />);
    expect(getDatasetPpi).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Minimum score"), { target: { value: "700" } });
    fireEvent.change(screen.getByLabelText("Genes"), { target: { value: "both" } });
    fireEvent.click(screen.getByRole("button", { name: /build network/i }));
    await waitFor(() => screen.getByTestId("ppi-summary"));
    expect(getDatasetPpi).toHaveBeenCalledWith("d1", "de:r1", { n: 100, direction: "both", score: 700 });
    expect(screen.getByTestId("ppi-summary").textContent).toBe(
      "CD14 Mono: STIM vs CTRL (Wilcoxon), up genes: 3 proteins, 1 interaction (0.1 expected by chance), p < 0.001; 1 community");
    expect(screen.getByText(/1 gene not in STRING \(NOTAGENE\)/)).toBeTruthy();
    expect(screen.getByText(/STRING v12.0 physical links/)).toBeTruthy();
    const table = screen.getByRole("table");
    const rows = within(table).getAllByRole("row");
    expect(rows).toHaveLength(4);
    fireEvent.click(within(table).getByRole("button", { name: "STAT1" }));
    expect(onGene).toHaveBeenCalledWith("STAT1");
    // DE lists are coloured by fold change to begin with.
    expect((screen.getByLabelText("log2FC") as HTMLInputElement).checked).toBe(true);
  });

  it("sorts the table from its headers", async () => {
    render(<ProteinNetwork datasetId="d1" sources={[{ value: "de:r1", label: "STIM" }]} showDirection />);
    fireEvent.click(screen.getByRole("button", { name: /build network/i }));
    await waitFor(() => screen.getByRole("table"));
    fireEvent.click(screen.getByRole("button", { name: "log2FC" }));
    const first = within(screen.getByRole("table")).getAllByRole("row")[1]!;
    expect(first.textContent).toContain("IRF7");
  });

  it("chooses among several lists and shows the server's sentence on failure", async () => {
    getDatasetPpi.mockRejectedValueOnce(new Error('API error 503: {"detail":"STRING is not downloaded yet"}'));
    render(<ProteinNetwork datasetId="d1" sources={[{ value: "program:1", label: "P1" }, { value: "program:2", label: "P2" }]} />);
    expect(screen.queryByLabelText("Genes")).toBeNull();
    fireEvent.change(screen.getByLabelText("List"), { target: { value: "program:2" } });
    fireEvent.click(screen.getByRole("button", { name: /build network/i }));
    await waitFor(() => screen.getByRole("alert"));
    expect(screen.getByRole("alert").textContent).toBe("STRING is not downloaded yet");
    expect(getDatasetPpi).toHaveBeenCalledWith("d1", "program:2", { n: 100, direction: "up", score: 400 });
  });
});
