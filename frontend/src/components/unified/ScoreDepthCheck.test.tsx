import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";

const client = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("@/api/client", () => client);

import { ScoreDepthCheck } from "./ScoreDepthCheck";

const c = (within: number | null, overall: number) => ({ within, overall, n_groups: 11 });
const CHECK = {
  n_cells: 13836, groupby: "scview_leiden_r0.5", n_groups: 11, min_group_cells: 20,
  depth: { total_counts: "obs['total_counts']", n_genes: "genes detected per cell in the log-normalised .raw" },
  metric: "n_genes",
  methods: {
    score_genes: { total_counts: c(-0.17, -0.19), n_genes: c(-0.43, -0.33) },
    aucell: { total_counts: c(0.04, -0.15), n_genes: c(0.06, -0.06) },
    gsva: { total_counts: c(0.02, 0.01), n_genes: c(0.03, 0.02) },
  },
  between: c(0.35, 0.54),
  agreement: { aucell: c(0.35, 0.54), gsva: c(0.61, 0.7) },
  skipped: { ssgsea: "ssGSEA scores up to 300 million non-zero values in one request" },
  max_rank: 703, share_below_max_rank: 0.637, genes_found: ["MS4A1"], genes_missing: [],
  sentence: "score_genes follows depth on this data (rho -0.43 with genes detected, within clusters); AUCell does not follow depth (rho 0.06).",
};

// Braces: a function returned from beforeEach runs as its teardown.
beforeEach(() => {
  client.apiFetch.mockReset();
});

describe("ScoreDepthCheck", () => {
  it("runs on request and shows the sentence, each method's rhos and the ranking caveat", async () => {
    client.apiFetch.mockResolvedValue(CHECK);
    render(<ScoreDepthCheck datasetId="kang" genes={["MS4A1", "CD79A"]} />);
    expect(client.apiFetch).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Check depth" }));
    expect(await screen.findByTestId("depth-check-sentence")).toHaveTextContent(
      `${CHECK.sentence} n = 13,836 cells.`,
    );
    expect(client.apiFetch).toHaveBeenCalledWith("/datasets/kang/genesets/score/compare", expect.objectContaining({ method: "POST" }));
    const table = screen.getByTestId("depth-check-table");
    const rows = within(table).getAllByRole("row");
    expect(rows).toHaveLength(4); // header, score_genes, AUCell, GSVA (ssGSEA left out)
    expect(rows[1]).toHaveTextContent(/^score_genes-0.17 \/ -0.19-0.43 \/ -0.33$/);
    expect(rows[2]).toHaveTextContent("AUCell0.04 / -0.150.06 / -0.060.35 / 0.54");
    expect(rows[3]).toHaveTextContent("GSVA0.02 / 0.010.03 / 0.020.61 / 0.70");
    expect(within(table).getByText("Genes detected")).toHaveAttribute("title", expect.stringContaining("median within groups / over all cells"));
    expect(within(table).getByText("vs score_genes")).toBeInTheDocument();
    const note = screen.getByText(/AUCell counts each cell's top 703 genes/);
    expect(note).toHaveTextContent("64% of cells detect fewer");
    expect(note).toHaveTextContent("ssGSEA left out: ssGSEA scores up to 300 million");
  });

  it("says what went wrong", async () => {
    client.apiFetch.mockImplementation(async () => {
      throw new Error("API error 400: None of the 2 provided genes were found in the dataset.");
    });
    render(<ScoreDepthCheck datasetId="kang" genes={["X", "Y"]} />);
    fireEvent.click(screen.getByRole("button", { name: "Check depth" }));
    expect(await screen.findByText(/None of the 2 provided genes/)).toBeInTheDocument();
  });
});
