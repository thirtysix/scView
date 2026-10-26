import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { DEResponse } from "@/api/de";
import type { DEAuditResult } from "@/api/deAudit";

const { runDEAudit } = vi.hoisted(() => ({ runDEAudit: vi.fn() }));
vi.mock("@/api/deAudit", () => ({ runDEAudit }));
vi.mock("@/components/plots/Plot", () => ({ default: () => <div data-testid="plot" /> }));

import { DEAudit } from "./DEAudit";

const RES: DEResponse = {
  n_selected: 2031,
  n_rest: 2194,
  label: "STIM (CD14 Mono)",
  genes: [],
  method: "wilcoxon",
  result_id: "r1",
  replicate_key: "Donor",
  n_donors_a: 8,
  n_donors_b: 8,
};

const AUDIT: DEAuditResult = {
  label: "CD14 Mono: STIM vs CTRL (Wilcoxon)",
  comparison: { group_a: "STIM", group_b: "CTRL" },
  donor_key: "Donor",
  method: "wilcoxon",
  scheme: "paired",
  scheme_text: "",
  n_donors_a: 8,
  n_donors_b: 8,
  n_cells_a: 2031,
  n_cells_b: 2194,
  n_genes_tested: 6000,
  observed: 1836,
  null_counts: [10, 310, 663],
  n_possible: 35,
  exhaustive: false,
  empirical_p: 0.25,
  stopped: null,
  permutation_note: null,
  qq: { expected: [3, 2, 1], observed: [60, 30, 2], permuted: [10, 5, 1] },
  pseudobulk: {
    method: "deseq2", name: "DESeq2", n_tested: 3454, n_samples_a: 8, n_samples_b: 8, design: "~Donor + condition",
    n_sig_cell: 1836, n_sig_pseudobulk: 2265, n_shared: 1565, jaccard: 0.6171, n_genes_common: 3454, spearman_lfc: 0.9946,
  },
  pseudobulk_note: null,
  cell_only: [
    { gene: "MYL6", cell_lfc: -0.52, cell_padj: 1e-109, pb_lfc: -0.11, pb_padj: 0.15, pb_tested: true, top_donor_share: 0.04, top_donor: "SNG-1039" },
    { gene: "G26", cell_lfc: 0.7, cell_padj: 1e-8, pb_lfc: null, pb_padj: null, pb_tested: false, top_donor_share: 0.46, top_donor: "d0" },
  ],
  n_cell_only: 271,
  verdict: "1,836 genes at cell level (adj. p < 0.05); 10-663 in 3 donor shuffles, median 310: trust the pseudobulk list.",
  seconds: 12.2,
  alpha: 0.05,
  null_share: 0.5,
  notes: [],
};

describe("DEAudit", () => {
  beforeEach(() => {
    runDEAudit.mockReset();
  });

  it("is disabled with the reason for a selection-vs-rest result", () => {
    render(<DEAudit datasetId="kang" result={RES} betweenGroups={false} donorColumns={2} onGeneClick={vi.fn()} />);
    expect(screen.getByRole("button", { name: /Audit this result/ })).toBeDisabled();
    expect(screen.getByText(/Needs a between-groups result/)).toBeInTheDocument();
  });

  it("runs on the kept result, then shows the verdict, plots, concordance and genes", async () => {
    runDEAudit.mockImplementation(async (_id, _body, h) => {
      h?.onProgress?.({ step: "shuffle", i: 1, n: 3, count: 10, message: "Donor shuffle 1 of 3: 10 genes" });
      return AUDIT;
    });
    const onGene = vi.fn();
    render(<DEAudit datasetId="kang" result={RES} betweenGroups donorColumns={2} replicateKey="Donor" onGeneClick={onGene} />);
    fireEvent.click(screen.getByRole("button", { name: /Audit this result/ }));
    await waitFor(() => expect(screen.getByTestId("de-audit-verdict")).toHaveTextContent("trust the pseudobulk list"));
    expect(runDEAudit.mock.calls[0]?.[1]).toEqual({ result_id: "r1" });
    expect(screen.getAllByTestId("plot")).toHaveLength(2);
    expect(screen.getByText(/10-663 \(median 310\)/)).toBeInTheDocument();
    expect(screen.getByText(/1,565 in both \(Jaccard 0.62\)/)).toBeInTheDocument();
    expect(screen.getByText("Top 2 of 271 by cell-level p; the results export has these.")).toBeInTheDocument();
    expect(screen.getByText("4% (SNG-1039)")).toBeInTheDocument();
    fireEvent.click(screen.getByText("MYL6"));
    expect(onGene).toHaveBeenCalledWith("MYL6");
  });

  it("sends the picked donor column when the result names none, and shows errors", async () => {
    runDEAudit.mockRejectedValue(new Error("Each group needs at least 3 cells with a donor in 'sample'."));
    render(
      <DEAudit datasetId="kang" result={{ ...RES, replicate_key: null }} betweenGroups donorColumns={1} replicateKey="sample" onGeneClick={vi.fn()} />,
    );
    fireEvent.click(screen.getByRole("button", { name: /Audit this result/ }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("at least 3 cells"));
    expect(runDEAudit.mock.calls[0]?.[1]).toEqual({ result_id: "r1", replicate_key: "sample" });
  });

  it("Cancel aborts the stream", async () => {
    let signal: AbortSignal | undefined;
    runDEAudit.mockImplementation(
      (_id, _body, h) =>
        new Promise((_res, rej) => {
          signal = h?.signal;
          h?.signal?.addEventListener("abort", () => rej(new Error("aborted")));
        }),
    );
    render(<DEAudit datasetId="kang" result={RES} betweenGroups donorColumns={2} onGeneClick={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /Audit this result/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(signal?.aborted).toBe(true);
    await waitFor(() => expect(screen.getByRole("button", { name: /Audit this result/ })).not.toBeDisabled());
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
