import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { CommunicationOptions } from "@/api/communication";
import type { DatasetInfo } from "@/api/types";
import { fakeCompare } from "@/test/communicationCompareFixture";

const { getCommunicationOptions, runCommunicationCompare, getCompareDonors } = vi.hoisted(() => ({
  getCommunicationOptions: vi.fn(),
  runCommunicationCompare: vi.fn(),
  getCompareDonors: vi.fn(),
}));
vi.mock("@/api/communication", async (orig) => ({
  ...(await orig<typeof import("@/api/communication")>()),
  getCommunicationOptions,
  runCommunicationCompare,
  getCompareDonors,
}));
const plotProps: { current: { data: Record<string, unknown>[] } | null } = { current: null };
vi.mock("@/components/plots/Plot", () => ({
  default: (props: { data: Record<string, unknown>[] }) => {
    if (props.data[0]?.type === "heatmap") plotProps.current = props;
    return <div data-testid="plot">{props.data.map((t) => String(t.type)).join(";")}</div>;
  },
}));

import { UnifiedCommunicationSubtab } from "./UnifiedCommunicationSubtab";
import { useCommunicationStore } from "@/stores/communicationStore";
import { useDatasetStore } from "@/stores/datasetStore";

const OPTIONS: CommunicationOptions = {
  conditions: [{ column: "stim", values: ["CTRL", "STIM"] }],
  species: "human",
  species_evidence: "Ensembl ids",
  resource: "consensus",
  max_groups: 60,
  max_perms: 5000,
  default_expr_prop: 0.1,
  default_n_perms: 1000,
  min_cells: 5,
};

describe("Communication: compare conditions", () => {
  beforeEach(() => {
    useCommunicationStore.getState().reset();
    plotProps.current = null;
    getCommunicationOptions.mockReset().mockResolvedValue(OPTIONS);
    runCommunicationCompare.mockReset().mockImplementation(async (_id, _body, h) => {
      h.onStart?.({ n_cells: 400, n_groups: 2, resource: "consensus", resource_label: "Consensus (human)",
                    species: "human", species_evidence: "", n_perms: 1000, estimate_seconds: 5, a: "CTRL", b: "STIM" });
      h.onProgress?.("CTRL", 500, 1000);
      return fakeCompare();
    });
    getCompareDonors.mockReset().mockResolvedValue({
      available: true, paired: true, sample_key: "Donor", test: "Wilcoxon signed-rank (paired donors)",
      points: [
        { donor: "d1", condition: "CTRL", value: 0.1, n_cells: 40 },
        { donor: "d1", condition: "STIM", value: 2.0, n_cells: 50 },
        { donor: "d2", condition: "CTRL", value: 0.2, n_cells: 30 },
        { donor: "d2", condition: "STIM", value: 2.4, n_cells: 35 },
      ],
      paired_test: { n_pairs: 2, mean_diff: 2.05, wilcoxon_p: 0.5, ttest_p: 0.05, caution: "2 pairs" },
      p_value: 0.0078, p_adj: 0.03, label: "Mono -> NK: CXCL10 -> CXCR3",
    });
    useDatasetStore.setState({
      currentDatasetId: "d1",
      currentDataset: {
        id: "d1", name: "d1", filename: "", n_cells: 400, n_genes: 900, status: "ready", available_embeddings: [],
        created_at: "",
        obs_columns: [
          { name: "cell", dtype: "category", n_unique: 2, values: ["Mono", "NK"] },
          { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
        ],
      } as DatasetInfo,
    });
  });

  it("compares two conditions: heatmap about 0, filtered table, donor plot", async () => {
    render(<UnifiedCommunicationSubtab groupByColumn="cell" />);
    fireEvent.click(screen.getByRole("button", { name: "Compare conditions" }));
    await waitFor(() => screen.getByLabelText("Reference (A)"));
    expect((screen.getByLabelText("Reference (A)") as HTMLSelectElement).value).toBe("CTRL");
    expect((screen.getByLabelText("Compared (B)") as HTMLSelectElement).value).toBe("STIM");
    fireEvent.click(screen.getByRole("button", { name: /^Compare$/ }));
    await waitFor(() => screen.getByText("CXCL10"));
    expect(runCommunicationCompare).toHaveBeenCalledWith(
      "d1",
      { groupby: "cell", condition: "stim", a: "CTRL", b: "STIM", expr_prop: 0.1 },
      expect.anything(),
    );
    expect(screen.getByText(/STIM vs CTRL: 1 gained, 1 lost, 0 changed, 1 unchanged/)).toBeTruthy();
    expect(screen.getByText(/Left out of both \(under 10 cells in a condition\): pDC \(12 \/ 4\)/)).toBeTruthy();
    expect(screen.getByTestId("ccc-cmp-replicates").textContent).toMatch(/Tested per donor .*Wilcoxon/);
    expect(screen.getByRole("note").textContent).toMatch(/Composition changed: NK 50.0% → 25.0%/);

    // Heatmap: B - A counts, symmetric range.
    const hm = plotProps.current!.data[0]!;
    expect(hm.z).toEqual([[3, 0], [-2, 0]]);
    expect([hm.zmin, hm.zmax]).toEqual([-3, 3]);

    // Default filter hides the unchanged row; "All" shows it.
    expect(screen.queryByText("CCL2")).toBeNull();
    fireEvent.change(screen.getByLabelText("Call"), { target: { value: "all" } });
    expect(screen.getByText("CCL2")).toBeTruthy();
    fireEvent.click(screen.getByRole("checkbox"));
    expect(screen.queryByText("IFNG")).toBeNull(); // adj. p 0.06

    // Per-donor plot for the picked row.
    fireEvent.click(screen.getByRole("button", { name: /Plot per donor: Mono → NK: CXCL10 → CXCR3/ }));
    await waitFor(() => screen.getByTestId("ccc-cmp-donors"));
    expect(getCompareDonors).toHaveBeenCalledWith("d1", "c1", 0);
    await waitFor(() => within(screen.getByTestId("ccc-cmp-donors")).getByTestId("donor-caption"));
    expect(screen.getByTestId("donor-caption").textContent).toMatch(/One line per Donor/);
    expect(screen.getByRole("link", { name: /CSV/ }).getAttribute("href")).toContain("/communication/compare/c1.csv");
  });

  it("says why there is no test without donors and disables the donor plot", async () => {
    runCommunicationCompare.mockResolvedValue(
      fakeCompare({
        replicates: {
          sample_key: null, paired: false, test: null, n_donors: {}, n_pairs: 0, caution: null,
          reason: "No donor or sample column: with no replicates there is no test.",
        },
      }),
    );
    useCommunicationStore.getState().update({ mode: "compare" });
    render(<UnifiedCommunicationSubtab groupByColumn="cell" />);
    await waitFor(() => screen.getByRole("button", { name: /^Compare$/ }));
    fireEvent.click(screen.getByRole("button", { name: /^Compare$/ }));
    await waitFor(() => screen.getByText("CXCL10"));
    expect(screen.getByTestId("ccc-cmp-replicates").textContent).toMatch(/no replicates there is no test/);
    expect((screen.getByRole("checkbox") as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: /Plot per donor: Mono → NK/ }) as HTMLButtonElement).disabled).toBe(true);
  });
});
