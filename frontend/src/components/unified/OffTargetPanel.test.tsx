import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { HpaSide, OffTargetGene, OffTargetResult } from "@/api/offtarget";

const { getOffTarget } = vi.hoisted(() => ({ getOffTarget: vi.fn() }));
vi.mock("@/api/offtarget", () => ({ getOffTarget }));
vi.mock("@/components/plots/Plot", () => ({
  default: (p: { data: { y: string[] }[] }) => <div data-testid="plot">{p.data[0]?.y.join("|")}</div>,
}));

import { OffTargetPanel } from "./OffTargetPanel";
import { categoryTitle, summaryText, topText } from "@/lib/offtarget";

const side = (category: HpaSide["category"], top: [string, number][], elevated: string[] = []): HpaSide => ({
  category, category_text: category.replace("_", " "), tau: 0.8, max: top[0]?.[1] ?? 0,
  top: top.map(([name, value]) => ({ name, value })), elevated,
});

const gene = (name: string, verdict: OffTargetGene["verdict"], over: Partial<OffTargetGene> = {}): OffTargetGene => ({
  gene: name, in_dataset_tau: 0.7, found: true, hpa_symbol: name, ensg: `ENSG_${name}`,
  url: `https://www.proteinatlas.org/ENSG_${name}`,
  tissue: side("low_specificity", [["blood vessel", 9000], ["colon", 7000.4], ["skin", 2.25]]),
  cell_type: side("enhanced", [["platelets", 5000]], ["platelets"]),
  verdict, verdict_text: verdict, flag: false, ...over,
});

const RESULT: OffTargetResult = {
  species: "human", note: null,
  genes: [
    gene("MS4A1", "restricted", { tissue: side("group_enriched", [["tonsil", 120]], ["tonsil", "lymph node"]) }),
    gene("ACTB", "broad", { flag: true }),
    gene("FAKE1", "not_in_hpa", { found: false, url: null, ensg: null, tissue: null, cell_type: null }),
  ],
  heatmap: { genes: ["MS4A1", "ACTB"], tissues: ["liver", "tonsil"], log2_ntpm: [[0, 7], [12, 11]], unit: "log2(nTPM + 1)" },
  summary: { n_genes: 3, counts: { restricted: 1, enhanced: 0, broad: 1, not_detected: 0, not_in_hpa: 1 }, n_flagged: 1, flagged: ["ACTB"] },
  attribution: "Human Protein Atlas v25.1", version: "25.1", licence: "CC BY-SA 4.0",
};

beforeEach(() => {
  getOffTarget.mockReset().mockResolvedValue(RESULT);
});

describe("OffTargetPanel", () => {
  it("waits to be asked, then follows the selected group", async () => {
    const onGeneClick = vi.fn();
    const { rerender } = render(<OffTargetPanel datasetId="ds" column="leiden" group="3" onGeneClick={onGeneClick} />);
    expect(getOffTarget).not.toHaveBeenCalled();
    expect(screen.getByText(/about 21 MB/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Look up in HPA" }));
    await screen.findByText(/1 of 3 restricted, 0 enhanced, 1 broad, 1 not in HPA; 1 specific here/);
    expect(getOffTarget).toHaveBeenCalledWith("ds", "leiden", "3", 20, "tissue");
    // the flagged row, the HPA link, the heatmap rows and the attribution
    expect(screen.getByTestId("offtarget-row-ACTB").className).toContain("bg-orange");
    expect(screen.getByTestId("offtarget-row-MS4A1").className).not.toContain("bg-orange");
    expect(screen.getByLabelText("MS4A1 in the Human Protein Atlas").getAttribute("href")).toBe(
      "https://www.proteinatlas.org/ENSG_MS4A1",
    );
    expect(screen.queryByLabelText("FAKE1 in the Human Protein Atlas")).toBeNull();
    expect(screen.getByTestId("plot").textContent).toBe("MS4A1|ACTB");
    expect(screen.getByRole("link", { name: "Human Protein Atlas" })).toBeTruthy();
    expect(screen.getByText(/CC BY-SA 4.0/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "ACTB" }));
    expect(onGeneClick).toHaveBeenCalledWith("ACTB");
    rerender(<OffTargetPanel datasetId="ds" column="leiden" group="5" onGeneClick={onGeneClick} />);
    await waitFor(() => expect(getOffTarget).toHaveBeenLastCalledWith("ds", "leiden", "5", 20, "tissue"));
    // the tissue level re-asks the server
    fireEvent.change(screen.getByLabelText("Tissue level"), { target: { value: "organ" } });
    await waitFor(() => expect(getOffTarget).toHaveBeenLastCalledWith("ds", "leiden", "5", 20, "organ"));
  });

  it("shows the server's reason when HPA cannot be fetched", async () => {
    getOffTarget.mockRejectedValue(new Error("The HPA tissue RNA consensus table is not downloaded yet"));
    render(<OffTargetPanel datasetId="ds" column="leiden" group="3" onGeneClick={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Look up in HPA" }));
    await screen.findByText(/not downloaded yet/);
    expect(screen.queryByTestId("plot")).toBeNull();
  });
});

describe("offtarget helpers", () => {
  it("formats the top tissues, the categories and the summary", () => {
    expect(topText(RESULT.genes[1]!.tissue)).toBe("blood vessel 9,000, colon 7,000, skin 2.3");
    expect(topText(null)).toBe("—");
    expect(categoryTitle(RESULT.genes[0]!)).toContain("Tissues: group enriched (tonsil, lymph node), tau 0.80");
    expect(categoryTitle(RESULT.genes[2]!)).toMatch(/Not in the Human Protein Atlas/);
    expect(summaryText({ ...RESULT, summary: { ...RESULT.summary, n_genes: 0 } })).toMatch(/no markers/);
  });
});
