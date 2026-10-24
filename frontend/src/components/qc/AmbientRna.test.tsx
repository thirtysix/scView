import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import type { AmbientResponse, AmbientSample } from "@/api/ambient";

const { getAmbient } = vi.hoisted(() => ({ getAmbient: vi.fn() }));
vi.mock("@/api/ambient", () => ({ getAmbient }));

import { AmbientRna } from "./AmbientRna";

const HELP = {
  rho: "Contamination fraction (rho): the share of each droplet's UMIs that is ambient RNA.",
  soup: "The soup: the RNA in empty droplets.",
  approximate: "No raw matrix is attached.",
  correction: "Removes rho x each cell's UMIs.",
};

const SAMPLE: AmbientSample = {
  sample: "pbmc_1k_v3",
  column: null,
  n_cells: 1222,
  source: "empty_droplets",
  top_genes: [
    { gene: "MALAT1", share: 0.029 },
    { gene: "LYZ", share: 0.012 },
    { gene: "HBB", share: 0.0042 },
  ],
  genes_covered: 1,
  n_droplets: 233233,
  n_umis: 905119,
  umi_max: 100,
  raw_files: ["pbmc_1k_v3_raw_feature_bc_matrix.h5"],
  rho: 0.053,
  rho_low: 0.014,
  rho_high: 0.113,
  rho_status: "ok",
  n_markers: 100,
  n_estimates: 1168,
  rho_genes: ["BIK", "GLDC"],
  notes: [],
};

function response(over: Partial<AmbientResponse> = {}): AmbientResponse {
  return {
    available: true,
    approximate: false,
    column: "leiden",
    counts_from: "X",
    samples: [SAMPLE],
    markers: { column: "leiden", approximate: false, groups: { "3": [{ gene: "LYZ", soup_rank: 2, ratio: 1.1 }] }, n_flagged: 1 },
    hb_note: null,
    notes: [],
    correction: null,
    help: HELP,
    method: "SoupX",
    url: "https://doi.org/10.1093/gigascience/giaa151",
    ...over,
  };
}

beforeEach(() => {
  getAmbient.mockReset(); // a block: a returned function would run as the teardown
});

describe("AmbientRna card", () => {
  it("shows each sample's rho with its range, the soup's top genes and the flagged markers", async () => {
    getAmbient.mockResolvedValue(response());
    render(<AmbientRna datasetId="d1" />);
    const row = await screen.findByTestId("ambient-sample");
    expect(within(row).getByText("5.3% (1.4-11.3%)")).toBeInTheDocument();
    expect(row.textContent).toContain("MALAT1 2.9%, LYZ 1.2%, HBB 0.42%");
    expect(screen.getByTestId("ambient-flag-count").textContent).toContain(
      "1 top marker in 1 cluster of leiden look ambient outside their cluster",
    );
    expect(screen.getByText(/n = 1,222 cells/)).toBeInTheDocument();
    // what rho means, on its column header
    expect(screen.getByText("Contamination (rho)").getAttribute("title")).toBe(HELP.rho);
    expect(getAmbient).toHaveBeenCalledWith("d1");
  });

  it("without a raw matrix says how to get rho and marks the profile approximate", async () => {
    getAmbient.mockResolvedValue(
      response({
        available: false,
        approximate: true,
        samples: [{ ...SAMPLE, source: "cells", rho: null, rho_low: null, rho_high: null, rho_status: "no_soup" }],
        markers: { column: "leiden", approximate: true, groups: {}, n_flagged: 0 },
      }),
    );
    render(<AmbientRna datasetId="d1" />);
    expect(await screen.findByTestId("ambient-no-raw")).toHaveTextContent("attach raw_feature_bc_matrix.h5");
    expect(screen.getByText("Attach the raw matrix to estimate it")).toBeInTheDocument();
    expect(screen.getByText("(approximate)")).toBeInTheDocument();
  });

  it("turns the opt-in correction step on, and only with a soup to subtract", async () => {
    getAmbient.mockResolvedValue(response());
    const onSelectedChange = vi.fn();
    const { rerender } = render(
      <AmbientRna datasetId="d1" step={{ selected: false, onSelectedChange, rho: null }} />,
    );
    const box = (await screen.findByTestId("ambient-step-toggle")) as HTMLInputElement;
    expect(box.checked).toBe(false);
    expect(screen.getByText(/each sample's own rho/)).toBeInTheDocument();
    fireEvent.click(box);
    expect(onSelectedChange).toHaveBeenCalledWith(true);

    getAmbient.mockResolvedValue(response({ available: false }));
    rerender(<AmbientRna datasetId="d2" step={{ selected: false, onSelectedChange, rho: 0.1 }} />);
    await screen.findByTestId("ambient-no-raw");
    expect((screen.getByTestId("ambient-step-toggle") as HTMLInputElement).disabled).toBe(true);
    expect(screen.getByText(/rho 10.0% for every sample/)).toBeInTheDocument();
  });

  it("says when the counts were corrected instead of offering the step", async () => {
    getAmbient.mockResolvedValue(
      response({
        correction: { samples: [{ sample: "pbmc_1k_v3", rho: 0.053, rho_source: "estimated" }], removed_share: 0.0512 },
      }),
    );
    render(<AmbientRna datasetId="d1" step={{ selected: false, onSelectedChange: vi.fn(), rho: null }} />);
    expect(await screen.findByTestId("ambient-corrected")).toHaveTextContent(
      "pbmc_1k_v3 rho 5.3% (estimated), 5.1% of the UMIs removed",
    );
    expect(screen.queryByTestId("ambient-step-toggle")).toBeNull();
  });

  it("shows a large-dataset refusal as a note", async () => {
    const why = "The ambient RNA estimate is not available for large datasets yet: it reads every cell's counts.";
    getAmbient.mockImplementation(async () => {
      throw new Error(why);
    });
    render(<AmbientRna datasetId="d1" />);
    const note = await screen.findByText(why);
    expect(note.className).toContain("text-slate-500");
  });
});
