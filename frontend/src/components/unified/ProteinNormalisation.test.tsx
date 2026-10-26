import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { ProteinInfo } from "@/api/types";
import type { DsbStatus } from "@/api/dsb";

const { getDsbStatus, runDsb, refresh } = vi.hoisted(() => ({
  getDsbStatus: vi.fn(),
  runDsb: vi.fn(),
  refresh: vi.fn(),
}));
vi.mock("@/api/dsb", () => ({ getDsbStatus, runDsb }));
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => refresh }));
vi.mock("@/components/plots/Plot", () => ({
  default: ({ layout }: { layout: { shapes?: { x0: number; x1: number; y0: number; y1: number }[] } }) => (
    <div data-testid="plot">{(layout.shapes ?? []).map((s) => `${s.x0}-${s.x1}/${s.y0}-${s.y1}`).join(";")}</div>
  ),
}));

import { ProteinNormalisation } from "./ProteinNormalisation";

const PROTEIN: ProteinInfo = {
  names: ["CD3", "CD19", "IgG1_control"],
  controls: ["IgG1_control"],
  layer: "protein",
  counts_layer: "protein_counts",
  normalisation: "CLR per cell",
  dsb_layer: null,
  dsb: null,
};

const DENSITY = { rna_bin: [3, 3, 25], prot_bin: [15, 29, 20], n: [100, 20, 5], bin_width: 0.1 };

const STATUS: DsbStatus = {
  available: true,
  reason: null,
  computed: false,
  empty_droplets: true,
  empty_droplets_reason: null,
  samples: [{ sample: "s1", n_cells: 900, n_droplets_default: 120, n_cells_excluded: 900, missing_antibodies: [], density: DENSITY }],
  isotypes: ["IgG1_control"],
  proteins: ["CD3", "CD19", "IgG1_control"],
  defaults: { rna_window: [0.3, 2.5], prot_window: [1.5, 3], bin_width: 0.1, rna_max: 4, prot_max: 5, min_droplets: 50 },
  record: null,
};

describe("ProteinNormalisation", () => {
  beforeEach(() => {
    getDsbStatus.mockReset().mockResolvedValue(STATUS);
    runDsb.mockReset().mockResolvedValue({});
    refresh.mockReset().mockResolvedValue({});
  });

  it("offers DSB only once computed and names how it was made", () => {
    const onNormChange = vi.fn();
    const { rerender } = render(
      <ProteinNormalisation datasetId="d1" protein={PROTEIN} norm="clr" onNormChange={onNormChange} />,
    );
    expect((screen.getByLabelText(/^DSB/) as HTMLInputElement).disabled).toBe(true);
    expect(screen.getByTestId("protein-norm-note").textContent).toMatch(/DSB is not computed yet/);
    const done: ProteinInfo = {
      ...PROTEIN,
      dsb_layer: "protein_dsb",
      dsb: { mode: "empty_droplets", isotypes: ["IgG1_control"], n_droplets: 120, denoise: true, method: "" },
    };
    rerender(<ProteinNormalisation datasetId="d1" protein={done} norm="clr" onNormChange={onNormChange} />);
    fireEvent.click(screen.getByLabelText(/^DSB/));
    expect(onNormChange).toHaveBeenCalledWith("dsb");
    rerender(<ProteinNormalisation datasetId="d1" protein={done} norm="dsb" onNormChange={onNormChange} />);
    expect(screen.getByTestId("protein-norm-note").textContent).toBe(
      "DSB against 120 empty droplets; per-cell noise removed with isotypes IgG1_control.",
    );
    expect(screen.getByRole("button", { name: "Recompute DSB…" })).toBeTruthy();
  });

  it("shows the window's droplets as it changes and computes with the settings", async () => {
    const onNormChange = vi.fn();
    render(<ProteinNormalisation datasetId="d1" protein={PROTEIN} norm="clr" onNormChange={onNormChange} />);
    fireEvent.click(screen.getByRole("button", { name: "Compute DSB…" }));
    await waitFor(() => screen.getByTestId("dsb-droplets"));
    expect(getDsbStatus).toHaveBeenCalledWith("d1");
    expect(screen.getByTestId("dsb-droplets").textContent).toMatch(/s1: 120 empty droplets in the window \(900 cells\)/);
    expect(screen.getByTestId("plot").textContent).toBe("0.3-2.5/1.5-3");
    // A protein floor of 2 leaves only 20: too few, the button waits
    fireEvent.change(screen.getByLabelText("Protein from"), { target: { value: "2" } });
    expect(screen.getByTestId("dsb-droplets").textContent).toMatch(/20 empty droplets .* at least 50 are needed/);
    expect((screen.getByRole("button", { name: "Compute DSB" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Protein from"), { target: { value: "1.5" } });
    expect((screen.getByRole("checkbox", { name: "IgG1_control" }) as HTMLInputElement).checked).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Compute DSB" }));
    await waitFor(() => expect(onNormChange).toHaveBeenCalledWith("dsb"));
    expect(runDsb).toHaveBeenCalledWith("d1", {
      mode: "empty_droplets", rna_window: [0.3, 2.5], prot_window: [1.5, 3], isotypes: ["IgG1_control"], denoise: true,
    });
    expect(refresh).toHaveBeenCalledWith("d1");
  });

  it("without empty droplets says why and runs the no-empty-droplet mode; errors are the backend's", async () => {
    getDsbStatus.mockResolvedValue({
      ...STATUS, empty_droplets: false, samples: [],
      empty_droplets_reason: "No raw matrix with antibody features is attached, so there are no empty droplets.",
    });
    runDsb.mockRejectedValue(new Error('API error 400: {"detail":"Not in the protein panel: IgG3."}'));
    render(<ProteinNormalisation datasetId="d1" protein={PROTEIN} norm="clr" onNormChange={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Compute DSB…" }));
    await waitFor(() => screen.getByTestId("dsb-no-empty"));
    expect(screen.getByTestId("dsb-no-empty").textContent).toMatch(/No raw matrix with antibody features/);
    fireEvent.click(screen.getByRole("checkbox", { name: "IgG1_control" }));
    expect(screen.getByText(/No isotypes ticked/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Compute DSB without empty droplets" }));
    await waitFor(() => screen.getByRole("alert"));
    expect(screen.getByRole("alert").textContent).toMatch(/Not in the protein panel: IgG3\./);
    expect(runDsb.mock.calls[0]![1]).toMatchObject({ mode: "model_negative", isotypes: [] });
  });
});
