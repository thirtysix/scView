import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const showDensity = vi.fn();
vi.mock("@/lib/density", async (orig) => ({
  ...(await orig<typeof import("@/lib/density")>()),
  showDensity: (...a: unknown[]) => showDensity(...a),
}));

import { DensityToggle, JointDensityButton } from "./DensityToggle";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

beforeEach(() => {
  showDensity.mockReset();
  useUnifiedViewStore.setState({ densityMode: false });
  useUnifiedViewStore.getState().clearOverlayState();
});

describe("DensityToggle", () => {
  it("turns density mode on and draws the gene's density", async () => {
    showDensity.mockResolvedValue({});
    const onGeneSelect = vi.fn();
    render(<DensityToggle datasetId="ds1" gene="CD3D" onGeneSelect={onGeneSelect} />);
    fireEvent.click(screen.getByRole("checkbox"));
    await waitFor(() => expect(showDensity).toHaveBeenCalledWith("ds1", ["CD3D"]));
    expect(useUnifiedViewStore.getState().densityMode).toBe(true);
    expect(onGeneSelect).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "What the density shows" })).toBeTruthy();
  });

  it("off: the gene's expression again", () => {
    useUnifiedViewStore.setState({ densityMode: true });
    const onGeneSelect = vi.fn();
    render(<DensityToggle datasetId="ds1" gene="CD3D" onGeneSelect={onGeneSelect} />);
    fireEvent.click(screen.getByRole("checkbox"));
    expect(useUnifiedViewStore.getState().densityMode).toBe(false);
    expect(onGeneSelect).toHaveBeenCalledWith("CD3D");
    expect(showDensity).not.toHaveBeenCalled();
  });

  it("says why when the server refuses, and stays off", async () => {
    showDensity.mockRejectedValue(new Error('API error 404: {"detail":"None of the genes are in layer \'X\'"}'));
    render(<DensityToggle datasetId="ds1" gene="CD3D" onGeneSelect={vi.fn()} />);
    fireEvent.click(screen.getByRole("checkbox"));
    expect((await screen.findByRole("alert")).textContent).toBe("Density: None of the genes are in layer 'X'");
    expect(useUnifiedViewStore.getState().densityMode).toBe(false);
  });
});

describe("JointDensityButton", () => {
  it("needs two genes", () => {
    render(<JointDensityButton datasetId="ds1" genes={["CD3D"]} />);
    expect((screen.getByRole("button", { name: /Joint density/ }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("draws the joint density after closing what covers the plot, and hides it again", async () => {
    showDensity.mockImplementation(async () => {
      useUnifiedViewStore.setState({
        scatterOverlay: { type: "density", genes: ["CD3D", "CD8A"], embedding: "X_umap", layer: "raw" },
        overlayDatasetId: "ds1",
        overlayValues: new Float32Array(2),
      });
      return {};
    });
    const onShow = vi.fn();
    render(<JointDensityButton datasetId="ds1" genes={["CD3D", "CD8A"]} onShow={onShow} />);
    fireEvent.click(screen.getByRole("button", { name: "Joint density" }));
    await waitFor(() => expect(showDensity).toHaveBeenCalledWith("ds1", ["CD3D", "CD8A"]));
    expect(onShow).toHaveBeenCalled();
    const hide = await screen.findByRole("button", { name: "Hide joint density" });
    expect(hide.getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(hide);
    expect(useUnifiedViewStore.getState().scatterOverlay).toBeNull();
  });
});
