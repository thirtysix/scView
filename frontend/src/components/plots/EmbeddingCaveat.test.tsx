import { describe, it, expect, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useViewStore } from "@/stores/viewStore";
import { EmbeddingCaveat } from "./EmbeddingCaveat";
import { hoverTip } from "@/test/hoverTip";

beforeEach(() => {
  useUnifiedViewStore.setState({ bottomPanelOpen: false, bottomView: "distribution" });
  useViewStore.setState({ activePanel: "expression" });
});

describe("EmbeddingCaveat", () => {
  it("notes on a UMAP or t-SNE plot that cluster gaps aren't distances", () => {
    const { rerender } = render(<EmbeddingCaveat embedding="X_umap" />);
    expect(screen.getByTestId("embedding-caveat")).toHaveTextContent("UMAP: gaps between clusters aren't distances");
    rerender(<EmbeddingCaveat embedding="X_tsne" />);
    expect(screen.getByTestId("embedding-caveat")).toHaveTextContent("t-SNE:");
  });

  it("explains the caveat when its text is hovered, with no icon", async () => {
    render(<EmbeddingCaveat embedding="X_umap" />);
    expect(screen.queryByRole("button", { name: /About distances/ })).toBeNull();
    expect(await hoverTip(screen.getByText("UMAP: gaps between clusters aren't distances"))).toContain(
      "keeps each cell's nearest neighbours",
    );
  });

  it("says nothing on a PCA plot, where distances hold", () => {
    render(<EmbeddingCaveat embedding="X_pca" />);
    expect(screen.queryByTestId("embedding-caveat")).toBeNull();
  });

  it("opens the cluster distances under the Unified View's plot", () => {
    render(<EmbeddingCaveat embedding="X_umap" />);
    fireEvent.click(screen.getByRole("button", { name: "Cluster distances" }));
    expect(useUnifiedViewStore.getState()).toMatchObject({ bottomPanelOpen: true, bottomView: "distances" });
    expect(useViewStore.getState().activePanel).toBe("unified");
  });
});
