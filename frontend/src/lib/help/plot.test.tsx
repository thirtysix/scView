import { describe, it, expect, beforeEach } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { DatasetInfo } from "@/api/types";
import { CompareColoringsControl } from "@/components/unified/CompareColoringsControl";
import { SelectByGroupControl } from "@/components/unified/SelectByGroup";
import { UnifiedToolbar } from "@/components/unified/UnifiedToolbar";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { PLOT_HELP } from "./plot";

const dataset: DatasetInfo = {
  id: "ds",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 12,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [
    { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
    { name: "cell_type", dtype: "category", n_unique: 3, values: ["B", "NK", "T"] },
    { name: "leiden", dtype: "category", n_unique: 4, values: ["0", "1", "2", "3"] },
  ],
  expression_layers: [{ key: "raw", label: "Log-normalized (raw)", n_genes: 10 }],
  default_expression_layer: "raw",
  created_at: "",
};

/** The tip behind an info icon, opened by a click. */
function tipOf(name: string) {
  fireEvent.click(screen.getByRole("button", { name }));
  return screen.getByRole("tooltip");
}

describe("plot and toolbar help", () => {
  beforeEach(() => {
    useDatasetStore.setState({ currentDataset: dataset, currentDatasetId: "ds" });
    useSettingsStore.setState({ embedding: "X_umap", colorBy: "cell_type", expressionLayer: "raw" });
  });

  it("puts a catalog tip beside each toolbar control, opening below the toolbar", () => {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <UnifiedToolbar numCells={12} />
      </QueryClientProvider>,
    );
    const tips: [string, string][] = [
      ["About the embedding", "plot.embedding"],
      ["About colouring the plot", "plot.colour-by"],
      ["About splitting the plot", "plot.split"],
      ["About the expression layer", "plot.layer"],
      ["About opacity", "plot.opacity"],
      ["About the colour schemes", "plot.colours"],
      ["About clipping the colour range", "plot.clip"],
      ["About the most cells drawn", "plot.max-cells"],
      ["About selecting cells", "plot.select"],
    ];
    for (const [name, id] of tips) {
      const tip = tipOf(name);
      expect(tip).toHaveTextContent(PLOT_HELP[id]!);
      expect(tip).toHaveClass("top-full"); // the toolbar is the top of the page: a tip above it is cut off
      fireEvent.click(screen.getByRole("button", { name })); // close it again
    }
  });

  it("keeps the dialogs' first focus on their first choice, the tip after it", () => {
    render(
      <SelectByGroupControl
        source={{ column: "cell_type", categories: ["B", "NK", "T"], onSelect: () => {} }}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "By group" }));
    const dialog = screen.getByRole("dialog", { name: "Select cells by group" });
    expect(within(dialog).getByRole("checkbox", { name: /^B/ })).toHaveFocus();
    expect(tipOf("About selecting by group")).toHaveTextContent(PLOT_HELP["plot.select-by-group"]!);
  });

  it("explains Compare inside its dialog without taking the first focus", () => {
    render(<CompareColoringsControl />);
    fireEvent.click(screen.getByRole("button", { name: /compare/i }));
    const dialog = screen.getByRole("dialog", { name: "Compare colourings" });
    expect(within(dialog).getAllByRole("checkbox")[0]).toHaveFocus();
    expect(tipOf("About comparing colourings")).toHaveTextContent(PLOT_HELP["plot.compare"]!);
  });
});
