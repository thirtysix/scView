import { describe, it, expect, beforeEach } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { hoverTip, unhover } from "@/test/hoverTip";
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

describe("plot and toolbar help", () => {
  beforeEach(() => {
    useDatasetStore.setState({ currentDataset: dataset, currentDatasetId: "ds" });
    useSettingsStore.setState({ embedding: "X_umap", colorBy: "cell_type", expressionLayer: "raw" });
  });

  it("puts a catalog tip on each toolbar label, opening below the toolbar", async () => {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <UnifiedToolbar numCells={12} />
      </QueryClientProvider>,
    );
    const tips: [string, string][] = [
      ["Embedding", "plot.embedding"],
      ["Color", "plot.colour-by"],
      ["Split", "plot.split"],
      ["Layer", "plot.layer"],
      ["Opacity", "plot.opacity"],
      ["Colours", "plot.colours"],
      ["Clip", "plot.clip"],
      ["Max", "plot.max-cells"],
      ["Lasso", "plot.lasso"],
    ];
    expect(document.querySelector("[data-infotip]")).toBeNull(); // no icons
    for (const [text, id] of tips) {
      const target = screen.getByText(text);
      expect(await hoverTip(target)).toBe(PLOT_HELP[id]);
      // the toolbar is the top of the page: the tip opens below it
      expect(screen.getByRole("tooltip")).toHaveAttribute("data-side", "below");
      // one tip only: no native title on the label or anything around it
      expect(target.closest("[title]"), text).toBeNull();
      unhover(target);
    }
    // the labels still name their controls
    expect(screen.getByLabelText("Clip")).toHaveAttribute("id", "colour-clip-default");
    expect(screen.getByLabelText("Split")).toHaveAttribute("id", "split-by");
  });

  it("gives the lasso one styled tip that names its L shortcut, and no native title", async () => {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <UnifiedToolbar numCells={12} />
      </QueryClientProvider>,
    );
    const lasso = screen.getByRole("button", { name: "Lasso selection" });
    expect(lasso).toHaveAttribute("aria-keyshortcuts", "L");
    expect(lasso.closest("[title]")).toBeNull();
    expect(lasso.querySelector("[title]")).toBeNull();
    // the icon opens it too, not only the word
    expect(await hoverTip(lasso.querySelector("svg")!)).toBe(PLOT_HELP["plot.lasso"]);
    expect(screen.getAllByRole("tooltip")).toHaveLength(1);
    for (const id of ["plot.lasso", "plot.lasso-grid", "plot.lasso-tissue"]) expect(PLOT_HELP[id], id).toMatch(/Press L/);
  });

  it("keeps the dialogs' first focus on their first choice; the tip is on the heading", async () => {
    render(
      <SelectByGroupControl
        source={{ column: "cell_type", categories: ["B", "NK", "T"], onSelect: () => {} }}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "By group" }));
    const dialog = screen.getByRole("dialog", { name: "Select cells by group" });
    expect(within(dialog).getByRole("checkbox", { name: /^B/ })).toHaveFocus();
    expect(screen.queryByRole("tooltip")).toBeNull();
    expect(await hoverTip(within(dialog).getByText(/^Select cells by/))).toBe(PLOT_HELP["plot.select-by-group"]);
  });

  it("explains Compare inside its dialog without taking the first focus", async () => {
    render(<CompareColoringsControl />);
    fireEvent.click(screen.getByRole("button", { name: /compare/i }));
    const dialog = screen.getByRole("dialog", { name: "Compare colourings" });
    expect(within(dialog).getAllByRole("checkbox")[0]).toHaveFocus();
    expect(screen.queryByRole("tooltip")).toBeNull();
    expect(await hoverTip(within(dialog).getByText("Compare colourings side by side"))).toBe(PLOT_HELP["plot.compare"]);
  });
});
