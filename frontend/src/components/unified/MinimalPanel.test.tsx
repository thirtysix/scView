import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const api = vi.hoisted(() => ({ getMarkerPanel: vi.fn(), runMarkerPanel: vi.fn() }));
vi.mock("@/api/markerPanel", async (orig) => ({
  ...(await orig<typeof import("@/api/markerPanel")>()),
  getMarkerPanel: api.getMarkerPanel,
  runMarkerPanel: api.runMarkerPanel,
}));
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { name?: string; type?: string }[] }) => (
    <div data-testid="plot">{data.map((t) => t.name ?? t.type).join("|")}</div>
  ),
}));
vi.mock("@/components/plots/MarkerPlot", () => ({
  MarkerPlot: ({ genes }: { genes: string[] }) => <div data-testid="marker-plot">{genes.join(",")}</div>,
}));

import { useGeneListsStore } from "@/stores/geneListsStore";
import { RESULT } from "@/lib/markerPanel.fixture";
import { MinimalPanel } from "./MinimalPanel";

beforeEach(() => {
  api.getMarkerPanel.mockReset().mockResolvedValue({ result: null, running: false });
  api.runMarkerPanel.mockReset();
  useGeneListsStore.setState({ lists: [], selectedId: null });
});

describe("MinimalPanel", () => {
  it("is collapsed until opened, then sends the choices and shows the result", async () => {
    api.runMarkerPanel.mockImplementation(async (_id, _body, h) => {
      h.onStart?.(RESULT);
      for (const s of RESULT.path) h.onStep?.(s);
      h.onBaseline?.(RESULT.baseline);
      h.onResult?.(RESULT);
    });
    render(<MinimalPanel datasetId="ds" groupby="cell_type" />);
    expect(screen.queryByText("Find panel")).toBeNull();
    const toggle = screen.getByRole("button", { name: /Minimal panel/ });
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    fireEvent.change(screen.getByLabelText(/Genes in the panel/), { target: { value: "2" } });
    fireEvent.change(screen.getByLabelText(/Required genes/), { target: { value: "CD3E, CD19" } });
    fireEvent.change(screen.getByLabelText(/Excluded genes/), { target: { value: "MALAT1" } });
    fireEvent.click(screen.getByRole("button", { name: "Find panel" }));
    await screen.findByTestId("panel-summary");
    expect(api.runMarkerPanel).toHaveBeenCalledWith(
      "ds",
      { groupby: "cell_type", size: 2, per_group: 10, required: ["CD3E", "CD19"], excluded: ["MALAT1"], restrict: null },
      expect.any(Object),
    );
    expect(screen.getByTestId("panel-summary").textContent).toContain("2 genes reach balanced accuracy 0.90");
    expect(screen.getByTestId("panel-summary").textContent).toContain("top 2 markers taken in turn: 0.70");
    expect(screen.getAllByTestId("plot")[0]!.textContent).toBe("selected panel|top markers in turn");
    expect(screen.getByTestId("marker-plot").textContent).toBe("MS4A1,GNLY");
    expect(screen.getByTestId("panel-genes").textContent).toContain("GNLY");
    expect(screen.getAllByText(/n = /).length).toBeGreaterThan(0);
  });

  it("restricts to a saved gene list and saves the panel as one", async () => {
    const id = useGeneListsStore.getState().create("surface", ["MS4A1", "CD3E"]);
    api.runMarkerPanel.mockImplementation(async (_id, _body, h) => h.onResult?.(RESULT));
    render(<MinimalPanel datasetId="ds" groupby="cell_type" />);
    fireEvent.click(screen.getByRole("button", { name: /Minimal panel/ }));
    fireEvent.change(screen.getByLabelText(/Only genes in a gene list/), { target: { value: id } });
    fireEvent.click(screen.getByRole("button", { name: "Find panel" }));
    await screen.findByTestId("panel-summary");
    expect(api.runMarkerPanel.mock.calls[0]![1].restrict).toEqual(["MS4A1", "CD3E"]);
    fireEvent.click(screen.getByRole("button", { name: /Save as gene list/ }));
    const saved = useGeneListsStore.getState().lists.find((l) => l.name.startsWith("Minimal panel"));
    expect(saved?.genes).toEqual(["MS4A1", "GNLY"]);
  });

  it("shows the server's reason", async () => {
    api.runMarkerPanel.mockRejectedValue(new Error("No marker genes for 'x' yet"));
    render(<MinimalPanel datasetId="ds" groupby="x" />);
    fireEvent.click(screen.getByRole("button", { name: /Minimal panel/ }));
    fireEvent.click(screen.getByRole("button", { name: "Find panel" }));
    expect((await screen.findByRole("alert")).textContent).toContain("No marker genes");
  });

  it("shows a panel held for the same column", async () => {
    api.getMarkerPanel.mockResolvedValue({ result: RESULT, running: false });
    render(<MinimalPanel datasetId="ds" groupby="cell_type" />);
    fireEvent.click(screen.getByRole("button", { name: /Minimal panel/ }));
    await waitFor(() => expect(screen.getByTestId("panel-summary")).toBeInTheDocument());
  });
});
