import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ColorLegend } from "./ColorLegend";
import type { ResolvedColorScale } from "@/lib/colorScale";

const categories = ["B cell", "Monocyte", "NK cells"];
const colors: [number, number, number][] = [
  [255, 0, 0],
  [0, 255, 0],
  [0, 0, 255],
];

describe("ColorLegend (categorical)", () => {
  it("renders each category label", () => {
    render(<ColorLegend type="categorical" categories={categories} categoryColors={colors} />);
    for (const c of categories) expect(screen.getByText(c)).toBeInTheDocument();
  });

  it("fires onCategoryClick with the clicked category", () => {
    const onClick = vi.fn();
    render(
      <ColorLegend
        type="categorical"
        categories={categories}
        categoryColors={colors}
        onCategoryClick={onClick}
      />,
    );
    fireEvent.click(screen.getByText("Monocyte"));
    expect(onClick).toHaveBeenCalledWith("Monocyte");
  });

  it("offers an ask-about-this button per category", () => {
    const onAsk = vi.fn();
    render(
      <ColorLegend
        type="categorical"
        categories={categories}
        categoryColors={colors}
        onAskAbout={onAsk}
      />,
    );
    const askButtons = screen.getAllByTitle("Ask the co-pilot about this");
    expect(askButtons).toHaveLength(categories.length);
    fireEvent.click(askButtons[1]!);
    expect(onAsk).toHaveBeenCalledWith("Monocyte");
  });

  it("inline-renames a category via the pencil", () => {
    const onRename = vi.fn();
    render(
      <ColorLegend
        type="categorical"
        categories={categories}
        categoryColors={colors}
        onRenameCategory={onRename}
      />,
    );
    fireEvent.click(screen.getAllByTitle("Rename label")[0]!);
    const input = screen.getByDisplayValue("B cell");
    fireEvent.change(input, { target: { value: "B lymphocyte" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onRename).toHaveBeenCalledWith("B cell", "B lymphocyte");
  });
});

describe("ColorLegend (continuous, with a resolved scale)", () => {
  const scale = (over: Partial<ResolvedColorScale> = {}): ResolvedColorScale => ({
    colormap: "RdBu",
    diverging: true,
    min: -2,
    max: 2,
    centre: 0,
    dataMin: -1.5,
    dataMax: 2,
    clip: null,
    n: 100,
    ...over,
  });

  it("labels the ends and the centre, and says the range is centred", () => {
    render(<ColorLegend type="continuous" scale={scale()} label="IFN score" />);
    expect(screen.getByText("-2")).toBeInTheDocument();
    expect(screen.getByText("0")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByTestId("colour-scale-caption")).toHaveTextContent(
      "Colour range -2 to 2, centred on 0 (data -1.5 to 2).",
    );
  });

  it("draws the gradient with the scale's map, not the settings' sequential one", () => {
    const { container } = render(<ColorLegend type="continuous" scale={scale()} />);
    const bar = container.querySelector("div.h-3") as HTMLElement;
    expect(bar.style.background).toContain("rgb(5, 48, 97) 0.0%"); // RdBu's blue end
    expect(bar.style.background).toContain("rgb(103, 0, 31) 100.0%");
  });

  it("states the clipping and the share of cells outside, and no centre on a sequential scale", () => {
    const s = scale({
      colormap: "viridis", diverging: false, centre: null, min: 0, max: 3.2, dataMin: 0, dataMax: 9.1,
      clip: { preset: "p1", fracBelow: 0, fracAbove: 0.012 },
    });
    render(<ColorLegend type="continuous" scale={s} label="CD3E" />);
    expect(screen.getByTestId("colour-scale-caption")).toHaveTextContent(
      "Colour range 0 to 3.2, clipped at the 1st–99th percentiles (data 0 to 9.1); 1.2% of cells above.",
    );
    expect(screen.queryByTitle("The centre of the diverging scale")).toBeNull();
  });

  it("offers the clip presets, a manual pair, and a way back to the default", () => {
    const onClipChange = vi.fn();
    const { rerender } = render(
      <ColorLegend type="continuous" scale={scale()} clip={{ preset: "none" }} onClipChange={onClipChange} />,
    );
    fireEvent.change(screen.getByLabelText("Colour range clipping"), { target: { value: "p2" } });
    expect(onClipChange).toHaveBeenLastCalledWith({ preset: "p2" });
    expect(screen.queryByText("default")).toBeNull();

    fireEvent.change(screen.getByLabelText("Colour range clipping"), { target: { value: "manual" } });
    // a manual range starts from the data range
    expect(onClipChange).toHaveBeenLastCalledWith({ preset: "manual", min: -1.5, max: 2 });

    rerender(
      <ColorLegend type="continuous" scale={scale()} clip={{ preset: "manual", min: -1.5, max: 2 }}
        onClipChange={onClipChange} clipOverridden />,
    );
    fireEvent.change(screen.getByLabelText("Colour range maximum"), { target: { value: "1.5" } });
    expect(onClipChange).toHaveBeenLastCalledWith({ preset: "manual", min: -1.5, max: 1.5 });
    fireEvent.change(screen.getByLabelText("Colour range minimum"), { target: { value: "" } });
    expect(onClipChange).toHaveBeenLastCalledWith({ preset: "manual", min: undefined, max: 2 });
    fireEvent.click(screen.getByText("default"));
    expect(onClipChange).toHaveBeenLastCalledWith(null);
  });

  it("keeps the plain min/max legend for callers without a scale", () => {
    render(<ColorLegend type="continuous" min={0} max={4.5} label="CD3E" />);
    expect(screen.getByText("0.00")).toBeInTheDocument();
    expect(screen.getByText("4.50")).toBeInTheDocument();
    expect(screen.queryByTestId("colour-scale-caption")).toBeNull();
    expect(screen.queryByTestId("clip-control")).toBeNull();
  });
});
