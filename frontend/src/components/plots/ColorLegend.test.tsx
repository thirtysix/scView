import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ColorLegend } from "./ColorLegend";

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
