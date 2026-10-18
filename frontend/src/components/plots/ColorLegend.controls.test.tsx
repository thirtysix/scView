import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import { ColorLegend } from "./ColorLegend";

const categories = ["B cell", "Monocyte", "NK cells"];
const colors: [number, number, number][] = [
  [255, 0, 0],
  [0, 255, 0],
  [0, 0, 255],
];

describe("ColorLegend multi-select", () => {
  it("sends Ctrl, Cmd and Shift clicks to the toggle and plain clicks to the click handler", () => {
    const onClick = vi.fn();
    const onToggle = vi.fn();
    render(
      <ColorLegend type="categorical" categories={categories} categoryColors={colors}
        onCategoryClick={onClick} onCategoryToggle={onToggle} />,
    );
    fireEvent.click(screen.getByText("Monocyte"), { ctrlKey: true });
    fireEvent.click(screen.getByText("NK cells"), { metaKey: true });
    fireEvent.click(screen.getByText("B cell"), { shiftKey: true });
    expect(onToggle.mock.calls.map((c) => c[0])).toEqual(["Monocyte", "NK cells", "B cell"]);
    expect(onClick).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText("Monocyte"));
    expect(onClick).toHaveBeenCalledWith("Monocyte");
  });

  it("without a toggle handler a modified click is a plain click", () => {
    const onClick = vi.fn();
    render(<ColorLegend type="categorical" categories={categories} categoryColors={colors} onCategoryClick={onClick} />);
    fireEvent.click(screen.getByText("Monocyte"), { ctrlKey: true });
    expect(onClick).toHaveBeenCalledWith("Monocyte");
  });

  it("marks the picked groups, dims the rest and offers select and clear", () => {
    const onSelect = vi.fn();
    const onClear = vi.fn();
    render(
      <ColorLegend type="categorical" categories={categories} categoryColors={colors}
        onCategoryToggle={vi.fn()} pickedCategories={["B cell", "NK cells"]} pickedCellCount={4512}
        onSelectPicked={onSelect} onClearPicked={onClear} />,
    );
    expect(screen.getByText("B cell")).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Monocyte")).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText("Monocyte").parentElement!.className).toContain("opacity-40");
    expect(screen.getByText("NK cells").parentElement!.className).not.toContain("opacity-40");
    expect(screen.getByText(/2 groups · 4,512 cells/)).toBeInTheDocument();
    fireEvent.click(screen.getByText("Select these cells"));
    fireEvent.click(screen.getByText("Clear"));
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onClear).toHaveBeenCalledTimes(1);
  });
});

describe("ColorLegend order mode", () => {
  const renderOrder = (props: Record<string, unknown> = {}) => {
    const onReorder = vi.fn();
    const onSort = vi.fn();
    render(
      <ColorLegend type="categorical" categories={categories} categoryColors={colors} label="cell_type"
        onReorder={onReorder} onSortCategories={onSort} {...props} />,
    );
    fireEvent.click(screen.getByLabelText("Reorder categories"));
    return { onReorder, onSort };
  };

  it("has no order control unless the parent can save one", () => {
    render(<ColorLegend type="categorical" categories={categories} categoryColors={colors} label="cell_type" />);
    expect(screen.queryByLabelText("Reorder categories")).toBeNull();
  });

  it("moves a category up or down and sends the full new order", () => {
    const { onReorder } = renderOrder();
    expect(screen.getByLabelText("Move B cell up")).toBeDisabled();
    expect(screen.getByLabelText("Move NK cells down")).toBeDisabled();
    fireEvent.click(screen.getByLabelText("Move NK cells up"));
    expect(onReorder).toHaveBeenLastCalledWith(["B cell", "NK cells", "Monocyte"]);
    fireEvent.click(screen.getByLabelText("Move B cell down"));
    expect(onReorder).toHaveBeenLastCalledWith(["Monocyte", "B cell", "NK cells"]);
  });

  it("drags a category to another place", () => {
    const { onReorder } = renderOrder();
    const rows = screen.getAllByTestId("legend-order-row");
    const dataTransfer = { effectAllowed: "" };
    fireEvent.dragStart(rows[2]!, { dataTransfer });
    fireEvent.dragOver(rows[0]!, { dataTransfer });
    fireEvent.drop(rows[0]!, { dataTransfer });
    expect(onReorder).toHaveBeenCalledWith(["NK cells", "B cell", "Monocyte"]);
  });

  it("offers the quick sorts and leaves order mode on Done", () => {
    const { onSort } = renderOrder();
    const group = screen.getByRole("group", { name: "Category order" });
    fireEvent.click(within(group).getByText("A–Z"));
    fireEvent.click(within(group).getByText("By size"));
    expect(onSort.mock.calls.map((c) => c[0])).toEqual(["natural", "size"]);
    fireEvent.click(within(group).getByText("Done"));
    expect(screen.queryByRole("group", { name: "Category order" })).toBeNull();
    expect(screen.queryAllByTestId("legend-order-row")).toHaveLength(0);
  });

  it("holds the controls while an order is being saved", () => {
    renderOrder({ orderBusy: true });
    expect(screen.getByLabelText("Move NK cells up")).toBeDisabled();
    expect(screen.getByText("By size")).toBeDisabled();
  });
});
