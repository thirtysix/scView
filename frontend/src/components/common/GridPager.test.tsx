import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { hoverTip } from "@/test/hoverTip";
import { useSplitViewStore } from "@/stores/splitViewStore";
import { GridPager } from "./GridPager";

describe("GridPager", () => {
  it("is hidden for four panels or fewer", () => {
    render(<GridPager n={4} perPage={6} page={0} onPerPageChange={vi.fn()} onPageChange={vi.fn()} />);
    expect(screen.queryByTestId("grid-pager")).toBeNull();
  });

  it("offers 4 / 6 / 9 / 12 / all and says which page is shown", () => {
    render(<GridPager n={24} perPage={6} page={1} onPerPageChange={vi.fn()} onPageChange={vi.fn()} />);
    const select = screen.getByLabelText("Panels per page") as HTMLSelectElement;
    expect([...select.options].map((o) => o.textContent)).toEqual(["4", "6", "9", "12", "All (24)"]);
    expect(select.value).toBe("6");
    expect(screen.getByText("Page 2 of 4")).toBeInTheDocument();
  });

  it("pages with Prev / Next, disabled at the ends", () => {
    const onPage = vi.fn();
    const { rerender } = render(
      <GridPager n={13} perPage={6} page={0} onPerPageChange={vi.fn()} onPageChange={onPage} />,
    );
    expect(screen.getByRole("button", { name: "Previous page" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    expect(onPage).toHaveBeenLastCalledWith(1);
    rerender(<GridPager n={13} perPage={6} page={2} onPerPageChange={vi.fn()} onPageChange={onPage} />);
    expect(screen.getByText("Page 3 of 3")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Next page" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Previous page" }));
    expect(onPage).toHaveBeenLastCalledWith(1);
  });

  it("a new page size keeps the current panels in view", () => {
    const onPerPage = vi.fn();
    render(<GridPager n={24} perPage={6} page={2} onPerPageChange={onPerPage} onPageChange={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Panels per page"), { target: { value: "4" } });
    expect(onPerPage).toHaveBeenLastCalledWith(4, 3);
    fireEvent.change(screen.getByLabelText("Panels per page"), { target: { value: "all" } });
    expect(onPerPage).toHaveBeenLastCalledWith("all", 0);
  });

  it("shows no Prev / Next when everything fits one page", () => {
    render(<GridPager n={12} perPage="all" page={0} onPerPageChange={vi.fn()} onPageChange={vi.fn()} />);
    expect(screen.getByLabelText("Panels per page")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Next page" })).toBeNull();
  });

  it("explains itself from the help catalog on hover", async () => {
    render(<GridPager n={24} perPage={6} page={0} onPerPageChange={vi.fn()} onPageChange={vi.fn()} />);
    expect(await hoverTip(screen.getByText("Per page"))).toMatch(/How many panels the grid shows at once/);
  });
});

describe("splitViewStore paging", () => {
  it("starts a new split or comparison on page 1, keeps the page for the same one", () => {
    const s = useSplitViewStore.getState;
    s().setSplitBy("sample");
    s().setPage(2);
    s().setSplitBy("sample");
    expect(s().page).toBe(2);
    s().setSplitBy("donor");
    expect(s().page).toBe(0);
    s().setCompareColumns(["a", "b"]);
    s().setPage(1);
    s().setCompareColumns(["a", "b"]);
    expect(s().page).toBe(1);
    s().setCompareColumns(["a", "c"]);
    expect(s().page).toBe(0);
  });

  it("keeps a gene grid's page as genes come and go, resets it for a blend", () => {
    const s = useSplitViewStore.getState;
    s().setGeneGrid({ datasetId: "d", kind: "features", genes: ["A", "B"] });
    s().setPage(1);
    s().setGeneGrid({ datasetId: "d", kind: "features", genes: ["A", "B", "C"] });
    expect(s().page).toBe(1);
    s().setGeneGrid({ datasetId: "d", kind: "blend", genes: ["A", "B"] });
    expect(s().page).toBe(0);
    s().setGeneGrid(null);
  });

  it("sets a page size with the page that keeps the panels in view", () => {
    const s = useSplitViewStore.getState;
    expect(s().panelsPerPage).toBe(6);
    s().setPanelsPerPage(12, 1);
    expect([s().panelsPerPage, s().page]).toEqual([12, 1]);
    s().setPanelsPerPage(6);
    expect(s().page).toBe(1);
  });
});
