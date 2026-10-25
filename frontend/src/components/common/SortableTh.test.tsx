import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { SortableTh, ariaSort } from "./SortableTh";

describe("SortableTh", () => {
  it("is a column header with aria-sort and a real button to sort by", () => {
    const onSort = vi.fn();
    render(
      <table>
        <thead>
          <tr>
            <SortableTh onSort={onSort} sort="descending" title="log2 fold change" align="right">
              LogFC
            </SortableTh>
            <SortableTh onSort={() => {}} sort="none">
              Gene
            </SortableTh>
          </tr>
        </thead>
      </table>,
    );
    const header = screen.getByRole("columnheader", { name: "LogFC" });
    expect(header).toHaveAttribute("aria-sort", "descending");
    expect(header).toHaveAttribute("title", "log2 fold change");
    expect(screen.getByRole("columnheader", { name: "Gene" })).toHaveAttribute("aria-sort", "none");
    fireEvent.click(screen.getByRole("button", { name: "LogFC" }));
    expect(onSort).toHaveBeenCalledTimes(1);
  });

  it("ariaSort: the direction for the sorted column, none for the others", () => {
    expect(ariaSort("gene", "gene", "asc")).toBe("ascending");
    expect(ariaSort("gene", "gene", "desc")).toBe("descending");
    expect(ariaSort("pval", "gene", "asc")).toBe("none");
  });
});
