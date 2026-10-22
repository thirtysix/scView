import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { FilterPreview, FilterThresholds } from "@/api/qc";
import type { DoubletChoices } from "@/api/doubletSource";
import { DoubletFilterControl } from "./DoubletFilterControl";

const T: FilterThresholds = { min_genes: 200, max_genes: 0, min_counts: 0, max_pct_mt: 20, drop_doublets: true };

const CHOICES: DoubletChoices = {
  source: "auto", used: { source: "hashing", label: "cell hashing", n: 4 }, auto: { source: "hashing", label: "cell hashing" },
  options: ["auto", "hashing", "scrublet", "either"], n_by_source: { auto: 4, hashing: 4, scrublet: 4, either: 6 },
  both: 2, n_cells: 20, hash_negatives: 2, drop_hash_negatives: false,
  negative_note: "Hashing negatives have no tag above background: usually empty droplets or cells whose tagging failed, not doublets.",
};

function preview(doublets: DoubletChoices | null | undefined, calls = true): FilterPreview {
  return {
    n_cells: 20, n_removed: 4, n_kept: 16, fraction_removed: 0.2, thresholds: { ...T, max_pct_mt: 20 }, has_mt: false,
    doublet_calls: calls, by_criterion: {}, samples: null, clusters: null, warnings: [], doublets,
  };
}

describe("DoubletFilterControl", () => {
  it("chooses whose doublets once hashing and Scrublet both exist, and offers the negatives", () => {
    const onChange = vi.fn();
    render(<DoubletFilterControl preview={preview(CHOICES)} thresholds={T} onChange={onChange} />);
    const select = screen.getByLabelText("Doublet source") as HTMLSelectElement;
    expect(select.value).toBe("auto");
    expect(Array.from(select.options).map((o) => o.textContent)).toEqual([
      "Auto: cell hashing (4)", "Cell hashing (4)", "Scrublet (4)", "Either: hashing or Scrublet (6)",
    ]);
    fireEvent.change(select, { target: { value: "either" } });
    expect(onChange).toHaveBeenCalledWith({ qc_doublet_source: "either" });
    expect(screen.getByTestId("doublet-filter-choice")).toHaveTextContent(
      "Drops 4 doublets by cell hashing; Scrublet calls 2 of them too.",
    );
    fireEvent.click(screen.getByLabelText("Also drop hashing negatives (2 cells)"));
    expect(onChange).toHaveBeenCalledWith({ drop_hash_negatives: true });
    expect(screen.getByText(/usually empty droplets or cells whose tagging failed/)).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Also drop predicted doublets"));
    expect(onChange).toHaveBeenCalledWith({ drop_doublets: false });
  });

  it("names Scrublet without a choice when there is no hashing", () => {
    const scrublet: DoubletChoices = {
      ...CHOICES, used: { source: "scrublet", label: "Scrublet", n: 40 }, auto: { source: "scrublet", label: "Scrublet" },
      options: ["auto", "scrublet"], n_by_source: { auto: 40, scrublet: 40 }, both: null, hash_negatives: null,
    };
    render(<DoubletFilterControl preview={preview(scrublet)} thresholds={T} onChange={vi.fn()} />);
    expect(screen.queryByLabelText("Doublet source")).not.toBeInTheDocument();
    expect(screen.getByText("(Scrublet)")).toBeInTheDocument();
    expect(screen.queryByText(/hashing negatives/)).not.toBeInTheDocument();
    expect(screen.getByTestId("doublet-filter-choice")).toHaveTextContent("Drops 40 doublets by Scrublet.");
  });

  it("keeps the plain checkbox for an older backend, and nothing without calls", () => {
    const { container, rerender } = render(
      <DoubletFilterControl preview={preview(undefined)} thresholds={T} onChange={vi.fn()} />,
    );
    expect(screen.getByLabelText("Also drop predicted doublets")).toBeChecked();
    expect(screen.queryByLabelText("Doublet source")).not.toBeInTheDocument();
    rerender(<DoubletFilterControl preview={preview(null, false)} thresholds={T} onChange={vi.fn()} />);
    expect(container.textContent).toBe("");
  });
});
