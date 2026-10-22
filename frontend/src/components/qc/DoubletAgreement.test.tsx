import { describe, it, expect } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import type { DoubletAgreement as Agreement } from "@/api/doubletSource";
import { DoubletAgreement } from "./DoubletAgreement";

// tests/test_doublet_source.py's fixture: both 2, hashing only 2, Scrublet only 2, neither 14.
const A: Agreement = {
  n_cells: 20, hashing: 4, scrublet: 4, both: 2, hashing_only: 2, scrublet_only: 2, neither: 14, negatives: 2,
  negatives_scrublet: 1, threshold: 0.5, column: "hash_sample",
  samples: [
    { name: "s1", n_cells: 10, hashing: 4, scrublet: 3, both: 2 },
    { name: "s2", n_cells: 9, hashing: 0, scrublet: 1, both: 0 },
  ],
  unassigned: { n_cells: 1, hashing: 0, scrublet: 0, both: 0 },
};

describe("DoubletAgreement", () => {
  it("shows the 2 x 2 counts and the samples, with cells without a sample", () => {
    render(<DoubletAgreement agreement={A} scrubletAt="Scrublet at 0.5" />);
    const two = screen.getByRole("table", { name: "Cell hashing x Scrublet doublet calls" });
    const rows = within(two).getAllByRole("row").map((r) => r.textContent);
    expect(rows).toEqual([
      "Scrublet doubletScrublet singletTotal", "Hashing doublet224", "Hashing not doublet21416", "Total41620",
    ]);
    expect(screen.getByTestId("doublet-agreement-text")).toHaveTextContent(
      "Of 4 hashing doublets, Scrublet at 0.5 also calls 2 (50.0%).",
    );
    expect(screen.getByTestId("doublet-agreement-text")).toHaveTextContent("n = 20 cells.");
    const per = screen.getByRole("table", { name: "Cell hashing x Scrublet per sample" });
    expect(within(per).getAllByRole("row").map((r) => r.textContent)).toEqual([
      "hash_sampleCellsHashingScrubletBoth", "s110432", "s29010", "(no value)1000",
    ]);
  });

  it("caps long sample lists behind show all", () => {
    const many = { ...A, unassigned: null,
      samples: Array.from({ length: 10 }, (_, i) => ({ name: `s${i}`, n_cells: 2, hashing: 0, scrublet: 0, both: 0 })) };
    render(<DoubletAgreement agreement={many} scrubletAt="Scrublet's saved call" />);
    const per = screen.getByRole("table", { name: "Cell hashing x Scrublet per sample" });
    expect(within(per).getAllByRole("row")).toHaveLength(9);
    fireEvent.click(screen.getByRole("button", { name: "Show all 10" }));
    expect(within(per).getAllByRole("row")).toHaveLength(11);
  });
});
