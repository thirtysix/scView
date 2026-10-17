import { describe, it, expect } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import type { FilterGroupLoss, FilterPreview, MadThresholds } from "@/api/qc";
import { heat, SampleQcTable } from "./SampleQcTable";

function group(name: string, n: number, genes: number, counts: number, mito: number, removed: number,
  flags: FilterGroupLoss["flags"] = [], reasons: string[] = []): FilterGroupLoss {
  return {
    name, n_cells: n, n_removed: removed, fraction: removed / n, by_criterion: { max_genes: removed },
    median_genes: genes, median_counts: counts, median_pct_mt: mito, flags, reasons,
  };
}

function preview(): FilterPreview {
  return {
    n_cells: 2100, n_removed: 60, n_kept: 2040, fraction_removed: 60 / 2100,
    thresholds: { min_genes: 200, max_genes: 0, min_counts: 0, max_pct_mt: 20, drop_doublets: false, mode: "mad" },
    has_mt: true, doublet_calls: false, by_criterion: {},
    samples: {
      column: "Donor", reason: "donor", compared: true,
      typical: { median_genes: 640, median_counts: 1700, median_pct_mt: 3.1 },
      flag_help: { high_mito: "Median % mito far above the other samples." },
      groups: [
        group("d1", 1000, 650, 1750, 3.0, 10),
        group("d2", 1000, 630, 1690, 9.5, 40, ["high_mito"], ["Median % mito 9.5% against 3.1% across samples (over 3 MADs above)."]),
        group("tiny", 100, 640, 1700, 3.2, 1, ["few_cells"], ["Few cells: 100 against a median of 1,000 per sample."]),
      ],
    },
    clusters: null, warnings: [],
  };
}

const MAD: MadThresholds = {
  column: "Donor", reason: "donor", k: 5, k_mt: 3, ceiling_pct_mt: 20, min_sample_cells: 100,
  samples: [{ name: "d1", n_cells: 1000, pooled: false, min_counts: 300, max_counts: 9000, min_genes: 200,
    max_genes: 1500, max_pct_mt: 7, skipped: [], n_removed: 10 }],
  pooled: { n_cells: 2100, min_counts: 300, max_counts: 9000, min_genes: 200, max_genes: 1500, max_pct_mt: 7, skipped: [] },
  n_unlabelled: 0,
};

describe("heat", () => {
  it("shades from 0 to the column's highest, in one hue per column", () => {
    expect(heat("genes", 0, 600)).toBe("rgba(37, 99, 235, 0.040)");
    expect(heat("genes", 600, 600)).toBe("rgba(37, 99, 235, 0.450)");
    expect(heat("pct_mt", 600, 600)).toBe("rgba(217, 119, 6, 0.450)");
    expect(heat("genes", null, 600)).toBeUndefined();
    expect(heat("genes", 5, 0)).toBeUndefined(); // an empty column: no shading
  });
});

describe("SampleQcTable", () => {
  it("lists each sample's medians, shaded per column, its loss and its flags", () => {
    render(<SampleQcTable preview={preview()} mad={null} />);
    expect(screen.getByText("2 of 3 samples stand out")).toBeInTheDocument();
    const table = screen.getByRole("table", { name: "QC per sample" });
    const rows = within(table).getAllByRole("row");
    const d2 = within(rows[2]!).getAllByRole("cell");
    expect(d2.map((c) => c.textContent)).toEqual(["d2", "1,000", "630", "1,690", "9.5", "40 (4.0%)", "High mito"]);
    // The highest % mito gets the darkest amber; the removed share its red.
    expect(d2[4]!.style.backgroundColor).toBe("rgba(217, 119, 6, 0.45)");
    expect(d2[5]!.style.backgroundColor).toBe("rgba(220, 38, 38, 0.45)");
    expect(d2[5]).toHaveAttribute("title", "Each threshold on its own: max genes 40");
    expect(within(rows[2]!).getByText("High mito")).toHaveAttribute(
      "title", "Median % mito 9.5% against 3.1% across samples (over 3 MADs above).",
    );
    expect(within(rows[3]!).getByText("Few cells")).toBeInTheDocument();
    expect(rows.at(-1)!.textContent).toBe("Typical6401,7003.1");
    // MAD thresholds view only in MAD mode.
    expect(screen.queryByRole("button", { name: "Thresholds" })).not.toBeInTheDocument();
  });

  it("leaves out % mito when it can't be measured", () => {
    const p = preview();
    p.has_mt = false;
    p.samples!.groups.forEach((g) => delete g.median_pct_mt);
    render(<SampleQcTable preview={p} mad={null} />);
    expect(screen.queryByText("% mito")).not.toBeInTheDocument();
  });

  it("switches to each sample's MAD thresholds in MAD mode", () => {
    render(<SampleQcTable preview={preview()} mad={MAD} />);
    expect(screen.getByRole("button", { name: "QC medians" })).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getByRole("button", { name: "Thresholds" }));
    const rows = within(screen.getByRole("table", { name: "Thresholds per sample" })).getAllByRole("row");
    expect(rows[1]!.textContent).toBe("d11,000300–9,000200–1,500< 7.010 (1.0%)");
  });

  it("says when there is nothing to compare or too many samples", () => {
    const p = preview();
    p.samples = null;
    const { unmount } = render(<SampleQcTable preview={p} mad={null} />);
    expect(screen.getByText(/No sample, donor or batch column found/)).toBeInTheDocument();
    unmount();
    const q = preview();
    q.samples = { ...q.samples!, groups: [], note: "'well' has 96 groups; the loss is shown for at most 50." };
    render(<SampleQcTable preview={q} mad={null} />);
    expect(screen.getByText(/96 groups/)).toBeInTheDocument();
  });

  it("does not claim a stand-out with fewer than 3 samples", () => {
    const p = preview();
    p.samples = { ...p.samples!, compared: false, groups: p.samples!.groups.slice(0, 2).map((g) => ({ ...g, flags: [] })) };
    render(<SampleQcTable preview={p} mad={null} />);
    expect(screen.getByText("2 samples: too few to say which stands out")).toBeInTheDocument();
  });
});
