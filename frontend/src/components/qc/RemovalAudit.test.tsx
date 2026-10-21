import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import type { FilterPreview, RecordedRemovalAudit, RemovalAudit as Audit, RemovalRow } from "@/api/qc";
import type { DatasetInfo } from "@/api/types";
import { RemovalAudit, criteriaText, formatP, testTitle } from "./RemovalAudit";
import { useDatasetStore } from "@/stores/datasetStore";

const RULE =
  "Flagged when a filter removes more than 30% of a group, or its removal rate differs between conditions by 10 or more percentage points with p < 0.01 (chi-square).";

function row(name: string | null, n: [number, number], r: [number, number], p: number | null): RemovalRow {
  const rates = [r[0] / n[0], r[1] / n[1]];
  const hi = rates[1]! >= rates[0]! ? 1 : 0;
  const names = ["control", "drug"];
  return {
    name,
    n_cells: n[0] + n[1],
    n_removed: r[0] + r[1],
    fraction: (r[0] + r[1]) / (n[0] + n[1]),
    by_criterion: { max_pct_mt: r[0] + r[1] },
    conditions: { n_cells: [...n], n_removed: [...r], by_criterion: { max_pct_mt: [...r] } },
    test: {
      compared: names, high: names[hi]!, high_rate: rates[hi]!, low: names[1 - hi]!, low_rate: rates[1 - hi]!,
      max_diff_pp: Math.round(1000 * Math.abs(rates[1]! - rates[0]!)) / 10,
      p, statistic: p === null ? null : 95.2, dof: p === null ? null : 1, low_expected: false,
    },
  };
}

function lab(): Audit {
  return {
    label: { column: "cell_type", reason: "labels", n_groups: 2 },
    condition: { column: "treatment", reason: "condition", names: ["control", "drug"] },
    by_type: [row("CAF", [500, 500], [15, 15], 1), row("LNCaP", [500, 500], [250, 400], 2e-21)],
    overall: row(null, [1000, 1000], [265, 415], 3e-12),
    n_unlabelled: 0,
    note: null,
    rule: RULE,
    flags: [
      {
        kind: "condition", column: "cell_type", group: "LNCaP", n_cells: 1000, n_removed: 650, fraction: 0.65,
        whole: false, criterion: "max_pct_mt", condition: "treatment", p: 2e-21, max_diff_pp: 30,
        message: "In LNCaP, Max % mito removes 80% of drug cells (400 of 500) but 50% of control cells (250 of 500) (p = 2e-21): the loss depends on 'treatment'.",
      },
      {
        kind: "loss", column: "cell_type", group: "LNCaP", n_cells: 1000, n_removed: 650, fraction: 0.65,
        whole: false, criterion: "max_pct_mt", message: "Removes 65% of LNCaP (650 of 1,000 cells), by Max % mito.",
      },
    ],
  };
}

function preview(extra: Partial<FilterPreview> = {}): FilterPreview {
  return {
    n_cells: 2000, n_removed: 680, n_kept: 1320, fraction_removed: 0.34,
    thresholds: { min_genes: 0, max_genes: 0, min_counts: 0, max_pct_mt: 10, drop_doublets: false },
    has_mt: true, doublet_calls: false, by_criterion: {}, samples: null, clusters: null, warnings: [],
    audit: lab(), recorded: null, ...extra,
  };
}

function recorded(extra: Partial<RecordedRemovalAudit> = {}): RecordedRemovalAudit {
  const a = lab();
  return {
    ...a,
    by_type: a.by_type.map((r) => ({ ...r, removed_median: { total_counts: 1200, n_genes: 450, pct_mt: 14.2 } })),
    timestamp: "2026-10-20T09:00:00+00:00", mode: "fixed", n_before: 2000, n_removed: 680,
    truncated: false, has_mt: true, stale: false, columns: ["cell_type", "treatment", "Donor"], ...extra,
  };
}

describe("RemovalAudit", () => {
  beforeEach(() => {
    useDatasetStore.setState({
      currentDataset: {
        obs_columns: [
          { name: "cell_type", dtype: "category", n_unique: 2 },
          { name: "treatment", dtype: "category", n_unique: 2 },
          { name: "Donor", dtype: "category", n_unique: 4 },
          { name: "barcode", dtype: "object", n_unique: 2000 },
        ],
      } as unknown as DatasetInfo,
    });
  });

  it("tabulates the loss per type x condition, flags LNCaP and explains the test", () => {
    render(<RemovalAudit preview={preview()} columns={{}} onColumns={() => {}} />);
    const card = screen.getByTestId("removal-audit");
    expect(card).toHaveTextContent("Who the current thresholds remove: 680 of 2,000 cells (34.0%).");
    expect(card).toHaveTextContent("Per cell_type group (the author labels) and per treatment (a condition column).");
    const flags = within(screen.getByRole("list", { name: "Removal audit flags" })).getAllByRole("listitem");
    expect(flags.map((f) => f.textContent)).toEqual([lab().flags[0]!.message, lab().flags[1]!.message]);

    const table = screen.getByRole("table", { name: "Cells removed per group and condition" });
    const headers = within(table).getAllByRole("columnheader");
    expect(headers.map((h) => h.textContent)).toEqual(["cell_type", "All", "control", "drug", "Δ pts", "p"]);
    expect(headers[5]).toHaveAttribute("title", expect.stringContaining("Chi-square test of removed vs kept by treatment"));
    const rows = within(table).getAllByRole("row");
    // Flagged first, highlighted; the all-cells row last.
    expect(rows[1]).toHaveAttribute("data-flagged", "condition");
    expect(rows[1]!.className).toContain("bg-amber-50");
    expect(rows[1]).toHaveTextContent("LNCaP65.0% 650/1,00050.0% 250/50080.0% 400/50030.02e-21");
    expect(rows[2]).toHaveTextContent("CAF");
    expect(rows[2]).not.toHaveAttribute("data-flagged");
    expect(rows[3]).toHaveTextContent("All cells");
    const p = within(rows[1]!).getByText("2e-21");
    expect(p).toHaveAttribute("title", expect.stringContaining("across control, drug (chi-square 95.2, df 1, Yates-corrected)"));
    expect(within(rows[1]!).getByText("80.0%").closest("td")).toHaveAttribute(
      "title", "400 of 500 cells removed. Each threshold on its own: Max % mito 400.",
    );
    expect(card).toHaveTextContent(RULE);
    expect(card).toHaveTextContent("n = 2,000 cells; 2 cell_type groups; 2 treatment groups.");
  });

  it("degrades to a per-type table without a condition (no p column, says why)", () => {
    const a = lab();
    const strip = (r: RemovalRow): RemovalRow => {
      const out = { ...r };
      delete out.conditions;
      delete out.test;
      return out;
    };
    const plain: Audit = { ...a, condition: null, by_type: a.by_type.map(strip), overall: strip(a.overall), flags: [a.flags[1]!] };
    render(<RemovalAudit preview={preview({ audit: plain })} columns={{}} onColumns={() => {}} />);
    const headers = screen.getAllByRole("columnheader").map((h) => h.textContent);
    expect(headers).toEqual(["cell_type", "All"]);
    expect(screen.getByTestId("removal-audit")).toHaveTextContent(
      "Without a condition or sample column, removal is not tested against one.",
    );
    expect(screen.getByLabelText("Columns column for the removal audit")).toHaveDisplayValue("None found");
  });

  it("shows the record of what the filtering step removed, the preview one click away", () => {
    render(<RemovalAudit preview={preview({ recorded: recorded() })} columns={{}} onColumns={() => {}} />);
    const card = screen.getByTestId("removal-audit");
    expect(card).toHaveTextContent(/What the filtering step removed on .*2026.*: 680 of 2,000 cells \(34\.0%\)\./);
    expect(screen.getByRole("button", { name: "What was removed" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("LNCaP")).toHaveAttribute(
      "title", "LNCaP: 1,000 cells. The removed cells' medians: 1,200 counts, 450 genes, 14.2% mito.",
    );
    // The recorded columns are the choices.
    const rows = screen.getByLabelText("Rows column for the removal audit");
    expect(within(rows).getAllByRole("option").map((o) => o.textContent)).toEqual([
      "Detected: cell_type", "cell_type (2)", "treatment (2)", "Donor (4)",
    ]);
    fireEvent.click(screen.getByRole("button", { name: "Current thresholds" }));
    expect(card).toHaveTextContent("Who the current thresholds remove");
  });

  it("says when the record no longer matches the dataset", () => {
    const stale = { ...recorded(), stale: true, note: "The dataset has 100 cells now, but 1,320 were kept." };
    render(<RemovalAudit preview={preview({ recorded: stale })} columns={{}} onColumns={() => {}} />);
    expect(screen.queryByRole("button", { name: "What was removed" })).not.toBeInTheDocument();
    expect(screen.getByTestId("removal-audit-stale")).toHaveTextContent("1,320 were kept");
  });

  it("changes the columns through onColumns", () => {
    const onColumns = vi.fn();
    render(<RemovalAudit preview={preview()} columns={{}} onColumns={onColumns} />);
    const select = screen.getByLabelText("Columns column for the removal audit");
    // Columns with more than 20 values (barcode) are not offered.
    expect(within(select).queryByText(/barcode/)).not.toBeInTheDocument();
    fireEvent.change(select, { target: { value: "Donor" } });
    expect(onColumns).toHaveBeenCalledWith({ condition_column: "Donor" });
  });

  it("renders nothing for an older backend without the audit", () => {
    const { container } = render(<RemovalAudit preview={preview({ audit: undefined })} columns={{}} onColumns={() => {}} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("formatting helpers", () => {
  it("formats p, the criteria and the test tooltip", () => {
    expect(formatP(2.1e-21)).toBe("2e-21");
    expect(formatP(0.0342)).toBe("0.034");
    expect(formatP(0)).toBe("< 1e-300");
    expect(formatP(null)).toBe("–");
    expect(criteriaText({ max_pct_mt: 3, min_genes: 0 })).toBe("Each threshold on its own: Max % mito 3.");
    expect(criteriaText({})).toBe("No threshold removes any of these cells.");
    expect(testTitle(null, "stim")).toMatch(/Fewer than two stim groups have 10 or more cells/);
    expect(testTitle({ ...row("x", [10, 10], [0, 0], null).test!, p: null }, "stim")).toMatch(/nothing to test/);
    expect(testTitle({ ...row("x", [10, 10], [1, 5], 0.2).test!, low_expected: true }, "stim")).toMatch(
      /approximate\.$/,
    );
  });
});
