import { describe, it, expect, vi } from "vitest";
import { render, renderHook, screen, waitFor, within } from "@testing-library/react";
import type { FilterPreview } from "@/api/qc";
import type { SexCheck, SexGroup } from "@/api/sampleQc";
import { SampleQcTable } from "./SampleQcTable";
import { SexCaption, useSexCheck } from "./SampleSexCheck";

const { getSexCheck } = vi.hoisted(() => ({ getSexCheck: vi.fn() }));
vi.mock("@/api/sampleQc", () => ({ getSexCheck }));

function sexGroup(name: string, call: SexGroup["call"], extra: Partial<SexGroup> = {}): SexGroup {
  return {
    name, n_cells: 1000, frac_female: call === "female" ? 0.7 : 0.01, frac_male: call === "male" ? 0.6 : 0.0,
    frac_both: 0, cpm_female: 400, cpm_male: 2, call, confidence: "high",
    evidence: `XIST in ${call === "female" ? "70%" : "1.0%"} of cells (1,000 cells).`, metadata: null,
    mismatch: false, ...extra,
  };
}

function sexCheck(groups: SexGroup[], extra: Partial<SexCheck> = {}): SexCheck {
  const summary = { female: 0, male: 0, mixed: 0, ambiguous: 0, undetermined: 0 };
  groups.forEach((g) => (summary[g.call] += 1));
  return {
    column: "Donor", reason: "donor", n_cells: 3000, species: "human", species_evidence: "",
    female_genes: ["XIST"], male_genes: ["RPS4Y1", "DDX3Y"], female_genes_missing: ["TSIX"],
    male_genes_missing: [], source: "layers['counts']", counts: true,
    thresholds: { on: 0.1, off: 0.02, min_cells: 10 }, metadata_column: null, groups, summary,
    confounding: [], warnings: [], notes: [], ...extra,
  };
}

function preview(): FilterPreview {
  const g = (name: string) => ({
    name, n_cells: 1000, n_removed: 0, fraction: 0, by_criterion: {}, median_genes: 600,
    median_counts: 1600, median_pct_mt: 3, flags: [], reasons: [],
  });
  return {
    n_cells: 3000, n_removed: 0, n_kept: 3000, fraction_removed: 0,
    thresholds: { min_genes: 200, max_genes: 0, min_counts: 0, max_pct_mt: 20, drop_doublets: false, mode: "mad" },
    has_mt: true, doublet_calls: false, by_criterion: {},
    samples: { column: "Donor", reason: "donor", compared: true, typical: {}, groups: [g("d1"), g("d2"), g("d3")] },
    clusters: null, warnings: [],
  };
}

describe("Sex column in the per-sample QC table", () => {
  it("shows each sample's call, marks a metadata mismatch, and captions the evidence", () => {
    const sex = sexCheck(
      [sexGroup("d1", "female"), sexGroup("d2", "male", { metadata: "female", mismatch: true }), sexGroup("d3", "mixed")],
      {
        metadata_column: "sex",
        warnings: ["'d2' is labelled female in 'sex' but its expression looks male: a possible sample swap or mislabel."],
      },
    );
    render(<SampleQcTable preview={preview()} mad={null} sex={sex} />);
    const table = screen.getByRole("table", { name: "QC per sample" });
    expect(within(table).getByText("Sex")).toBeInTheDocument();
    expect(screen.getByTestId("sex-d1")).toHaveTextContent("F");
    const d2 = screen.getByTestId("sex-d2");
    expect(d2).toHaveTextContent("M ≠");
    expect(d2.getAttribute("title")).toContain("sex: female, which DISAGREES");
    expect(screen.getByTestId("sex-d3")).toHaveTextContent("F+M");
    const caption = screen.getByTestId("sex-caption");
    expect(caption).toHaveTextContent("1 female, 1 male, 1 mixed");
    expect(caption).toHaveTextContent("XIST vs Y genes (RPS4Y1, DDX3Y)");
    expect(caption).toHaveTextContent("possible sample swap");
  });

  it("leaves the column out when the check is over other samples, and works without it", () => {
    const { rerender } = render(
      <SampleQcTable preview={preview()} mad={null} sex={sexCheck([sexGroup("CTRL", "female")], { column: "stim" })} />,
    );
    expect(within(screen.getByRole("table", { name: "QC per sample" })).queryByText("Sex")).toBeNull();
    expect(screen.queryByTestId("sex-caption")).toBeNull();
    rerender(<SampleQcTable preview={preview()} mad={null} />);
    expect(screen.queryByTestId("sex-caption")).toBeNull();
  });
});

describe("SexCaption", () => {
  it("states missing markers and an all-cells check", () => {
    render(
      <SexCaption
        sex={sexCheck([sexGroup("All cells", "undetermined", { confidence: "none" })], {
          column: null,
          female_genes: [],
          male_genes: [],
          notes: ["None of the sex marker genes (XIST, TSIX, Y-chromosome genes) are in the data."],
        })}
      />,
    );
    const caption = screen.getByTestId("sex-caption");
    expect(caption).toHaveTextContent("all cells: 1 undetermined");
    expect(caption).toHaveTextContent("None of the sex marker genes");
  });
});

describe("useSexCheck", () => {
  it("asks once per sample column, waits until enabled, and drops a failure", async () => {
    getSexCheck.mockResolvedValue(sexCheck([sexGroup("d1", "female")]));
    const { result, rerender } = renderHook(
      ({ column, enabled }: { column: string | null; enabled: boolean }) => useSexCheck("ds", column, 0, enabled),
      { initialProps: { column: null as string | null, enabled: false } },
    );
    expect(getSexCheck).not.toHaveBeenCalled();
    rerender({ column: "Donor", enabled: true });
    await waitFor(() => expect(result.current?.groups[0]?.name).toBe("d1"));
    rerender({ column: "Donor", enabled: true });
    expect(getSexCheck).toHaveBeenCalledTimes(1);
    expect(getSexCheck).toHaveBeenCalledWith("ds", "Donor");
    getSexCheck.mockRejectedValue(new Error("boom"));
    rerender({ column: "stim", enabled: true });
    await waitFor(() => expect(result.current).toBeNull());
  });
});
