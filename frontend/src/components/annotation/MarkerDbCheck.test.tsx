import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { MarkerDbResult, MarkerDbSource, MarkerDbStatus } from "@/api/annotation";

const { getMarkerDbStatus, runMarkerDbCheck } = vi.hoisted(() => ({
  getMarkerDbStatus: vi.fn(),
  runMarkerDbCheck: vi.fn(),
}));
vi.mock("@/api/annotation", () => ({ getMarkerDbStatus, runMarkerDbCheck }));

import { MarkerDbCheck, markerDbSourceText } from "./MarkerDbCheck";

const SRC: MarkerDbSource[] = [
  {
    name: "panglao", label: "PanglaoDB", release: "27 Mar 2020", home: "https://panglaodb.se/",
    paper: "Franzén et al. 2019, Database (baz046)", doi: "10.1093/database/baz046", url: "u", downloaded: true,
    fetched_at: "2026-10-19T03:00:00+00:00", served_from: "u", sha256: "s", rows: 1, types: { human: 1, mouse: 1 },
    download_bytes: 1, error: null, failed_at: null,
  },
  {
    name: "cellmarker2", label: "CellMarker 2.0", release: "2.0", home: "http://117.50.127.228/CellMarker/",
    paper: "Hu et al. 2023, Nucleic Acids Res (51:D870)", doi: "10.1093/nar/gkac947", url: "u", downloaded: true,
    fetched_at: "2026-10-18T03:00:00+00:00", served_from: "u", sha256: "s", rows: 1, types: { human: 1, mouse: 1 },
    download_bytes: 1, error: null, failed_at: null,
  },
  {
    name: "act", label: "ACT", release: "21 Feb 2024", home: "http://xteam.xbio.top/ACT/",
    paper: "Quan et al. 2023, Genome Med (15:91)", doi: "10.1186/s13073-023-01249-5", url: "u", downloaded: true,
    fetched_at: "2026-10-17T03:00:00+00:00", served_from: "u", sha256: "s", rows: 1, types: { human: 1, mouse: 1 },
    download_bytes: 1, error: null, failed_at: null,
  },
];
const ONTO = { downloaded: true, url: "u", version: null, fetched_at: null, error: null };
const READY: MarkerDbStatus = { state: "ready", auto_download: true, sources: SRC, ontology: ONTO, dataset_species: "human" };
const MISSING: MarkerDbStatus = {
  ...READY, state: "missing", sources: SRC.map((s) => ({ ...s, downloaded: false, fetched_at: null })),
};

const T = { cell_type: "T cells", cl_id: null, sources: ["PanglaoDB"], p: 1e-9, overlap: ["CD3D", "CD3E"], n_db_genes: 40 };
const NK = { cell_type: "NK cells", cl_id: null, sources: ["PanglaoDB"], p: 1e-12, overlap: ["NKG7", "GNLY"], n_db_genes: 40 };
const RESULT: MarkerDbResult = {
  column: "cell_type", species: "human", species_note: null,
  labels: [
    { label: "T", markers: ["CD3D"], match: { types: ["T cells"], how: "exact", how_text: "same name", via_parent: false },
      db_type: "T cells", cl_id: null, sources: ["PanglaoDB"], p: 1e-9, overlap: ["CD3D", "CD3E"], n_db_genes: 40,
      verdict: "supported", verdict_text: "supported", suggestions: [T] },
    { label: "CD8 T", markers: ["NKG7"],
      match: { types: ["T cells"], how: "cl-ancestor", how_text: "matched via parent term", via_parent: true },
      db_type: "T cells", cl_id: null, sources: ["PanglaoDB"], p: 1, overlap: [], n_db_genes: 40,
      verdict: "no_support", verdict_text: "no database support", suggestions: [NK, T] },
    { label: "Zorblax", markers: ["X"], match: { types: [], how: null, how_text: null, via_parent: false },
      db_type: null, cl_id: null, sources: [], p: null, overlap: [], n_db_genes: 0,
      verdict: "not_in_db", verdict_text: "not in database", suggestions: [] },
  ],
  verdict_counts: { supported: 1, no_support: 1, too_few: 0, not_in_db: 1, no_markers: 0 },
  n_measured: 100, n_db_types: 10, ontology: true, sources: SRC,
  verdict_sources: ["PanglaoDB", "CellMarker 2.0", "ACT"], suggestion_sources: ["PanglaoDB", "CellMarker 2.0"], params: {}, note: "A reason to look.", elapsed_seconds: 0.2,
};

describe("markerDbSourceText", () => {
  it("names the sources on disk and the latest fetch", () => {
    expect(markerDbSourceText(SRC)).toMatch(
      /^PanglaoDB \(27 Mar 2020\) \+ CellMarker 2\.0 \+ ACT \(21 Feb 2024\); fetched .*19.*2026$/, // the latest fetch
    );
    expect(markerDbSourceText([{ ...SRC[0]!, downloaded: false }, SRC[1]!])).toMatch(/^CellMarker 2\.0; fetched/);
    expect(markerDbSourceText([])).toBe("");
  });
});

describe("MarkerDbCheck", () => {
  beforeEach(() => {
    getMarkerDbStatus.mockReset().mockResolvedValue(READY);
    runMarkerDbCheck.mockReset().mockResolvedValue(RESULT);
  });

  it("checks the column and shows verdicts, matches and suggestions", async () => {
    render(<MarkerDbCheck datasetId="d1" column="cell_type" />);
    await waitFor(() => expect(screen.getByTestId("marker-db-source").textContent).toContain("PanglaoDB (27 Mar 2020)"));
    // Links go to each database's site and its paper, not to licence terms
    const links = within(screen.getByTestId("marker-db-source")).getAllByRole("link").map((a) => a.getAttribute("href"));
    expect(links).toEqual([
      "https://panglaodb.se/", "https://doi.org/10.1093/database/baz046",
      "http://117.50.127.228/CellMarker/", "https://doi.org/10.1093/nar/gkac947",
      "http://xteam.xbio.top/ACT/", "https://doi.org/10.1186/s13073-023-01249-5",
    ]);
    expect(screen.getByTestId("marker-db-source").textContent).toContain(
      "Suggestions rank PanglaoDB + CellMarker 2.0 types only",
    );
    fireEvent.click(screen.getByRole("button", { name: "Check against marker databases" }));
    await waitFor(() => expect(screen.getByTestId("marker-db-result")).toBeTruthy());
    expect(runMarkerDbCheck).toHaveBeenCalledWith("d1", "cell_type", {});
    const rows = within(screen.getByTestId("marker-db-labels")).getAllByRole("row").slice(1);
    expect(rows.map((r) => (r as HTMLTableRowElement).cells[1]!.textContent)).toEqual([
      "supported", "no database support", "not in database",
    ]);
    expect(screen.queryByText(/wrong/i)).toBeNull();
    expect(rows[0]!.textContent).toContain("CD3D, CD3E");
    expect(rows[1]!.textContent).toContain("matched via parent term");
    expect(rows[1]!.textContent).toContain("0 of 40 measured");
    expect((rows[1] as HTMLTableRowElement).cells[4]!.textContent).toMatch(/^NK cells.*T cells/);
  });

  it("downloads on first use and offers a retry when a download fails", async () => {
    getMarkerDbStatus.mockResolvedValue(MISSING);
    runMarkerDbCheck.mockRejectedValueOnce(
      new Error('API error 503: {"detail":"The marker databases could not be downloaded (PanglaoDB: timeout)."}'),
    );
    render(<MarkerDbCheck datasetId="d1" column="cell_type" />);
    const button = await screen.findByRole("button", { name: "Download databases and check" });
    fireEvent.click(button);
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("could not be downloaded"));
    fireEvent.click(screen.getByRole("button", { name: /Retry download/ }));
    await waitFor(() => expect(screen.getByTestId("marker-db-result")).toBeTruthy());
    expect(runMarkerDbCheck).toHaveBeenLastCalledWith("d1", "cell_type", { retry: true });
  });

  it("says when ACT is missing but the other two work", async () => {
    getMarkerDbStatus.mockResolvedValue({
      ...READY, state: "partial", sources: SRC.map((s) => (s.name === "act" ? { ...s, downloaded: false } : s)),
    });
    render(<MarkerDbCheck datasetId="d1" column="cell_type" />);
    await waitFor(() => expect(screen.getByTestId("marker-db-source").textContent).toContain("ACT not downloaded."));
    expect(screen.getByRole("button", { name: "Check against marker databases" })).toBeTruthy();
  });

  it("says so when downloading is off", async () => {
    getMarkerDbStatus.mockResolvedValue({ ...MISSING, state: "disabled", auto_download: false });
    render(<MarkerDbCheck datasetId="d1" column="cell_type" />);
    await waitFor(() => expect(screen.getByTestId("marker-db-disabled")).toBeTruthy());
    expect((screen.getByRole("button", { name: "Download databases and check" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("sends a chosen species and drops the result when the column changes", async () => {
    const { rerender } = render(<MarkerDbCheck datasetId="d1" column="cell_type" />);
    await waitFor(() => expect(screen.getByRole("option", { name: "From the genes (human)" })).toBeTruthy());
    fireEvent.change(screen.getByLabelText("Species"), { target: { value: "mouse" } });
    fireEvent.click(screen.getByRole("button", { name: "Check against marker databases" }));
    await waitFor(() => expect(screen.getByTestId("marker-db-result")).toBeTruthy());
    expect(runMarkerDbCheck).toHaveBeenCalledWith("d1", "cell_type", { species: "mouse" });
    rerender(<MarkerDbCheck datasetId="d1" column="leiden" />);
    await waitFor(() => expect(screen.queryByTestId("marker-db-result")).toBeNull());
  });
});
