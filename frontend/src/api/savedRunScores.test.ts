import { describe, it, expect, vi, beforeEach } from "vitest";

const client = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("./client", () => client);

import { getProgramUsage } from "./programs";
import { getChromVarMotif } from "./atacMotifs";
import { onSavedScore } from "@/lib/savedScores";

beforeEach(() => {
  client.apiFetch.mockReset().mockResolvedValue({});
});

describe("drawing a program or a motif saves it, as a gene set's score is saved", () => {
  it("asks to save a drawn program unless told not to (a regroup, a session)", async () => {
    await getProgramUsage("ds1", 3);
    await getProgramUsage("ds1", 3, "cell_type");
    await getProgramUsage("ds1", 3, "cell_type", false);
    await getProgramUsage("ds1", 3, undefined, false);
    expect(client.apiFetch.mock.calls.map(([url]) => url)).toEqual([
      "/datasets/ds1/programs/3/usage?save=true",
      "/datasets/ds1/programs/3/usage?groupby=cell_type&save=true",
      "/datasets/ds1/programs/3/usage?groupby=cell_type",
      "/datasets/ds1/programs/3/usage",
    ]);
  });

  it("asks to save a drawn motif unless told not to", async () => {
    await getChromVarMotif("ds1", "MA0080.6");
    await getChromVarMotif("ds1", "MA0080.6", "cell_type", false);
    expect(client.apiFetch.mock.calls.map(([url]) => url)).toEqual([
      "/datasets/ds1/atac/motifs/chromvar/motif?motif_id=MA0080.6&save=true",
      "/datasets/ds1/atac/motifs/chromvar/motif?motif_id=MA0080.6&groupby=cell_type",
    ]);
  });

  it("announces a saved column, so Compare and Colour by list it at once", async () => {
    const heard: [string, string | null][] = [];
    const off = onSavedScore((d, c) => heard.push([d, c]));
    client.apiFetch
      .mockResolvedValueOnce({ saved_column: "program_k12_3" })
      .mockResolvedValueOnce({ saved_column: "chromvar_SPI1_MA0080.6" })
      .mockResolvedValueOnce({ saved_column: null, save_note: "Not saved: the dataset is being processed." });
    await getProgramUsage("ds1", 3);
    await getChromVarMotif("ds1", "MA0080.6");
    await getProgramUsage("ds1", 4);
    off();
    expect(heard).toEqual([["ds1", "program_k12_3"], ["ds1", "chromvar_SPI1_MA0080.6"]]);
  });
});
