import { describe, it, expect, vi, beforeEach } from "vitest";

const client = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("./client", () => client);

import { checkScoreDepth, isScoreMethod, methodLabel, scoreGeneSet, scoreLabel } from "./genesetScore";
import { onSavedScore } from "@/lib/savedScores";

beforeEach(() => {
  client.apiFetch.mockReset().mockResolvedValue({});
});

describe("gene-set score API", () => {
  it("labels an AUCell score and leaves score_genes (the old default) bare", () => {
    expect(scoreLabel("IFN response", "aucell")).toBe("IFN response (AUCell)");
    expect(scoreLabel("IFN response", "score_genes")).toBe("IFN response");
    expect(scoreLabel("IFN response", undefined)).toBe("IFN response");
    expect(methodLabel("aucell")).toBe("AUCell");
    expect(scoreLabel("IFN response", "ssgsea")).toBe("IFN response (ssGSEA)");
    expect(scoreLabel("IFN response", "gsva")).toBe("IFN response (GSVA)");
    expect(isScoreMethod("gsva") && isScoreMethod("ssgsea") && !isScoreMethod("zscore")).toBe(true);
    expect(methodLabel(undefined)).toBe("score_genes");
  });

  it("sends the method, score_genes unless told otherwise, and saves unless told not to", async () => {
    await scoreGeneSet("ds1", ["ISG15"], "IFN");
    await scoreGeneSet("ds1", ["ISG15"], "IFN", "aucell");
    await scoreGeneSet("ds1", ["ISG15"], "IFN", "aucell", false);
    const bodies = client.apiFetch.mock.calls.map(([, init]) => JSON.parse(init.body));
    expect(client.apiFetch.mock.calls[0]![0]).toBe("/datasets/ds1/genesets/score");
    expect(bodies).toEqual([
      { gene_set: ["ISG15"], score_name: "IFN", method: "score_genes", save: true },
      { gene_set: ["ISG15"], score_name: "IFN", method: "aucell", save: true },
      { gene_set: ["ISG15"], score_name: "IFN", method: "aucell", save: false },
    ]);
  });

  it("announces a saved column, so the dataset's columns are read again", async () => {
    const heard: [string, string | null][] = [];
    const off = onSavedScore((d, c) => heard.push([d, c]));
    client.apiFetch.mockResolvedValueOnce({ saved_column: "score_IFN" }).mockResolvedValueOnce({ saved_column: null });
    await scoreGeneSet("ds1", ["ISG15"], "IFN");
    await scoreGeneSet("ds1", ["ISG15"], "IFN");
    off();
    expect(heard).toEqual([["ds1", "score_IFN"]]);
  });

  it("asks for the depth check, with a grouping only when given", async () => {
    await checkScoreDepth("ds1", ["A", "B"]);
    await checkScoreDepth("ds1", ["A"], "cell_type");
    expect(client.apiFetch.mock.calls[0]![0]).toBe("/datasets/ds1/genesets/score/compare");
    expect(JSON.parse(client.apiFetch.mock.calls[0]![1].body)).toEqual({ gene_set: ["A", "B"] });
    expect(JSON.parse(client.apiFetch.mock.calls[1]![1].body)).toEqual({ gene_set: ["A"], groupby: "cell_type" });
  });
});
