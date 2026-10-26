import { describe, it, expect } from "vitest";
import { describePbModel, formatR, pbMethodLabel, PB_METHODS } from "./pseudobulk";

describe("pseudobulk labels", () => {
  it("names each model, DESeq2 for older servers", () => {
    expect(PB_METHODS.map((m) => m.value)).toEqual(["deseq2", "voom", "voombygroup"]);
    expect(pbMethodLabel("voombygroup")).toBe("voomByGroup");
    expect(pbMethodLabel(undefined)).toBe("DESeq2");
  });

  it("describes the model that ran", () => {
    expect(describePbModel({ pb_method: "deseq2", design: "~Donor + condition", ruv_k: 0 })).toBe(
      "Pseudobulk DESeq2 (Wald test, design ~Donor + condition)",
    );
    expect(describePbModel({ pb_method: "voom", design: "~W1 + Donor + condition", ruv_k: 1 })).toBe(
      "Pseudobulk limma-voom (moderated t on TMM log-CPM, 1 RUV factor, design ~W1 + Donor + condition)",
    );
    expect(describePbModel({ design: "~condition" })).toContain("DESeq2 (Wald test");
  });

  it("formats correlations, n/a when missing", () => {
    expect(formatR(0.6234)).toBe("0.62");
    expect(formatR(null)).toBe("n/a");
    expect(formatR(Number.NaN)).toBe("n/a");
  });
});
