import { describe, it, expect } from "vitest";
import { columnNameError, labelError, suggestColumnName } from "./labels";

describe("labelError", () => {
  it("accepts a trimmed label and refuses empty, long or invisible ones", () => {
    expect(labelError("  CD4 T ")).toBeNull();
    expect(labelError("   ")).toMatch(/Type a label/);
    expect(labelError("x".repeat(81))).toMatch(/too long/);
    expect(labelError("T​cell")).toMatch(/invisible/);
  });
});

describe("columnNameError", () => {
  const existing = ["leiden", "cell_type"];

  it("accepts plain names", () => {
    for (const name of ["my labels", "Cell_type-2", "a.b"]) expect(columnNameError(name, existing)).toBeNull();
  });

  it("refuses names the backend would refuse", () => {
    for (const name of ["", "2nd", "a/b", "scview_mine", "x".repeat(65), "_index"]) {
      expect(columnNameError(name, existing)).not.toBeNull();
    }
  });

  it("refuses an existing column whatever its case", () => {
    expect(columnNameError("Leiden", existing)).toMatch(/already exists/);
  });
});

describe("suggestColumnName", () => {
  it("offers the first free manual_labels name", () => {
    expect(suggestColumnName(["leiden"])).toBe("manual_labels");
    expect(suggestColumnName(["manual_labels", "Manual_Labels_2"])).toBe("manual_labels_3");
  });
});
