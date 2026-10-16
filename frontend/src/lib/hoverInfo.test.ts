import { describe, it, expect } from "vitest";
import { colorByLine, formatHoverNumber } from "./hoverInfo";

describe("formatHoverNumber", () => {
  it("shows integers whole, fractions to 3 places, and no value as N/A", () => {
    expect(formatHoverNumber(12)).toBe((12).toLocaleString());
    expect(formatHoverNumber(2.15349)).toBe("2.153");
    expect(formatHoverNumber(0)).toBe("0");
    expect(formatHoverNumber(0.00042)).toBe("4.20e-4");
    expect(formatHoverNumber(12345.67)).toBe((12345.7).toLocaleString());
    expect(formatHoverNumber(NaN)).toBe("N/A");
  });
});

describe("colorByLine", () => {
  const categories = ["B", "NK", "T"];
  const codes = Int32Array.from([0, 1, 2, -1]);
  const colorOf = (code: number) => [code, code, code] as const;

  it("names the cell's category with the colour the plot draws it in", () => {
    expect(
      colorByLine({ label: "Cell type", values: codes, index: 1, type: "categorical", categories, colorOf }),
    ).toEqual({ label: "Cell type", value: "NK", color: [1, 1, 1] });
  });

  it("shows a cell without a category as N/A", () => {
    expect(
      colorByLine({ label: "Cell type", values: codes, index: 3, type: "categorical", categories, colorOf }),
    ).toEqual({ label: "Cell type", value: "N/A", color: null });
  });

  it("takes the value from the cell's metadata when the categories are not listed", () => {
    const line = colorByLine({ label: "Barcode", values: codes, index: 2, type: "categorical", fallback: "AAAC-1" });
    expect(line).toEqual({ label: "Barcode", value: "AAAC-1", color: null });
    expect(colorByLine({ label: "Barcode", values: codes, index: 2, type: "categorical" })).toBeNull();
  });

  it("shows the number for a gene, score or numeric column", () => {
    const expr = Float32Array.from([0, 2.5, NaN]);
    expect(colorByLine({ label: "CD3E expression", values: expr, index: 1, type: "continuous" })).toEqual({
      label: "CD3E expression",
      value: "2.500",
      color: null,
    });
    expect(colorByLine({ label: "CD3E expression", values: expr, index: 2, type: "continuous" })?.value).toBe("N/A");
  });

  it("treats float values as numbers while the colour type still says categorical", () => {
    const expr = Float32Array.from([0.5, 1.25]);
    expect(
      colorByLine({ label: "x", values: expr, index: 1, type: "categorical", categories, colorOf })?.value,
    ).toBe("1.250");
  });

  it("returns null when there is nothing to colour by or the index is out of range", () => {
    expect(colorByLine({ label: "x", values: null, index: 0, type: "categorical" })).toBeNull();
    expect(colorByLine({ label: "x", values: codes, index: 9, type: "categorical", categories })).toBeNull();
  });
});
