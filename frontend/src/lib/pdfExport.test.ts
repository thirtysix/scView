import { beforeAll, describe, it, expect } from "vitest";
import { pdfSafeSvg, pdfSafeText, svgToPdf } from "./pdfExport";
import { buildScatterSvg } from "./scatterSvg";

describe("pdfSafeText", () => {
  it("keeps Latin-1 and WinAnsi, spells out what Helvetica lacks", () => {
    expect(pdfSafeText("Zoë – 5 µm, 37 °C")).toBe("Zoë – 5 µm, 37 °C");
    expect(pdfSafeText("−2.5")).toBe("-2.5");
    expect(pdfSafeText("IFN-β log₂FC ≥ 1")).toBe("IFN-beta log2FC >= 1");
    expect(pdfSafeText("漢")).toBe("?");
  });

  it("rewrites the text nodes of an SVG", () => {
    const doc = new DOMParser().parseFromString(
      `<svg xmlns="http://www.w3.org/2000/svg"><text>−1</text><g><text>Δ score</text></g></svg>`,
      "image/svg+xml",
    );
    pdfSafeSvg(doc.documentElement);
    expect([...doc.querySelectorAll("text")].map((t) => t.textContent)).toEqual(["-1", "Delta score"]);
  });
});

describe("svgToPdf", () => {
  // jsdom has no SVG layout; svg2pdf measures centred text with getBBox.
  beforeAll(() => {
    (SVGElement.prototype as unknown as { getBBox: () => DOMRect }).getBBox = function (this: SVGElement) {
      return { x: 0, y: 0, width: (this.textContent ?? "").length * 6, height: 12 } as DOMRect;
    };
  });

  it("produces a non-empty PDF from a scatter SVG", async () => {
    const svg = buildScatterSvg({
      width: 200,
      height: 150,
      panels: [
        {
          view: { x: 0, y: 0, width: 200, height: 150 },
          sets: [
            {
              xy: Float32Array.from([10, 10, 100, 80]),
              colors: Uint8Array.from([255, 0, 0, 204, 0, 0, 255, 204]),
              radius: 2,
            },
          ],
          labels: [{ text: "B cells", x: 100, y: 70, faded: false }],
        },
      ],
      legend: { type: "categorical", title: "cell type", categories: ["B", "T"], colors: [[255, 0, 0], [0, 0, 255]] },
      colormap: "viridis",
      title: "IFN-β",
      axes: ["UMAP1", "UMAP2"],
      mode: "vector",
    });
    const blob = await svgToPdf(svg.svg, svg.width, svg.height);
    const bytes = new Uint8Array(await blob.arrayBuffer());
    expect(bytes.length).toBeGreaterThan(500);
    expect(new TextDecoder().decode(bytes.slice(0, 5))).toBe("%PDF-");
  });
});
