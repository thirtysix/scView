import { describe, expect, it } from "vitest";
import { parseTranscripts, shownText, sizeText, timeText, TRANSCRIPT_COLORS } from "./transcripts";

function wire(gene: number[], x: number[], y: number[]): ArrayBuffer {
  const n = gene.length;
  const pad = (4 - ((4 + n) % 4)) % 4;
  const out = new ArrayBuffer(4 + n + pad + 8 * n);
  new Int32Array(out, 0, 1)[0] = n;
  new Uint8Array(out, 4, n).set(gene);
  new Float32Array(out, 4 + n + pad, n).set(x);
  new Float32Array(out, 4 + n + pad + 4 * n, n).set(y);
  return out;
}

describe("the transcripts answer", () => {
  it.each([[1], [3], [4], [5]])("reads %i points past the padding, y flipped to the tissue world", (n) => {
    const gene = Array.from({ length: n }, (_, i) => i % 2);
    const x = gene.map((_, i) => i + 0.5);
    const y = gene.map((_, i) => 10 + i);
    const p = parseTranscripts(wire(gene, x, y));
    expect(p.n).toBe(n);
    expect([...p.positions.slice(0, 2)]).toEqual([0.5, -10]);
    expect([...p.colors.slice(0, 4)]).toEqual([...TRANSCRIPT_COLORS[0]!, 255]);
    if (n > 1) expect([...p.colors.slice(4, 7)]).toEqual([...TRANSCRIPT_COLORS[1]!]);
  });

  it("takes an empty view and refuses a short answer", () => {
    expect(parseTranscripts(wire([], [], [])).n).toBe(0);
    expect(() => parseTranscripts(wire([0], [1], [2]).slice(0, 10))).toThrow(/announces/);
  });
});

describe("the words", () => {
  it("say how many of those in view are shown", () => {
    expect(shownText(5, 5)).toBe("5 transcripts in view, all shown.");
    expect(shownText(1, 1)).toBe("1 transcript in view, all shown.");
    expect(shownText(300000, 3348837)).toContain("Showing 300,000 of 3,348,837 transcripts in view");
  });

  it("say sizes and times", () => {
    expect(sizeText(312 * 1024 ** 2)).toBe("312 MB");
    expect(sizeText(2.5 * 1024 ** 3)).toBe("2.5 GB");
    expect(timeText(14.2)).toBe("14 s");
    expect(timeText(600)).toBe("10 min");
  });
});
