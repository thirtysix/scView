import { describe, expect, it } from "vitest";
import type { MotifState } from "@/api/atacMotifs";
import { chromvarEstimateText, formatFold, formatPct, motifLabel, scanBlockedText, scanDownloadText, scanEstimateText } from "./atacMotifs";
import { parseSession, type SessionOverlay } from "./session";

const STATE: MotifState = {
  available: true,
  scan: { ready: false, running: false, progress: null, n_peaks: 143887, n_bases: 121861766, estimate_seconds: 110, estimate_bytes: 1.1e9, meta: null, p_cutoff: 5e-5 },
  files: {
    genome: { genome: "mm10", source: "UCSC mm10", url: "u", size: 714784109, downloaded: true }, genome_auto: true,
    jaspar: { name: "JASPAR", url: "u", size: 283814, downloaded: false }, jaspar_auto: true,
  },
  chromvar_estimate: { n_cells: 11208, estimate_seconds: 540, estimate_bytes: 9.5e8, disk_bytes: 1.5e8, iterations: 50 },
};

describe("motif card wording", () => {
  it("names only what is missing, and nothing when both are here", () => {
    expect(scanDownloadText(STATE)).toBe("The first run downloads the JASPAR 2024 motifs (1 MB), once.");
    const both = { ...STATE, files: { ...STATE.files!, jaspar: { ...STATE.files!.jaspar, downloaded: true } } };
    expect(scanDownloadText(both)).toBeNull();
  });

  it("says a switch blocks the download only when the file is missing", () => {
    expect(scanBlockedText(STATE)).toBeNull();
    const off = { ...STATE, files: { ...STATE.files!, jaspar_auto: false } };
    expect(scanBlockedText(off)).toContain("TF_MOTIFS_AUTO_DOWNLOAD is off");
    const genomeOff = { ...STATE, files: { ...STATE.files!, genome_auto: false } };
    expect(scanBlockedText(genomeOff)).toBeNull();
  });

  it("gives the estimates", () => {
    expect(scanEstimateText(STATE)).toContain("About 1 min 50 s and 1.1 GB");
    expect(chromvarEstimateText(STATE)).toContain("9 min, 950 MB beyond the open dataset and 150 MB on disk");
  });

  it("draws no number for a missing fold or TF", () => {
    expect(formatFold(null)).toBe("–");
    expect(formatFold(2.345)).toBe("2.35");
    expect(formatPct(null)).toBe("–");
    expect(formatPct(0.62)).toBe("62%");
    expect(motifLabel("SPI1")).toBe("SPI1 chromVAR z");
  });

  it("a session keeps a chromVAR overlay, and refuses one with no motif", () => {
    const session = (overlay: unknown) => ({ format: "scview-session", version: 1, view: { overlay } });
    const o: SessionOverlay = { type: "atac_motif", motif_id: "MA0080.7", motif: "SPI1" };
    expect(parseSession(session(o)).view?.overlay).toEqual(o);
    expect(() => parseSession(session({ type: "atac_motif" }))).toThrow();
  });
});
