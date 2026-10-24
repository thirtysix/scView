import { describe, it, expect } from "vitest";
import type { DoubletAgreement, DoubletChoices, DoubletSourceInfo } from "@/api/doubletSource";
import { agreementText, asDoubletSource, choiceText, sourceOptionLabel, sourceText } from "./doubletSource";

// The backend fixture of tests/test_doublet_source.py: 20 cells, hashing calls 4
// doublets and 2 negatives, Scrublet 4 doublets, 2 cells by both.
const HASHED: DoubletSourceInfo = {
  source: "hashing", label: "cell hashing", columns: ["hash_class"], n_cells: 20, n_doublets: 4, fraction: 0.2,
  hashing: { n_doublets: 4, fraction: 0.2, n_negatives: 2 }, scrublet: { n_doublets: 4, fraction: 0.2 },
  both: { n: 2, fraction: 0.1 }, options: ["auto", "hashing", "scrublet", "either"],
};

const CHOICES: DoubletChoices = {
  source: "auto", used: { source: "hashing", label: "cell hashing", n: 4 }, auto: { source: "hashing", label: "cell hashing" },
  options: ["auto", "hashing", "scrublet", "either"], n_by_source: { auto: 4, hashing: 4, scrublet: 4, either: 6 },
  both: 2, n_cells: 20, hash_negatives: 2, drop_hash_negatives: false, negative_note: "Not doublets.",
};

describe("doublet source texts", () => {
  it("names the source and how Scrublet agrees", () => {
    expect(sourceText(HASHED)).toBe("20.0% doublets by cell hashing; Scrublet flags 20.0%, 10.0% by both.");
    expect(sourceText({ ...HASHED, source: "scrublet", label: "Scrublet", hashing: null, both: null })).toBe(
      "20.0% doublets by Scrublet.",
    );
    expect(sourceText(null)).toBe("");
  });

  it("labels each choice with what it stands for and its count", () => {
    expect(sourceOptionLabel("auto", CHOICES)).toBe("Auto: cell hashing (4)");
    expect(sourceOptionLabel("either", CHOICES)).toBe("Either: hashing or Scrublet (6)");
    expect(sourceOptionLabel("scrublet", { ...CHOICES, n_by_source: {} })).toBe("Scrublet");
  });

  it("says what the choice drops", () => {
    expect(choiceText(CHOICES)).toBe("Drops 4 doublets by cell hashing; Scrublet calls 2 of them too.");
    expect(choiceText({ ...CHOICES, source: "either", used: { source: "either", label: "cell hashing or Scrublet", n: 6 } }))
      .toBe("Drops 6 cells that cell hashing or Scrublet calls doublets (2 by both).");
    expect(choiceText({ ...CHOICES, source: "hashing", used: null })).toMatch(/No cell hashing calls/);
    expect(choiceText({ ...CHOICES, used: { source: "scrublet", label: "Scrublet", n: 4 }, both: null })).toBe(
      "Drops 4 doublets by Scrublet.",
    );
    expect(choiceText({ ...CHOICES, used: { source: "scrublet", label: "Scrublet", n: 4 }, both: null, unscored: 40 }))
      .toBe("Drops 4 doublets by Scrublet. 40 cells Scrublet did not score (channels too small for it) are kept.");
  });

  it("names the two detectors' choices", () => {
    // tests/test_doublet_source.py with the classifier, no hashing: Scrublet 4, the classifier 4, both 2;
    // auto = the classifier, the choice here: both detectors
    const det: DoubletChoices = {
      ...CHOICES, hash_negatives: null, both: null,
      used: { source: "consensus", label: "both Scrublet and the doublet classifier", n: 2 },
      source: "consensus", auto: { source: "classifier", label: "the doublet classifier" },
      options: ["auto", "scrublet", "classifier", "consensus", "either"],
      n_by_source: { auto: 4, scrublet: 4, classifier: 4, consensus: 2, either: 6 },
    };
    expect(sourceOptionLabel("auto", det)).toBe("Auto: the doublet classifier (4)");
    expect(sourceOptionLabel("consensus", det)).toBe("Both detectors (Scrublet and classifier) (2)");
    expect(sourceOptionLabel("classifier", det)).toBe("Classifier (4)");
    expect(sourceOptionLabel("either", det)).toBe("Either: Scrublet or classifier (6)");
    expect(sourceOptionLabel("either", { ...det, options: [...det.options, "hashing"] })).toBe(
      "Either: hashing, Scrublet or classifier (6)",
    );
    expect(choiceText(det)).toBe("Drops 2 cells that both Scrublet and the classifier call doublets.");
    expect(choiceText({ ...det, used: { source: "classifier", label: "the doublet classifier", n: 4 }, unscored: 1 }))
      .toBe("Drops 4 doublets by the doublet classifier. 1 cells the classifier did not score (channels too small for it) are kept.");
    expect(
      sourceText({
        ...HASHED, source: "consensus", label: "both Scrublet and the doublet classifier", n_doublets: 2, fraction: 0.1,
        hashing: null, both: null, classifier: { n_doublets: 4, fraction: 0.2, n_unscored: 1 },
      }),
    ).toBe("10.0% doublets by both Scrublet and the doublet classifier (Scrublet 20.0%, the classifier 20.0%).");
  });

  it("reads anything else as auto", () => {
    expect(asDoubletSource("consensus")).toBe("consensus");
    expect(asDoubletSource("either")).toBe("either");
    expect(asDoubletSource("nope")).toBe("auto");
    expect(asDoubletSource(undefined)).toBe("auto");
  });

  it("explains the agreement table", () => {
    const a: DoubletAgreement = {
      n_cells: 20, hashing: 4, scrublet: 4, both: 2, hashing_only: 2, scrublet_only: 2, neither: 14, negatives: 2,
      negatives_scrublet: 1, samples: [], unassigned: null, threshold: 0.5, column: null,
    };
    expect(agreementText(a, "Scrublet at 0.5")).toBe(
      "Of 4 hashing doublets, Scrublet at 0.5 also calls 2 (50.0%). It calls 2 cells that hashing does not: " +
        "doublets of two cells from one sample, which hashing cannot see, or false calls; 1 of those are hashing " +
        "negatives (no tag).",
    );
  });
});
