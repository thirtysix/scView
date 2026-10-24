/** The doublet step's detectors (backend core/doublets.py METHODS). */
export type DoubletMethod = "scrublet" | "scdblfinder-style" | "both";

export const DOUBLET_METHODS: readonly DoubletMethod[] = ["scrublet", "scdblfinder-style", "both"];

export const DOUBLET_METHOD_LABELS: Record<DoubletMethod, string> = {
  scrublet: "Scrublet (default)",
  "scdblfinder-style": "scDblFinder-style classifier",
  both: "Both, and compare",
};

/** What each choice does, for the step's info tip. */
export const DOUBLET_METHOD_POINTS: readonly string[] = [
  "Scrublet scores each cell by how many simulated doublets are among its nearest neighbours.",
  "The scDblFinder-style classifier (Germain et al. 2021) trains a model on artificial doublets " +
    "(two cells of different clusters added together) and scores each cell with it. It uses more " +
    "clues than Scrublet: how many artificial doublets are near the cell, its RNA content, and genes " +
    "seen together that rarely are. It sets its own threshold from the expected doublet rate.",
  "Both: each runs in every capture channel and the doublet card shows where they agree. " +
    "Filtering's Auto then drops only cells both call.",
  "Doublets of two cells of one type look like one cell of it: neither finds them.",
];

const pct = (f: number) => `${(100 * f).toFixed(1)}%`;

/** Under the detectors' agreement table: what they share, and what to make of the rest. */
export function detectorAgreementText(
  m: { both: number; scrublet: number; classifier: number; scrublet_only: number; classifier_only: number },
  scrubletAt: string,
): string {
  const union = m.both + m.scrublet_only + m.classifier_only;
  const share = union ? ` (${pct(m.both / union)} of the cells either calls)` : "";
  return (
    `${scrubletAt} calls ${m.scrublet.toLocaleString()} cells doublets, the classifier ${m.classifier.toLocaleString()}; ` +
    `both call ${m.both.toLocaleString()}${share}. A cell both call is a doublet with more confidence; ` +
    "one called by a single detector may be a doublet the other missed, or a false call."
  );
}

/** One line on the classifier's own calls (its threshold is its own, per channel). */
export function classifierText(c: { n_called: number; fraction: number; n_scored: number; n_unscored: number }): string {
  const unscored = c.n_unscored
    ? ` ${c.n_unscored.toLocaleString()} cells were not scored (a channel too small for it, or under 3 genes).`
    : "";
  return (
    `The classifier calls ${c.n_called.toLocaleString()} cells doublets (${pct(c.fraction)}), at the threshold it ` +
    `set in each channel from its artificial doublets and the expected rate.${unscored}`
  );
}

/** The History chip for a recorded doublet_method. */
export function doubletMethodText(value: unknown): string {
  switch (value) {
    case "scdblfinder-style":
      return "doublets: scDblFinder-style classifier";
    case "both":
      return "doublets: Scrublet + classifier";
    default:
      return "doublets: Scrublet";
  }
}
