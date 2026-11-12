/**
 * The app's help text: one catalog of plain-language tips, keyed by a stable id.
 *
 * Each group of tabs keeps its own file, so work on one group never touches another's.
 * Keys: `tab.<subtab id>` for a Unified View tab (shown when hovering the tab), and
 * `<area>.<thing>` for a tip inside it (shown by `<Help id="..."/>`). `help.test.ts`
 * checks every used id exists, every entry is used, and every Unified View tab has one.
 *
 * Who reads it: biologists new to single-cell analysis. Say what the thing does, when
 * it is useful and how to read it, in one to three short sentences.
 */
import { GENES_HELP } from "./genes";
import { MODALITIES_HELP } from "./modalities";
import { PLOT_HELP } from "./plot";
import { QC_HELP } from "./qc";
import { SPATIAL_HELP } from "./spatial";

export const HELP_GROUPS = {
  plot: PLOT_HELP,
  genes: GENES_HELP,
  modalities: MODALITIES_HELP,
  spatial: SPATIAL_HELP,
  qc: QC_HELP,
};

export const HELP: Readonly<Record<string, string>> = Object.assign({}, ...Object.values(HELP_GROUPS));

/** The tip for an id, or undefined (the test catches a missing id before it ships). */
export function helpText(id: string): string | undefined {
  return HELP[id];
}
