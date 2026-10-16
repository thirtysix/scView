import type { QcByGroup } from "@/api/qc";
import type { ObsColumnInfo } from "@/api/types";
import { rgbCss, sortByCategoryOrder } from "@/lib/categoryColors";

/** The backend's cap (core/qc_groups.py MAX_GROUPS). */
export const QC_MAX_GROUPS = 50;

export interface QcGroupOption {
  name: string;
  n: number;
  /** Over the group cap: listed, so the user sees why it can't be picked. */
  disabled: boolean;
}

/** Columns to group QC by: categorical, text, bool or integer (a donor id can be an
 *  int, e.g. Kang's `ind`) with 2..100 values (more are per-cell ids), those over
 *  `maxGroups` disabled. */
export function qcGroupOptions(
  cols: ObsColumnInfo[] | undefined,
  maxGroups = QC_MAX_GROUPS,
): QcGroupOption[] {
  return (cols ?? [])
    .filter(
      (c) =>
        (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool" || /^u?int/.test(c.dtype)) &&
        c.n_unique >= 2 &&
        c.n_unique <= 100,
    )
    .map((c) => ({ name: c.name, n: c.n_unique, disabled: c.n_unique > maxGroups }));
}

/** Why the column was picked, for the caption. */
export function qcGroupReason(reason: QcByGroup["reason"]): string {
  switch (reason) {
    case "donor":
      return "the donor / sample column";
    case "batch":
      return "the batch column";
    case "clustering":
      return "the clustering";
    default:
      return "";
  }
}

type Rgb = readonly [number, number, number];

/** One scattergl trace per group of the sampled counts-vs-genes points, in the
 *  column's category order and colours, so the legend can hide a group. */
export function groupScatterTraces(
  scatter: NonNullable<QcByGroup["scatter"]>,
  groups: QcByGroup["groups"],
  order: readonly string[],
  colorOf: (name: string) => Rgb,
) {
  const xs = groups.map(() => [] as number[]);
  const ys = groups.map(() => [] as number[]);
  scatter.group.forEach((g, i) => {
    if (g < 0 || g >= groups.length) return;
    xs[g]!.push(scatter.x[i]!);
    ys[g]!.push(scatter.y[i]!);
  });
  const index = new Map(groups.map((g, i) => [g.name, i]));
  return sortByCategoryOrder(
    groups.map((g) => g.name),
    order,
  )
    .map((name) => index.get(name)!)
    .filter((i) => xs[i]!.length > 0)
    .map((i) => {
      const name = groups[i]!.name;
      return {
        type: "scattergl" as const,
        mode: "markers" as const,
        name,
        x: xs[i]!,
        y: ys[i]!,
        marker: { size: 3, color: rgbCss(colorOf(name)), opacity: 0.6 },
        // The name via fullData, not pasted in: a "%{" in a label would break the template.
        hovertemplate: "%{fullData.name}<br>counts %{x}<br>genes %{y}<extra></extra>",
      };
    });
}
