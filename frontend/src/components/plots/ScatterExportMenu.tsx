import { useMemo } from "react";
import { ExportMenu, type ExportMenuItem } from "@/components/common/ExportMenu";
import { HYBRID_THRESHOLD, defaultPointMode, type PointMode } from "@/lib/scatterSvg";

export type FigureFormat = "svg" | "pdf";

/** A paged grid's second set of items: every panel on one sheet. */
export interface AllPanelsExport {
  /** Panels on the sheet */
  count: number;
  /** Points the sheet's figure would draw */
  nPoints: number;
  onPng: () => void | Promise<void>;
  onFigure: (format: FigureFormat, mode: PointMode) => Promise<void>;
}

interface ScatterExportMenuProps {
  /** Points the figure would draw (decides whether points as an image is offered first) */
  nPoints: number;
  onPng: () => void | Promise<void>;
  onFigure: (format: FigureFormat, mode: PointMode) => Promise<void>;
  /** What is downloaded, for the button's tooltip, e.g. "Download figure" or "Download grid" */
  label?: string;
  /** Extra line on the SVG / PDF items, e.g. what they leave out */
  figureNote?: string;
  disabled?: boolean;
  className?: string;
  onOpenChange?: (open: boolean) => void;
  /** A paged grid: what the first items save, said first in their hints (e.g. "This page, panels 7–12 of 24.") */
  pageNote?: string;
  /** A paged grid: PNG / SVG / PDF of every panel on one sheet, after the page's items */
  allPanels?: AllPanelsExport | null;
}

/**
 * The one download menu of an embedding scatter or grid: PNG, SVG and PDF.
 * Above HYBRID_THRESHOLD points the SVG / PDF items with the points as one
 * embedded high-resolution image (text and legend still vector) come first,
 * as the recommended choice; all-vector stays available below them. A paged
 * grid adds the same three formats for all its panels on one sheet.
 */
export function ScatterExportMenu({
  nPoints,
  onPng,
  onFigure,
  label = "Download figure",
  figureNote,
  disabled = false,
  className = "",
  onOpenChange,
  pageNote,
  allPanels = null,
}: ScatterExportMenuProps) {
  const items = useMemo<ExportMenuItem[]>(() => {
    const page = figureItems({ nPoints, onPng, onFigure, figureNote, lead: pageNote });
    if (!allPanels) return page;
    const all = figureItems({
      nPoints: allPanels.nPoints,
      onPng: allPanels.onPng,
      onFigure: allPanels.onFigure,
      figureNote,
      keyPrefix: "all-",
      labelPrefix: `All ${allPanels.count} panels: `,
      lead: "Every panel on one sheet.",
    });
    return [...page, { ...all[0]!, dividerBefore: true }, ...all.slice(1)];
  }, [nPoints, onPng, onFigure, figureNote, pageNote, allPanels]);

  // The wrapper takes the placement (ExportMenu's own root is `relative`).
  return (
    <div className={className}>
      <ExportMenu
        items={items}
        label={label}
        compact
        disabled={disabled}
        className="align-top"
        onOpenChange={onOpenChange}
      />
    </div>
  );
}

/** PNG, then SVG and PDF (points as an image first above HYBRID_THRESHOLD points). */
function figureItems(o: {
  nPoints: number;
  onPng: () => void | Promise<void>;
  onFigure: (format: FigureFormat, mode: PointMode) => Promise<void>;
  figureNote?: string;
  keyPrefix?: string;
  labelPrefix?: string;
  /** Said first in every hint, e.g. which panels are saved */
  lead?: string;
}): ExportMenuItem[] {
  const { nPoints, onPng, onFigure, figureNote, keyPrefix = "", labelPrefix = "", lead } = o;
  const big = defaultPointMode(nPoints) === "raster";
  const note = (hint: string) => [lead, hint, figureNote].filter(Boolean).join(" ");
  const png: ExportMenuItem = {
    key: `${keyPrefix}png`,
    label: `${labelPrefix}PNG (high resolution)`,
    hint: lead ? `${lead} White background, with legend` : "White background, with legend",
    icon: "image",
    onSelect: onPng,
  };
  const vector = (format: FigureFormat): ExportMenuItem => ({
    key: `${keyPrefix}${format}-vector`,
    label: `${labelPrefix}${big ? `${format.toUpperCase()}, every point vector` : `${format.toUpperCase()} (vector)`}`,
    hint: big
      ? note(`${nPoints.toLocaleString()} points: a large file, slow to open.`)
      : note(format === "pdf" ? "Text selectable, white background." : "Editable, white background."),
    icon: "image",
    onSelect: () => onFigure(format, "vector"),
  });
  const raster = (format: FigureFormat, first: boolean): ExportMenuItem => ({
    key: `${keyPrefix}${format}-raster`,
    label: `${labelPrefix}${format.toUpperCase()}, points as image`,
    hint: note(
      first
        ? `Recommended above ${HYBRID_THRESHOLD.toLocaleString()} points: points as one high-resolution image, text and legend vector (what journals accept).`
        : "Points as one high-resolution image, text and legend vector.",
    ),
    icon: "image",
    dividerBefore: first,
    onSelect: () => onFigure(format, "raster"),
  });
  if (!big) return [png, { ...vector("svg"), dividerBefore: true }, vector("pdf")];
  return [png, raster("svg", true), raster("pdf", false), { ...vector("svg"), dividerBefore: true }, vector("pdf")];
}
