import { useMemo } from "react";
import { ExportMenu, type ExportMenuItem } from "@/components/common/ExportMenu";
import { HYBRID_THRESHOLD, defaultPointMode, type PointMode } from "@/lib/scatterSvg";

export type FigureFormat = "svg" | "pdf";

interface ScatterExportMenuProps {
  /** Points the figure would draw (decides whether points as an image is offered first) */
  nPoints: number;
  onPng: () => void;
  onFigure: (format: FigureFormat, mode: PointMode) => Promise<void>;
  /** What is downloaded, for the button's tooltip, e.g. "Download figure" or "Download grid" */
  label?: string;
  /** Extra line on the SVG / PDF items, e.g. what they leave out */
  figureNote?: string;
  disabled?: boolean;
  className?: string;
  onOpenChange?: (open: boolean) => void;
}

/**
 * The one download menu of an embedding scatter or grid: PNG, SVG and PDF.
 * Above HYBRID_THRESHOLD points the SVG / PDF items with the points as one
 * embedded high-resolution image (text and legend still vector) come first,
 * as the recommended choice; all-vector stays available below them.
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
}: ScatterExportMenuProps) {
  const items = useMemo<ExportMenuItem[]>(() => {
    const big = defaultPointMode(nPoints) === "raster";
    const note = (hint: string) => (figureNote ? `${hint} ${figureNote}` : hint);
    const png: ExportMenuItem = {
      key: "png",
      label: "PNG (high resolution)",
      hint: "White background, with legend",
      icon: "image",
      onSelect: onPng,
    };
    const vector = (format: FigureFormat): ExportMenuItem => ({
      key: `${format}-vector`,
      label: big ? `${format.toUpperCase()}, every point vector` : `${format.toUpperCase()} (vector)`,
      hint: big
        ? note(`${nPoints.toLocaleString()} points: a large file, slow to open.`)
        : note(format === "pdf" ? "Text selectable, white background." : "Editable, white background."),
      icon: "image",
      onSelect: () => onFigure(format, "vector"),
    });
    const raster = (format: FigureFormat, first: boolean): ExportMenuItem => ({
      key: `${format}-raster`,
      label: `${format.toUpperCase()}, points as image`,
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
  }, [nPoints, onPng, onFigure, figureNote]);

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
