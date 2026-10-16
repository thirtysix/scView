// Kept apart from plotConfig.ts so the deck.gl scatter can name its PNG
// without importing the Plotly bundle.

function stem(text: string): string {
  return text
    .replace(/[^A-Za-z0-9._-]+/g, "_")
    .replace(/_{2,}/g, "_")
    .replace(/^[._-]+|[._-]+$/g, "")
    .slice(0, 80)
    .replace(/[._-]+$/, "");
}

/**
 * `scview_<dataset>_<plot>` with anything but letters, digits, `.`, `_` and `-`
 * collapsed to one `_` (the backend's `file_stem` uses the same rule, so figures
 * and tables from one dataset sort together). Empty parts are left out.
 */
export function exportFilename(dataset: string | null | undefined, plot: string): string {
  return ["scview", stem(dataset ?? ""), stem(plot)].filter(Boolean).join("_");
}
