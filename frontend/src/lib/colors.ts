// Curated 20-color palette for single-cell biology — vibrant, perceptually
// distinct, no greys, optimized for UMAP/bar/violin on white & dark backgrounds.
export const CATEGORICAL_COLORS = [
  [0, 117, 220],     // vivid blue
  [230, 25, 75],     // crimson
  [60, 180, 75],     // green
  [255, 127, 0],     // orange
  [145, 30, 180],    // violet
  [0, 183, 183],     // teal
  [240, 163, 10],    // amber
  [233, 90, 196],    // pink
  [128, 128, 0],     // olive
  [70, 153, 144],    // dark teal
  [0, 51, 128],      // navy
  [255, 0, 16],      // red
  [43, 206, 72],     // vivid green
  [157, 204, 0],     // lime
  [194, 0, 136],     // magenta
  [116, 10, 255],    // bright violet
  [153, 63, 0],      // brown
  [0, 153, 143],     // sea green
  [255, 164, 5],     // golden
  [94, 106, 211],    // periwinkle
] as const;

// Viridis colormap — polynomial approximation matching matplotlib's viridis.
// Maps t in [0, 1] to [R, G, B] in [0, 255].
export function viridisColor(t: number): [number, number, number] {
  // Clamp t
  const s = Math.max(0, Math.min(1, t));
  // R: dark purple → blue → teal → green → yellow
  const r = Math.round(
    255 * Math.max(0, Math.min(1,
      0.267004 + s * (0.003263 + s * (-2.2956 + s * (14.4694 + s * (-25.5699 + s * 14.7853))))
    ))
  );
  // G: low → rising through teal/green → high
  const g = Math.round(
    255 * Math.max(0, Math.min(1,
      0.004874 + s * (0.8383 + s * (1.6024 + s * (-9.1894 + s * (14.4694 + s * -6.7275))))
    ))
  );
  // B: high in purple/blue → fading through green → low at yellow
  const b = Math.round(
    255 * Math.max(0, Math.min(1,
      0.329415 + s * (1.5023 + s * (-5.4402 + s * (9.3068 + s * (-7.9672 + s * 2.3286))))
    ))
  );
  return [r, g, b];
}

export function mapCategoryToColor(index: number): [number, number, number, number] {
  const safeIdx = Number.isFinite(index)
    ? ((Math.floor(index) % CATEGORICAL_COLORS.length) + CATEGORICAL_COLORS.length) % CATEGORICAL_COLORS.length
    : 0;
  const color = CATEGORICAL_COLORS[safeIdx]!;
  return [color[0], color[1], color[2], 255];
}

export function mapValueToColor(
  value: number,
  min: number,
  max: number
): [number, number, number, number] {
  const t = max > min ? (value - min) / (max - min) : 0;
  const [r, g, b] = viridisColor(Math.max(0, Math.min(1, t)));
  return [r, g, b, 255];
}
