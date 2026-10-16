// 20-colour categorical palette, colour-vision-deficiency (CVD) aware.
// 1-8: Okabe-Ito (Okabe & Ito 2008) with black replaced by mid grey, since the
// scatter background toggles white / dark slate. 9-20: Paul Tol muted + light
// schemes. Only the first 8 are CVD-checked (colors.cvd.test.ts); later entries
// are for >8 categories, where separation relies on hue/lightness alone.
export const CATEGORICAL_COLORS = [
  [230,159,0],      // orange
  [86,180,233],     // sky blue
  [0,158,115],      // bluish green
  [240,228,66],     // yellow
  [0,114,178],      // blue
  [213,94,0],       // vermillion
  [204,121,167],    // reddish purple
  [153,153,153],    // mid grey (Okabe-Ito black dropped: invisible on the dark background)
  [51,34,136],      // indigo
  [136,204,238],    // cyan
  [68,170,153],     // teal
  [17,119,51],      // green
  [153,153,51],     // olive
  [221,204,119],    // sand
  [204,102,119],    // rose
  [136,34,85],      // wine
  [170,68,153],     // purple
  [119,170,221],    // light blue
  [238,136,102],    // light orange
  [187,204,51],     // pear
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
  // A missing value (e.g. pseudotime for a cell the trajectory never reached).
  if (!Number.isFinite(value)) return [200, 200, 200, 255];
  const t = max > min ? (value - min) / (max - min) : 0;
  const [r, g, b] = viridisColor(Math.max(0, Math.min(1, t)));
  return [r, g, b, 255];
}
