// 20-colour categorical palette, colour-vision-deficiency (CVD) aware.
// 1-8: Okabe-Ito (Okabe & Ito 2008) with black replaced by mid grey, since the
// scatter background toggles white / dark slate. 9-20: Paul Tol muted + light
// schemes. Only the first 8 are CVD-checked (colors.cvd.test.ts); later entries
// are for >8 categories, where separation relies on hue/lightness alone.
const CVD_PALETTE: readonly (readonly [number, number, number])[] = [
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
];


export type CategoricalPaletteName = "cvd" | "classic" | "tableau20";
export type ColormapName = "viridis" | "cividis" | "magma" | "plasma";
type RGB = readonly [number, number, number];

export const CATEGORICAL_PALETTE_OPTIONS: { value: CategoricalPaletteName; label: string }[] = [
  { value: "cvd", label: "Colour-blind safe" },
  { value: "classic", label: "scView classic" },
  { value: "tableau20", label: "Tableau 20" },
];
export const COLORMAP_OPTIONS: { value: ColormapName; label: string }[] = [
  { value: "viridis", label: "Viridis" },
  { value: "cividis", label: "Cividis (colour-blind optimised)" },
  { value: "magma", label: "Magma" },
  { value: "plasma", label: "Plasma" },
];

// The pre-CVD palette (commit dd72971), restored verbatim.
const CLASSIC_PALETTE: readonly RGB[] = [[0,117,220],[230,25,75],[60,180,75],[255,127,0],[145,30,180],[0,183,183],[240,163,10],[233,90,196],[128,128,0],[70,153,144],[0,51,128],[255,0,16],[43,206,72],[157,204,0],[194,0,136],[116,10,255],[153,63,0],[0,153,143],[255,164,5],[94,106,211]];

// Tableau 20 (the standard Tableau 20 hex values).
const TABLEAU20_PALETTE: readonly RGB[] = [[78,121,167],[160,203,232],[242,142,43],[255,190,125],[89,161,79],[140,209,125],[182,153,45],[241,206,99],[73,152,148],[134,188,182],[225,87,89],[255,157,154],[121,112,110],[186,176,172],[211,114,149],[250,191,210],[176,122,161],[212,166,200],[157,118,96],[215,181,166]];

export const CATEGORICAL_PALETTES: Record<CategoricalPaletteName, readonly RGB[]> = {
  cvd: CVD_PALETTE,
  classic: CLASSIC_PALETTE,
  tableau20: TABLEAU20_PALETTE,
};

/** Default (colour-blind safe) palette; kept as a named export for tests and defaults. */
export const CATEGORICAL_COLORS = CVD_PALETTE;

export function getPalette(name: CategoricalPaletteName | undefined): readonly RGB[] {
  return CATEGORICAL_PALETTES[name ?? "cvd"] ?? CVD_PALETTE;
}

// 32 evenly spaced RGB stops per colormap, sampled from matplotlib 3.10.9
// (colormaps[name](linspace(0,1,32)), rounded to 0-255). Linearly interpolated.
export const COLORMAP_STOPS: Record<ColormapName, readonly RGB[]> = {
  viridis: [
    [68,1,84],
    [71,13,96],
    [72,24,106],
    [72,35,116],
    [71,46,124],
    [69,56,130],
    [66,65,134],
    [62,74,137],
    [58,84,140],
    [54,93,141],
    [50,101,142],
    [46,109,142],
    [43,117,142],
    [40,125,142],
    [37,132,142],
    [34,140,141],
    [31,148,140],
    [30,156,137],
    [32,163,134],
    [37,171,130],
    [46,179,124],
    [58,186,118],
    [72,193,110],
    [88,199,101],
    [108,205,90],
    [127,211,78],
    [147,215,65],
    [168,219,52],
    [192,223,37],
    [213,226,26],
    [234,229,26],
    [253,231,37],
  ],
  cividis: [
    [0,34,78],
    [0,40,91],
    [0,46,106],
    [5,51,113],
    [28,57,111],
    [41,63,110],
    [51,68,109],
    [60,74,108],
    [69,80,108],
    [77,85,108],
    [85,91,109],
    [92,97,110],
    [100,103,112],
    [107,109,114],
    [114,114,116],
    [120,120,119],
    [128,127,120],
    [136,133,120],
    [144,139,120],
    [151,145,119],
    [160,152,117],
    [168,158,115],
    [176,165,113],
    [185,171,109],
    [194,179,105],
    [203,185,101],
    [211,192,95],
    [220,200,89],
    [230,208,81],
    [239,215,72],
    [248,223,60],
    [254,232,56],
  ],
  magma: [
    [0,0,4],
    [3,3,18],
    [10,8,34],
    [19,13,52],
    [30,17,73],
    [42,17,92],
    [56,16,108],
    [69,16,119],
    [84,19,125],
    [96,24,128],
    [109,29,129],
    [121,34,130],
    [136,39,129],
    [148,44,128],
    [161,48,126],
    [174,52,123],
    [189,57,119],
    [202,62,114],
    [214,69,108],
    [226,77,102],
    [236,88,96],
    [243,101,92],
    [248,116,92],
    [251,131,95],
    [253,148,103],
    [254,163,111],
    [254,178,122],
    [254,193,133],
    [254,209,148],
    [253,224,161],
    [252,238,176],
    [252,253,191],
  ],
  plasma: [
    [13,8,135],
    [34,6,144],
    [49,5,151],
    [63,4,156],
    [78,2,162],
    [91,1,165],
    [103,0,168],
    [116,1,168],
    [129,4,167],
    [141,11,165],
    [152,20,160],
    [162,29,154],
    [173,39,147],
    [182,48,139],
    [191,57,132],
    [199,66,124],
    [207,76,116],
    [214,85,109],
    [221,94,102],
    [227,104,95],
    [233,114,87],
    [239,124,81],
    [243,135,74],
    [247,145,67],
    [250,158,59],
    [252,169,52],
    [253,181,46],
    [253,194,41],
    [252,208,37],
    [249,221,37],
    [245,235,39],
    [240,249,33],
  ],
};

/** Maps t in [0, 1] (clamped) to [R, G, B] in [0, 255] on the named colormap. */
export function colormapColor(name: ColormapName | undefined, t: number): [number, number, number] {
  const stops = COLORMAP_STOPS[name ?? "viridis"] ?? COLORMAP_STOPS.viridis;
  const s = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0)) * (stops.length - 1);
  const i = Math.min(Math.floor(s), stops.length - 2);
  const f = s - i;
  const a = stops[i]!;
  const b = stops[i + 1]!;
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

export function viridisColor(t: number): [number, number, number] {
  return colormapColor("viridis", t);
}

/** Plotly colorscale array ([[pos, "rgb(...)"], ...]) built from the same stops. */
export function plotlyColorscale(name: ColormapName | undefined): [number, string][] {
  const stops = COLORMAP_STOPS[name ?? "viridis"] ?? COLORMAP_STOPS.viridis;
  return stops.map((c, i) => [i / (stops.length - 1), `rgb(${c[0]},${c[1]},${c[2]})`]);
}

/**
 * RGBA for category code `index`. `slots` (the column's `color_slots`) maps a
 * code to its palette slot when a saved order pinned colours to labels; without
 * it the slot is the code.
 */
export function mapCategoryToColor(
  code: number,
  paletteName?: CategoricalPaletteName,
  slots?: readonly number[],
): [number, number, number, number] {
  const palette = getPalette(paletteName);
  const index = slots?.[code] ?? code;
  const safeIdx = Number.isFinite(index)
    ? ((Math.floor(index) % palette.length) + palette.length) % palette.length
    : 0;
  const color = palette[safeIdx]!;
  return [color[0], color[1], color[2], 255];
}

export function mapValueToColor(
  value: number,
  min: number,
  max: number,
  colormap?: ColormapName,
): [number, number, number, number] {
  // A missing value (e.g. pseudotime for a cell the trajectory never reached).
  if (!Number.isFinite(value)) return [200, 200, 200, 255];
  const t = max > min ? (value - min) / (max - min) : 0;
  const [r, g, b] = colormapColor(colormap, t);
  return [r, g, b, 255];
}
