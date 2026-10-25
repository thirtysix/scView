/**
 * SVG -> PDF for figure downloads (the embedding scatter, the split grids and
 * every Plotly plot), with jsPDF + svg2pdf.js (both MIT). Both are loaded on
 * the first PDF download only, so they stay out of the main bundle.
 *
 * Text stays text (selectable, editable in Illustrator/Inkscape) in the PDF
 * standard font Helvetica. That font covers Latin-1 and a few more signs
 * (WinAnsi), so pdfSafeText first spells out what it lacks: the minus sign
 * Plotly uses on axes, Greek letters, sub- and superscripts.
 */

/** CSS pixels -> PDF points (96 px = 72 pt = one inch). */
export const PT_PER_PX = 0.75;

// WinAnsi's characters outside Latin-1 (jsPDF's standard fonts encode these).
const WIN_ANSI_EXTRA = new Set("€‚ƒ„…†‡ˆ‰Š‹ŒŽ‘’“”•–—˜™š›œžŸ");

const GREEK: Record<string, string> = {
  α: "alpha", β: "beta", γ: "gamma", δ: "delta", ε: "epsilon", ζ: "zeta", η: "eta", θ: "theta",
  ι: "iota", κ: "kappa", λ: "lambda", μ: "µ", ν: "nu", ξ: "xi", π: "pi", ρ: "rho", σ: "sigma",
  τ: "tau", υ: "upsilon", φ: "phi", χ: "chi", ψ: "psi", ω: "omega",
  Γ: "Gamma", Δ: "Delta", Θ: "Theta", Λ: "Lambda", Ξ: "Xi", Π: "Pi", Σ: "Sigma", Φ: "Phi",
  Ψ: "Psi", Ω: "Omega",
};
const SIGNS: Record<string, string> = {
  "−": "-", "‐": "-", "‑": "-", "‒": "-", " ": " ", " ": " ",
  " ": " ", " ": " ", "​": "", "≥": ">=", "≤": "<=", "≠": "!=", "≈": "~", "∼": "~",
  "→": "->", "←": "<-", "↑": "^", "↓": "v", "∞": "inf", "√": "sqrt", "✓": "v", "✗": "x",
  "⁰": "0", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+",
  "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
  "₋": "-", "₊": "+",
};

/** `text` with every character Helvetica (WinAnsi) lacks spelled out, else "?". */
export function pdfSafeText(text: string): string {
  let out = "";
  for (const ch of text) {
    const code = ch.codePointAt(0)!;
    if ((code >= 0x20 && code <= 0x7e) || (code >= 0xa0 && code <= 0xff) || WIN_ANSI_EXTRA.has(ch) || ch === "\n") {
      out += ch;
    } else {
      out += GREEK[ch] ?? SIGNS[ch] ?? "?";
    }
  }
  return out;
}

/** Make every text node of an SVG safe for the PDF's standard font, and its
 *  weights ones that font has (svg2pdf knows normal and bold, not 600). */
export function pdfSafeSvg(root: Element): void {
  for (const el of root.querySelectorAll("[font-weight], [style*='font-weight']")) {
    const w = el.getAttribute("font-weight");
    if (w && /^\d+$/.test(w)) el.setAttribute("font-weight", Number(w) >= 600 ? "bold" : "normal");
    const style = el.getAttribute("style");
    if (style && /font-weight:\s*\d/.test(style)) {
      el.setAttribute(
        "style",
        style.replace(/font-weight:\s*(\d+)/g, (_, n: string) => `font-weight: ${Number(n) >= 600 ? "bold" : "normal"}`),
      );
    }
  }
  const walker = root.ownerDocument.createTreeWalker(root, 4 /* NodeFilter.SHOW_TEXT */);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = node.nodeValue ?? "";
    const safe = pdfSafeText(text);
    if (safe !== text) node.nodeValue = safe;
  }
}

/**
 * Turn an SVG (as text) of `width` x `height` CSS pixels into a one-page PDF of
 * the same size. The SVG is parsed and briefly attached off-screen (svg2pdf
 * reads computed styles), then removed.
 */
export async function svgToPdf(svgText: string, width?: number, height?: number): Promise<Blob> {
  const [{ jsPDF }, { svg2pdf }] = await Promise.all([import("jspdf"), import("svg2pdf.js")]);
  const parsed = new DOMParser().parseFromString(svgText, "image/svg+xml");
  const svg = parsed.documentElement;
  if (svg.nodeName !== "svg") throw new Error("Could not read the figure for the PDF.");
  pdfSafeSvg(svg);
  // Without a size given, the SVG's own (Plotly's exports carry one)
  const own = (attr: string) => Number.parseFloat(svg.getAttribute(attr) ?? "");
  const pxW = width ?? own("width");
  const pxH = height ?? own("height");
  if (!(pxW > 0 && pxH > 0)) throw new Error("The figure has no size to make a PDF of.");
  const w = pxW * PT_PER_PX;
  const h = pxH * PT_PER_PX;
  const doc = new jsPDF({
    unit: "pt",
    format: [w, h],
    orientation: w >= h ? "landscape" : "portrait",
    compress: true,
  });
  const urls = await imagesToBlobUrls(svg);
  const host = document.createElement("div");
  host.style.cssText = "position:fixed;left:-100000px;top:0;width:0;height:0;overflow:hidden";
  const node = document.importNode(svg, true);
  host.appendChild(node);
  document.body.appendChild(host);
  try {
    await svg2pdf(node, doc, { x: 0, y: 0, width: w, height: h });
  } finally {
    host.remove();
    urls.forEach((u) => URL.revokeObjectURL(u));
  }
  return doc.output("blob");
}

/**
 * Point every embedded (data: URL) image at a blob: URL instead. svg2pdf
 * parses a data URL with a regular expression that overflows the stack on a
 * multi-megabyte image (the points of a big scatter); a blob URL it fetches
 * instead. The `#.png` tail is how it learns the format (from the URL's ending).
 */
async function imagesToBlobUrls(svg: Element): Promise<string[]> {
  const urls: string[] = [];
  for (const img of svg.querySelectorAll("image")) {
    const attr = img.hasAttribute("href") ? "href" : "xlink:href";
    const href = img.getAttribute(attr) ?? img.getAttributeNS("http://www.w3.org/1999/xlink", "href") ?? "";
    if (!href.startsWith("data:image/")) continue;
    const format = href.slice("data:image/".length).split(/[;,+]/)[0] || "png";
    const blob = await (await fetch(href)).blob();
    const url = URL.createObjectURL(blob);
    urls.push(url);
    img.removeAttribute("href");
    img.setAttributeNS("http://www.w3.org/1999/xlink", "xlink:href", `${url}#.${format}`);
  }
  return urls;
}

/** Save a blob as a download named `filename` (+ `ext` when missing). */
export function saveBlob(blob: Blob, filename: string, ext: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename.endsWith(`.${ext}`) ? filename : `${filename}.${ext}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/** Save an SVG (text) as `<filename>.svg`, or converted, as `<filename>.pdf`. */
export async function saveFigure(
  format: "svg" | "pdf",
  svg: { svg: string; width: number; height: number },
  filename: string,
): Promise<void> {
  if (format === "svg") {
    saveBlob(new Blob([svg.svg], { type: "image/svg+xml;charset=utf-8" }), filename, "svg");
    return;
  }
  saveBlob(await svgToPdf(svg.svg, svg.width, svg.height), filename, "pdf");
}
