import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";

/**
 * The co-pilot's feature guide (backend/src/scview/core/feature_guide.py) names
 * each feature by its navigation path. Every segment of every path must be a
 * label the UI shows, so a renamed tab fails here instead of the model sending
 * the user to a menu that does not exist.
 */
const GUIDE = resolve(__dirname, "../../../backend/src/scview/core/feature_guide.py");
const SRC = resolve(__dirname, "..");

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) return sources(p);
    return /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name) ? [p] : [];
  });
}

function guidePaths(): string[] {
  const py = readFileSync(GUIDE, "utf8");
  const body = py.slice(py.indexOf("FEATURES:"), py.indexOf("\n)\n", py.indexOf("FEATURES:")));
  return [...body.matchAll(/^ {4}\("([^"]+)",/gm)].map((m) => m[1]);
}

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Shown as a label: a whole string literal, or the whole text of a JSX element. */
function shown(label: string, code: string): boolean {
  const e = escape(label).replace(/&/g, "(?:&|&amp;)");
  return new RegExp(`["'\`]${e}["'\`]|>\\s*${e}\\s*(?:<|\\{|$)`, "m").test(code);
}

describe("co-pilot feature guide", () => {
  const code = sources(SRC).map((p) => readFileSync(p, "utf8")).join("\n");
  const paths = guidePaths();

  it("lists the features", () => {
    expect(paths.length).toBeGreaterThan(40);
    expect(paths).toContain("Unified View > Activity > Cytokine response");
  });

  it("names only labels the UI shows", () => {
    const missing = paths.flatMap((p) => p.split(" > ").filter((seg) => !shown(seg, code)).map((seg) => `${p}: ${seg}`));
    expect(missing).toEqual([]);
  });

  it("catches a label that does not exist", () => {
    expect(shown("Cell-cell communication", code)).toBe(false);
    expect(shown("Communication", code)).toBe(true);
  });
});
