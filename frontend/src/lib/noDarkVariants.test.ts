import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";

/**
 * Dark theme remaps the LIGHT colour classes (index.css, `.dark`), so a `dark:` variant
 * applies a colour that is itself remapped: `dark:bg-orange-950` comes out near white,
 * `dark:text-slate-200` dark on dark. Components use light classes only.
 */
const SRC = resolve(__dirname, "..");

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) return sources(p);
    return /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name) ? [p] : [];
  });
}

describe("theme classes", () => {
  it("no component uses a dark: variant", () => {
    const hits = sources(SRC).flatMap((p) =>
      readFileSync(p, "utf8").split("\n").flatMap((line, i) =>
        /(?:^|[\s"'`])dark:[a-z]/.test(line) ? [`${relative(SRC, p)}:${i + 1}`] : [],
      ),
    );
    expect(hits).toEqual([]);
  });
});
