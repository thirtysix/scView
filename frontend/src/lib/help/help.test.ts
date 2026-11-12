import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { HELP, HELP_GROUPS } from "./index";

/** Every .tsx under src (tests left out): where `<Help id="..."/>` is used. */
const SRC = resolve(__dirname, "../..");
function tsxFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) return tsxFiles(p);
    return p.endsWith(".tsx") && !p.endsWith(".test.tsx") ? [p] : [];
  });
}
const sources = tsxFiles(SRC).map((p) => ({ p, code: readFileSync(p, "utf8") }));
const allCode = sources.map((s) => s.code).join("\n");

const unified = readFileSync(join(SRC, "components/panels/UnifiedViewPanel.tsx"), "utf8");
const subtabIds = [...unified.slice(unified.indexOf("const SUBTABS")).matchAll(/id: "([a-z0-9_]+)" as const/g)]
  .map((m) => m[1]!);

describe("help catalog", () => {
  it("has no id in two group files", () => {
    const seen = new Map<string, string>();
    for (const [group, entries] of Object.entries(HELP_GROUPS)) {
      for (const id of Object.keys(entries)) {
        expect(seen.get(id), `${id} in ${group} and ${seen.get(id)}`).toBeUndefined();
        seen.set(id, group);
      }
    }
  });

  it("covers every Unified View tab", () => {
    expect(subtabIds.length).toBeGreaterThan(20);
    expect(subtabIds.filter((id) => !HELP[`tab.${id}`])).toEqual([]);
  });

  it("uses only literal ids that exist", () => {
    const uses = sources.flatMap(({ p, code }) =>
      [...code.matchAll(/<Help\b([^>]*)>/g)].map((m) => ({ p, attrs: m[1]! })),
    );
    const iconStyle = uses.filter((u) => u.attrs.trimEnd().endsWith("/")).map((u) => u.p);
    expect(iconStyle, "wrap the thing explained: <Help id=...>Label</Help>, not <Help id=... />").toEqual([]);
    const dynamic = uses.filter((u) => !/\bid="[^"]+"/.test(u.attrs)).map((u) => u.p);
    expect(dynamic, "use a literal id, so this test can check it").toEqual([]);
    const missing = uses.map((u) => u.attrs.match(/\bid="([^"]+)"/)![1]!).filter((id) => !HELP[id]);
    expect(missing).toEqual([]);
  });

  it("has no entry nobody shows", () => {
    const unused = Object.keys(HELP).filter(
      (id) => !(id.startsWith("tab.") && subtabIds.includes(id.slice(4))) && !allCode.includes(`"${id}"`),
    );
    expect(unused).toEqual([]);
  });

  it("keeps each tip short and plain", () => {
    for (const [id, text] of Object.entries(HELP)) {
      expect(text.split(/\s+/).length, id).toBeLessThanOrEqual(60);
      expect(text, `${id}: no em or en dashes`).not.toMatch(/[–—]/);
      expect(text.trim(), id).toBe(text);
      expect(text, id).toMatch(/[.?)]$/);
    }
  });
});
