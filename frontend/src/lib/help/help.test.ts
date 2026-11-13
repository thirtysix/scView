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

const INFO_ICONS = "Info|InfoIcon|BadgeInfo|HelpCircle|CircleHelp|CircleQuestionMark";
const TRIGGER_ATTR = /\b(title|onMouseEnter|onMouseOver|onFocus|tabIndex|aria-describedby)=/;

/**
 * Lines (1-based) where an info-style lucide icon is a tip trigger: it carries tip props itself,
 * it is the first child of a HoverTip / Help / TipPopup or of an element with tip props, or it
 * is imported under another name (which would hide it from this check). An icon beside its own
 * text (a note or warning level) is a status icon and passes.
 */
function tipIcons(code: string): number[] {
  const lineOf = (i: number) => code.slice(0, i).split("\n").length;
  const hits = new Set<number>();
  for (const m of code.matchAll(new RegExp(`\\b(?:${INFO_ICONS}) as \\w+`, "g"))) hits.add(lineOf(m.index));
  for (const m of code.matchAll(new RegExp(`<(?:${INFO_ICONS})\\b((?:[^<>]|=>)*)>`, "g"))) {
    const before = code.slice(Math.max(0, m.index - 400), m.index);
    const parent = /<([A-Za-z]\w*)\b((?:[^<>]|=>)*)>\s*$/.exec(before);
    const wrapped =
      parent != null &&
      !parent[2]!.trimEnd().endsWith("/") &&
      (/^(HoverTip|Help|TipPopup)$/.test(parent[1]!) || TRIGGER_ATTR.test(parent[2]!));
    if (TRIGGER_ATTR.test(m[1]!) || wrapped) hits.add(lineOf(m.index));
  }
  return [...hits].sort((a, b) => a - b);
}

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

  it("draws no info icons: tips go on the thing itself (<Help> or HoverTip)", () => {
    const icons = sources.filter(({ p, code }) => !p.endsWith("common/InfoTip.tsx") && /<InfoTip\b/.test(code)).map((s) => s.p);
    expect(icons).toEqual([]);
  });

  it("uses no info-style icon as a tip trigger", () => {
    const hits = sources
      .filter(({ p }) => !p.endsWith("common/InfoTip.tsx"))
      .flatMap(({ p, code }) => tipIcons(code).map((line) => `${p.slice(SRC.length + 1)}:${line}`));
    expect(hits, "put the tip on the label: <Help id=...>Label</Help> or HoverTip, no icon").toEqual([]);
  });

  it("tells a tip-trigger icon from a status icon", () => {
    expect(tipIcons('<HoverTip tip="x">\n  <Info className="h-3 w-3" />\n</HoverTip>')).toEqual([2]);
    expect(tipIcons('<Help id="a.b"><Info /></Help>')).toEqual([1]);
    expect(tipIcons('<span title="What this means"><Info className="h-3 w-3" /></span>')).toEqual([1]);
    expect(tipIcons('<Info className="h-3 w-3" title="Why" />')).toEqual([1]);
    expect(tipIcons("<button onMouseEnter={open}><HelpCircle /></button>")).toEqual([1]);
    expect(tipIcons("import { Info as I } from \"lucide-react\";")).toEqual([1]);
    // status icons beside their own text are fine
    expect(tipIcons('<li className="flex"><Info className="h-3 w-3" aria-label="Note" /> <span>{t}</span></li>')).toEqual([]);
    expect(tipIcons('{warn ? (<AlertTriangle />) : (\n  <Info className="h-3 w-3" />\n)}')).toEqual([]);
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
