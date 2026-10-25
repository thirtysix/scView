/**
 * axe-core over a rendered tree (jsdom), for the accessibility tests in
 * src/test/a11y. Colour contrast is left out: jsdom does no layout or
 * cascade, so axe cannot measure it there (accessibility mode covers it).
 */
import axe from "axe-core";

export interface AxeFinding {
  id: string;
  impact: string | null | undefined;
  help: string;
  nodes: string[];
}

const RANK = ["critical", "serious", "moderate", "minor"];

/** Every violation axe finds in `root`, worst first. */
export async function axeViolations(root: Element = document.body): Promise<AxeFinding[]> {
  const result = await axe.run(root, {
    rules: { "color-contrast": { enabled: false } },
    resultTypes: ["violations"],
  });
  return result.violations
    .map((v) => ({
      id: v.id,
      impact: v.impact,
      help: v.help,
      nodes: v.nodes.map((n) => n.html.slice(0, 160)),
    }))
    .sort((a, b) => RANK.indexOf(a.impact ?? "minor") - RANK.indexOf(b.impact ?? "minor"));
}

/** The serious and critical violations only (what the tests assert is empty). */
export async function seriousViolations(root?: Element): Promise<AxeFinding[]> {
  return (await axeViolations(root)).filter((v) => v.impact === "serious" || v.impact === "critical");
}
