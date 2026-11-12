import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { Claim } from "@/api/assistant";
import { balancedInline, findClaimMarks } from "@/lib/claimMarks";
import { hoverTip } from "@/test/hoverTip";
import { tipEvidence } from "./ClaimMark";
import { MarkdownLite } from "./MarkdownLite";

const claim = (id: number, span: string, verdict: Claim["verdict"], over: Partial<Claim> = {}): Claim => ({
  id, span, verdict, text: span, cites: ["result:groups:stim"], refs: ["result:groups:stim"],
  reason: `reason ${id}`, evidence: "- **stim**: STIM (7,263), CTRL (6,573)", checked_by: "rules", ...over,
});

const ANSWER =
  "**STIM** has 7,263 cells [result:groups:stim]. CTRL has 6,000 cells [result:groups:stim]. " +
  "That is a balanced design.\n\n- CD14 Mono shift is unclear [result:composition].";
const CLAIMS = [
  claim(0, "**STIM** has 7,263 cells [result:groups:stim].", "supported"),
  claim(1, "CTRL has 6,000 cells [result:groups:stim].", "refuted", {
    reason: "says CTRL 6,000; the fact has 6,573.",
  }),
  claim(2, "CD14 Mono shift is unclear [result:composition].", "uncertain"),
];

describe("finding checked sentences", () => {
  it("finds each sentence once, in order, and never inside a split bold", () => {
    const used = new Set<number>();
    const text = "A is 5. B is 6. A is 5.";
    const marks = findClaimMarks(text, [claim(1, "B is 6.", "supported"), claim(0, "A is 5.", "refuted")], used);
    expect(marks.map((m) => [m.start, m.claim.id])).toEqual([[0, 0], [8, 1]]);
    expect(findClaimMarks(text, [claim(0, "A is 5.", "refuted")], used)).toEqual([]); // marked once
    // cutting "**A. B**" after "A." would leave stray asterisks: left unmarked
    expect(findClaimMarks("**A is 5. B is 6.**", [claim(3, "B is 6.**", "supported")], new Set())).toEqual([]);
    expect(balancedInline("**a** `b` *c*")).toBe(true);
    expect(balancedInline("**a")).toBe(false);
  });

  it("keeps a tip's evidence short", () => {
    const long = Array.from({ length: 8 }, (_, i) => `line ${i} ${"x".repeat(200)}`).join("\n");
    const out = tipEvidence(long).split("\n");
    expect(out).toHaveLength(5);
    expect(out[0]!.endsWith("…")).toBe(true);
    expect(out[4]).toBe("…");
  });
});

describe("inline claim marks", () => {
  it("marks each checked sentence by its verdict and leaves prose alone", () => {
    const { container } = render(<MarkdownLite text={ANSWER} claims={CLAIMS} onCitation={() => {}} />);
    const marks = [...container.querySelectorAll("[data-claim]")];
    expect(marks.map((m) => m.getAttribute("data-claim"))).toEqual(["supported", "refuted", "uncertain"]);
    expect(marks[0]!.querySelector("strong")?.textContent).toBe("STIM"); // inline markdown kept
    expect(marks[1]!.textContent).toContain("CTRL has 6,000 cells");
    expect(marks[1]!.className).toMatch(/decoration-wavy/);
    expect(marks[0]!.className).toMatch(/decoration-emerald-500\/30/);
    expect(marks[2]!.closest("li")).not.toBeNull();
    expect(container.textContent).toContain("That is a balanced design.");
    expect(container.querySelectorAll("[data-claim]")).toHaveLength(3);
    // the citation chips still work inside a mark
    expect(marks[1]!.querySelector("button")).not.toBeNull();
  });

  it("shows the verdict and the fact on hover", async () => {
    const { container } = render(<MarkdownLite text={ANSWER} claims={CLAIMS} />);
    const refuted = container.querySelector('[data-claim="refuted"]')!;
    const tip = await hoverTip(refuted);
    expect(tip).toContain("Refuted");
    expect(tip).toContain("says CTRL 6,000; the fact has 6,573.");
    expect(tip).toContain("CTRL (6,573)");
  });

  it("renders as before without claims or spans", () => {
    const onCitation = vi.fn();
    const { container } = render(
      <MarkdownLite text={ANSWER} onCitation={onCitation} claims={CLAIMS.map((c) => ({ ...c, span: undefined }))} />,
    );
    expect(container.querySelector("[data-claim]")).toBeNull();
    fireEvent.click(screen.getAllByRole("button", { name: /stim/i })[0]!);
    expect(onCitation).toHaveBeenCalledWith("result:groups:stim");
  });
});
