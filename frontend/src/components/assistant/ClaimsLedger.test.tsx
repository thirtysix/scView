import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { Claim } from "@/api/assistant";
import { ClaimsLedger, MAX_CHECK, mergeVerdicts, tally, toCheck } from "./ClaimsLedger";

const claim = (id: number, verdict: Claim["verdict"], over: Partial<Claim> = {}): Claim => ({
  id,
  text: `claim ${id}`,
  cites: ["result:groups:stim"],
  refs: ["result:groups:stim"],
  verdict,
  reason: `reason ${id}`,
  evidence: "- **stim**: STIM (7,263), CTRL (6,573)",
  checked_by: "rules",
  ...over,
});

const LEDGER = [claim(0, "supported"), claim(1, "refuted"), claim(2, "uncertain"), claim(3, "uncertain")];

describe("ledger helpers", () => {
  it("tallies verdicts", () => {
    expect(tally(LEDGER)).toEqual({ supported: 1, refuted: 1, uncertain: 2 });
  });

  it("sends only rule-uncertain claims, at most MAX_CHECK, and never twice", () => {
    expect(toCheck(LEDGER).map((c) => c.id)).toEqual([2, 3]);
    const many = Array.from({ length: MAX_CHECK + 3 }, (_, i) => claim(i, "uncertain"));
    expect(toCheck(many)).toHaveLength(MAX_CHECK);
    const judged = mergeVerdicts(LEDGER, [{ id: 2, verdict: "uncertain", reason: "not stated" }]);
    expect(toCheck(judged).map((c) => c.id)).toEqual([3]);
  });

  it("merges the model's verdicts and marks them", () => {
    const out = mergeVerdicts(LEDGER, [{ id: 3, verdict: "supported", reason: "sum of counts" }]);
    expect(out[3]).toMatchObject({ verdict: "supported", reason: "sum of counts", checked_by: "model" });
    expect(out[2]).toBe(LEDGER[2]);
  });
});

describe("ClaimsLedger", () => {
  it("renders nothing for an answer without claims", () => {
    const { container } = render(<ClaimsLedger claims={[]} canCheck />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows counts, keeps refuted claims visible, and expands to the full list", () => {
    render(<ClaimsLedger claims={LEDGER} canCheck={false} />);
    expect(screen.getByText("1 supported")).toBeInTheDocument();
    expect(screen.getByText("2 uncertain")).toBeInTheDocument();
    expect(screen.getByText("claim 1")).toBeInTheDocument(); // refuted, collapsed
    expect(screen.queryByText("claim 0")).not.toBeInTheDocument();
    expect(screen.queryByText("Check claims")).not.toBeInTheDocument(); // no model
    fireEvent.click(screen.getByRole("button", { name: /Claims/ }));
    expect(screen.getByText("claim 0")).toBeInTheDocument();
    expect(screen.getByText("claim 3")).toBeInTheDocument();
  });

  it("offers Check claims while rule-uncertain claims remain", () => {
    const onCheck = vi.fn();
    const { rerender } = render(<ClaimsLedger claims={LEDGER} canCheck onCheck={onCheck} />);
    fireEvent.click(screen.getByText("Check claims"));
    expect(onCheck).toHaveBeenCalledOnce();
    rerender(<ClaimsLedger claims={LEDGER} canCheck checking onCheck={onCheck} />);
    expect(screen.getByText("Checking…")).toBeInTheDocument();
    const done = mergeVerdicts(LEDGER, [
      { id: 2, verdict: "supported", reason: "" },
      { id: 3, verdict: "uncertain", reason: "not stated" },
    ]);
    rerender(<ClaimsLedger claims={done} canCheck onCheck={onCheck} />);
    expect(screen.queryByText("Check claims")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Claims/ }));
    expect(screen.getAllByText("checked by AI")).toHaveLength(2);
  });

  it("shows a failed check", () => {
    render(<ClaimsLedger claims={LEDGER} canCheck error="Check failed: 503" />);
    expect(screen.getByText("Check failed: 503")).toBeInTheDocument();
  });
});
