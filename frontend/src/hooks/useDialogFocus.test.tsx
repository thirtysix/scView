import { render, screen } from "@testing-library/react";
import { useRef } from "react";
import { describe, expect, it } from "vitest";
import { InfoTip } from "@/components/common/InfoTip";
import { useDialogFocus } from "./useDialogFocus";

function Dialog() {
  const ref = useRef<HTMLDivElement>(null);
  useDialogFocus(ref, () => {});
  return (
    <div ref={ref}>
      <h2>
        Filter cells <InfoTip label="About filters">Tip text</InfoTip>
      </h2>
      <button type="button">Close</button>
    </div>
  );
}

describe("useDialogFocus", () => {
  it("does not put first focus on an info icon (its tip would open over the dialog)", () => {
    render(<Dialog />);
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("tooltip")).toBeNull();
  });

  it("puts first focus on `initial` when given (a confirm's Cancel)", () => {
    function Confirm() {
      const ref = useRef<HTMLDivElement>(null);
      const cancel = useRef<HTMLButtonElement>(null);
      useDialogFocus(ref, () => {}, { initial: cancel });
      return (
        <div ref={ref}>
          <button type="button">Delete</button>
          <button type="button" ref={cancel}>
            Cancel
          </button>
        </div>
      );
    }
    render(<Confirm />);
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveFocus();
  });
});
