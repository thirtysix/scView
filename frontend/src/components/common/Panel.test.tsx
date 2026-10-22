import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import { Panel } from "@/components/common/Panel";

describe("Panel height", () => {
  it("keeps a caller's own height instead of h-full", () => {
    const { container } = render(<Panel title="x" className="h-[300px]">body</Panel>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("h-[300px]");
    expect(root.className.split(/\s+/)).not.toContain("h-full");
  });

  it("fills its parent when no height is given", () => {
    const { container } = render(<Panel title="x">body</Panel>);
    expect((container.firstElementChild as HTMLElement).className.split(/\s+/)).toContain("h-full");
  });
});
