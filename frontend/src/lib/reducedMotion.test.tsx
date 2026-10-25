import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { act, render, screen } from "@testing-library/react";
import { prefersReducedMotion, useReducedMotion } from "./reducedMotion";
import { useA11yStore } from "./accessibility";

function stubMedia(matches: boolean) {
  const listeners = new Set<() => void>();
  const mq = {
    matches,
    addEventListener: (_: string, cb: () => void) => listeners.add(cb),
    removeEventListener: (_: string, cb: () => void) => listeners.delete(cb),
  };
  vi.stubGlobal("matchMedia", vi.fn(() => mq));
  return {
    set(v: boolean) {
      mq.matches = v;
      listeners.forEach((cb) => cb());
    },
  };
}

function Probe() {
  return <span data-testid="rm">{String(useReducedMotion())}</span>;
}

beforeEach(() => useA11yStore.setState({ enabled: false }));
afterEach(() => vi.unstubAllGlobals());

describe("reduced motion", () => {
  it("follows the system setting, live", () => {
    const media = stubMedia(false);
    expect(prefersReducedMotion()).toBe(false);
    render(<Probe />);
    expect(screen.getByTestId("rm")).toHaveTextContent("false");
    act(() => media.set(true));
    expect(screen.getByTestId("rm")).toHaveTextContent("true");
  });

  it("is on in accessibility mode too, and off without matchMedia", () => {
    vi.stubGlobal("matchMedia", undefined);
    expect(prefersReducedMotion()).toBe(false);
    render(<Probe />);
    expect(screen.getByTestId("rm")).toHaveTextContent("false");
    act(() => useA11yStore.setState({ enabled: true }));
    expect(screen.getByTestId("rm")).toHaveTextContent("true");
  });
});
