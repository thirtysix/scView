import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import {
  A11Y_MIN_FONT,
  A11Y_STORAGE_KEY,
  a11yLayout,
  initA11y,
  readStoredA11y,
  useA11yStore,
} from "@/lib/accessibility";
import { AccessibilityToggle } from "@/components/layout/AccessibilityToggle";
import { useSettingsStore } from "@/stores/settingsStore";

afterEach(() => {
  useA11yStore.getState().setEnabled(false);
  localStorage.removeItem(A11Y_STORAGE_KEY);
  document.documentElement.classList.remove("a11y");
});

describe("accessibility mode", () => {
  it("is off by default and remembered per browser", () => {
    expect(readStoredA11y()).toBe(false);
    useA11yStore.getState().setEnabled(true);
    expect(localStorage.getItem(A11Y_STORAGE_KEY)).toBe("on");
    expect(readStoredA11y()).toBe(true);
  });

  it("puts .a11y on <html> and follows the switch", () => {
    const stop = initA11y();
    expect(document.documentElement.classList.contains("a11y")).toBe(false);
    useA11yStore.getState().setEnabled(true);
    expect(document.documentElement.classList.contains("a11y")).toBe(true);
    useA11yStore.getState().setEnabled(false);
    expect(document.documentElement.classList.contains("a11y")).toBe(false);
    stop();
  });

  it("turning it on picks the colour-blind-safe palette", () => {
    useSettingsStore.getState().setCategoricalPalette("classic");
    useA11yStore.getState().setEnabled(true);
    expect(useSettingsStore.getState().categoricalPalette).toBe("cvd");
  });

  it("raises every plot font under 12 px, and leaves larger ones", () => {
    const layout = {
      font: { size: 10 },
      xaxis: { tickfont: { size: 9 }, title: { text: "x", font: { size: 11 } } },
      yaxis2: { tickfont: { size: 14 } },
      legend: { font: { size: 10 } },
      annotations: [{ text: "a", font: { size: 8 } }, { text: "b" }],
    };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- nested layout fields read in the asserts
    const out = a11yLayout(layout as never, true) as Record<string, any>;
    expect(out.font.size).toBe(A11Y_MIN_FONT);
    expect(out.xaxis.tickfont.size).toBe(A11Y_MIN_FONT);
    expect(out.xaxis.title.font.size).toBe(A11Y_MIN_FONT);
    expect(out.yaxis2.tickfont.size).toBe(14);
    expect(out.legend.font.size).toBe(A11Y_MIN_FONT);
    expect(out.annotations[0].font.size).toBe(A11Y_MIN_FONT);
    expect(layout.font.size).toBe(10); // input not mutated
    expect(a11yLayout({ font: {} } as never, true)!.font).toEqual({ size: A11Y_MIN_FONT });
  });

  it("off, the layout is the same object", () => {
    const layout = { font: { size: 10 } };
    expect(a11yLayout(layout as never, false)).toBe(layout);
  });

  it("the switch toggles the mode", () => {
    render(createElement(AccessibilityToggle, { collapsed: false }));
    const sw = screen.getByRole("switch", { name: "Accessibility mode" });
    expect(sw.getAttribute("aria-checked")).toBe("false");
    fireEvent.click(sw);
    expect(useA11yStore.getState().enabled).toBe(true);
    expect(sw.getAttribute("aria-checked")).toBe("true");
  });

  it("index.css carries the contrast, size and focus rules", () => {
    const css = readFileSync(resolve(__dirname, "../index.css"), "utf8");
    expect(css).toContain(".a11y:not(.dark) .text-slate-400");
    expect(css).toContain(".a11y .text-\\[10px\\]");
    expect(css).toContain(".a11y :focus-visible");
  });
});
