import { act, fireEvent, screen } from "@testing-library/react";

/**
 * Hover `el` long enough for its HoverTip / <Help> tip to open and return the tip's text.
 * Uses real timers (the hover delay is 300 ms), so `await` it.
 */
export async function hoverTip(el: Element): Promise<string> {
  fireEvent.mouseEnter(el);
  await act(() => new Promise((r) => setTimeout(r, 350)));
  return screen.getByRole("tooltip").textContent ?? "";
}

/** Move the pointer off `el` (closes its tip). */
export function unhover(el: Element): void {
  fireEvent.mouseLeave(el);
}
