import { useEffect, useRef, type RefObject } from "react";

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), ' +
  'textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** The elements of `root` that Tab can reach, in order. */
export function focusableIn(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
    (el) => !el.closest("[inert]") && el.getAttribute("aria-hidden") !== "true",
  );
}

/**
 * Keyboard behaviour of a dialog or popover, from the moment it mounts:
 * focus moves into it (the first field or button, unless something in it
 * already has focus), Tab and Shift+Tab stay inside it (`trap`; off for a
 * floating panel used alongside the plot), Escape calls `onClose`, and on
 * unmount focus goes back to where it was before.
 */
export function useDialogFocus(
  ref: RefObject<HTMLElement | null>,
  onClose: () => void,
  { trap = true }: { trap?: boolean } = {},
): void {
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    if (!root.contains(document.activeElement)) (focusableIn(root)[0] ?? root).focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        closeRef.current();
        return;
      }
      if (e.key !== "Tab" || !trap) return;
      const items = focusableIn(root);
      if (items.length === 0) {
        e.preventDefault();
        return;
      }
      const first = items[0]!;
      const last = items[items.length - 1]!;
      if (e.shiftKey && (document.activeElement === first || !root.contains(document.activeElement))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    root.addEventListener("keydown", onKey);
    return () => {
      root.removeEventListener("keydown", onKey);
      // Not when the user has since put focus somewhere else (a click outside).
      const active = document.activeElement;
      const lost = !active || active === document.body || root.contains(active);
      if (lost && opener?.isConnected) opener.focus();
    };
  }, [ref, trap]);
}
