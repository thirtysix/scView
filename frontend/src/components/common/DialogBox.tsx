import { useRef, type ReactNode } from "react";
import { useDialogFocus } from "@/hooks/useDialogFocus";

interface DialogBoxProps {
  /** The dialog's name, read out on entry */
  label: string;
  onClose: () => void;
  /** Keep Tab inside (default); off for a panel used alongside the page */
  trap?: boolean;
  className?: string;
  children: ReactNode;
}

/**
 * A popover dialog with the keyboard behaviour of hooks/useDialogFocus: focus
 * moves in when it opens, Tab stays inside, Escape closes it and focus goes
 * back to the button that opened it. Render it only while open.
 */
export function DialogBox({ label, onClose, trap = true, className, children }: DialogBoxProps) {
  const ref = useRef<HTMLDivElement>(null);
  useDialogFocus(ref, onClose, { trap });
  return (
    <div ref={ref} role="dialog" aria-label={label} className={className}>
      {children}
    </div>
  );
}
