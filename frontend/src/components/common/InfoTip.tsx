import { useState, useId, type ReactNode } from "react";
import { Info } from "lucide-react";

/**
 * Small explanatory popover on an info icon.
 *
 * Native `title=` is used elsewhere in the app for short hints, but it is slow to
 * appear, unstyled, truncates, and never reaches keyboard users. Anything that has
 * to actually teach the reader something belongs here instead.
 */
export function InfoTip({
  children,
  align = "left",
  width = 260,
  label = "More information",
}: {
  children: ReactNode;
  align?: "left" | "right";
  width?: number;
  label?: string;
}) {
  const [open, setOpen] = useState(false);
  const id = useId();

  return (
    <span className="relative inline-flex">
      <button
        type="button"
        aria-label={label}
        aria-expanded={open}
        aria-describedby={open ? id : undefined}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}
        className="inline-flex items-center text-slate-300 transition-colors hover:text-slate-500 focus:text-slate-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 rounded-sm"
      >
        <Info className="h-3 w-3" />
      </button>
      {open && (
        <span
          id={id}
          role="tooltip"
          style={{ width }}
          className={`absolute bottom-full z-30 mb-1.5 block rounded-md border border-slate-200 bg-white p-2 text-[10px] font-normal leading-relaxed text-slate-600 shadow-lg ${
            align === "right" ? "right-0" : "left-0"
          }`}
        >
          {children}
        </span>
      )}
    </span>
  );
}
