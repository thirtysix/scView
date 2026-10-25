import { Accessibility } from "lucide-react";
import { useA11yStore } from "@/lib/accessibility";

const TIP =
  "Accessibility mode: stronger text contrast (WCAG AA), no text under 12 px, a visible focus " +
  "outline, underlined links, no transitions and colour-blind-safe group colours. Off, the " +
  "settings are tuned for clarity and looks.";

/** Accessibility mode switch at the foot of the sidebar, under Theme. */
export function AccessibilityToggle({ collapsed }: { collapsed: boolean }) {
  const enabled = useA11yStore((s) => s.enabled);
  const setEnabled = useA11yStore((s) => s.setEnabled);
  const label = `Accessibility mode: ${enabled ? "on" : "off"}`;

  if (collapsed) {
    return (
      <div className="flex justify-center py-1">
        <button
          role="switch"
          aria-checked={enabled}
          aria-label="Accessibility mode"
          onClick={() => setEnabled(!enabled)}
          title={`${label}. ${TIP}`}
          className={`rounded p-1 hover:bg-sidebar-accent ${
            enabled ? "text-sidebar-foreground" : "text-sidebar-foreground/60 hover:text-sidebar-foreground"
          }`}
        >
          <Accessibility className="h-4 w-4" />
        </button>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-between gap-2 px-4 pb-2" title={TIP}>
      <span className="text-[11px] text-sidebar-foreground/60">Accessibility</span>
      <button
        role="switch"
        aria-checked={enabled}
        aria-label="Accessibility mode"
        onClick={() => setEnabled(!enabled)}
        className={`relative h-4 w-7 rounded-full border border-white/10 ${enabled ? "bg-primary" : "bg-white/10"}`}
      >
        <span
          className={`absolute top-0.5 h-2.5 w-2.5 rounded-full bg-white ${enabled ? "left-3.5" : "left-0.5"}`}
        />
      </button>
    </div>
  );
}
