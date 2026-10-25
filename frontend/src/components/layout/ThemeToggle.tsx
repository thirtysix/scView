import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme, type ThemePreference } from "@/lib/theme";

const OPTIONS: { value: ThemePreference; label: string; Icon: React.ElementType }[] = [
  { value: "light", label: "Light", Icon: Sun },
  { value: "dark", label: "Dark", Icon: Moon },
  { value: "system", label: "System", Icon: Monitor },
];

/** Light / Dark / System theme switch at the foot of the sidebar. Collapsed,
 *  it is one button that steps through the three. */
export function ThemeToggle({ collapsed }: { collapsed: boolean }) {
  const { preference, setPreference } = useTheme();

  if (collapsed) {
    const i = OPTIONS.findIndex((o) => o.value === preference);
    const current = OPTIONS[i]!;
    const next = OPTIONS[(i + 1) % OPTIONS.length]!;
    return (
      <div className="flex justify-center border-t border-white/10 py-2">
        <button
          onClick={() => setPreference(next.value)}
          title={`Theme: ${current.label} (click for ${next.label})`}
          aria-label={`Theme: ${current.label}`}
          className="rounded p-1 text-sidebar-foreground/60 hover:bg-sidebar-accent hover:text-sidebar-foreground"
        >
          <current.Icon className="h-4 w-4" />
        </button>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-between gap-2 border-t border-white/10 px-4 py-2">
      <span className="text-[11px] text-sidebar-foreground/60">Theme</span>
      <div role="radiogroup" aria-label="Theme" className="flex overflow-hidden rounded-md border border-white/10">
        {OPTIONS.map(({ value, label, Icon }) => {
          const on = preference === value;
          return (
            <button
              key={value}
              role="radio"
              aria-checked={on}
              aria-label={label}
              title={value === "system" ? "System (follows your computer's setting)" : label}
              onClick={() => setPreference(value)}
              className={`p-1.5 transition-colors ${
                on
                  ? "bg-sidebar-accent text-sidebar-foreground"
                  : "text-sidebar-foreground/50 hover:text-sidebar-foreground"
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
            </button>
          );
        })}
      </div>
    </div>
  );
}
