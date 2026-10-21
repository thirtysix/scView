import { CLIP_PRESET_OPTIONS, type ClipPreset, type ClipSetting } from "@/lib/colorScale";

interface ClipControlProps {
  /** The clipping in force (the view's override, or the settings default) */
  clip: ClipSetting;
  /** The data range, as the manual inputs' placeholders */
  dataMin?: number;
  dataMax?: number;
  /** null: back to the settings default */
  onChange: (clip: ClipSetting | null) => void;
  /** The view overrides the settings default (a "default" link shows) */
  overridden?: boolean;
}

/** Up to three significant digits, for an input's placeholder. */
function short(v: number | undefined): string {
  return v === undefined || !Number.isFinite(v) ? "" : String(Number(v.toPrecision(3)));
}

/**
 * The colour-range clipping of one view: a preset (none, percentiles) or a
 * manual min/max pair, beside the continuous legend. Values outside the range
 * take the end colours; the legend's caption says how many cells that is.
 */
export function ClipControl({ clip, dataMin, dataMax, onChange, overridden = false }: ClipControlProps) {
  const setPreset = (preset: ClipPreset) => {
    if (preset === "manual") onChange({ preset, min: clip.min ?? dataMin, max: clip.max ?? dataMax });
    else onChange({ preset });
  };
  const setEnd = (end: "min" | "max", raw: string) => {
    const v = raw.trim() === "" ? undefined : Number(raw);
    if (v !== undefined && !Number.isFinite(v)) return;
    onChange({ ...clip, preset: "manual", [end]: v });
  };
  const input = "w-14 rounded border border-slate-300 bg-white px-1 py-0 text-[10px] tabular-nums text-slate-700 focus:border-blue-400 focus:outline-none";
  return (
    <div className="flex flex-wrap items-center gap-1 text-[10px] text-slate-500" data-testid="clip-control">
      <label htmlFor="colour-clip" title="Narrow the colour range so a few extreme cells do not wash out the rest; cells outside it take the end colours">
        Clip
      </label>
      <select
        id="colour-clip"
        aria-label="Colour range clipping"
        value={clip.preset}
        onChange={(e) => setPreset(e.target.value as ClipPreset)}
        className="rounded border border-slate-300 bg-white px-1 py-0 text-[10px] text-slate-700 focus:border-blue-400 focus:outline-none"
      >
        {CLIP_PRESET_OPTIONS.map((o) => (
          <option key={o.value} value={o.value} title={o.title}>
            {o.label}
          </option>
        ))}
      </select>
      {clip.preset === "manual" && (
        <>
          <input
            type="number"
            step="any"
            aria-label="Colour range minimum"
            value={clip.min ?? ""}
            placeholder={short(dataMin)}
            onChange={(e) => setEnd("min", e.target.value)}
            className={input}
          />
          <span>to</span>
          <input
            type="number"
            step="any"
            aria-label="Colour range maximum"
            value={clip.max ?? ""}
            placeholder={short(dataMax)}
            onChange={(e) => setEnd("max", e.target.value)}
            className={input}
          />
        </>
      )}
      {overridden && (
        <button
          type="button"
          onClick={() => onChange(null)}
          title="Back to the default clipping (toolbar: Colours)"
          className="text-slate-400 hover:text-primary hover:underline"
        >
          default
        </button>
      )}
    </div>
  );
}
