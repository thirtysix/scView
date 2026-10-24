import { InfoTip } from "@/components/common/InfoTip";
import { REGRESS_HELP } from "@/lib/regressOut";

/** The scaling step's regress-out covariates: a checkbox each (none = off). */
export function RegressOutChoice({
  options,
  value,
  onChange,
  disabled,
  pearson,
}: {
  options: { key: string; label: string }[];
  value: string[];
  onChange: (next: string[]) => void;
  disabled?: boolean;
  /** Pearson residuals chosen: regress-out works on the log values only. */
  pearson?: boolean;
}) {
  const off = disabled || pearson;
  return (
    <div className="flex flex-col gap-1">
      <div className="flex max-h-40 flex-col gap-1 overflow-auto rounded border border-slate-100 bg-slate-50 p-2">
        {options.map((o) => {
          const checked = value.includes(o.key);
          return (
            <label key={o.key} className="flex items-center gap-1.5 text-xs text-slate-700">
              <input
                type="checkbox"
                checked={checked}
                disabled={off}
                onChange={() => onChange(checked ? value.filter((k) => k !== o.key) : [...value, o.key])}
                className="h-3.5 w-3.5 rounded border-slate-300 accent-primary disabled:opacity-50"
              />
              {o.label}
            </label>
          );
        })}
      </div>
      <p className="flex items-center gap-1 text-[10px] text-slate-500">
        {pearson
          ? "Off with Pearson residuals: they already model sequencing depth (Normalization step)."
          : value.length
            ? `Regressed out of the variable genes before PCA: ${value.length} covariate${value.length === 1 ? "" : "s"}.`
            : "None: off (the default)."}
        <InfoTip width={300} label="When regress-out helps">{REGRESS_HELP}</InfoTip>
      </p>
    </div>
  );
}
