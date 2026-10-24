import { InfoTip } from "@/components/common/InfoTip";
import { DOUBLET_METHOD_POINTS } from "@/lib/doubletMethods";

/** The info icon beside the doublet step's detector choice. */
export function DoubletMethodInfo() {
  return (
    <InfoTip width={320} label="About the doublet detectors">
      <span className="block space-y-1">
        {DOUBLET_METHOD_POINTS.map((p) => (
          <span key={p} className="block">
            {p}
          </span>
        ))}
      </span>
    </InfoTip>
  );
}
