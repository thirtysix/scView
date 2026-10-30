import type { SpatialInfo } from "@/api/types";
import { InfoTip } from "@/components/common/InfoTip";
import { activeLibrary, unitWord } from "@/lib/spatial";
import { useTissueViewStore } from "@/stores/tissueViewStore";

/**
 * Toolbar controls of the tissue plot (Embedding "Tissue"): which section is
 * shown (several libraries: one at a time), and the H&E image under the spots.
 */
export function TissueControls({ spatial }: { spatial: SpatialInfo }) {
  const wanted = useTissueViewStore((s) => s.library);
  const setLibrary = useTissueViewStore((s) => s.setLibrary);
  const showImage = useTissueViewStore((s) => s.showImage);
  const setShowImage = useTissueViewStore((s) => s.setShowImage);
  const imageOpacity = useTissueViewStore((s) => s.imageOpacity);
  const setImageOpacity = useTissueViewStore((s) => s.setImageOpacity);
  const library = activeLibrary(spatial, wanted);
  const hasImage = !!(library?.images.lowres || library?.images.hires);
  const units = unitWord(spatial);

  return (
    <div className="flex items-center gap-1.5">
      {spatial.libraries.length > 1 && (
        <>
          <label htmlFor="tissue-section" className="text-[11px] font-medium text-slate-500">
            Section
          </label>
          <select
            id="tissue-section"
            value={library?.id ?? ""}
            onChange={(e) => setLibrary(e.target.value)}
            title={`One tissue section at a time; the ${units} of the others are hidden`}
            className="max-w-[160px] rounded border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700 focus:border-blue-400 focus:outline-none"
          >
            {spatial.libraries.map((l) => (
              <option key={l.id} value={l.id}>
                {l.id} ({l.n_obs.toLocaleString()} {unitWord(spatial, l.n_obs !== 1)})
              </option>
            ))}
          </select>
        </>
      )}
      <label
        className={`flex items-center gap-1 text-[11px] font-medium ${hasImage ? "text-slate-500" : "text-slate-400"}`}
        title={hasImage ? "Show the tissue image under the " + units : "This section has no tissue image"}
      >
        <input
          type="checkbox"
          checked={hasImage && showImage}
          disabled={!hasImage}
          onChange={(e) => setShowImage(e.target.checked)}
          className="accent-blue-500"
        />
        Image
      </label>
      {hasImage && showImage && (
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={imageOpacity}
          onChange={(e) => setImageOpacity(parseFloat(e.target.value))}
          aria-label="Image opacity"
          title="Image opacity"
          className="w-14 accent-blue-500"
        />
      )}
      <InfoTip label="About the tissue plot">
        {spatial.unit === "spot"
          ? "Each dot is a Visium spot (55 µm), drawn at its real size on the H&E image; one spot holds several cells. Size 2 is the real spot size."
          : spatial.unit === "bin"
            ? `Each dot is a Visium HD bin${spatial.bin_size_um ? ` (${spatial.bin_size_um} µm square)` : ""}, as wide as the real bin on the H&E image; a bin is a square of the capture grid, not a segmented cell. Size 2 is the real bin width.`
            : "Each dot is a cell at its position in the section."}{" "}
        Colours, overlays, lasso and labels work as on the UMAP. Zooming in loads the sharper image.
      </InfoTip>
    </div>
  );
}
