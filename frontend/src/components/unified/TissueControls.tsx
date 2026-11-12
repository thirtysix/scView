import type { SpatialInfo } from "@/api/types";
import { Help } from "@/components/common/Help";
import { activeLibrary, unitWord } from "@/lib/spatial";
import { useTissueViewStore, type TissueDraw } from "@/stores/tissueViewStore";

/** What the tissue plot shows (the toolbar's Embedding label carries it while the tissue is shown). */
export function TissuePlotTip({ spatial }: { spatial: SpatialInfo }) {
  const outlines = spatial.outlines ?? [];
  return (
    <>
      {spatial.unit === "spot"
        ? "Each dot is a Visium spot (55 µm), drawn at its real size on the H&E image; one spot holds several cells. Size 2 is the real spot size."
        : spatial.unit === "bin"
          ? `Each dot is a Visium HD bin${spatial.bin_size_um ? ` (${spatial.bin_size_um} µm square)` : ""}, as wide as the real bin on the H&E image; a bin is a square of the capture grid, not a segmented cell. Size 2 is the real bin width.`
          : "Each dot is a cell at its position in the section."}{" "}
      Colours, overlays, lasso and labels work as on the UMAP. Zooming in loads the sharper image.
      {outlines.length > 0 && " Draw: the whole section stays points; the segmented shapes are drawn once few enough cells are in view."}
      {outlines.length === 0 && spatial.outlines_note && ` ${spatial.outlines_note}`}
    </>
  );
}

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
  const draw = useTissueViewStore((s) => s.draw);
  const setDraw = useTissueViewStore((s) => s.setDraw);
  const library = activeLibrary(spatial, wanted);
  const outlines = spatial.outlines ?? [];
  const drawChoices: [TissueDraw, string][] = [
    ["points", "Points"],
    ...(outlines.includes("cell") ? [["cell", "Cells"] as [TissueDraw, string]] : []),
    ...(outlines.includes("nucleus") ? [["nucleus", "Nuclei"] as [TissueDraw, string]] : []),
  ];
  const hasImage = !!(library?.images.lowres || library?.images.hires);
  const units = unitWord(spatial);

  return (
    <div className="flex items-center gap-1.5">
      {spatial.libraries.length > 1 && (
        <>
          <label htmlFor="tissue-section" className="text-[11px] font-medium text-slate-500">
            <Help id="tissue.section" side="below">Section</Help>
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
        title={hasImage ? undefined : "This section has no tissue image"}
      >
        <input
          type="checkbox"
          checked={hasImage && showImage}
          disabled={!hasImage}
          onChange={(e) => setShowImage(e.target.checked)}
          title={hasImage ? "Show the tissue image under the " + units : undefined}
          className="accent-blue-500"
        />
        {hasImage ? <Help id="tissue.image" side="below">Image</Help> : "Image"}
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
      {outlines.length > 0 && (
        <div className="flex items-center gap-1" role="group" aria-labelledby="tissue-draw">
          <span id="tissue-draw" className="text-[11px] font-medium text-slate-500">
            <Help id="tissue.draw" side="below">Draw</Help>
          </span>
          {drawChoices.map(([value, label]) => (
            <button
              key={value}
              type="button"
              aria-pressed={draw === value}
              onClick={() => setDraw(value)}
              title={value === "points" ? `Each ${unitWord(spatial, false)} a dot` : `The segmented ${label.toLowerCase()}, once few enough are in view`}
              className={`rounded border px-1.5 py-0.5 text-[11px] ${draw === value ? "border-blue-400 bg-blue-50 text-blue-700" : "border-slate-300 bg-white text-slate-600 hover:bg-slate-50"}`}
            >
              {label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
