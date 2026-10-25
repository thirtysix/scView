import { useCallback, useId, useState, type KeyboardEvent } from "react";
import {
  cameraAfterKey,
  cellsInScreenRect,
  crosshairAfterKey,
  rectPolygon,
  scatterKeyHelp,
} from "@/lib/scatterKeyboard";

type ViewState = Record<string, unknown>;

interface Options {
  /** Off: the plot is not focusable (a compact reference map) */
  enabled: boolean;
  /** The plot's accessible name (lib/scatterKeyboard scatterAriaLabel) */
  label: string;
  viewState: ViewState | null;
  /** Show this camera, as a user's drag would */
  onCamera: (viewState: ViewState) => void;
  /** Back to the automatic fit ("0") */
  onFit: () => void;
  positions: Float32Array | null;
  dimensions: 2 | 3;
  size: { width: number; height: number };
  /** The cells a keyboard rectangle selects; without it, S does nothing */
  onSelect?: (indices: Set<number>) => void;
}

type RectState = { stage: "first"; at: [number, number] } | { stage: "second"; at: [number, number]; from: [number, number] };

const MESSAGES = {
  first: "Rectangle selection: move the crosshair to one corner with the arrow keys, then press Enter.",
  second: "First corner set. Move to the opposite corner, then press Enter.",
  cancel: "Rectangle selection cancelled.",
};

/**
 * Keyboard use of the embedding scatter: the props for its focusable wrapper
 * (role, name, key help, key handler) and the crosshair / rectangle drawn while
 * a rectangle selection is under way. Arrow keys pan (3D: turn), + / - zoom,
 * 0 fits; S starts a rectangle selection whose corners are set with Enter.
 */
export function useScatterKeyboard({
  enabled,
  label,
  viewState,
  onCamera,
  onFit,
  positions,
  dimensions,
  size,
  onSelect,
}: Options) {
  const helpId = useId();
  const [rect, setRect] = useState<RectState | null>(null);
  const [message, setMessage] = useState("");

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      if (e.target !== e.currentTarget || e.altKey || e.ctrlKey || e.metaKey) return;
      if (rect) {
        if (e.key === "Escape") {
          // Not the panel's Escape (which clears the selection) as well.
          e.preventDefault();
          e.stopPropagation();
          setRect(null);
          setMessage(MESSAGES.cancel);
          return;
        }
        if (e.key === "Enter") {
          e.preventDefault();
          if (rect.stage === "first") {
            setRect({ stage: "second", at: rect.at, from: rect.at });
            setMessage(MESSAGES.second);
            return;
          }
          setRect(null);
          setMessage("");
          if (positions && viewState && onSelect) {
            onSelect(cellsInScreenRect(positions, dimensions, rect.from, rect.at, viewState, size));
          }
          return;
        }
        const at = crosshairAfterKey(rect.at, e.key, e.shiftKey, size);
        if (at) {
          e.preventDefault();
          setRect({ ...rect, at });
        }
        return;
      }
      if ((e.key === "s" || e.key === "S") && onSelect && positions) {
        e.preventDefault();
        setRect({ stage: "first", at: [size.width / 2, size.height / 2] });
        setMessage(MESSAGES.first);
        return;
      }
      if (e.key === "0") {
        e.preventDefault();
        onFit();
        return;
      }
      if (!viewState) return;
      const next = cameraAfterKey(viewState, e.key, size, dimensions);
      if (next) {
        e.preventDefault();
        onCamera(next);
      }
    },
    [rect, positions, viewState, onSelect, dimensions, size, onFit, onCamera],
  );

  if (!enabled) return { containerProps: {}, overlay: null };

  const containerProps = {
    tabIndex: 0,
    role: "application",
    "aria-roledescription": "scatter plot",
    "aria-label": label,
    "aria-describedby": helpId,
    onKeyDown,
    onBlur: () => {
      if (rect) {
        setRect(null);
        setMessage(MESSAGES.cancel);
      }
    },
  } as const;

  const box = rect?.stage === "second" ? rectPolygon(rect.from, rect.at) : null;
  const overlay = (
    <>
      <span id={helpId} className="sr-only">
        {scatterKeyHelp(dimensions, !!onSelect)}
      </span>
      <span role="status" aria-live="polite" className="sr-only">
        {message}
      </span>
      {rect && (
        <svg className="pointer-events-none absolute inset-0 z-20 h-full w-full" aria-hidden="true" data-testid="keyboard-rect">
          {box && (
            <rect
              x={box[0]![0]}
              y={box[0]![1]}
              width={box[2]![0] - box[0]![0]}
              height={box[2]![1] - box[0]![1]}
              fill="rgba(255, 165, 0, 0.15)"
              stroke="rgba(255, 140, 0, 0.9)"
              strokeWidth={1.5}
              strokeDasharray="4 2"
            />
          )}
          <line x1={rect.at[0] - 10} x2={rect.at[0] + 10} y1={rect.at[1]} y2={rect.at[1]} stroke="#ea580c" strokeWidth={2} />
          <line x1={rect.at[0]} x2={rect.at[0]} y1={rect.at[1] - 10} y2={rect.at[1] + 10} stroke="#ea580c" strokeWidth={2} />
        </svg>
      )}
    </>
  );
  return { containerProps, overlay };
}
