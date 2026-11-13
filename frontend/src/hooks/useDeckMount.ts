import { useMemo, useState } from "react";
import type { Layer } from "@deck.gl/core";

/**
 * A key for a conditionally rendered `<DeckGL>` and the layers to hand it.
 *
 * deck.gl refuses a layer instance that another Deck has already initialised
 * (`assert(!this.internalState)` in Layer._initialize: "deck.gl: assertion
 * failed"). A plot whose memoised layers outlive its `<DeckGL>` (unmounted
 * while a resize measured the area at 0, or for a context-lost Retry) failed
 * every layer on the remount. The key goes up each time `shown` turns true, so
 * every mount is a new Deck, and that Deck gets clones: the same props, no state.
 */
export function useDeckMount<L extends Layer>(
  shown: boolean,
  layers: readonly L[],
): { key: number; layers: L[] } {
  const [mount, setMount] = useState({ shown, key: 0 });
  let key = mount.key;
  if (shown !== mount.shown) {
    // Adjusting state while rendering (React's pattern for "state from the previous render")
    key = shown ? mount.key + 1 : mount.key;
    setMount({ shown, key });
  }
  const fresh = useMemo(
    // (a stand-in without `clone`, as test fakes are, passes through)
    () => layers.map((l) => (typeof l.clone === "function" ? (l.clone({}) as L) : l)),
    // `key` is the point: a new Deck needs new instances even when the layers did not change
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [layers, key],
  );
  return { key, layers: fresh };
}
