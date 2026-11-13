import { describe, expect, it } from "vitest";
import { renderHook } from "@testing-library/react";
import { ScatterplotLayer } from "@deck.gl/layers";
import { useDeckMount } from "./useDeckMount";
import { isContextLoss } from "@/lib/webgl";

const layers = [new ScatterplotLayer({ id: "split-context" }), new ScatterplotLayer({ id: "split-fg-0" })];

describe("useDeckMount", () => {
  it("gives each new Deck its own layer instances with the same props", () => {
    const { result, rerender } = renderHook(({ shown }) => useDeckMount(shown, layers), {
      initialProps: { shown: true },
    });
    const first = result.current;
    expect(first.layers.map((l) => l.id)).toEqual(["split-context", "split-fg-0"]);
    expect(first.layers[0]).not.toBe(layers[0]);

    // Still mounted: same key, same instances (no needless re-initialisation)
    rerender({ shown: true });
    expect(result.current.key).toBe(first.key);
    expect(result.current.layers[0]).toBe(first.layers[0]);

    // Unmounted (area measured 0 mid-resize, or a lost context) and back: a new Deck, new instances
    rerender({ shown: false });
    rerender({ shown: true });
    expect(result.current.key).toBe(first.key + 1);
    expect(result.current.layers[0]).not.toBe(first.layers[0]);
    expect(result.current.layers[0]!.props.id).toBe("split-context");
  });
});

describe("isContextLoss", () => {
  it("is not fooled by a layer called split-context", () => {
    expect(
      isContextLoss(new Error("initialization of ScatterplotLayer({id: 'split-context'}): deck.gl: assertion failed.")),
    ).toBe(false);
  });
  it("matches deck.gl's lost and refused context errors", () => {
    expect(isContextLoss(new Error("WebGL context is lost"))).toBe(true);
    expect(isContextLoss(new Error("WebGL context creation failed"))).toBe(true);
  });
});
