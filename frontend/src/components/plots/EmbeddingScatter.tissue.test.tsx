import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen } from "@testing-library/react";
import type { SpatialLibrary } from "@/api/types";

// jsdom has no WebGL: capture what the scatter hands deck.gl instead.
const { deckProps } = vi.hoisted(() => ({
  deckProps: { current: null as Record<string, unknown> | null },
}));
vi.mock("@deck.gl/react", async () => {
  const { forwardRef } = await import("react");
  return {
    DeckGL: forwardRef<HTMLDivElement, Record<string, unknown>>((props, ref) => {
      deckProps.current = props;
      return <div ref={ref} data-testid="deck" />;
    }),
  };
});

// The image layer as constructed (deck.gl resolves `image`, an async prop, on load).
vi.mock("@deck.gl/layers", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  BitmapLayer: class {
    id: string;
    props: Record<string, unknown>;
    constructor(props: Record<string, unknown>) {
      this.id = props.id as string;
      this.props = props;
    }
  },
}));

import { EmbeddingScatter, type TissueMode } from "./EmbeddingScatter";

class FakeResizeObserver {
  constructor(private cb: ResizeObserverCallback) {}
  observe() {
    this.cb([{ contentRect: { width: 400, height: 300 } } as ResizeObserverEntry], this as never);
  }
  disconnect() {}
  unobserve() {}
}

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", FakeResizeObserver);
  deckProps.current = null;
});
afterEach(() => vi.unstubAllGlobals());

const LIB: SpatialLibrary = {
  id: "V1 node",
  n_obs: 3,
  images: {
    lowres: { width: 600, height: 500, scalef: 0.02 },
    hires: { width: 2000, height: 1666, scalef: 0.0666 },
  },
  spot_diameter_fullres: 90,
  bounds: [1000, 1000, 20000, 20000],
};
const tissue = (over: Partial<TissueMode> = {}): TissueMode => ({
  datasetId: "ds1",
  library: LIB,
  showImage: true,
  imageOpacity: 0.7,
  unit: "spot",
  ...over,
});
// Tissue world (x, -y); the middle cell belongs to another section (no position).
const positions = Float32Array.from([1000, -1000, NaN, NaN, 20000, -20000]);

type AnyLayer = { id: string; props: Record<string, unknown> };
const layers = () => deckProps.current!.layers as AnyLayer[];
const layer = (id: string) => layers().find((l) => l.id === id);

const renderTissue = (t: TissueMode, pointSize = 2) =>
  render(
    <EmbeddingScatter
      positions={positions}
      colorValues={Int32Array.from([0, 1, 0])}
      colorType="categorical"
      categories={["A", "B"]}
      pointSize={pointSize}
      embedding="spatial"
      tissue={t}
    />,
  );

describe("EmbeddingScatter tissue mode", () => {
  it("draws the lowres image under the spots, over its full-res extent", async () => {
    renderTissue(tissue());
    await screen.findByTestId("deck");
    const ids = layers().map((l) => l.id);
    expect(ids.indexOf("tissue-image")).toBeLessThan(ids.indexOf("embedding-scatter"));
    const img = layer("tissue-image")!;
    expect(img.props.image).toBe("/api/v1/datasets/ds1/spatial/V1%20node/image?res=lowres");
    const [l, b, r, t] = img.props.bounds as number[];
    expect([l, t]).toEqual([0, 0]);
    expect(r).toBeCloseTo(600 / 0.02);
    expect(b).toBeCloseTo(-500 / 0.02);
    expect(img.props.opacity).toBe(0.7);
  });

  it("switches to the hires image once zoomed in past the lowres pixels, and keeps it", async () => {
    renderTissue(tissue());
    await screen.findByTestId("deck");
    const move = (zoom: number) =>
      act(() =>
        (deckProps.current!.onViewStateChange as (a: unknown) => void)({
          viewState: { target: [10000, -10000, 0], zoom },
        }),
      );
    move(Math.log2(0.02) + 1); // a lowres pixel spans 2 screen pixels
    let img = layer("tissue-image")!;
    expect(img.props.image).toBe("/api/v1/datasets/ds1/spatial/V1%20node/image?res=hires");
    const [, b, r] = img.props.bounds as number[];
    expect(r).toBeCloseTo(2000 / 0.0666);
    expect(b).toBeCloseTo(-1666 / 0.0666);
    move(Math.log2(0.02) - 2); // zoomed out again: no reload of the lowres one
    img = layer("tissue-image")!;
    expect(img.props.image).toContain("res=hires");
  });

  it("draws spots in world units: radius = half the spot diameter, scaled by the size", async () => {
    renderTissue(tissue(), 3);
    await screen.findByTestId("deck");
    const pts = layer("embedding-scatter")!.props;
    expect(pts.radiusUnits).toBe("common");
    expect(pts.getRadius).toBeCloseTo(45 * 1.5);
    expect(pts.radiusMaxPixels).toBeGreaterThan(1e6);
  });

  it("draws every Visium HD bin at half its side, past the sampling cap, and names bins", async () => {
    render(
      <EmbeddingScatter
        positions={positions}
        colorValues={Int32Array.from([0, 1, 0])}
        colorType="categorical"
        categories={["A", "B"]}
        pointSize={2}
        embedding="spatial"
        maxRenderedCells={1}
        tissue={tissue({ unit: "bin", library: { ...LIB, spot_diameter_fullres: 29.2 } })}
      />,
    );
    await screen.findByTestId("deck");
    const dots = layer("embedding-scatter")!;
    expect(dots.props.getRadius).toBeCloseTo(14.6);
    expect(dots.props.radiusUnits).toBe("common");
    // The two with a position in this section: never sampled down to the cap of 1.
    expect((dots.props.data as { length: number }).length).toBe(2);
    expect(screen.queryByText(/sampled/)).toBeNull();
  });

  it("still samples cells over the cap", async () => {
    render(
      <EmbeddingScatter
        positions={Float32Array.from([1, 1, 2, 2, 3, 3])}
        colorValues={Int32Array.from([0, 1, 0])}
        colorType="categorical"
        categories={["A", "B"]}
        pointSize={2}
        embedding="spatial"
        maxRenderedCells={1}
        tissue={tissue({ unit: "spot" })}
      />,
    );
    await screen.findByTestId("deck");
    expect((layer("embedding-scatter")!.props.data as { length: number }).length).toBe(1);
  });

  it("draws cells in screen pixels when the spot size is unknown", async () => {
    renderTissue(tissue({ library: { ...LIB, spot_diameter_fullres: null }, unit: "cell" }), 3);
    await screen.findByTestId("deck");
    const pts = layer("embedding-scatter")!.props;
    expect(pts.radiusUnits).toBe("pixels");
    expect(pts.getRadius).toBe(3);
  });

  it("leaves out the cells of the other sections and fits the camera to the shown one", async () => {
    renderTissue(tissue());
    await screen.findByTestId("deck");
    const data = layer("embedding-scatter")!.props.data as { length: number; attributes: { getPosition: { value: Float32Array } } };
    expect(data.length).toBe(2);
    expect(Array.from(data.attributes.getPosition.value).every(Number.isFinite)).toBe(true);
    const vs = deckProps.current!.viewState as { target: number[] };
    expect(vs.target[0]).toBeCloseTo(10500);
    expect(vs.target[1]).toBeCloseTo(-10500);
  });

  it("hides the image when turned off, and names spots in the tooltip", async () => {
    renderTissue(tissue({ showImage: false }));
    await screen.findByTestId("deck");
    expect(layer("tissue-image")).toBeUndefined();
    const pos = (layer("embedding-scatter")!.props.data as { attributes: { getPosition: { value: Float32Array } } })
      .attributes.getPosition.value;
    const slot = pos[0] === 1000 ? 0 : 1;
    act(() => (deckProps.current!.onHover as (i: unknown) => void)({ picked: true, index: slot, x: 5, y: 5 }));
    expect(screen.getByText("Spot 0")).toBeInTheDocument();
  });

  it("is the plain scatter without tissue mode", async () => {
    render(<EmbeddingScatter positions={Float32Array.from([0, 0, 1, 1])} colorValues={null} colorType="categorical" />);
    await screen.findByTestId("deck");
    expect(layer("tissue-image")).toBeUndefined();
    expect(layer("embedding-scatter")!.props.radiusUnits).toBe("pixels");
  });
});
