"""PNG figures for the ``plot`` tool, drawn server-side with matplotlib (Agg).

The values come from the dataset's adaptor, as the embedding and expression
routes read them; a dataset with more than ``MAX_POINTS`` cells is drawn from
a random sample of them, and the caption says so. Each figure is at most
``MAX_PX`` pixels a side and is redrawn smaller until under ``MAX_BYTES``.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd

MAX_PX = 1600
MAX_BYTES = 300_000
MAX_POINTS = 50_000
MAX_LEGEND = 40
LABEL_MIN_SHARE = 0.002  # groups under 0.2% of the cells get a legend entry, no label
LABEL_HEIGHT = 0.035  # of the axis: a label's line height, for moving labels apart
SMALL_GROUP = 0.01  # groups under 1% of the cells: drawn last, with larger points
SEED = 0


def _figure(width: float, height: float):
    from matplotlib.figure import Figure  # no pyplot: no global state across threads

    return Figure(figsize=(width, height), dpi=100)


def _png(fig, max_px: int = MAX_PX) -> bytes:
    w, h = fig.get_size_inches()
    dpi = min(150.0, max_px / max(w, h))
    for _ in range(6):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        data = buf.getvalue()
        if len(data) <= MAX_BYTES:
            return data
        dpi *= 0.75
    return data


def sample(n: int, max_points: int | None = None) -> np.ndarray | None:
    """Row positions to draw (sorted), or None for every cell."""
    max_points = max_points or MAX_POINTS
    if n <= max_points:
        return None
    return np.sort(np.random.default_rng(SEED).choice(n, size=max_points, replace=False))


def _palette(n: int) -> list:
    from matplotlib import colormaps

    names = ["tab20", "tab20b", "tab20c"]
    colours = [c for name in names for c in colormaps[name].colors]
    return [colours[i % len(colours)] for i in range(n)]


def is_categorical(values: Any) -> bool:
    """Coloured by group (not a colour scale): categories, text, booleans."""
    if not isinstance(values, pd.Series):
        return False
    return (isinstance(values.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(values)
            or not pd.api.types.is_numeric_dtype(values))


def _label_spots(xy: np.ndarray, codes: np.ndarray, names: list[str]) -> list[tuple[str, float, float]]:
    """Where to write each group's name: on the group's cell nearest its median (so a
    scattered group's label is never in empty space), skipped for a group under
    LABEL_MIN_SHARE of the cells, and moved up or down off a label already placed
    (largest group first); a small group's label sits just above it, so its few cells
    stay visible; a label with no free spot near its group is left out
    (its legend entry stays). Cells with no coordinates (NaN) are left out."""
    ok = np.isfinite(xy).all(axis=1)
    xy, codes = xy[ok], codes[ok]
    if len(xy) == 0:
        return []
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    placed: list[tuple[float, float, float]] = []  # x, y (0-1), half width
    out = []
    sizes = np.bincount(codes[codes >= 0], minlength=len(names))
    for i in np.argsort(-sizes):
        hit = codes == i
        if sizes[i] < max(5, LABEL_MIN_SHARE * len(codes)):
            continue
        pts = xy[hit]
        med = np.median(pts, axis=0)
        dist = ((pts - med) ** 2).sum(axis=1)
        x, y = (pts[np.argmin(dist)] - lo) / span
        if sizes[i] < SMALL_GROUP * len(codes):  # just above a small group's core, not over it
            core = pts[dist <= np.median(dist)]
            y = (core[:, 1].max() - lo[1]) / span[1] + 0.6 * LABEL_HEIGHT
        half = 0.0075 * len(names[i]) + 0.01
        for dy in (0, 1, -1, 2, -2, 3, -3):
            yy = y + dy * LABEL_HEIGHT
            if all(abs(x - px) > half + ph or abs(yy - py) > LABEL_HEIGHT for px, py, ph in placed):
                placed.append((x, yy, half))
                out.append((names[i], lo[0] + x * span[0], lo[1] + yy * span[1]))
                break
    return out


def embedding(xy: np.ndarray, values: pd.Series | np.ndarray | None, title: str,
              color_label: str, caption: str, focus: np.ndarray | None = None) -> bytes:
    """Cells on two embedding dimensions, coloured by a category (legend, and labels on
    the groups) or a number (colour bar). ``focus`` (a mask): only those cells are
    coloured, the others drawn grey beneath them."""
    fig = _figure(8.5, 7)
    ax = fig.add_subplot(1, 1, 1)
    size = max(0.5, min(12.0, 30000.0 / max(len(xy), 1)))
    if focus is not None:
        ax.scatter(xy[~focus, 0], xy[~focus, 1], s=size, c="#d9d9d9", linewidths=0, rasterized=True)
        xy = xy[focus]
        if values is not None:
            values = values[focus]
    if values is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=size, c="#4a6fa5", linewidths=0, rasterized=True)
    elif is_categorical(values):
        cats = values.astype("category")
        if focus is not None:
            cats = cats.cat.remove_unused_categories()
        names = [str(c) for c in cats.cat.categories]
        codes = cats.cat.codes.to_numpy()
        colours = _palette(len(names))
        # Random order, so no large group is drawn over all others; a small group (under
        # SMALL_GROUP of the cells) last and with larger points, or a tight one is not seen.
        sizes = np.bincount(codes[codes >= 0], minlength=len(names))
        small = (codes >= 0) & (sizes[np.maximum(codes, 0)] < SMALL_GROUP * len(codes))
        order = np.random.default_rng(SEED).permutation(len(xy))
        order = np.concatenate([order[~small[order]], order[small[order]]])
        rgba = np.array([colours[c] if c >= 0 else (0.8, 0.8, 0.8) for c in codes[order]],
                        dtype=float).reshape(-1, 3)
        dot = np.where(small[order], max(3.0, 4 * size), size)
        ax.scatter(xy[order, 0], xy[order, 1], s=dot, c=rgba, linewidths=0, rasterized=True)
        if len(names) <= MAX_LEGEND:
            from matplotlib.lines import Line2D

            handles = [Line2D([], [], marker="o", ls="", color=colours[i], label=n)
                       for i, n in enumerate(names)]
            ax.legend(handles=handles, title=color_label, loc="center left", bbox_to_anchor=(1.0, 0.5),
                      fontsize=8, title_fontsize=9, frameon=False, markerscale=1.2)
            for n, x, y in _label_spots(xy, codes, names):
                ax.text(x, y, n, fontsize=8, ha="center", va="center", weight="bold",
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.6})
        else:
            caption += f" {len(names)} groups: no legend."
    else:
        v = np.asarray(values, dtype=np.float64)
        order = np.argsort(np.nan_to_num(v, nan=-np.inf))  # highest on top
        sc = ax.scatter(xy[order, 0], xy[order, 1], s=size, c=v[order], cmap="viridis",
                        linewidths=0, rasterized=True)
        fig.colorbar(sc, ax=ax, shrink=0.6, label=color_label)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.01, 0.005, caption, fontsize=8, color="#555555")
    return _png(fig)


def violin(values: np.ndarray, groups: pd.Series, title: str, ylabel: str, caption: str) -> bytes:
    cats = groups.astype("category")
    names = [str(c) for c in cats.cat.categories]
    codes = cats.cat.codes.to_numpy()
    data = [values[codes == i] for i in range(len(names))]
    keep = [i for i, d in enumerate(data) if len(d)]
    fig = _figure(max(6.0, min(16.0, 0.45 * len(keep) + 2)), 5)
    ax = fig.add_subplot(1, 1, 1)
    colours = _palette(len(names))
    parts = ax.violinplot([data[i] if np.ptp(data[i]) > 0 else data[i] + 1e-9 for i in keep],
                          showmedians=True, showextrema=False, widths=0.8)
    for body, i in zip(parts["bodies"], keep):
        body.set_facecolor(colours[i])
        body.set_alpha(0.75)
    ax.set_xticks(range(1, len(keep) + 1))
    ax.set_xticklabels([names[i] for i in keep], rotation=60 if len(keep) > 6 else 0, ha="right"
                       if len(keep) > 6 else "center", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.01, -0.02, caption, fontsize=8, color="#555555")
    return _png(fig)


def dotplot(result: dict[str, Any], title: str, caption: str) -> bytes:
    """Genes x groups from the by-group route: dot size = fraction expressing, colour =
    mean over all of a group's cells."""
    genes, groups = result["genes"], result["groups"]
    mean = np.asarray(result["mean"], dtype=float)  # [group][gene]
    frac = np.asarray(result["fraction"], dtype=float)
    fig = _figure(max(5.0, min(16.0, 0.42 * len(genes) + 3)), max(3.5, min(14.0, 0.32 * len(groups) + 2)))
    ax = fig.add_subplot(1, 1, 1)
    gx, gy = np.meshgrid(np.arange(len(genes)), np.arange(len(groups)))
    sc = ax.scatter(gx.ravel(), gy.ravel(), s=(frac.ravel() * 220) + 1, c=mean.ravel(), cmap="Reds",
                    edgecolors="#888888", linewidths=0.3)
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=8)
    ax.set_xlim(-0.6, len(genes) - 0.4)
    ax.set_ylim(len(groups) - 0.4, -0.6)
    fig.colorbar(sc, ax=ax, shrink=0.5, label=f"mean ({result.get('layer_label') or result.get('layer')})")
    for f in (0.25, 0.5, 1.0):
        ax.scatter([], [], s=f * 220 + 1, c="#bbbbbb", label=f"{int(f * 100)}%")
    ax.legend(title="expressing", loc="upper left", bbox_to_anchor=(1.25, 1.0), fontsize=7,
              title_fontsize=8, frameon=False, labelspacing=1.2)
    ax.set_title(title, fontsize=11)
    fig.text(0.01, -0.02, caption, fontsize=8, color="#555555")
    return _png(fig)
