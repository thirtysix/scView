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
              color_label: str, caption: str, focus: np.ndarray | None = None, tissue: bool = False,
              arrows: dict[str, Any] | None = None, graph: dict[str, Any] | None = None,
              label_nodes: bool = True) -> bytes:
    """Cells on two embedding dimensions, coloured by a category (legend, and labels on
    the groups) or a number (colour bar). ``focus`` (a mask): only those cells are
    coloured, the others drawn grey beneath them. ``tissue``: positions on a section,
    y downwards as in its image. ``arrows``: velocity on a grid (``GET /velocity/grid``)
    drawn over the cells; ``graph``: a PAGA graph placed on this embedding (``GET
    /trajectory/paga``), its nodes named unless ``label_nodes`` is False."""
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
    if arrows is not None:
        _arrows(ax, arrows)
    if graph is not None:
        _paga(ax, graph, label_nodes)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    if tissue:
        ax.invert_yaxis()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.01, 0.005, caption, fontsize=8, color="#555555")
    return _png(fig)


ARROW_STEPS = 1.6  # the 95th-percentile arrow is drawn this many grid steps long
PAGA_MIN_EDGE = 0.1  # connectivity below which an edge outside the tree is not drawn


def _arrows(ax, grid: dict[str, Any]) -> None:
    """Velocity arrows on a grid, scaled so the 95th-percentile arrow spans ARROW_STEPS grid steps."""
    x, y = np.asarray(grid.get("x") or [], float), np.asarray(grid.get("y") or [], float)
    u, v = np.asarray(grid.get("u") or [], float), np.asarray(grid.get("v") or [], float)
    if not x.size:
        return
    p95 = float(grid.get("length_p95") or np.percentile(np.hypot(u, v), 95) or 1.0)
    step = float(grid.get("step") or 1.0)
    ax.quiver(x, y, u, v, angles="xy", scale_units="xy", scale=max(p95, 1e-12) / (ARROW_STEPS * step),
              width=0.0022, headwidth=4, headlength=4.5, headaxislength=4, color="#111111", alpha=0.85, zorder=3)


def _paga(ax, view: dict[str, Any], label_nodes: bool) -> None:
    """PAGA's groups at their median position (size by cells), tree edges solid, other edges
    with connectivity >= PAGA_MIN_EDGE dashed; line width by connectivity."""
    rec, pos = view.get("paga") or {}, view.get("positions") or []
    groups, sizes = rec.get("groups") or [], rec.get("sizes") or []
    tree = {(int(i), int(j)) for i, j in rec.get("tree") or []}
    for i, j, w in rec.get("edges") or []:
        i, j = int(i), int(j)
        if i >= len(pos) or j >= len(pos) or pos[i] is None or pos[j] is None:
            continue
        in_tree = (min(i, j), max(i, j)) in tree
        if not in_tree and w < PAGA_MIN_EDGE:
            continue
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="#111111" if in_tree else "#555555",
                lw=0.6 + 4.0 * float(w), ls="-" if in_tree else "--", alpha=0.9 if in_tree else 0.55, zorder=4)
    top = max(sizes or [1]) or 1
    for k, g in enumerate(groups):
        if k >= len(pos) or pos[k] is None:
            continue
        ax.scatter([pos[k][0]], [pos[k][1]], s=40 + 360 * np.sqrt((sizes[k] if k < len(sizes) else 0) / top),
                   c="white", edgecolors="#111111", linewidths=1.2, zorder=5)
        if label_nodes:  # beside the node, not over it
            ax.annotate(str(g), (pos[k][0], pos[k][1]), xytext=(9, 9), textcoords="offset points", fontsize=8,
                        weight="bold", zorder=6,
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75})


def phase(result: dict[str, Any], title: str, caption: str) -> bytes:
    """A gene's phase portrait (``GET /velocity/phase``): unspliced against spliced moments per
    cell, coloured by a column when given, with the fitted steady-state line u = gamma s."""
    s = np.asarray(result.get("spliced") or [], dtype=float)
    u = np.asarray(result.get("unspliced") or [], dtype=float)
    fig = _figure(7.5, 6)
    ax = fig.add_subplot(1, 1, 1)
    size = max(1.0, min(14.0, 20000.0 / max(len(s), 1)))
    color = result.get("color")
    if not color:
        ax.scatter(s, u, s=size, c="#4a6fa5", linewidths=0, rasterized=True)
    elif color.get("kind") == "numeric":
        v = np.asarray([np.nan if x is None else x for x in color.get("values") or []], dtype=float)
        sc = ax.scatter(s, u, s=size, c=v, cmap="viridis", linewidths=0, rasterized=True)
        fig.colorbar(sc, ax=ax, shrink=0.6, label=color.get("column"))
    else:
        names = [str(c) for c in color.get("categories") or []]
        codes = np.asarray(color.get("codes") or [], dtype=int)
        colours = _palette(len(names))
        order = np.random.default_rng(SEED).permutation(len(s))
        rgba = np.array([colours[c] for c in codes[order]], dtype=float).reshape(-1, 3)
        ax.scatter(s[order], u[order], s=size, c=rgba, linewidths=0, rasterized=True)
        if len(names) <= MAX_LEGEND:
            from matplotlib.lines import Line2D

            present = set(codes.tolist())
            handles = [Line2D([], [], marker="o", ls="", color=colours[i], label=n)
                       for i, n in enumerate(names) if i in present]
            ax.legend(handles=handles, title=color.get("column"), loc="center left", bbox_to_anchor=(1.0, 0.5),
                      fontsize=8, title_fontsize=9, frameon=False, markerscale=1.2)
    gamma = result.get("gamma")
    if gamma is not None and s.size:
        top = float(np.nanmax(s)) if np.isfinite(s).any() else 1.0
        ax.plot([0, top], [0, gamma * top], color="#c0392b", lw=1.5, label=f"steady state, gamma {gamma:.3g}")
        if not color or color.get("kind") == "numeric":
            ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_xlabel("spliced (neighbour mean)")
    ax.set_ylabel("unspliced (neighbour mean)")
    ax.set_title(title, fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.01, -0.02, caption, fontsize=8, color="#555555")
    return _png(fig)


MAX_PANELS = 12  # sections side by side in one figure


def sections(xy: np.ndarray, values: pd.Series | np.ndarray | None, panel: np.ndarray, order: list[str],
             title: str, color_label: str, caption: str, focus: np.ndarray | None = None) -> bytes:
    """Tissue sections side by side, each on its own axes (their positions overlap: each
    section has its own pixel frame), one colour scale or legend for all. ``panel`` names
    each row's section; ``order`` the sections drawn, in that order."""
    import math

    n = len(order)
    ncol = min(4, max(1, math.ceil(math.sqrt(n))))
    nrow = math.ceil(n / ncol)
    cats = None
    colours: list = []
    legend = values is not None and is_categorical(values)
    if legend:  # the groups present among the coloured cells, one colour each across the sections
        base = values.astype("category").reset_index(drop=True)
        coloured = base if focus is None else base[focus]
        cats = base.cat.set_categories([c for c in base.cat.categories if (coloured == c).any()])
        colours = _palette(len(cats.cat.categories))
    fig = _figure(3.3 * ncol + (2.2 if legend or values is not None else 0.4), 3.3 * nrow + 0.9)
    vmin = vmax = None
    if values is not None and not legend:
        v_all = np.asarray(values, dtype=np.float64)
        shown = v_all[focus] if focus is not None else v_all
        finite = shown[np.isfinite(shown)]
        vmin, vmax = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
    sc = None
    for k, name in enumerate(order):
        ax = fig.add_subplot(nrow, ncol, k + 1)
        rows = np.flatnonzero(panel == name)
        size = max(0.5, min(10.0, 6000.0 / max(len(rows), 1)))
        if focus is not None:  # the other cells grey beneath
            others = rows[~focus[rows]]
            ax.scatter(xy[others, 0], xy[others, 1], s=size, c="#d9d9d9", linewidths=0, rasterized=True)
        draw = rows if focus is None else rows[focus[rows]]
        if values is None:
            ax.scatter(xy[draw, 0], xy[draw, 1], s=size, c="#4a6fa5", linewidths=0, rasterized=True)
        elif len(draw):
            if legend:
                order_ = draw[np.random.default_rng(SEED).permutation(len(draw))]
                codes = cats.cat.codes.to_numpy()[order_]
                rgba = np.array([colours[c] if c >= 0 else (0.8, 0.8, 0.8) for c in codes], dtype=float).reshape(-1, 3)
                ax.scatter(xy[order_, 0], xy[order_, 1], s=size, c=rgba, linewidths=0, rasterized=True)
            else:
                v = np.asarray(values, dtype=np.float64)[draw]
                o = np.argsort(np.nan_to_num(v, nan=-np.inf))
                sc = ax.scatter(xy[draw[o], 0], xy[draw[o], 1], s=size, c=v[o], cmap="viridis", vmin=vmin,
                                vmax=vmax, linewidths=0, rasterized=True)
        ax.set_title(f"{name} ({len(rows):,})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_yaxis()
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    if legend:
        names = [str(c) for c in cats.cat.categories]
        if len(names) <= MAX_LEGEND:
            from matplotlib.lines import Line2D

            handles = [Line2D([], [], marker="o", ls="", color=colours[i], label=nm) for i, nm in enumerate(names)]
            fig.legend(handles=handles, title=color_label, loc="center left", bbox_to_anchor=(0.995, 0.5),
                       fontsize=8, title_fontsize=9, frameon=False, markerscale=1.2)
        else:
            caption += f" {len(names)} groups: no legend."
    elif sc is not None:
        fig.colorbar(sc, ax=fig.axes, shrink=0.6, label=color_label)
    fig.suptitle(title, fontsize=11)
    fig.text(0.01, 0.003, caption, fontsize=8, color="#555555")
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
    fig.colorbar(sc, ax=ax, shrink=0.5, label=f"mean, {result.get('layer_label') or result.get('layer')}")
    for f in (0.25, 0.5, 1.0):
        ax.scatter([], [], s=f * 220 + 1, c="#bbbbbb", label=f"{int(f * 100)}%")
    ax.legend(title="expressing", loc="upper left", bbox_to_anchor=(1.25, 1.0), fontsize=7,
              title_fontsize=8, frameon=False, labelspacing=1.2)
    ax.set_title(title, fontsize=11)
    fig.text(0.01, -0.02, caption, fontsize=8, color="#555555")
    return _png(fig)


MAX_TRACKS = 16  # coverage tracks drawn; the caption says how many there are
MAX_GENE_ARROWS = 12


def coverage(result: dict[str, Any], title: str, caption: str) -> bytes:
    """ATAC coverage tracks (``GET /atac/coverage``): one filled track per group on a shared
    y-axis (normalised depth), the peaks in the window as bars, the genes as arrows."""
    from matplotlib.patches import Rectangle

    tracks = list(result.get("tracks") or [])[:MAX_TRACKS]
    x = np.asarray(result.get("x") or [], dtype=float)
    reg = result.get("region") or {}
    n = max(1, len(tracks))
    fig = _figure(10, 0.75 * n + 1.6)
    axes = fig.subplots(n + 2, 1, sharex=True,
                        gridspec_kw={"height_ratios": [1.0] * n + [0.35, 0.6], "hspace": 0.08})
    colours = _palette(n)
    top = max((max(t.get("values") or [0]) for t in tracks), default=0.0)
    ymax = float(result.get("ymax") or 0) or float(top) or 1.0
    width = float(x[1] - x[0]) if x.size > 1 else 1.0
    for ax, t, colour in zip(axes[:n], tracks, colours):
        v = np.asarray(t.get("values") or [], dtype=float)
        if v.size and v.size == x.size:
            ax.fill_between(x, 0, v, step="mid", color=colour, alpha=0.85, linewidth=0)
        ax.set_ylim(0, ymax)
        ax.set_yticks([])
        ax.set_ylabel(str(t.get("name")), rotation=0, ha="right", va="center", fontsize=8)
    pax, gax = axes[n], axes[n + 1]
    for p in result.get("peaks") or []:
        pax.add_patch(Rectangle((p["start"], 0.2), max(p["end"] - p["start"], width / 4), 0.6,
                                color="#555555", linewidth=0))
    pax.set_ylim(0, 1)
    lo, hi = reg.get("start", 0), reg.get("end", 1)
    for i, g in enumerate((result.get("genes") or [])[:MAX_GENE_ARROWS]):
        y = 0.75 - 0.5 * (i % 2)
        start, end = max(g["start"], lo), min(g["end"], hi)
        forward = g.get("strand") != "-"
        gax.annotate("", xy=(end if forward else start, y), xytext=(start if forward else end, y),
                     arrowprops={"arrowstyle": "->", "color": "#1f4e79", "lw": 1.2})
        gax.text((start + end) / 2, y + 0.1, g.get("name"), ha="center", va="bottom", fontsize=7)
    gax.set_ylim(0, 1.15)
    for ax, name in ((pax, "peaks"), (gax, "genes")):
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    for ax in axes:
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    gax.set_xlim(lo, hi)
    gax.set_xlabel(f"{reg.get('chrom', '')} position (bp)", fontsize=8)
    gax.ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[0].set_title(title, fontsize=11)
    import textwrap

    # wrapped: a long caption would widen the figure past MAX_PX
    fig.text(0.01, -0.02, textwrap.fill(caption, 150), fontsize=8, color="#555555", va="top")
    return _png(fig)
