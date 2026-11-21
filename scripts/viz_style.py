"""Shared look for the generated notebook images: palette, matplotlib defaults, and layout helpers.

Used by the make_*_concept_images.py scripts in this folder (one per lesson folder). Palette: blue/orange/aqua
categorical slots, a one-hue blue sequential ramp for heatmaps, recessive gray axes and text, light text cards.
"""
from fractions import Fraction
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
CARD = "#f2f1ed"
BLUE_LIGHT = "#b7d3f6"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f3f7fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": MUTED, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlepad": 10, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "mathtext.fontset": "dejavusans",
})


# ---------------------------------------------------------------- helpers
def new_figure(title, subtitle):
    fig = plt.figure(figsize=(16, 10))
    fig.text(0.5, 0.965, title, ha="center", va="center", fontsize=22, weight="bold")
    fig.text(0.5, 0.925, subtitle, ha="center", va="center", fontsize=13, color=INK2)
    return fig


def grid(fig, rows=2, cols=3, **kw):
    params = dict(top=0.79, bottom=0.06, left=0.05, right=0.975, hspace=0.5, wspace=0.28)
    params.update(kw)
    return fig.add_gridspec(rows, cols, **params)


def save(fig, folder, name):
    """Write the figure to folder/name at 100 dpi (1600 x 1000 px)."""
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / name, dpi=100)
    plt.close(fig)
    print("wrote", folder.name + "/" + name)


def frac_label(value, max_den=36):
    if abs(value) < 1e-12:
        return "0"
    f = Fraction(value).limit_denominator(max_den)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def heat(ax, matrix, rows, cols, title, row_name, col_name, labels=None, vmax=None,
         highlight_rows=(), highlight_cols=(), highlight_cells=(), fontsize=11):
    """Annotated heatmap table: one-hue sequential fill, 2px surface gaps, orange highlights."""
    matrix = np.asarray(matrix, float)
    vmax = vmax or matrix.max() or 1.0
    n_rows, n_cols = matrix.shape
    ax.pcolormesh(np.arange(n_cols + 1), np.arange(n_rows + 1), matrix, cmap=SEQ,
                  vmin=0, vmax=vmax, edgecolors=SURFACE, linewidth=2)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks(np.arange(n_cols) + 0.5, cols)
    ax.set_yticks(np.arange(n_rows) + 0.5, rows)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(col_name, labelpad=6)
    ax.set_ylabel(row_name)
    ax.set_title(title, pad=34)
    for i in range(n_rows):
        for j in range(n_cols):
            value = matrix[i, j]
            text = labels[i][j] if labels is not None else (f"{value:.2f}" if value else "0")
            if abs(value) < 1e-12:
                color = MUTED
            elif value / vmax > 0.55:
                color = "white"
            else:
                color = INK
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=fontsize, color=color)
    box = dict(fill=False, edgecolor=ORANGE, linewidth=3, clip_on=False)
    for i in highlight_rows:
        ax.add_patch(Rectangle((0, i), n_cols, 1, **box))
    for j in highlight_cols:
        ax.add_patch(Rectangle((j, 0), 1, n_rows, **box))
    for i, j in highlight_cells:
        ax.add_patch(Rectangle((j, i), 1, 1, **box))


def bars(ax, x, y, title, xlabel, ylabel="probability", color=BLUE, ylim=None,
         label_fmt=frac_label, highlight=None):
    positions = np.arange(len(x))
    colors = [ORANGE if highlight is not None and k == highlight else color for k in range(len(x))]
    ax.bar(positions, y, width=0.62, color=colors, zorder=2)
    ax.set_xticks(positions, x)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", zorder=0)
    top = ylim if ylim else max(y) * 1.25
    ax.set_ylim(0, top)
    for p, v in zip(positions, y):
        if v > 0:
            ax.text(p, v + top * 0.02, label_fmt(v), ha="center", va="bottom", fontsize=10, color=INK)


def grouped_bars(ax, x, series, title, xlabel, ylim=1.0, label_fmt=lambda v: frac_label(v, 12)):
    positions = np.arange(len(x))
    width = 0.38
    for k, (name, values, color) in enumerate(series):
        offset = (k - 0.5) * width
        ax.bar(positions + offset, values, width=width - 0.04, color=color, label=name, zorder=2)
        for p, v in zip(positions, values):
            if v > 0:
                ax.text(p + offset, v + ylim * 0.02, label_fmt(v), ha="center",
                        va="bottom", fontsize=9, color=INK)
    ax.set_xticks(positions, x)
    ax.set_ylim(0, ylim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", fontsize=10)


def panel(ax, heading, lines, fontsize=12.5, top=1.0, spacing=0.085):
    """Text card: a bold heading, then one entry per line (mathtext allowed)."""
    ax.axis("off")
    ax.add_patch(Rectangle((-0.03, -0.02), 1.06, top + 0.04, transform=ax.transAxes,
                           facecolor=CARD, edgecolor="none", zorder=0, clip_on=False))
    ax.text(0.0, top - 0.04, heading, transform=ax.transAxes, fontsize=14.5,
            weight="bold", va="top")
    y = top - 0.04 - 0.11
    for line in lines:
        if line == "":
            y -= spacing * 0.5
            continue
        is_code = line.startswith("`")
        text = line.strip("`")
        ax.text(0.0, y, text, transform=ax.transAxes, va="top", color=INK if is_code else INK2,
                fontsize=fontsize - (1 if is_code else 0),
                family="monospace" if is_code else None)
        y -= spacing


def trapezoid(values, grid_points, axis=-1):
    """np.trapezoid with the numpy<2 fallback."""
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return integrate(values, grid_points, axis=axis)


def density_image(fig, ax, values, title, vmax, extent=(0, 1, 0, 1), label="density"):
    image = ax.imshow(values.T, origin="lower", extent=extent, cmap=SEQ, vmin=0, vmax=vmax,
                      aspect="equal")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(label)
    colorbar.outline.set_visible(False)
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image
