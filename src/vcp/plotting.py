"""Shared figure style (validated categorical palette, recessive chrome, CJK-capable font)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# reference palette (light mode), fixed slot order - never cycled
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
GRAY_LIGHT = "#c3c2b7"
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"     # diverging blue <-> red, gray midpoint
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# colour follows the entity (model family), never its rank
MODEL_COLORS = {
    "PerturbFormer": BLUE,
    "kNN": ORANGE,
    "Linear": AQUA,
    "PerturbNet": YELLOW,
    "Mean-shift": MUTED,
    "No-change": GRAY_LIGHT,
}


def model_color(name: str) -> str:
    for k, c in MODEL_COLORS.items():
        if name.startswith(k) or k in name:
            return c
    return INK2


def setup():
    cjk = [f.name for f in font_manager.fontManager.ttflist if "Noto Sans CJK SC" in f.name]
    if not cjk:
        for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                pass
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans", "Arial"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.facecolor": SURFACE,
        "figure.facecolor": "white",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def bar_h(ax, labels, values, colors, err=None, fmt="{:.3f}", xlim=None):
    """Horizontal bars (thin, value label at the tip, text in ink - never series colour)."""
    y = range(len(labels))
    ax.barh(list(y), values, height=0.6, color=colors, xerr=err,
            error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 0})
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    span = (xlim[1] - xlim[0]) if xlim else max(abs(v) for v in values)
    for i, v in enumerate(values):
        off = (err[i] if err is not None else 0) + 0.01 * span
        ax.text(v + off if v >= 0 else v - off, i, fmt.format(v), va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=INK2)
    if xlim:
        ax.set_xlim(*xlim)
