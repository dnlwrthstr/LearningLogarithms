#!/usr/bin/env python3
"""Concept sheets for modules 02 (graphs, equations and scales) and 03 (computing logarithms).

One page per notebook that summarizes its central idea, in exponent form, in log form and as a
picture (log-spaced number lines, log axes). Every number is computed here from the same data,
formulas, parameters and random seeds the notebook uses (rng draws replayed in the notebook's
order) and is checked against the notebook's printed outputs with asserts.
Images are written to modules/02_graphs_equations_and_scales/img/ and
modules/03_computing_logarithms/img/.

Run from anywhere:  python scripts/make_modules_02_03_concept_images.py
"""
import math
import random
from decimal import Decimal, getcontext
from functools import partial

import numpy as np
import sympy as sp
from matplotlib.patches import Rectangle, Circle

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, CARD, BLUE, BLUE_LIGHT, ORANGE, AQUA, GRID,
                       new_figure, grid)
from viz_style import save as _save

MOD02 = REPO / "modules" / "02_graphs_equations_and_scales"
MOD03 = REPO / "modules" / "03_computing_logarithms"
save02 = partial(_save, folder=MOD02 / "img")
save03 = partial(_save, folder=MOD03 / "img")

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#a9e3cd"
GRAY = "#5b5a56"
RAMP = ["#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"]


# ================================================================ helpers
def panel(ax, heading, lines, fontsize=12.5, top=1.0, spacing=0.085, cols=(0.0, 0.33, 0.67)):
    """viz_style.panel, but a line holding a fraction is set in display size (\\dfrac) and gets
    a taller line, so stacked fractions stay readable and never touch the next line."""
    ax.axis("off")
    ax.add_patch(Rectangle((-0.03, -0.02), 1.06, top + 0.04, transform=ax.transAxes,
                           facecolor=CARD, edgecolor="none", zorder=0, clip_on=False))
    ax.text(0.0, top - 0.04, heading, transform=ax.transAxes, fontsize=14.5, weight="bold", va="top")
    y = top - 0.04 - 0.11
    for line in lines:
        if line == "":
            y -= spacing * 0.5
            continue
        if isinstance(line, tuple):            # a table row: one entry per column position
            for x, cell in zip(cols, line):
                bold = cell.startswith("*")
                ax.text(x, y, cell.lstrip("*"), transform=ax.transAxes, va="top", fontsize=fontsize,
                        color=INK if bold else INK2, weight="bold" if bold else None)
            y -= spacing
            continue
        is_code = line.startswith("`")
        text = line.strip("`")
        tall = r"\frac" in text
        if tall:
            text = text.replace(r"\frac", r"\dfrac")
            y -= spacing * 0.15
        ax.text(0.0, y, text, transform=ax.transAxes, va="top", color=INK if is_code else INK2,
                fontsize=fontsize - (1 if is_code else 0), family="monospace" if is_code else None)
        y -= spacing * (1.75 if tall else 1.0)


def finish(ax, title, xlabel, ylabel, legend_loc="best", legend_size=9.5, grid_which="major"):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which=grid_which)
    ax.set_axisbelow(True)
    if legend_loc is not None:
        ax.legend(loc=legend_loc, fontsize=legend_size, frameon=True, facecolor=SURFACE,
                  edgecolor="none", framealpha=0.9)


def axes_cross(ax):
    ax.axhline(0, color=MUTED, linewidth=0.9, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=0.9, zorder=1)


def close(a, b, tol):
    return abs(a - b) <= tol


def bisect(function, low, high, steps=200):
    for _ in range(steps):
        middle = (low + high) / 2
        if function(low) * function(middle) <= 0:
            high = middle
        else:
            low = middle
    return (low + high) / 2


def repeated_square_roots(start, count):
    value = Decimal(start)
    for _ in range(count):
        value = value.sqrt()
    return value


# ================================================================ 02/01
def sheet_02_01_graphs():
    # 4.1: swapped points of 2^x land on log2 x
    for x in [-2, -1, 0, 0.5, 1, 2, 3]:
        assert math.isclose(math.log2(2 ** x), x, abs_tol=1e-12)
    # 4.3: every log is a stretched ln
    stretches = {b: 1 / math.log(b) for b in [2, math.e, 10, 0.5]}
    assert [f"{v:+.4f}" for v in stretches.values()] == ["+1.4427", "+1.0000", "+0.4343", "-1.4427"]
    # 4.5: the googol
    googol = 10 ** 100
    assert len(str(googol)) == 101 and math.log10(googol) == 100.0
    ln_googol, log2_googol = math.log(googol), math.log2(googol)
    assert round(ln_googol, 2) == 230.26 and round(log2_googol, 2) == 332.19
    assert f"{math.exp(100):.3e}" == "2.688e+43"
    # Example 3: x-intercept of 2 log10(x + 3) - 1
    x_intercept = math.sqrt(10) - 3
    assert round(x_intercept, 3) == 0.162

    fig = new_figure("The logarithm is the exponential, mirrored in the line y = x",
                     r"$b^{\,y} = x \;\Leftrightarrow\; y = \log_b x$:  every point $(p, q)$ on $y = b^x$ "
                     r"is a point $(q, p)$ on $y = \log_b x$.   Every $\log_b x = \ln x / \ln b$ is one "
                     r"curve, stretched by $1/\ln b$.")
    gs = grid(fig)

    # mirror
    ax = fig.add_subplot(gs[0, 0])
    xe = np.linspace(-4, 3.2, 400)
    xl = np.linspace(0.01, 9, 400)
    ax.plot(xe, 2 ** xe, color=BLUE, linewidth=2.4, label=r"$y = 2^x$")
    ax.plot(xl, np.log2(xl), color=ORANGE, linewidth=2.4, label=r"$y = \log_2 x$")
    ax.plot([-4, 9], [-4, 9], color=MUTED, linestyle="--", linewidth=1.2, label="mirror  y = x")
    for p in [-2, 0, 1, 2, 3]:
        q = 2.0 ** p
        ax.plot([p, q], [q, p], ":", color=MUTED, linewidth=1)
        ax.plot(p, q, "o", color=BLUE, markersize=5, zorder=4)
        ax.plot(q, p, "o", color=ORANGE, markersize=5, zorder=4)
    ax.annotate("(3, 8)", (3, 8), xytext=(6, -4), textcoords="offset points", fontsize=9.5, color=BLUE)
    ax.annotate("(8, 3)", (8, 3), xytext=(-8, -16), textcoords="offset points", fontsize=9.5, color=ORANGE)
    axes_cross(ax)
    ax.set_xlim(-4, 9)
    ax.set_ylim(-4, 9)
    ax.set_aspect("equal")
    finish(ax, "Swap x and y: the same points", "x", "y", legend_loc="upper left", legend_size=9)

    # the base
    ax = fig.add_subplot(gs[0, 1])
    xs = np.linspace(0.02, 6, 400)
    colors = [AQUA, BLUE, "#1c5cab", ORANGE]
    for base, color, label in zip([1.5, 2, np.e, 10], colors, ["1.5", "2", "e", "10"]):
        ax.plot(xs, np.log(xs) / np.log(base), color=color, linewidth=2.2,
                label=rf"$\log_{{{label}}} x$:  stretch $1/\ln b$ = {1 / np.log(base):.2f}")
        ax.plot(xs, np.log(xs) / np.log(1 / base), color=color, linewidth=1.3, linestyle="--")
    ax.plot(1, 0, "o", color=INK, markersize=6, zorder=5)
    axes_cross(ax)
    ax.set_ylim(-4, 4)
    ax.set_xlim(0, 6)
    finish(ax, r"Base $b$ solid, $1/b$ dashed: all through (1, 0)", "x", "y", legend_loc="lower right",
           legend_size=8.5)

    panel(fig.add_subplot(gs[0, 2]), "Two readings of one table", [
        ("", r"*$y = b^x$", r"*$y = \log_b x$"),
        ("domain", r"all real $x$", r"$x > 0$"),
        ("range", r"$y > 0$", r"all real $y$"),
        ("landmarks", r"$(0, 1),\ (1, b)$", r"$(1, 0),\ (b, 1)$"),
        ("asymptote", r"$y = 0$", r"$x = 0$"),
        "",
        r"Why $x > 0$: $b^{\,y} > 0$ for every real $y$,",
        r"so $2^{\,?} = -4$ has no answer: $\log_2(-4)$ is undefined.",
        "",
        r"Base $1/b$:  $\log_{1/b} x = -\log_b x$,  $(1/b)^x = b^{-x}$.",
        r"Base $1$ is banned: $1^x = 1$ is not one-to-one.",
    ], fontsize=12, spacing=0.083)

    # transformations
    ax = fig.add_subplot(gs[1, 0])
    xb = np.linspace(0.005, 12, 600)
    xsft = np.linspace(-3 + 0.005, 12, 600)
    ax.plot(xb, np.log10(xb), color=MUTED, linewidth=1.8, label=r"$\log_{10} x$")
    ax.plot(xsft, np.log10(xsft + 3), color=BLUE_LIGHT, linewidth=1.8, linestyle="--",
            label=r"$\log_{10}(x+3)$: shift left 3")
    ax.plot(xsft, 2 * np.log10(xsft + 3) - 1, color=BLUE, linewidth=2.6,
            label=r"$2\log_{10}(x+3) - 1$")
    ax.axvline(-3, color=ORANGE, linestyle=":", linewidth=2, label="asymptote x = −3")
    for (px, py), text, off in [((-2, -1), "(−2, −1)", (8, -14)), ((7, 1), "(7, 1)", (-10, 8)),
                                ((x_intercept, 0), f"x = √10 − 3 ≈ {x_intercept:.3f}", (8, -16))]:
        ax.plot(px, py, "o", color=INK, markersize=6, zorder=5)
        ax.annotate(text, (px, py), xytext=off, textcoords="offset points", fontsize=9.5, color=INK2)
    axes_cross(ax)
    ax.set_xlim(-4, 12)
    ax.set_ylim(-4, 3)
    finish(ax, "Only the horizontal shift moves the asymptote", "x", "y", legend_loc="lower right",
           legend_size=9)

    # slowness
    ax = fig.add_subplot(gs[1, 1])
    exps = np.linspace(0, 100, 500)
    big = 10.0 ** exps
    ax.plot(big, np.log2(big), color=AQUA, linewidth=2.2, label=r"$\log_2 x$")
    ax.plot(big, np.log(big), color=ORANGE, linewidth=2.2, label=r"$\ln x$")
    ax.plot(big, np.log10(big), color=BLUE, linewidth=2.4, label=r"$\log_{10} x$")
    for value, color, text in [(log2_googol, AQUA, f"{log2_googol:.2f}"),
                               (ln_googol, ORANGE, f"{ln_googol:.2f}"), (100, BLUE, "100")]:
        ax.plot(1e100, value, "o", color=color, markersize=6, zorder=5, clip_on=False)
        ax.annotate(text, (1e100, value), xytext=(-8, 6), textcoords="offset points", ha="right",
                    fontsize=10, color=color)
    ax.set_xscale("log")
    ax.set_xlim(1, 1e100)
    ax.set_ylim(0, 360)
    ax.set_xticks([10.0 ** k for k in range(0, 101, 20)])
    finish(ax, r"Out to a googol $10^{100}$: the logs reach 100 to 332", "x (log scale)",
           "y (linear scale)", legend_loc="upper left")

    panel(fig.add_subplot(gs[1, 2]), "How slowly it grows", [
        r"Raise $\log_{10} x$ by 1  $\Leftrightarrow$  multiply $x$ by 10.",
        r"Equal steps up cost equal ratios across.",
        "",
        r"$\log_{10} 10^{100} = 100$,  $\ln 10^{100} = 230.26$,",
        r"$\log_2 10^{100} = 332.19$.",
        "",
        r"$\ln x = 100$ needs $x = e^{100} \approx 2.688 \times 10^{43}$.",
        "",
        r"Unbounded ($\log_b x = M$ at $x = b^M$), yet the",
        r"slope $\frac{d}{dx}\ln x = \frac{1}{x}$ is one millionth at $x = 10^6$.",
    ], fontsize=12, spacing=0.083, cols=(0.0, 0.3, 0.68))
    save02(fig, name="01_graphs_of_exp_and_log.png")


# ================================================================ 02/02
def sheet_02_02_solving():
    t_minutes = 20 * math.log(1e6 / 1000) / math.log(2)
    n_years = math.log(2) / math.log(1.05)
    x_cross = math.log(3) / (math.log(3) - math.log(2))
    assert f"{t_minutes:.1f}" == "199.3" and f"{n_years:.2f}" == "14.21"
    assert f"{1.05 ** 14:.4f}" == "1.9799" and f"{1.05 ** 15:.4f}" == "2.0789"
    assert f"{x_cross:.4f}" == "2.7095" and f"{2 ** x_cross:.4f}" == "6.5410"

    def hidden_quadratic(a, b, c, base):
        kept, dropped = [], []
        for u in np.roots([a, b, c]):
            (kept if u.real > 0 else dropped).append(u.real)
        return sorted(math.log(u) / math.log(base) for u in kept), dropped

    sol1, drop1 = hidden_quadratic(1, -4, 3, 3)
    sol2, drop2 = hidden_quadratic(1, -2, -8, 2)
    assert np.allclose(sol1, [0, 1]) and not drop1
    assert np.allclose(sol2, [2]) and np.allclose(drop2, [-2])

    def f(x):
        return x * np.exp(x) - 5

    def fp(x):
        return (x + 1) * np.exp(x)

    reference = 1.3267246652422002
    assert close(float(sp.LambertW(5).evalf(20)), reference, 1e-16)
    bis_mid, bis_err = [], []
    low, high = 1.0, 2.0
    for _ in range(40):
        middle = (low + high) / 2
        bis_mid.append(middle)
        bis_err.append(abs(middle - reference))
        if f(low) * f(middle) <= 0:
            high = middle
        else:
            low = middle
    assert [f"{e:.1e}" for e in bis_err[:10]] == ["1.7e-01", "7.7e-02", "4.8e-02", "1.4e-02", "1.7e-02",
                                                  "1.4e-03", "6.4e-03", "2.5e-03", "5.5e-04", "4.2e-04"]
    newton, newton_err = [1.0], [abs(1.0 - reference)]
    for _ in range(5):
        newton.append(newton[-1] - f(newton[-1]) / fp(newton[-1]))
        newton_err.append(abs(newton[-1] - reference))
    assert [f"{x:.15f}" for x in newton[1:5]] == ["1.419698602928606", "1.332595108383704",
                                                  "1.326749220851704", "1.326724665673260"]
    assert [f"{e:.1e}" for e in newton_err[1:]] == ["9.3e-02", "5.9e-03", "2.5e-05", "4.3e-10", "0.0e+00"]
    xx_root = float(sp.nsolve(sp.Symbol("x") ** sp.Symbol("x") - 100, sp.Symbol("x"), 3.5))
    assert close(xx_root, 3.59728502354042, 1e-13)

    fig = new_figure("Taking a logarithm brings the unknown down from the exponent",
                     r"$b^{\,x} = c \;\Leftrightarrow\; x = \log_b c = \ln c / \ln b$  (only for $c > 0$).   "
                     r"Log equations: state the domain first and check every candidate.")
    gs = grid(fig)

    # bacteria on a log axis
    ax = fig.add_subplot(gs[0, 0])
    t = np.linspace(0, 240, 400)
    ax.plot(t, 1000 * 2 ** (t / 20), color=BLUE, linewidth=2.6, label=r"$N(t) = 1000 \cdot 2^{t/20}$")
    ax.axhline(1e6, color=ORANGE, linestyle="--", linewidth=1.8, label=r"target $10^6$")
    ax.axvline(t_minutes, color=MUTED, linestyle=":", linewidth=1.4)
    ax.plot(t_minutes, 1e6, "o", color=ORANGE, markersize=8, zorder=5)
    ax.annotate(rf"$t = 20\log_2 1000 = {t_minutes:.1f}$ min", (t_minutes, 1e6), xytext=(235, 4e3),
                textcoords="data", ha="right", fontsize=10.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_yscale("log")
    ax.set_xlim(0, 240)
    finish(ax, "“When?” = a line meets a curve (log axis: straight)", "t (minutes)",
           "bacteria (log scale)", legend_loc="upper left")

    # loan
    ax = fig.add_subplot(gs[0, 1])
    years = np.arange(0, 17)
    balance = 10000 * 1.05 ** years / 1000
    colors = [ORANGE if n == 15 else (BLUE if n == 14 else BLUE_LIGHT) for n in years]
    ax.bar(years, balance, width=0.7, color=colors, zorder=2)
    ax.axhline(20, color=INK, linestyle="--", linewidth=1.4, zorder=3, label="double the loan")
    ax.axvline(n_years, color=ORANGE, linestyle=":", linewidth=2, zorder=3,
               label=rf"$n = \ln 2/\ln 1.05 = {n_years:.2f}$")
    for n in (14, 15):
        ax.text(n, 1.0, rf"$1.05^{{{n}}} = {1.05 ** n:.4f}$", ha="center", va="bottom", rotation=90,
                fontsize=10, color="white")
    ax.set_ylim(0, 25)
    ax.set_xticks(range(0, 17, 2))
    finish(ax, r"Loan at 5%: $1.05^n = 2$", "years n", "balance (thousands)", legend_loc="upper left")

    panel(fig.add_subplot(gs[0, 2]), "Exponent form  →  log form", [
        r"$2^{t/20} = 1000$  →  $t = 20 \log_2 1000 = 199.3$",
        r"$1.05^n = 2$  →  $n = \ln 2 / \ln 1.05 = 14.21$",
        r"$2^x = 3^{x-1}$  →  $x\ln 2 = (x-1)\ln 3$,  $x = 2.7095$",
        "",
        r"Hidden quadratic, $u = 3^x > 0$:",
        r"$3^{2x} - 4\cdot 3^x + 3 = 0$  →  $u \in \{1, 3\}$,  $x \in \{0, 1\}$",
        r"$4^x - 2^{x+1} - 8 = 0$  →  $u \in \{4, -2\}$:",
        r"$2^x = -2$ is impossible, so only $x = 2$.",
        "",
        r"Isolate the power first:  $\ln(A + B) \neq \ln A + \ln B$.",
    ], fontsize=12, spacing=0.083)

    # extraneous root
    ax = fig.add_subplot(gs[1, 0])
    xr = np.linspace(2.001, 7, 400)
    xlft = np.linspace(-5, -0.001, 400)
    ax.axvspan(0, 2, color=GRID, alpha=0.7, zorder=0)
    ax.plot(xlft, np.log2(xlft * (xlft - 2)), color=ORANGE, linewidth=2, linestyle="--",
            label=r"combined $\log_2(x(x-2))$: extra branch")
    ax.plot(xr, np.log2(xr * (xr - 2)), color=BLUE, linewidth=2.6,
            label=r"original $\log_2 x + \log_2(x-2)$, $x > 2$")
    ax.axhline(3, color=MUTED, linestyle=":", linewidth=1.4)
    ax.plot(4, 3, "o", color=AQUA, markersize=9, zorder=5)
    ax.plot(-2, 3, "X", color=ORANGE, markersize=11, zorder=5)
    ax.annotate("x = 4 ✓", (4, 3), xytext=(8, -18), textcoords="offset points", fontsize=10.5, color=INK)
    ax.annotate("x = −2: extraneous", (-2, 3), xytext=(12, 10), textcoords="offset points",
                fontsize=10.5, color=ORANGE)
    ax.set_xlim(-5, 7)
    ax.set_ylim(-4, 6)
    finish(ax, r"$\log_2 x + \log_2(x-2) = 3$: combining adds a root", "x", "y",
           legend_loc="lower right", legend_size=9)

    # newton vs bisection
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(range(1, 41), bis_err, "o-", color=BLUE, markersize=4, linewidth=1.6,
            label="bisection on [1, 2]: halves the bracket")
    shown = [max(e, 1e-17) for e in newton_err]
    ax.plot(range(0, 6), shown, "s-", color=ORANGE, markersize=6, linewidth=2.2,
            label=r"Newton from $x_0 = 1$: digits double")
    ax.annotate("exact to the last bit\n(drawn at $10^{-17}$)", (5, 1e-17), xytext=(10, 8),
                textcoords="offset points", fontsize=9.5, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(3e-18, 3)
    ax.set_xlim(-1, 41)
    finish(ax, rf"No algebra for $x e^x = 5$: root $W(5) = {reference:.5f}$", "step",
           "absolute error (log scale)", legend_loc="upper right", legend_size=9)

    panel(fig.add_subplot(gs[1, 2]), "The two traps, and the way out", [
        r"Combining logs can ADD roots:",
        r"$\log_b x + \log_b y = \log_b(xy)$ needs $x, y > 0$,",
        r"but $xy > 0$ also holds when both are negative.",
        "",
        r"Splitting logs can LOSE roots:",
        r"$\ln(x^2) = \ln 9$ has $x = \pm 3$;  $2\ln x$ finds only $3$.",
        r"Correct: $\ln(x^2) = 2\ln|x|$.",
        "",
        r"Mixed in and out of the exponent: graph, then",
        r"bisect or use Newton $x_{k+1} = x_k - f(x_k)/f'(x_k)$:",
        rf"$x e^x = 5$ → {reference:.5f},   $x^x = 100$ → {xx_root:.5f}.",
    ], fontsize=12, spacing=0.081)
    save02(fig, name="02_solving_exponential_and_log_equations.png")


# ================================================================ 02/03
def sheet_02_03_log_scales():
    # 4.1 positions in a decade
    positions = [math.log10(v) for v in range(1, 11)]
    assert [f"{p:.3f}" for p in positions] == ["0.000", "0.301", "0.477", "0.602", "0.699", "0.778",
                                               "0.845", "0.903", "0.954", "1.000"]
    # 4.2 Kepler
    planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    a = np.array([0.387, 0.723, 1.000, 1.524, 5.203, 9.537, 19.19, 30.07])
    T = np.array([0.2408, 0.6152, 1.000, 1.8808, 11.862, 29.457, 84.01, 164.8])
    k_fit, c_fit = np.polyfit(np.log10(a), np.log10(T), 1)
    assert f"{k_fit:.4f}" == "1.4998" and f"{c_fit:.5f}" == "0.00005" and f"{10 ** c_fit:.4f}" == "1.0001"
    assert [f"{v:.4f}" for v in T ** 2 / a ** 3] == ["1.0004", "1.0014", "1.0000", "0.9994", "0.9990",
                                                     "1.0003", "0.9987", "0.9989"]
    # 4.3 / 4.4 synthetic Moore's law
    rng = np.random.default_rng(seed=1971)
    years = np.arange(1971, 2022, 2)
    transistors = 2300 * 2 ** ((years - 1971) / 2.0) * np.exp(rng.normal(0.0, 0.3, size=years.size))
    slope, intercept = np.polyfit(years, np.log10(transistors), 1)
    doubling = np.log10(2) / slope
    assert years.size == 26 and f"{transistors[0]:.3g}" == "2.95e+03" and f"{transistors[-1]:.3g}" == "8.94e+10"
    assert f"{slope:.4f}" == "0.1501" and f"{10 ** slope:.3f}" == "1.413" and f"{doubling:.2f}" == "2.01"
    share = transistors ** 2 / np.sum(transistors ** 2)
    assert f"{share[-3:].sum():.1%}" == "98.7%" and f"{share[:20].sum():.2e}" == "1.67e-04"
    rel = np.abs(10 ** (intercept + slope * years) / transistors - 1)
    assert f"{np.median(rel):.1%}" == "13.8%" and f"{rel.max():.1%}" == "57.2%"
    # Example 1
    k1 = math.log10(300_000 / 3) / math.log10(1000 / 10)
    c1 = 3 / 10 ** k1
    assert f"{k1:.3f}" == "2.500" and f"{c1:.6f}" == "0.009487"
    # crossings of 2^x and x^3 (section 5.2 text)
    g = lambda x: x * math.log(2) - 3 * math.log(x)
    cross_lo, cross_hi = bisect(g, 1.0, 2.0), bisect(g, 5.0, 15.0)
    assert round(cross_lo, 2) == 1.37 and round(cross_hi, 2) == 9.94

    fig = new_figure("A log axis turns exponentials and power laws into straight lines",
                     r"Semi-log: $y = A\,b^x \Leftrightarrow \log_{10} y = \log_{10} A + x\log_{10} b$.   "
                     r"Log-log: $y = C x^k \Leftrightarrow \log_{10} y = \log_{10} C + k\log_{10} x$, "
                     r"slope = the exponent $k$.")
    gs = grid(fig)

    # log-spaced number line
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 2], [1, 1], color=INK, linewidth=1.4)
    for v in list(range(1, 10)) + list(range(10, 101, 10)):
        p = math.log10(v)
        major = v in (1, 10, 100)
        ax.plot([p, p], [1, 1.13 if major else 1.08], color=INK, linewidth=1.4 if major else 0.9)
        if v in (1, 2, 3, 5, 10, 20, 30, 50, 100):
            ax.text(p, 1.19, str(v), ha="center", va="bottom", fontsize=10, color=INK)
    for d in (0, 1):
        ax.annotate("", xy=(d + 1, 0.86), xytext=(d, 0.86),
                    arrowprops=dict(arrowstyle="<->", color=BLUE, linewidth=1.4))
        ax.text(d + 0.5, 0.79, "one decade = ×10", ha="center", va="top", fontsize=9.5, color=BLUE)
    doubles = [1, 2, 4, 8, 16, 32, 64]
    for lo, hi in zip(doubles[:-1], doubles[1:]):
        x0, x1 = math.log10(lo), math.log10(hi)
        ax.annotate("", xy=(x1, 0.48), xytext=(x0, 0.48),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=1.6, mutation_scale=12,
                                    connectionstyle="arc3,rad=-0.5", shrinkA=0, shrinkB=0))
    for v in doubles:
        ax.plot(math.log10(v), 0.48, "o", color=ORANGE, markersize=5)
        ax.text(math.log10(v), 0.40, str(v), ha="center", va="top", fontsize=9.5, color=ORANGE)
    ax.text(1.0, 0.18, r"×2 is always the same step: $\log_{10} 2 = 0.301$ of a decade",
            ha="center", fontsize=10, color=INK2)
    ax.text(1.0, 0.04, r"value $y$ sits at position $\log_{10} y$;  0 is at $-\infty$",
            ha="center", fontsize=10, color=INK2)
    ax.set_xlim(-0.08, 2.08)
    ax.set_ylim(-0.02, 1.4)
    ax.axis("off")
    ax.set_title("A log axis: equal ratios, equal distances")

    # Kepler
    ax = fig.add_subplot(gs[0, 1])
    a_line = np.logspace(np.log10(0.25), np.log10(50), 100)
    ax.plot(a_line, 10 ** c_fit * a_line ** k_fit, "--", color=ORANGE, linewidth=1.8,
            label=rf"fit: $T = {10 ** c_fit:.4f}\,a^{{{k_fit:.4f}}}$")
    ax.plot(a, T, "o", color=BLUE, markersize=8, zorder=4, label="8 planets")
    for name, x, y in zip(planets, a, T):
        ax.annotate(name, (x, y), xytext=(7, -11), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.25, 60)
    ax.set_ylim(0.12, 600)
    finish(ax, r"Kepler on log-log axes: slope 1.5, $T^2 = a^3$", "distance a (AU, log scale)",
           "period T (years, log scale)", legend_loc="upper left")

    # Moore
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(years, transistors, "o", color=BLUE, markersize=6, zorder=4, label="synthetic data (seed 1971)")
    ax.plot(years, 10 ** (intercept + slope * years), color=ORANGE, linewidth=2,
            label=f"fit: slope {slope:.4f} decades/yr → doubling {doubling:.2f} yr")
    ax.set_yscale("log")
    finish(ax, r"Semi-log: an exponential is a straight line", "year",
           "transistors (log scale)", legend_loc="upper left", legend_size=9, grid_which="major")

    # diagnostic: semi-log and log-log
    xs = np.linspace(1, 20, 300)
    for col, (xscale, title) in enumerate([("linear", r"Semi-log ($y$ log): $2^x$ straight, $x^3$ bends over"),
                                           ("log", r"Log-log: $x^3$ straight, $2^x$ bends up")]):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(xs, 2.0 ** xs, color=AQUA, linewidth=2.6, label=r"exponential $2^x$")
        ax.plot(xs, xs ** 3, color="#1c5cab", linewidth=2.6, label=r"power law $x^3$")
        for c in (cross_lo, cross_hi):
            ax.plot(c, c ** 3, "o", color=ORANGE, markersize=7, zorder=5)
        ax.annotate(f"crossings x ≈ {cross_lo:.2f} and {cross_hi:.2f}", (cross_hi, cross_hi ** 3),
                    xytext=(-20, 40), ha="right", textcoords="offset points", fontsize=9.5, color=ORANGE,
                    arrowprops=dict(arrowstyle="->", color=ORANGE))
        ax.set_xscale(xscale)
        ax.set_yscale("log")
        finish(ax, title, f"x ({xscale} scale)", "y (log scale)", legend_loc="upper left")

    panel(fig.add_subplot(gs[1, 2]), "Reading the slope", [
        r"Semi-log slope $m$ (decades per unit of $x$):",
        r"factor $b = 10^{m}$,  doubling time $\log_{10} 2 / m$.",
        r"Log-log slope = exponent, in any base:",
        r"$k = \frac{\log_{10}(y_2/y_1)}{\log_{10}(x_2/x_1)}$, e.g. $(10, 3)$ to $(1000, 3\cdot 10^5)$:",
        rf"$k = 5/2 = {k1:.1f}$,  $C = 3/10^{{2.5}} = {c1:.6f}$.",
        "",
        r"Fit logs (np.polyfit on $\log_{10} y$): every point's",
        r"relative error counts the same. Raw least squares:",
        rf"the last 3 of 26 points carry {share[-3:].sum():.1%} of the weight.",
    ], fontsize=12, spacing=0.083)
    save02(fig, name="03_log_scales_and_power_laws.png")


# ================================================================ 02/04
def sheet_02_04_growth():
    # 4.1 the operation-count table for n = 10^6
    n6 = 1_000_000
    assert math.floor(n6 * math.log10(2)) + 1 == 301_030
    assert f"{n6 * math.log2(n6) / 1e9 * 1e3:.1f}" == "19.9"
    # 4.3 crossovers
    t1 = bisect(lambda t: t - math.exp(0.1 * t), 1, 2)
    t2 = bisect(lambda t: t - math.exp(0.1 * t), 20, 60)
    assert f"{t1:.4f}" == "1.1183" and f"{t2:.4f}" == "35.7715"
    assert f"{math.exp(t1):.4g}" == "3.06" and f"{math.exp(t2):.4g}" == "3.431e+15"
    t3 = bisect(lambda t: t - math.exp(0.01 * t), 100, 2000)
    assert f"{t3 / math.log(10):.1f}" == "281.1"
    n_cross = bisect(lambda n: 100 * math.log(n) - n * math.log(1.01), 1e3, 1e7)
    assert f"{n_cross:,.0f}" == "117,308" and f"{100 * math.log10(n_cross):.1f}" == "506.9"
    n_ab = bisect(lambda n: n - 100 * math.log2(n), 100, 1e4)
    assert f"{n_ab:.1f}" == "996.0"
    # 4.4 maxima of ln n / n^eps
    eps_list = [0.5, 1 / math.e, 0.3, 0.1, 0.01]
    maxima = [1 / (e * math.e) for e in eps_list]
    assert [f"{m:.4f}" for m in maxima] == ["0.7358", "1.0000", "1.2263", "3.6788", "36.7879"]
    # 4.5 comparisons counted on random lists (random.seed(4), one list per n in this order)

    def insertion(values):
        items = list(values)
        count = 0
        for i in range(1, len(items)):
            current = items[i]
            j = i - 1
            while j >= 0:
                count += 1
                if items[j] <= current:
                    break
                items[j + 1] = items[j]
                j -= 1
            items[j + 1] = current
        return count

    def merge(values):
        if len(values) <= 1:
            return list(values), 0
        mid = len(values) // 2
        left, cl = merge(values[:mid])
        right, cr = merge(values[mid:])
        out, count = [], cl + cr
        i = j = 0
        while i < len(left) and j < len(right):
            count += 1
            if left[i] <= right[j]:
                out.append(left[i])
                i += 1
            else:
                out.append(right[j])
                j += 1
        out.extend(left[i:])
        out.extend(right[j:])
        return out, count

    random.seed(4)
    sizes = [100, 200, 400, 800, 1600, 3200]
    ins_counts, mer_counts = [], []
    for n in sizes:
        data = [random.random() for _ in range(n)]
        ins_counts.append(insertion(data))
        mer_counts.append(merge(data)[1])
    assert ins_counts == [2597, 9804, 42608, 157666, 652759, 2532961]
    assert mer_counts == [538, 1292, 2974, 6713, 15039, 33265]

    fig = new_figure(r"Growth rates: $\ln n \ll n^{\varepsilon} \ll n \ll n\ln n \ll n^2 \ll 2^n$",
                     r"$f \ll g \Leftrightarrow f/g \to 0$;  compare logs: $f \ll g$ when $\ln f - \ln g \to -\infty$.   "
                     r"$\max\, \ln n / n^{\varepsilon} = 1/(\varepsilon e)$ at $n = e^{1/\varepsilon}$: "
                     r"“eventually” can be very late.")
    gs = grid(fig)

    # the race
    ax = fig.add_subplot(gs[0, 0])
    n = np.logspace(0, 6, 400)
    curves = [(r"$\ln n$", np.log(n), AQUA), (r"$\sqrt{n}$", np.sqrt(n), BLUE_LIGHT), (r"$n$", n, BLUE),
              (r"$n \ln n$", n * np.log(n), "#1c5cab"), (r"$n^2$", n ** 2, INK),
              (r"$2^n$", 2.0 ** np.minimum(n, 100), ORANGE)]
    for label, values, color in curves:
        ax.plot(n, np.where(values > 0, values, np.nan), color=color, linewidth=2.2)
        if label == r"$2^n$":
            ax.text(55, 3e13, label, fontsize=12, color=color)
        else:
            below = label == r"$n$"
            ax.text(7e5, values[-1] / 2.5 if below else values[-1] * 2.2, label, ha="right",
                    va="top" if below else "bottom", fontsize=12, color=color)
    ax.axhline(1e12, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(1.5, 2.5e12, r"$10^{12}$ operations ≈ 17 min at $10^9$/s", fontsize=9.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e6)
    ax.set_ylim(1e-1, 1e15)
    finish(ax, "The race on log-log axes", "n (log scale)", "operations (log scale)", legend_loc=None)

    # ln n vs n^0.1 ratio
    ax = fig.add_subplot(gs[0, 1])
    n = np.logspace(0, 18, 600)
    ratio = np.log(n) / n ** 0.1
    ax.fill_between(n, 1, ratio, where=ratio > 1, color=ORANGE_LIGHT, alpha=0.7, label=r"$\ln n$ ahead")
    ax.plot(n, ratio, color=BLUE, linewidth=2.4, label=r"$\ln n \,/\, n^{0.1}$")
    ax.axhline(1, color=INK, linewidth=0.9)
    ax.plot(np.exp(10), maxima[3], "o", color=ORANGE, markersize=7, zorder=5)
    ax.annotate(rf"max $1/(0.1e) = {maxima[3]:.2f}$ at $n = e^{{10}}$", (np.exp(10), maxima[3]),
                xytext=(-10, 10), textcoords="offset points", fontsize=10, color=INK)
    for c, text, off, ha in [(math.exp(t1), f"n ≈ {math.exp(t1):.2f}", (6, -30), "left"),
                             (math.exp(t2), rf"n ≈ ${math.exp(t2) / 1e15:.2f} \times 10^{{15}}$", (-6, -26),
                              "right")]:
        ax.axvline(c, color=MUTED, linestyle=":", linewidth=1.3)
        ax.annotate(text, (c, 1), xytext=off, textcoords="offset points", ha=ha, fontsize=9.5,
                    color=INK2)
    ax.set_xscale("log")
    ax.set_ylim(0, 4.7)
    ax.set_xlim(1, 1e18)
    finish(ax, r"$\ln n \ll n^{0.1}$, yet $\ln n$ leads until $3.4\times 10^{15}$", "n (log scale)",
           "ratio", legend_loc="center right", legend_size=9.5)

    # n^100 vs 1.01^n
    ax = fig.add_subplot(gs[0, 2])
    n = np.linspace(1, 200_000, 2000)
    ax.plot(n / 1000, 100 * np.log10(n), color=BLUE, linewidth=2.4,
            label=r"$\log_{10}(n^{100}) = 100\log_{10} n$")
    ax.plot(n / 1000, n * np.log10(1.01), color=ORANGE, linewidth=2.4,
            label=r"$\log_{10}(1.01^n) = n\log_{10} 1.01$")
    ax.plot(n_cross / 1000, 100 * math.log10(n_cross), "o", color=INK, markersize=7, zorder=5)
    ax.annotate(f"n = {n_cross:,.0f}:\nboth ≈ $10^{{{100 * math.log10(n_cross):.1f}}}$",
                (n_cross / 1000, 100 * math.log10(n_cross)), xytext=(14, -40), textcoords="offset points",
                fontsize=10, color=INK)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 900)
    finish(ax, r"$n^{100}$ vs $1.01^n$, through their logs", "n (thousands)", r"$\log_{10}$ of the value",
           legend_loc="upper left", legend_size=9.5)

    # sorting counts
    ax = fig.add_subplot(gs[1, 0])
    sz = np.array(sizes, float)
    ax.plot(sz, sz ** 2 / 4, color=BLUE_LIGHT, linewidth=6, alpha=0.8, label=r"$n^2/4$")
    ax.plot(sz, ins_counts, "o-", color=BLUE, linewidth=2, markersize=6, label="insertion sort (counted)")
    ax.plot(sz, sz * np.log2(sz), color=ORANGE_LIGHT, linewidth=6, alpha=0.9, label=r"$n\log_2 n$")
    ax.plot(sz, mer_counts, "s-", color=ORANGE, linewidth=2, markersize=6, label="merge sort (counted)")
    ax.annotate(f"{ins_counts[-1]:,}", (3200, ins_counts[-1]), xytext=(-6, 6), textcoords="offset points",
                ha="right", fontsize=9.5, color=BLUE)
    ax.annotate(f"{mer_counts[-1]:,}", (3200, mer_counts[-1]), xytext=(-6, 8), textcoords="offset points",
                ha="right", fontsize=9.5, color=ORANGE)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes, [str(s) for s in sizes])
    finish(ax, "Comparisons on random lists: slope 2 vs slope ≈ 1", "list length n (log scale)",
           "comparisons (log scale)", legend_loc="upper left", legend_size=9, grid_which="major")

    # max of ln n / n^eps vs eps
    ax = fig.add_subplot(gs[1, 1])
    eps = np.geomspace(0.005, 1, 300)
    ax.plot(eps, 1 / (eps * np.e), color=BLUE, linewidth=2.4, label=r"largest $\ln n / n^{\varepsilon} = 1/(\varepsilon e)$")
    ax.axhline(1, color=INK, linewidth=0.9)
    ax.axvline(1 / math.e, color=ORANGE, linestyle=":", linewidth=2, label=r"threshold $\varepsilon = 1/e \approx 0.368$")
    offsets = [((-10, -16), "right"), ((10, 4), "left"), ((-10, -16), "right"), ((8, 4), "left"),
               ((8, 4), "left")]
    names = ["0.5", "1/e", "0.3", "0.1", "0.01"]
    for e_, m_, (off, ha), name in zip(eps_list, maxima, offsets, names):
        ax.plot(e_, m_, "o", color=ORANGE if m_ > 1 + 1e-12 else AQUA, markersize=7, zorder=5)
        ax.annotate(f"ε = {name}: {m_:.4g}", (e_, m_), xytext=off, ha=ha, textcoords="offset points",
                    fontsize=9, color=INK2)
    ax.text(0.007, 0.45, r"$\ln n < n^{\varepsilon}$ for every $n$", fontsize=9.5, color=AQUA)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 150)
    finish(ax, r"Does $\ln n$ ever beat $n^{\varepsilon}$?", r"$\varepsilon$ (log scale)",
           "largest ratio (log scale)", legend_loc="upper right", legend_size=9)

    panel(fig.add_subplot(gs[1, 2]), "n = 10⁶ records at 10⁹ operations/s", [
        (r"$\log_2 n$", "binary search", "20 ns"),
        (r"$n$", "one pass", "1 ms"),
        (r"$n\log_2 n$", "merge sort", "19.9 ms"),
        (r"$n^2$", "all pairs", "1000 s ≈ 17 min"),
        (r"$2^n$", "all subsets", "301,030 digits"),
        "",
        r"Base of a log: a constant factor, $\log_2 n = \ln n/\ln 2$.",
        r"Base of an exponential: decisive, $2^n/3^n \to 0$.",
        "",
        rf"Crossovers: $\ln n = n^{{0.01}}$ at $n = 10^{{{t3 / math.log(10):.1f}}}$;",
        rf"$100\,n\log_2 n = n^2$ at $n = {n_ab:.0f}$ (constants matter).",
    ], fontsize=12, spacing=0.081, cols=(0.0, 0.2, 0.55))
    save02(fig, name="04_growth_rates_and_asymptotics.png")


# ================================================================ 03/01
def sheet_03_01_briggs():
    getcontext().prec = 60
    roots_ten = {c: repeated_square_roots(10, c) for c in [1, 2, 3, 4, 10, 20, 54]}
    assert str(roots_ten[4])[:42] == "1.1547819846894581796664828872955082815669"
    assert str(roots_ten[10])[:42] == "1.0022511482929129154656736388665711924542"
    assert str(roots_ten[54])[:42] == "1.0000000000000001278191493200323453724568"
    eps_ten = roots_ten[54] - 1
    k_briggs = 1 / (Decimal(2) ** 54 * eps_ten)
    k_exact = Decimal(1) / Decimal(10).ln()
    assert f"{eps_ten:.10e}" == "1.2781914932e-16"
    assert str(k_briggs)[:22] == "0.43429448190325179989" and str(k_exact)[:22] == "0.43429448190325182765"
    assert f"{abs(k_briggs / k_exact - 1):.2e}" == "6.39e-17" and f"{eps_ten / 2:.2e}" == "6.39e-17"
    eps_two = repeated_square_roots(2, 54) - 1
    log2_briggs = eps_two / eps_ten
    assert str(log2_briggs)[:22] == "0.30102999566398118176"
    assert f"{abs(log2_briggs - Decimal(2).log10()):.1e}" == "1.3e-17"
    # Example 3 (ten roots)
    e2_10, e10_10 = repeated_square_roots(2, 10) - 1, repeated_square_roots(10, 10) - 1
    assert f"{float(e2_10):.9f}" == "0.000677131"  # notebook Example 3 prints 0.000677130 (rounding slip)
    assert f"{float(e2_10 / e10_10):.4f}" == "0.3008"
    # 4.4 the table from three primes
    briggs = lambda x: (repeated_square_roots(x, 54) - 1) / (repeated_square_roots(10, 54) - 1)
    l2, l3, l7 = briggs(2), briggs(3), briggs(7)
    table = {2: l2, 3: l3, 4: 2 * l2, 5: 1 - l2, 6: l2 + l3, 7: l7, 8: 3 * l2, 9: 2 * l3}
    assert all(math.isclose(float(v), math.log10(x), abs_tol=1e-15) for x, v in table.items())
    # 5.1 / 5.3 curves
    counts = np.arange(0, 41)
    eps10_curve, eps2_curve = [], []
    v10, v2 = Decimal(10), Decimal(2)
    for c in counts:
        if c > 0:
            v10, v2 = v10.sqrt(), v2.sqrt()
        eps10_curve.append(float(v10 - 1))
        eps2_curve.append(float(v2 - 1))
    eps10_curve, eps2_curve = np.array(eps10_curve), np.array(eps2_curve)
    k_curve = 2.0 ** (-counts) / eps10_curve
    true_value = Decimal(2).log10()
    err_counts = np.arange(1, 61)
    dec_err, flt_err = [], []
    d2, d10, f2, f10 = Decimal(2), Decimal(10), 2.0, 10.0
    for _ in err_counts:
        d2, d10 = d2.sqrt(), d10.sqrt()
        f2, f10 = math.sqrt(f2), math.sqrt(f10)
        dec_err.append(float(abs((d2 - 1) / (d10 - 1) - true_value)))
        flt_err.append(abs((f2 - 1) / (f10 - 1) - float(true_value)) if f10 > 1.0 else np.nan)
    best_float = int(err_counts[np.nanargmin(flt_err)])
    assert best_float == 27 and f"{np.nanmin(flt_err):.1e}" == "5.4e-10" and f"{dec_err[53]:.1e}" == "1.3e-17"
    # Napier
    nap = lambda x: 10 ** 7 * math.log(10 ** 7 / x)
    assert round(nap(10 ** 7 / 2), 1) == 6931471.8

    fig = new_figure("Briggs: square-root until the logarithm is a straight line, then divide",
                     r"$\log_{10} x^{1/2^n} = \log_{10} x \,/\, 2^n$;   near 1, $\log_{10}(1+\varepsilon) \approx k\,\varepsilon$ "
                     r"with $k = 1/\ln 10$;   so after the same 54 roots  $\log_{10} 2 \approx \varepsilon_2 / \varepsilon_{10}$.")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(counts, eps10_curve, "o-", color=BLUE, markersize=3.5, linewidth=1.6,
            label=r"$\varepsilon_{10} = 10^{1/2^n} - 1$")
    ax.plot(counts, eps2_curve, "s-", color=ORANGE, markersize=3.5, linewidth=1.6,
            label=r"$\varepsilon_{2} = 2^{1/2^n} - 1$")
    ax.plot(counts, math.log(10) / 2.0 ** counts, "--", color=MUTED, linewidth=1.2,
            label=r"$\ln 10 / 2^n$: halves per root")
    ax.set_yscale("log")
    finish(ax, "Each square root halves the distance to 1", "number of square roots n",
           "distance from 1 (log scale)", legend_loc="upper right", legend_size=9.5)

    ax = fig.add_subplot(gs[0, 1])
    xs = np.linspace(0.85, 3.4, 400)
    k = 1 / math.log(10)
    ax.plot(xs, np.log10(xs), color=BLUE, linewidth=2.6, label=r"$\log_{10} x$")
    ax.plot(xs, k * (xs - 1), "--", color=ORANGE, linewidth=2, label=r"tangent $k(x-1)$, $k = 0.4343$")
    for c in range(1, 6):
        x = float(repeated_square_roots(10, c))
        ax.plot(x, 1 / 2 ** c, "o", color=INK, markersize=6, zorder=5)
        if c <= 3:
            off, ha = ((-10, 6), "right") if c == 1 else ((8, -12), "left")
            ax.annotate(rf"$10^{{1/{2 ** c}}}$: log $= 1/{2 ** c}$", (x, 1 / 2 ** c), xytext=off, ha=ha,
                        textcoords="offset points", fontsize=9.5, color=INK2)
    ax.annotate(r"$10^{1/16}, 10^{1/32}$, ...: curve and line merge", (1.075, 1 / 32), xytext=(1.45, -0.05),
                textcoords="data", fontsize=9.5, color=INK2, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlim(0.85, 3.4)
    ax.set_ylim(-0.08, 0.7)
    finish(ax, "Roots of 10 walk down to where the log is straight", "x", "y",
           legend_loc="upper left", legend_size=9.5)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(counts, k_curve, "o-", color=BLUE, markersize=4, linewidth=1.6,
            label=r"$\log_{10}(1+\varepsilon_n)/\varepsilon_n = 2^{-n}/\varepsilon_n$")
    ax.axhline(k, color=ORANGE, linestyle="--", linewidth=1.6, label=r"$k = 1/\ln 10 = 0.43429\ldots$")
    ax.annotate(f"4 roots: {k_curve[4]:.3f}", (4, k_curve[4]), xytext=(10, -14), textcoords="offset points",
                fontsize=9.5, color=INK2)
    ax.set_ylim(0.1, 0.48)
    finish(ax, "Log ÷ distance settles on Briggs's constant k", "number of square roots n", "ratio",
           legend_loc="lower right", legend_size=9.5)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(err_counts, dec_err, "o-", color=BLUE, markersize=3.5, linewidth=1.6, label="60-digit decimals")
    ax.plot(err_counts, flt_err, "s-", color=ORANGE, markersize=3.5, linewidth=1.6, label="16-digit floats")
    ax.axvline(54, color=MUTED, linestyle=":", linewidth=1.4)
    ax.annotate(f"Briggs: 54 roots, error {dec_err[53]:.1e}", (54, dec_err[53]), xytext=(25, 2e-19),
                textcoords="data", fontsize=9.5, color=INK2, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate(f"floats: best after {best_float} roots, error {np.nanmin(flt_err):.1e};\n"
                "then 1 + ε cannot hold ε's digits",
                (best_float, np.nanmin(flt_err)), xytext=(8, 1e1), textcoords="data", fontsize=9.5,
                color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.set_yscale("log")
    ax.set_ylim(3e-20, 1e3)
    finish(ax, r"Error in $\log_{10} 2$: keep the digits after the zeros", "number of square roots n",
           "error (log scale)", legend_loc="lower left", legend_size=9.5)

    # Napier's moving points
    ax = fig.add_subplot(gs[1, 1])
    times = np.arange(0, 8) * 0.25
    remaining = np.exp(-times)
    q_pos = times
    ax.plot([0, 1], [1, 1], color=BLUE, linewidth=3)
    ax.plot([0, 1.95], [0, 0], color=ORANGE, linewidth=3)
    for x, y in zip(remaining, q_pos):
        ax.plot([1 - x, y], [1, 0], color=MUTED, linewidth=0.9, alpha=0.7)
        ax.plot(1 - x, 1, "o", color=BLUE, markersize=6, zorder=4)
        ax.plot(y, 0, "o", color=ORANGE, markersize=6, zorder=4)
        ax.text(y, -0.13, f"{y:.2f}", ha="center", va="top", fontsize=8.5, color=ORANGE)
    ax.text(1.0, 1.08, "B", fontsize=11, weight="bold", ha="center")
    ax.text(0.0, 1.27, r"P: distance left $x$ shrinks by the same RATIO each step", fontsize=10, color=BLUE)
    ax.text(0.0, -0.42, r"Q: distance covered $y$ grows by the same STEP", fontsize=10, color=ORANGE)
    ax.text(0.0, -0.58, r"NapLog$(x) = 10^7 \ln(10^7/x)$:  NapLog$(10^7/2) = 6{,}931{,}472$",
            fontsize=10, color=INK2)
    ax.set_xlim(-0.05, 2.0)
    ax.set_ylim(-0.68, 1.4)
    ax.axis("off")
    ax.set_title("Napier (1614): a geometric and an arithmetic motion")

    panel(fig.add_subplot(gs[1, 2]), "Briggs's numbers (60-digit replay)", [
        r"$10^{1/2^{54}} = 1.000\,000\,000\,000\,000\,1278\ldots$",
        rf"$\varepsilon_{{10}} = {float(eps_ten) * 1e16:.10f} \times 10^{{-16}}$",
        r"$k = \frac{1}{2^{54}\,\varepsilon_{10}} = 0.434294481903251\ldots$",
        r"$\log_{10} 2 \approx \varepsilon_2/\varepsilon_{10} = 0.3010299956639811\ldots$",
        r"error $1.3 \times 10^{-17}$: relative error $\approx \varepsilon/2$",
        "",
        r"Only primes need the mill:  $\log_{10} 4 = 2\log_{10} 2$,",
        r"$\log_{10} 5 = 1 - \log_{10} 2$,  $\log_{10} 6 = \log_{10} 2 + \log_{10} 3$,",
        r"$\log_{10} 8 = 3\log_{10} 2$,  $\log_{10} 9 = 2\log_{10} 3$.",
    ], fontsize=12, spacing=0.083)
    save03(fig, name="01_napier_and_briggs_tables.png")


# ================================================================ 03/02
def draw_scale(ax, name, y, offset=0.0, decades=1, ticks_up=True, color=INK, fill="#fbf7ea"):
    """One strip of a slide rule: value v sits at offset + log10(v) / decades (rule length 1)."""
    ax.add_patch(Rectangle((offset, y - 0.5), 1.0, 1.0, facecolor=fill, edgecolor=MUTED, linewidth=1))
    sign = 1 if ticks_up else -1
    base = y - 0.5 if ticks_up else y + 0.5
    for decade in range(decades):
        factor = 10 ** decade
        for whole in range(1, 10):
            step = 0.1 if whole < 5 else 0.5
            for fraction in np.arange(0, 1, step):
                value = (whole + fraction) * factor
                is_whole = np.isclose(fraction, 0)
                height = 0.45 if is_whole else (0.3 if np.isclose(fraction, 0.5) else 0.18)
                x = offset + math.log10(value) / decades
                ax.plot([x, x], [base, base + sign * height], color=color, linewidth=0.8)
                if is_whole:
                    ax.text(x, base + sign * (height + 0.06), f"{int(round(value))}", ha="center",
                            va="bottom" if ticks_up else "top", fontsize=8.5, color=color)
    x = offset + 1.0
    ax.plot([x, x], [base, base + sign * 0.45], color=color, linewidth=0.8)
    ax.text(x, base + sign * 0.51, f"{10 ** decades}", ha="center", va="bottom" if ticks_up else "top",
            fontsize=8.5, color=color)
    ax.text(offset - 0.025, y, name, ha="right", va="center", fontsize=12, weight="bold", color=color)


def sheet_03_02_slide_rule():
    L = 25.0
    d = lambda x: L * math.log10(x)

    def multiply(a, b):
        ma, ea = a / 10 ** math.floor(math.log10(a)), math.floor(math.log10(a))
        mb, eb = b / 10 ** math.floor(math.log10(b)), math.floor(math.log10(b))
        cursor, e = d(ma) + d(mb), ea + eb
        if cursor >= L:
            cursor, e = cursor - L, e + 1
        return 10 ** (cursor / L) * 10 ** e, cursor

    assert [f"{multiply(a, b)[1]:5.2f}" for a, b in [(2, 3), (4, 5), (3.7, 4.6), (37, 460)]] == \
        ["19.45", " 7.53", " 5.77", " 5.77"]
    assert f"{multiply(37, 460)[0]:.6g}" == "17020"
    assert f"{d(8) - d(2):5.2f}" == "15.05" and f"{d(3) - d(7) + L:5.2f}" == "15.80"
    assert f"{L / 2 * math.log10(50):.2f}" == "21.24" and f"{10 ** (L / 2 * math.log10(50) / L):.4f}" == "7.0711"
    # 4.3: three readings, seed 1
    rng = np.random.default_rng(seed=1)
    stds, reads = [], []
    for a, b in [(2, 3), (3.7, 4.6), (1.2, 1.3), (9.1, 8.7)]:
        errors = rng.normal(0, 0.01, size=(100_000, 3)).sum(axis=1)
        results = 10 ** ((L * (math.log10(a) + math.log10(b)) + errors) / L)
        stds.append(f"{(results / (a * b) - 1).std():.5f}")
        reads.append(float(f"{results[0]:.3g}"))
    assert stds == ["0.00159"] * 4 and reads == [6.01, 17.0, 1.56, 79.2]
    predicted = math.sqrt(3) * math.log(10) * 0.01 / L
    assert f"{predicted:.5f}" == "0.00160"
    # 5.4: log rule vs linear rule, seed 2 (25 values first, then the histogram samples)
    rng = np.random.default_rng(seed=2)
    true_values = np.geomspace(1.05, 9.5, 25)
    log_spread, lin_spread = [], []
    for value in true_values:
        pe = rng.normal(0, 0.01, size=20_000) * math.sqrt(3)
        log_spread.append(np.std(10 ** (math.log10(value) + pe / L) / value - 1))
        lin_spread.append(np.std((value + pe * 9 / L) / value - 1))
    samples = 10 ** (math.log10(3.7 * 4.6) + rng.normal(0, 0.01, size=(20_000, 3)).sum(axis=1) / L)
    assert f"{samples.std():.3f}" == "0.027" and f"{samples.std() / 17.02:.4f}" == "0.0016"
    lin_better = true_values[np.array(lin_spread) < np.array(log_spread)]
    print(f"03/02 linear rule beats the log rule for x >= {lin_better.min():.2f}")

    fig = new_figure("A slide rule multiplies by adding logarithmic lengths",
                     r"$x$ is printed at $d(x) = L\log_{10} x$:   $d(a) + d(b) = d(ab)$, i.e. "
                     r"$10^{\alpha}\cdot 10^{\beta} = 10^{\alpha+\beta}$.   A fixed reading error $\delta$ is a "
                     r"fixed relative error $\ln 10\,\delta/L$.")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0:2])
    shift = math.log10(2)
    draw_scale(ax, "C", y=1.0, offset=shift, ticks_up=False, color="#1c5cab", fill="#eef4fc")
    draw_scale(ax, "D", y=0.0, ticks_up=True)
    cursor = shift + math.log10(3)
    ax.plot([cursor, cursor], [-0.6, 1.6], color=ORANGE, linewidth=2)
    ax.text(cursor + 0.01, 1.62, "cursor: 3 on C over 6 on D", color=ORANGE, fontsize=11)
    for x0, x1, color, text in [(0, shift, AQUA, r"$L\log_{10} 2 = 7.53$ cm"),
                                (shift, cursor, BLUE, r"$L\log_{10} 3 = 11.93$ cm")]:
        ax.annotate("", xy=(x1, -0.78), xytext=(x0, -0.78),
                    arrowprops=dict(arrowstyle="<->", color=color, linewidth=1.6))
        ax.text((x0 + x1) / 2, -0.88, text, ha="center", va="top", fontsize=10.5, color=color)
    ax.text(cursor / 2, -1.22, r"sum $= 19.45$ cm $= L\log_{10} 6$", ha="center", va="top", fontsize=10.5,
            color=INK)
    ax.text(1.12, 0.0, "every pair is now in the ratio 2:\nC 1.5 over D 3, C 4 over D 8,\n"
            "so one setting is a whole ×2 table", fontsize=10, color=INK2, va="center")
    ax.set_xlim(-0.06, 1.45)
    ax.set_ylim(-1.45, 1.9)
    ax.axis("off")
    ax.set_title("2 × 3 on a 25 cm rule: slide C by d(2), read D under C's 3")

    panel(fig.add_subplot(gs[0, 2]), "The moves", [
        r"Off the end, $4 \times 5$: $d(4) + d(5) = 32.52 > 25$ cm.",
        r"Use C's right index: $32.52 - 25 = 7.53$ cm → “2”;",
        r"the estimate $\approx 20$ places the point: 20.",
        r"Divide $8/2$: $22.58 - 7.53 = 15.05$ cm → 4.",
        "",
        r"A scale: $d_A(x) = \frac{L}{2}\log_{10} x$ shows $y^2$ over $y$ on D;",
        r"$\sqrt{50}$: 50 on A at 21.24 cm, D reads 7.07.",
        "",
        r"Digits only: $37 \times 460$ is set like $3.7 \times 4.6$ → 17,020.",
    ], fontsize=12, spacing=0.085)

    # circular rule
    ax = fig.add_subplot(gs[1, 0])
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color=INK, linewidth=1.2))
    for value in np.concatenate([np.arange(1, 2, 0.1), np.arange(2, 5, 0.2), np.arange(5, 10, 0.5)]):
        angle = np.pi / 2 - 2 * np.pi * np.log10(value)
        whole = np.isclose(value, round(value))
        inner = 0.87 if whole else 0.94
        ax.plot([inner * np.cos(angle), np.cos(angle)], [inner * np.sin(angle), np.sin(angle)], color=INK,
                linewidth=0.8)
        if whole:
            ax.text(1.12 * np.cos(angle), 1.12 * np.sin(angle), f"{int(round(value))}", ha="center",
                    va="center", fontsize=11)
    for start, length, color, radius, text in [(0, math.log10(4), AQUA, 0.74, r"$\log_{10} 4$"),
                                               (math.log10(4), math.log10(5), BLUE, 0.6, r"$\log_{10} 5$")]:
        fr = np.linspace(start, start + length, 100)
        ang = np.pi / 2 - 2 * np.pi * fr
        ax.plot(radius * np.cos(ang), radius * np.sin(ang), color=color, linewidth=3.2)
        mid = np.pi / 2 - 2 * np.pi * (start + length * 0.72)
        ax.text((radius - 0.17) * np.cos(mid), (radius - 0.17) * np.sin(mid), text, ha="center", va="center",
                fontsize=10, color=color)
    end = np.pi / 2 - 2 * np.pi * (math.log10(4) + math.log10(5))
    ax.plot([0, 0.88 * np.cos(end)], [0, 0.88 * np.sin(end)], color=ORANGE, linewidth=2)
    ax.text(1.3, 0.35, "4 × 5 → digits 2\n(= 20)", ha="left", fontsize=10, color=ORANGE)
    ax.text(1.3, -0.15, "$1.30103\\ \\mathrm{mod}\\ 1$\n$= 0.30103$\n$= \\log_{10} 2$", ha="left", va="top",
            fontsize=10, color=INK2)
    ax.set_xlim(-1.25, 2.2)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Circular rule: angle = 2π log₁₀ x, no end to run off")

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(samples, bins=60, color=BLUE, alpha=0.85, zorder=2)
    ax.axvline(17.02, color=ORANGE, linewidth=2, zorder=3, label="exact 17.02")
    ax.axvline(17.02 - samples.std(), color=INK, linestyle=":", linewidth=1.2, zorder=3)
    ax.axvline(17.02 + samples.std(), color=INK, linestyle=":", linewidth=1.2, zorder=3,
               label=f"±1 std = ±{samples.std():.3f} ({samples.std() / 17.02:.2%})")
    finish(ax, "3.7 × 4.6 by 20,000 simulated users", "slide-rule result (3 readings, δ = 0.1 mm)",
           "count", legend_loc="upper left", legend_size=9.5)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(true_values, log_spread, "o-", color=BLUE, markersize=5, linewidth=2, label="logarithmic rule")
    ax.plot(true_values, lin_spread, "s-", color=ORANGE, markersize=5, linewidth=2,
            label="linear rule, same 25 cm")
    ax.axhline(predicted, color=MUTED, linestyle="--", linewidth=1.2,
               label=rf"$\sqrt{{3}}\,\ln 10\,\delta/L = {predicted:.4f}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    ax.set_yticks([5e-4, 1e-3, 2e-3, 5e-3], ["0.05%", "0.1%", "0.2%", "0.5%"])
    ax.minorticks_off()
    finish(ax, "Equal relative precision everywhere", "value being read (log scale)",
           "relative error, std (log scale)", legend_loc="upper right", legend_size=9.5)
    save03(fig, name="02_the_slide_rule.png")


# ================================================================ 03/03
SQRT_HALF = math.sqrt(0.5)


def reduce_range(x):
    m, k = math.frexp(x)
    if m < SQRT_HALF:
        m, k = 2 * m, k - 1
    return m, k


def sheet_03_03_computers():
    getcontext().prec = 40
    LN2 = float(Decimal(2).ln())

    def artanh_tail(s2, terms):
        total = 0.0
        for n in reversed(range(1, terms)):
            total = total * s2 + 1 / (2 * n + 1)
        return total

    def my_ln(x, terms=11):
        m, k = reduce_range(x)
        s = (m - 1) / (m + 1)
        s2 = s * s
        ln_m = 2 * s + 2 * s * s2 * artanh_tail(s2, terms)   # same order of operations as the notebook
        return k * LN2 + ln_m

    assert reduce_range(10) == (1.25, 3) and reduce_range(1e-300)[1] == -997 and reduce_range(5e-324) == (1.0, -1074)
    assert f"{my_ln(10): .16f}" == " 2.3025850929940455" and f"{my_ln(1e-300): .16f}" == "-690.7755278982136815"
    # Example 2
    terms3 = 2 * (1 / 9 + 1 / (3 * 9 ** 3) + 1 / (5 * 9 ** 5))
    assert abs(terms3 - math.log(1.25)) < 1e-7 and f"{math.log(1.25):.7f}" == "0.2231436"
    rng = np.random.default_rng(seed=0)
    pts = np.exp(rng.uniform(-50, 50, size=100_000))
    ulp = np.array([abs(my_ln(x) - math.log(x)) / np.spacing(abs(math.log(x))) for x in pts])
    assert ulp.max() == 2.0 and f"{np.mean(ulp == 0):.1%}" == "85.3%" and f"{np.mean(ulp <= 1):.3%}" == "99.993%"
    rng = np.random.default_rng(seed=0)
    pts20 = np.exp(rng.uniform(-50, 50, size=20_000))
    signed = np.array([(my_ln(x) - math.log(x)) / np.spacing(abs(math.log(x))) for x in pts20])
    values, hist_counts = np.unique(np.round(signed), return_counts=True)
    # 5.2 error vs terms
    getcontext().prec = 50
    term_counts = np.arange(1, 61)

    def mercator_error(f, terms):
        partial_sum = sum(Decimal(-1) ** (n + 1) * Decimal(f) ** n / n for n in range(1, terms + 1))
        return float(abs(partial_sum - (1 + Decimal(f)).ln()))

    def artanh_error(s, terms):
        partial_sum = 2 * sum(Decimal(s) ** (2 * n + 1) / (2 * n + 1) for n in range(terms))
        return float(abs(partial_sum - ((1 + Decimal(s)) / (1 - Decimal(s))).ln()))

    worst = 3 - 2 * Decimal(2).sqrt()
    curves = [("Mercator, m in [0.5, 1)", [mercator_error(-0.5, n) for n in term_counts], MUTED),
              ("artanh, m in [0.5, 1)", [artanh_error(-Decimal(1) / 3, n) for n in term_counts], BLUE),
              ("artanh, m in [√½, √2)", [artanh_error(worst, n) for n in term_counts], ORANGE)]
    needed = [next(n for n, e in zip(term_counts, errs) if e < 2.0 ** -53) for _, errs, _ in curves]
    assert needed == [48, 16, 10]
    # 4.6 AGM gaps
    getcontext().prec = 60
    a, b = Decimal(1), Decimal(4) / Decimal(2) ** 100
    gaps = []
    for _ in range(1, 40):
        a, b = (a + b) / 2, (a * b).sqrt()
        gaps.append(abs(a - b))
        if abs(a - b) < Decimal(10) ** -55:
            break
    assert [f"{g:.2e}" for g in gaps] == ["5.00e-1", "2.50e-1", "1.25e-1", "5.93e-2", "1.86e-2", "1.90e-3",
                                          "2.00e-5", "2.20e-9", "2.68e-17", "3.95e-33", "1.00e-61"]
    # 4.7 bit length
    n = 2 ** 60 - 1
    assert n.bit_length() - 1 == 59 and math.floor(math.log2(n)) == 60

    fig = new_figure(r"How math.log works: $\ln x = k\ln 2 + \ln m$, then a short series",
                     r"$x = m\cdot 2^k \Leftrightarrow \ln x = k\ln 2 + \ln m$ with $m \in [\sqrt{1/2}, \sqrt{2})$;   "
                     r"$\ln m = 2(s + s^3/3 + s^5/5 + \cdots)$, $s = \frac{m-1}{m+1}$, $|s| \leq 0.1716$.")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0])
    xv = np.geomspace(2 ** -4, 2 ** 4, 2000)
    red = [reduce_range(x) for x in xv]
    kv = np.array([k for _, k in red])
    lnm = np.array([math.log(m) for m, _ in red])
    ax.axhspan(-0.347, 0.347, color=ORANGE_LIGHT, alpha=0.5, zorder=0)
    ax.plot(xv, np.log(xv), color=INK, linewidth=2.4, label=r"$\ln x$")
    ax.step(xv, kv * LN2, where="post", color=BLUE, linewidth=2, label=r"staircase $k\ln 2$ (exponent bits)")
    ax.plot(xv, lnm, color=ORANGE, linewidth=2, label=r"sawtooth $\ln m$, inside $\pm 0.347$")
    ax.set_xscale("log", base=2)
    finish(ax, "Range reduction: a staircase plus a wiggle", "x (log scale, base 2)", "value",
           legend_loc="upper left", legend_size=9.5)

    ax = fig.add_subplot(gs[0, 1])
    for (label, errs, color), need in zip(curves, needed):
        ax.plot(term_counts, errs, "o-", color=color, markersize=3, linewidth=1.6,
                label=f"{label}: {need} terms")
    ax.axhline(2.0 ** -53, color=INK, linestyle="--", linewidth=1.2, label=r"double precision $2^{-53}$")
    ax.set_yscale("log")
    ax.set_ylim(1e-30, 10)
    finish(ax, "Small s and odd powers only: 10 terms suffice", "number of terms",
           "worst-case error (log scale)", legend_loc="upper right", legend_size=9)

    panel(fig.add_subplot(gs[0, 2]), r"$\ln 10$ by hand", [
        r"frexp: $10 = 0.625\cdot 2^4$; re-centre: $10 = 1.25\cdot 2^3$.",
        r"$s = \frac{1.25 - 1}{1.25 + 1} = \frac{1}{9}$",
        r"$\ln 1.25 = 2\left(\frac{1}{9} + \frac{1}{3\cdot 9^3} + \frac{1}{5\cdot 9^5} + \cdots\right) = 0.2231436$",
        r"$\ln 10 = 3\ln 2 + \ln 1.25 = 2.0794415 + 0.2231436$",
        r"$= 2.3025851$ (three terms: within $10^{-7}$).",
        "",
        r"Mercator with $f = 0.25$, three terms: $0.2239583$.",
        r"$10^{-300}$: $k = -997$, $m = 1.339$; size lives in $k$.",
    ], fontsize=12, spacing=0.08)

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(values, hist_counts, width=0.6, color=[AQUA if v == 0 else BLUE for v in values], zorder=2)
    for v, c in zip(values, hist_counts):
        ax.text(v, c * 1.25, f"{c:,}", ha="center", fontsize=10, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1e5)
    ax.set_xticks(values)
    finish(ax, "my_ln vs math.log: 20,000 x from $e^{-50}$ to $e^{50}$", "error (ulps)",
           "count (log scale)", legend_loc=None)
    ax.text(0.02, 0.97, f"100,000 points: {np.mean(ulp == 0):.1%} identical,\nmax {ulp.max():.0f} ulps",
            transform=ax.transAxes, va="top", fontsize=10, color=INK2)

    ax = fig.add_subplot(gs[1, 1])
    steps = np.arange(1, len(gaps) + 1)
    ax.plot(steps, [float(g) for g in gaps], "o-", color=BLUE, linewidth=2, markersize=6)
    for st, g in zip(steps[6:], gaps[6:]):
        ax.annotate(f"{g:.0e}".replace("e-0", "e-"), (st, float(g)), xytext=(-8, -4), ha="right",
                    textcoords="offset points", fontsize=9, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(steps)
    finish(ax, r"AGM$(1, 4/2^{100})$: the gap is squared each step", "AGM step",
           r"gap $a_n - b_n$ (log scale)", legend_loc=None)

    panel(fig.add_subplot(gs[1, 2]), "Libraries, AGM and integers", [
        r"fdlibm: same $m$ and $s$, the series tail replaced",
        r"by a minimax polynomial; $\ln 2$ stored in two parts.",
        r"Accuracy is measured in ulps, $\approx |y|\cdot 2^{-52}$.",
        "",
        r"AGM: $\ln s \approx \frac{\pi}{2\,\mathrm{AGM}(1, 4/s)}$; digits double per step:",
        r"11 steps give $\ln 2$ to 59 digits.",
        "",
        r"Integers: $\lfloor \log_2 n \rfloor$ = n.bit_length() − 1, exact.",
        r"$n = 2^{60} - 1$: bit_length → 59, math.log2 → 60.",
    ], fontsize=12, spacing=0.08)
    save03(fig, name="03_how_computers_compute_logs.png")


# ================================================================ 03/04
def sheet_03_04_floats():
    probs = np.full(1000, 1 / 6)
    naive = 1.0
    for p in probs:
        naive *= p
    log_p = np.sum(np.log(probs))
    assert naive == 0.0 and f"{log_p / math.log(10):.2f}" == "-778.15" and round(log_p, 2) == -1791.76
    info = np.finfo(np.float64)
    assert info.eps == 2.0 ** -52 and round(math.log10(np.nextafter(0.0, 1.0)), 1) == -323.3
    assert round(math.log10(info.max), 1) == 308.3 and np.spacing(1e6) == 2.0 ** -33
    # 4.4
    rows = {}
    for x in [1e-3, 1e-6, 1e-10, 1e-15, 1e-17]:
        rows[x] = abs(math.log(1 + x) / math.log1p(x) - 1)
    assert [f"{v:.1e}" for v in rows.values()] == ["1.1e-13", "8.2e-11", "8.3e-08", "1.1e-01", "1.0e+00"]
    p, n = 1e-18, 1e15
    assert (1 - p) ** n == 1.0 and f"{math.exp(n * math.log1p(-p)):.7f}" == "0.9990005"
    # 4.6 model comparison, seed 5
    rng = np.random.default_rng(seed=5)
    loaded = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.25])
    fair = np.full(6, 1 / 6)
    rolls = rng.choice(6, size=2000, p=loaded)
    ll_fair, ll_loaded = np.sum(np.log(fair[rolls])), np.sum(np.log(loaded[rolls]))
    assert round(ll_fair, 2) == -3583.52 and round(ll_loaded, 2) == -3547.51
    evidence = np.logaddexp(ll_fair, ll_loaded)
    p_fair = np.exp(ll_fair - evidence)
    assert f"{p_fair:.4e}" == "2.3016e-16" and np.prod(fair[rolls]) == 0.0
    # 5.1 toy floats
    toy = np.array(sorted((1 + j / 8) * 2.0 ** e for e in range(-3, 4) for j in range(8)))
    assert len(toy) == 56 and toy[0] == 0.125 and toy[-1] == 15.0
    # 5.4 logsumexp edges
    shifts = np.linspace(-1000, 1000, 801)
    with np.errstate(over="ignore", divide="ignore"):
        naive_lse = np.array([np.log(np.exp(a) + np.exp(a - 1)) for a in shifts])
    stable_lse = np.array([max(a, a - 1) + np.log(np.exp(a - max(a, a - 1)) + np.exp(a - 1 - max(a, a - 1)))
                           for a in shifts])
    finite = np.isfinite(naive_lse)
    assert shifts[finite][0] == -745.0 and shifts[finite][-1] == 707.5

    fig = new_figure("Floats are a logarithmic ruler: work in logs, carefully",
                     r"$x = (1.f)_2\cdot 2^{e} \Leftrightarrow \log_2 x = e + \log_2(1.f)_2$;   "
                     r"$\ln\prod p_i = \sum \ln p_i$;   $\ln\sum e^{a_i} = M + \ln\sum e^{a_i - M}$, $M = \max a_i$.")
    gs = grid(fig)

    sub = gs[0, 0].subgridspec(2, 1, hspace=0.9)
    for row, (scale, color, title) in enumerate([("linear", BLUE, "56 toy floats (3 fraction bits): linear axis"),
                                                 ("log", ORANGE, "the same floats on a log axis (base 2)")]):
        ax = fig.add_subplot(sub[row])
        ax.plot(toy, np.zeros_like(toy), "|", markersize=22, markeredgewidth=1.6, color=color)
        if scale == "log":
            ax.set_xscale("log", base=2)
            ax.grid(True, axis="x")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("x" if scale == "linear" else "x (log scale)", fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    xn = np.geomspace(0.25, 16, 3000)
    ax.plot(xn, np.spacing(xn) / xn * 1e16, color=ORANGE, linewidth=2, label="np.spacing(x) / x")
    ax.axhline(2.0 ** -52 * 1e16, color=MUTED, linestyle=":", linewidth=1.4, label=r"$2^{-52} = 2.2\times 10^{-16}$")
    ax.axhline(2.0 ** -53 * 1e16, color=MUTED, linestyle="--", linewidth=1.4, label=r"$2^{-53} = 1.1\times 10^{-16}$")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.9, 3.0)
    finish(ax, "Relative spacing: one sawtooth per binade", "x (log scale, base 2)",
           r"spacing / x  (units of $10^{-16}$)",
           legend_loc="upper right", legend_size=9)

    panel(fig.add_subplot(gs[0, 2]), "Landmarks of a double", [
        (r"machine $\varepsilon$", r"$2^{-52} \approx 2.22\times 10^{-16}$"),
        ("largest", r"$\approx 1.80\times 10^{308}$"),
        ("smallest normal", r"$2^{-1022} \approx 2.23\times 10^{-308}$"),
        ("smallest subnormal", r"$2^{-1074} \approx 4.9\times 10^{-324}$"),
        (r"$\mathrm{ulp}(10^6)$", r"$2^{19-52} = 2^{-33} \approx 1.16\times 10^{-10}$"),
        "",
        r"Log form of the range: $\log_{10} x \in [-323.3,\ 308.3]$.",
        r"1000 die rolls, $(1/6)^{1000} = 10^{-778.15}$, become 0.0,",
        r"but $\ln P = 1000\ln\frac{1}{6} = -1791.76$ is an ordinary float.",
    ], fontsize=12, spacing=0.083, cols=(0.0, 0.44))

    ax = fig.add_subplot(gs[1, 0])
    xs = np.geomspace(1e-18, 1, 400)
    log_err = np.abs(np.log(1 + xs) / np.log1p(xs) - 1)
    exp_err = np.abs((np.exp(xs) - 1) / np.expm1(xs) - 1)
    ax.plot(xs, np.maximum(log_err, 1e-17), ".", color=BLUE, markersize=3.5, label="log(1 + x) vs log1p(x)")
    ax.plot(xs, np.maximum(exp_err, 1e-17), ".", color=ORANGE, markersize=3.5, label="exp(x) − 1 vs expm1(x)")
    ax.plot(xs, np.minimum(2.0 ** -53 / xs, 1), "--", color=INK, linewidth=1.2, label=r"$\varepsilon/(2x)$")
    ax.plot(1e-10, rows[1e-10], "o", color=INK, markersize=6, zorder=5)
    ax.annotate(f"x = 1e-10: {rows[1e-10]:.1e},\nabout 7 of 16 digits left", (1e-10, rows[1e-10]),
                xytext=(1e-17, 1e-13), textcoords="data", arrowprops=dict(arrowstyle="->", color=MUTED),
                fontsize=9.5, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    finish(ax, "Cancellation in 1 + x: use log1p / expm1", "x (log scale)",
           "relative error of naive (log scale)", legend_loc="upper right", legend_size=9)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(shifts, stable_lse - shifts, color=BLUE_LIGHT, linewidth=5, label="stable logsumexp − a")
    ax.plot(shifts[finite], naive_lse[finite] - shifts[finite], ".", color=BLUE, markersize=3,
            label="naive − a (where finite)")
    ax.axhline(np.log1p(np.exp(-1)), color=INK, linestyle="--", linewidth=1,
               label=r"exact $\ln(1 + e^{-1}) = 0.3133$")
    for edge in (shifts[finite][0], shifts[finite][-1]):
        ax.axvline(edge, color=ORANGE, linestyle=":", linewidth=1.8)
    ax.text(-765, 0.06, "naive: −inf", color=ORANGE, fontsize=10, ha="right")
    ax.text(728, 0.06, "naive: inf", color=ORANGE, fontsize=10, ha="left")
    ax.set_ylim(0, 0.55)
    finish(ax, r"$\ln(e^{a} + e^{a-1})$: naive works only for −745 ≤ a ≤ 707.5", "shift a",
           r"result $- a$", legend_loc="upper center", legend_size=9)

    panel(fig.add_subplot(gs[1, 2]), "Working in the log domain", [
        r"2000 rolls of a die loaded towards six:",
        r"both likelihoods are $\approx 10^{-1550}$ → 0.0 as floats.",
        rf"log-likelihoods: fair ${ll_fair:.2f}$, loaded ${ll_loaded:.2f}$,",
        rf"ratio ${ll_loaded - ll_fair:.2f}$;  P(fair) $= {p_fair * 1e16:.1f}\times 10^{{-16}}$ via logaddexp.",
        "",
        r"$(1 - 10^{-18})^{10^{15}}$: naive gives 1.0;",
        r"exp(n·log1p(−p)) gives $0.9990005$.",
        "",
        r"logsumexp: every $a_i - M \leq 0$, the largest term is 1,",
        r"so the sum is in $[1, n]$: no overflow, never $\ln 0$.",
    ], fontsize=12, spacing=0.083)
    save03(fig, name="04_floating_point_and_numerical_stability.png")


if __name__ == "__main__":
    sheet_02_01_graphs()
    sheet_02_02_solving()
    sheet_02_03_log_scales()
    sheet_02_04_growth()
    sheet_03_01_briggs()
    sheet_03_02_slide_rule()
    sheet_03_03_computers()
    sheet_03_04_floats()
