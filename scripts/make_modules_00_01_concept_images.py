#!/usr/bin/env python3
"""Concept sheets for modules/00_foundations (notebooks 01-04) and modules/01_the_natural_logarithm (01-05).

One page per notebook that summarizes its central idea. Every number is computed here from the same
tables, bases, step sizes, series and random seed the notebook uses, and is checked against the
notebook's printed outputs with asserts. Images are written to modules/<module>/img/<notebook stem>.png.

Run from anywhere:  python scripts/make_modules_00_01_concept_images.py
"""
import math
from fractions import Fraction
from functools import partial

import numpy as np
from matplotlib.patches import Rectangle

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, CARD, BLUE, BLUE_LIGHT, ORANGE, AQUA, GRID,
                       new_figure, grid)
from viz_style import save as _save

FOUNDATIONS = REPO / "modules" / "00_foundations"
NATURAL_LOG = REPO / "modules" / "01_the_natural_logarithm"
save_00 = partial(_save, folder=FOUNDATIONS / "img")
save_01 = partial(_save, folder=NATURAL_LOG / "img")

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#a9e3cd"
PURPLE = "#7b5cc4"


# ================================================================ helpers
def panel(ax, heading, lines, fontsize=12.5, top=1.0, spacing=0.085):
    """Text card: bold heading, one entry per line (mathtext allowed). A line holding a fraction is
    set in display size and gets twice the line height so stacked fractions never touch."""
    ax.axis("off")
    ax.add_patch(Rectangle((-0.03, -0.02), 1.06, top + 0.04, transform=ax.transAxes,
                           facecolor=CARD, edgecolor="none", zorder=0, clip_on=False))
    ax.text(0.0, top - 0.04, heading, transform=ax.transAxes, fontsize=14.5, weight="bold", va="top")
    y = top - 0.04 - 0.11
    for line in lines:
        if line == "":
            y -= spacing * 0.5
            continue
        is_code = line.startswith("`")
        text = line.strip("`")
        tall = r"\frac" in text
        if tall:
            text = text.replace(r"\frac", r"\dfrac")
            y -= spacing * 0.15
        ax.text(0.0, y, text, transform=ax.transAxes, va="top", color=INK if is_code else INK2,
                fontsize=fontsize - (1 if is_code else 0), family="monospace" if is_code else None)
        y -= spacing * (1.95 if tall else 1.0)


def close(a, b, tol):
    return abs(a - b) <= tol


def arrow(ax, start, end, color, width=1.8, style="<->"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, linewidth=width, shrinkA=0, shrinkB=0))


def number_line(ax, y, x0, x1):
    ax.plot([x0, x1], [y, y], color=INK2, linewidth=1.2, zorder=1)


def axes_cross(ax):
    ax.axhline(0, color=MUTED, linewidth=0.9, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=0.9, zorder=1)


def bare(ax):
    ax.axis("off")


# ================================================================ 00/01
def sheet_00_01_why_logarithms_exist():
    # 4.1 Stifel's table
    log_table = {2 ** n: n for n in range(21)}
    antilog_table = {n: 2 ** n for n in range(21)}
    assert antilog_table[log_table[32] + log_table[128]] == 4096
    assert antilog_table[log_table[1024] + log_table[512]] == 524288
    assert antilog_table[log_table[512] // 3] == 8 and 3 not in log_table

    # 4.2 Buergi table with base 1.0001
    base = 1.0001
    n_max = int(np.ceil(np.log(100) / np.log(base)))
    table_powers = base ** np.arange(n_max + 1)
    assert len(table_powers) == 46056

    def lookup_exponent(x):
        position = np.searchsorted(table_powers, x)
        candidates = [max(position - 1, 0), min(position, len(table_powers) - 1)]
        return min(candidates, key=lambda i: abs(table_powers[i] - x))

    rows_3, rows_5 = lookup_exponent(3), lookup_exponent(5)
    product_35 = table_powers[rows_3 + rows_5]
    assert (rows_3, rows_5, rows_3 + rows_5) == (10987, 16095, 27082)
    assert f"{product_35:.5f}" == "15.00022"
    for (x, y), printed in zip([(2.5, 3.7), (7.25, 9.1)], ["9.24968", "65.97597"]):
        assert f"{table_powers[lookup_exponent(x) + lookup_exponent(y)]:.5f}" == printed

    # 4.4 four-figure common logs
    log_x, log_y = round(math.log10(3.172), 4), round(math.log10(47.85), 4)
    assert (log_x, log_y) == (0.5013, 1.6799)
    product_from_logs = 10 ** (log_x + log_y)
    assert f"{product_from_logs:.4f}" == "151.7749" and f"{3.172 * 47.85:.4f}" == "151.7802"

    # 4.5 digits by logarithm
    digit_checks = {e: math.floor(e * math.log10(2)) + 1 for e in [10, 64, 100, 1000]}
    assert digit_checks == {10: 4, 64: 20, 100: 31, 1000: 302}
    assert all(d == len(str(2 ** e)) for e, d in digit_checks.items())

    # 4.6 prosthaphaeresis
    alpha, beta = math.acos(0.5), math.acos(0.866)
    tycho = (math.cos(alpha - beta) + math.cos(alpha + beta)) / 2
    assert f"{tycho:.6f}" == "0.433000"

    # 5.3 bases close to 1
    bases = np.array([2, 1.1, 1.01, 1.001, 1.0001, 1.00001])
    rows_1_to_10 = np.log(10) / np.log(bases)
    worst_error = np.sqrt(bases) - 1
    assert round(rows_1_to_10[4]) == 23027

    fig = new_figure("A logarithm turns multiplication into addition",
                     r"$b^m \cdot b^n = b^{m+n}$   read backwards is   $\log_b(xy) = \log_b x + \log_b y$:"
                     "   a table of powers multiplies by adding exponents (Stifel 1544, Napier 1614)")
    gs = grid(fig, top=0.845)

    # --- Stifel's two rows
    ax = fig.add_subplot(gs[0, 0])
    bare(ax)
    for n in range(11):
        highlight = n in (4, 6, 10)
        color = ORANGE if highlight else INK
        ax.text(n, 1.0, str(n), ha="center", va="center", fontsize=12, color=color,
                weight="bold" if highlight else None)
        ax.text(n, 0.0, str(2 ** n), ha="center", va="center", fontsize=10.5 if n < 10 else 9.5,
                color=color, weight="bold" if highlight else None)
        ax.plot([n, n], [0.78, 0.22], color=GRID, linewidth=1.2, zorder=0)
    ax.text(-0.4, 1.38, "exponent n  (add)", fontsize=10.5, color=BLUE)
    ax.text(-0.4, -0.42, r"$2^n$  (multiply)", fontsize=10.5, color=BLUE)
    ax.annotate("", xy=(10, 1.18), xytext=(4, 1.18),
                arrowprops=dict(arrowstyle="->", color=ORANGE, connectionstyle="arc3,rad=-0.25", lw=1.6))
    ax.text(7.2, 1.72, "4 + 6 = 10", fontsize=11, color=ORANGE, ha="center")
    ax.text(7.0, -0.82, "16 × 64 = 1024", fontsize=11, color=ORANGE, ha="center")
    ax.set_xlim(-0.6, 10.6)
    ax.set_ylim(-1.05, 2.0)
    ax.set_title("Stifel's two rows: add on top, multiply below")

    # --- base close to 1
    ax = fig.add_subplot(gs[0, 1])
    labels = [f"{b:g}" for b in bases]
    pos = np.arange(len(bases))
    ax.bar(pos, rows_1_to_10, width=0.6, color=BLUE, zorder=2, label="rows between 1 and 10")
    for p, r in zip(pos, rows_1_to_10):
        ax.text(p, r * 1.25, f"{r:,.0f}" if r > 10 else f"{r:.1f}", ha="center", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(1, 3e7)
    ax.set_xticks(pos, labels, fontsize=9.5)
    ax.set_xlabel("table base b")
    ax.set_ylabel("rows from 1 to 10 (log scale)")
    ax.grid(axis="y", which="major", zorder=0)
    twin = ax.twinx()
    twin.plot(pos, worst_error, "o-", color=ORANGE, linewidth=2, label="worst relative error")
    twin.set_yscale("log")
    twin.set_ylim(1e-6, 3)
    twin.set_ylabel(r"worst relative error $\sqrt{b} - 1$", color=ORANGE)
    twin.spines["right"].set_visible(True)
    ax.plot([], [], "o-", color=ORANGE, label="worst relative error")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Bürgi/Napier: base near 1 fills the gaps")

    # --- log ruler
    ax = fig.add_subplot(gs[0, 2])
    bare(ax)
    number_line(ax, 0, 0, 2)
    for decade in [1, 10]:
        for digit in range(1, 10):
            value = digit * decade
            p = math.log10(value)
            major = digit == 1
            ax.plot([p, p], [0, 0.22 if major else 0.12], color=INK, linewidth=1)
            if digit in (1, 2, 3, 5) or major:
                ax.text(p, 0.3, str(value), ha="center", fontsize=9)
    ax.plot([2, 2], [0, 0.22], color=INK, linewidth=1)
    ax.text(2, 0.3, "100", ha="center", fontsize=9)
    for k, (start, color) in enumerate([(1, BLUE), (2, ORANGE), (4, AQUA)]):
        left, right = math.log10(start), math.log10(2 * start)
        arrow(ax, (left, -0.2 - 0.0 * k), (right, -0.2), color)
        ax.text((left + right) / 2, -0.42, f"×2", ha="center", fontsize=10, color=color)
    arrow(ax, (0, -0.75), (1, -0.75), INK2)
    arrow(ax, (1, -0.75), (2, -0.75), INK2)
    ax.text(0.5, -0.97, "×10", ha="center", fontsize=10, color=INK2)
    ax.text(1.5, -0.97, "×10", ha="center", fontsize=10, color=INK2)
    ax.text(1.0, 0.62, r"position of x  =  $\log_{10} x$", ha="center", fontsize=11, color=INK2)
    ax.set_xlim(-0.05, 2.08)
    ax.set_ylim(-1.15, 0.8)
    ax.set_title("A log ruler: equal ratios, equal lengths")

    # --- digits of 2^n
    ax = fig.add_subplot(gs[1, 0])
    n_values = np.unique(np.logspace(0, 3, 200).astype(int))
    digits = np.floor(n_values * math.log10(2)) + 1
    ax.plot(n_values, digits, color=BLUE, linewidth=2.2, label=r"digits of $2^n$ = $\lfloor n \log_{10} 2 \rfloor + 1$")
    ax.plot(n_values, n_values * math.log10(2), color=ORANGE, linestyle="--", linewidth=1.5,
            label=r"$n \log_{10} 2 = 0.30103\,n$")
    offsets = {10: (8, -14), 64: (-8, 6), 100: (8, -14), 1000: (-10, -4)}
    for e, d in digit_checks.items():
        ax.plot(e, d, "o", color=INK, markersize=6, zorder=4)
        ax.annotate(f"$2^{{{e}}}$: {d} digits", (e, d), textcoords="offset points",
                    xytext=offsets[e], ha="left" if offsets[e][0] > 0 else "right", fontsize=9.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("exponent n (log scale)")
    ax.set_ylabel("number of digits (log scale)")
    ax.grid(True, which="major")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Counting digits with one logarithm")

    panel(fig.add_subplot(gs[1, 1]), "Four ways to multiply", [
        "Stifel, powers of 2:  32 × 128",
        "   exponents 5 + 7 = 12  →  4096",
        "Bürgi, base 1.0001:  3 × 5",
        f"   rows {rows_3} + {rows_5} = {rows_3 + rows_5}  →  {product_35:.5f}",
        r"Briggs, four-figure $\log_{10}$:  3.172 × 47.85",
        f"   {log_x} + {log_y} = {log_x + log_y:.4f}  →  {product_from_logs:.4f}",
        f"   (exact {3.172 * 47.85:.4f}: four figures in, four out)",
        "Tycho, prosthaphaeresis:  0.5 × 0.866",
        f"   [cos 30° + cos 90°] / 2 = {tycho:.3f}  (four lookups)",
    ], fontsize=12, spacing=0.092)

    panel(fig.add_subplot(gs[1, 2]), "Powers need two inverses", [
        "3 + x = 10:  undo addition by subtraction",
        "3 · x = 12:  undo multiplication by division",
        r"$x^3 = 8$:  the root recovers the base, $x = \sqrt[3]{8} = 2$",
        r"$2^x = 8$:  the log recovers the exponent, $x = \log_2 8 = 3$",
        "",
        r"$b^y = x \;\Leftrightarrow\; y = \log_b x$",
        "logos + arithmos: the number that counts ratios.",
        "Each operation drops one level: × → +, ÷ → −,",
        "power → ×, root → ÷.",
    ], fontsize=12, spacing=0.092)
    save_00(fig, name="01_why_logarithms_exist.png")


# ================================================================ 00/02
def sheet_00_02_logarithm_as_inverse():
    # 4.1 bisection for log2(10)
    def log_by_bisection(x, base, low, high, steps):
        for _ in range(steps):
            middle = (low + high) / 2
            if base ** middle < x:
                low = middle
            else:
                high = middle
        return (low + high) / 2

    printed = {1: "3.2500000000", 2: "3.3750000000", 3: "3.3125000000", 5: "3.3281250000",
               10: "3.3217773438", 20: "3.3219285011", 40: "3.3219280949"}
    for steps, text in printed.items():
        assert f"{log_by_bisection(10, 2, 3, 4, steps):.10f}" == text
    target = math.log2(10)
    halvings = np.arange(1, 41)
    bisection_error = np.array([abs(log_by_bisection(10, 2, 3, 4, s) - target) for s in halvings])

    # 4.2 rational exponents -> 2^sqrt(2)
    sqrt_two = math.sqrt(2)
    rational_powers = []
    for digits in range(1, 9):
        exponent = Fraction(math.floor(sqrt_two * 10 ** digits), 10 ** digits)
        rational_powers.append((2 ** (1 / exponent.denominator)) ** exponent.numerator)
    assert f"{rational_powers[0]:.10f}" == "2.6390158215" and f"{rational_powers[-1]:.10f}" == "2.6651441363"
    assert f"{2 ** sqrt_two:.10f}" == "2.6651441427"

    # 4.4 / 4.5
    assert (-8) ** (1 / 3) == complex(1.0000000000000002, 1.7320508075688772)
    assert abs(math.log2(10) - (1 + math.log(5) / math.log(2))) < 1e-14

    fig = new_figure(r"The logarithm is the exponent: $\log_b x$ undoes $b^y$",
                     r"$b^y = x \;\Leftrightarrow\; y = \log_b x$   ($b > 0,\ b \neq 1,\ x > 0$)"
                     r"        $b^{\log_b x} = x$,   $\log_b(b^y) = y$        domain and range swap")
    gs = grid(fig, top=0.845)

    # --- mirror image
    ax = fig.add_subplot(gs[0, 0])
    ys = np.linspace(-4, 3.25, 300)
    xs = np.linspace(0.02, 9.5, 300)
    ax.plot(ys, 2.0 ** ys, color=BLUE, linewidth=2.3, label=r"$y = 2^x$")
    ax.plot(xs, np.log2(xs), color=ORANGE, linewidth=2.3, label=r"$y = \log_2 x$")
    ax.plot([-4, 9.5], [-4, 9.5], color=MUTED, linestyle="--", linewidth=1, label="y = x")
    for e in [-2, -1, 0, 1, 2, 3]:
        v = 2.0 ** e
        ax.plot([e, v], [v, e], ":", color=INK2, linewidth=1)
        ax.plot(e, v, "o", color=BLUE, markersize=5)
        ax.plot(v, e, "o", color=ORANGE, markersize=5)
    ax.annotate("(3, 8)", (3, 8), textcoords="offset points", xytext=(8, -4), fontsize=10)
    ax.annotate("(8, 3)", (8, 3), textcoords="offset points", xytext=(-4, -20), fontsize=10)
    axes_cross(ax)
    ax.set_xlim(-4, 9.5)
    ax.set_ylim(-4, 9.5)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Mirror images in the line y = x")

    # --- bisection
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(halvings, bisection_error, "o", color=BLUE, markersize=4, label="|estimate − log₂ 10|")
    ax.semilogy(halvings, 2.0 ** -halvings, color=ORANGE, linestyle="--", label=r"interval width $2^{-n}$")
    ax.text(0.03, 0.04, "after 1, 2, 3 halvings: 3.25, 3.375, 3.3125\n"
            f"after 40 halvings: {log_by_bisection(10, 2, 3, 4, 40):.10f}",
            transform=ax.transAxes, fontsize=9.5, color=INK2)
    ax.set_xlabel("halvings n")
    ax.set_ylabel("error (log scale)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title(r"Solving $2^x = 10$ by halving [3, 4]")

    # --- rational exponents
    ax = fig.add_subplot(gs[0, 2])
    digits = np.arange(1, 9)
    err = 2 ** sqrt_two - np.array(rational_powers)
    ax.semilogy(digits, err, "o-", color=BLUE, linewidth=2, label=r"$2^{\sqrt{2}} - 2^{r}$")
    for d, rp, e in zip(digits, rational_powers, err):
        if d in (1, 2, 4, 8):
            r = math.floor(sqrt_two * 10 ** d) / 10 ** d
            ax.annotate(f"$2^{{{r:g}}}$ = {rp:.7f}", (d, e), textcoords="offset points",
                        xytext=(6, 4), fontsize=9, color=INK2)
    ax.set_xlim(0.5, 11.8)
    ax.set_xlabel(r"decimals of $\sqrt{2}$ kept in the exponent r")
    ax.set_ylabel("gap to the limit (log scale)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(r"Real exponents as limits: $2^{\sqrt{2}} = 2.6651441\ldots$")

    # --- several bases
    ax = fig.add_subplot(gs[1, 0])
    xv = np.linspace(0.05, 12, 400)
    for b, color, name in [(2, BLUE, "2"), (math.e, AQUA, "e"), (10, ORANGE, "10"), (0.5, PURPLE, "1/2")]:
        ax.plot(xv, np.log(xv) / np.log(b), color=color, linewidth=2,
                label=(r"$\ln x$" if name == "e" else rf"$\log_{{{name}}} x$"))
        if b > 1:
            ax.plot(b, 1, "o", color=color, markersize=6)
    ax.plot(1, 0, "o", color=INK, markersize=6)
    ax.annotate(r"(1, 0): $\log_b 1 = 0$", (1, 0), textcoords="offset points", xytext=(8, -18), fontsize=9.5)
    ax.annotate(r"(b, 1): $\log_b b = 1$", (10, 1), textcoords="offset points", xytext=(-10, 10),
                ha="right", fontsize=9.5)
    ax.axhline(1, color=MUTED, linestyle=":", linewidth=1)
    axes_cross(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("x (linear scale)")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=9.5, ncol=2)
    ax.set_title("Every log curve passes (1, 0) and (b, 1)")

    panel(fig.add_subplot(gs[1, 1]), "Exponent form  ↔  log form", [
        r"$b^0 = 1 \;\Leftrightarrow\; \log_b 1 = 0$",
        r"$b^1 = b \;\Leftrightarrow\; \log_b b = 1$",
        r"$b^{-1} = 1/b \;\Leftrightarrow\; \log_b (1/b) = -1$",
        r"$b^{1/2} = \sqrt{b} \;\Leftrightarrow\; \log_b \sqrt{b} = 1/2$",
        "",
        r"$27^{2/3} = 9 \;\Leftrightarrow\; \log_{27} 9 = 2/3$",
        r"$2^{-3} = 1/8 \;\Leftrightarrow\; \log_2 (1/8) = -3$",
        r"$(1/3)^{-4} = 81 \;\Leftrightarrow\; \log_{1/3} 81 = -4$",
        r"$4^{5/2} = 32 \;\Leftrightarrow\; \log_4 32 = 5/2$",
    ], fontsize=12.5, spacing=0.088)

    panel(fig.add_subplot(gs[1, 2]), "Why b > 0, b ≠ 1 and x > 0", [
        r"$1^y = 1$ for every y: constant, no inverse.",
        r"$(-8)^{1/3} = -2$ but $(-8)^{2/6} = \sqrt[6]{64} = 2$:",
        "negative bases give no continuous function.",
        "Python: (-8) ** (1/3) is complex, 1 + 1.732j.",
        r"$b^y > 0$ always, so $\log_b 0$, $\log_b(-3)$ do not exist.",
        "`math.log(0)    -> ValueError: math domain error`",
        "",
        "The log may be negative or fractional;",
        r"its argument may not:  $\log_{10} 0.001 = -3$.",
    ], fontsize=12, spacing=0.09)
    save_00(fig, name="02_logarithm_as_inverse_of_exponentiation.png")


# ================================================================ 00/03
def sheet_00_03_laws_of_logarithms():
    log10 = math.log10
    x, y = 1000.0, 10.0
    false_laws = [("log(x + y)", log10(x + y), "log x + log y", log10(x) + log10(y)),
                  ("log(x · y)", log10(x * y), "log x · log y", log10(x) * log10(y)),
                  ("log(x / y)", log10(x / y), "log x / log y", log10(x) / log10(y)),
                  ("(log x)²", log10(x) ** 2, "2 log x", 2 * log10(x)),
                  ("log(x − y)", log10(x - y), "log x − log y", log10(x) - log10(y))]
    assert [f"{a:.4f}" for _, a, _, _ in false_laws] == ["3.0043", "4.0000", "2.0000", "9.0000", "2.9956"]
    assert [f"{b:.4f}" for _, _, _, b in false_laws] == ["4.0000", "3.0000", "3.0000", "6.0000", "2.0000"]

    log_two_power = 1000 * log10(2)
    log_three_power = 630 * log10(3)
    ratio = 10 ** (log_two_power - log_three_power)
    assert (f"{log_two_power:.4f}", f"{log_three_power:.4f}", f"{ratio:.4f}") == ("301.0300", "300.5864", "2.7772")
    pair_logs = np.array([log10((50 + k) / k) for k in range(1, 51)])
    log_binomial = sum(log10(k) for k in range(51, 101)) - sum(log10(k) for k in range(1, 51))
    assert f"{log_binomial:.4f}" == "29.0039" and abs(pair_logs.sum() - log_binomial) < 1e-11
    assert len(str(math.comb(100, 50))) == math.floor(log_binomial) + 1 == 30
    assert f"{10 ** (log_binomial % 1):.4f}" == "1.0089"
    years = log10(2) / log10(1.05)
    assert f"{years:.4f}" == "14.2067"

    fig = new_figure("Products add, quotients subtract, powers multiply",
                     r"$\log_b(xy) = \log_b x + \log_b y$,   $\log_b \frac{x}{y} = \log_b x - \log_b y$,   "
                     r"$\log_b(x^k) = k \log_b x$   —   the exponent laws $b^m b^n = b^{m+n}$, $(b^m)^k = b^{mk}$ in log form")
    gs = grid(fig, top=0.845)

    # --- log ruler: lengths add
    ax = fig.add_subplot(gs[0, 0])
    bare(ax)
    number_line(ax, 0, 0, 1)
    for value in range(1, 11):
        p = log10(value)
        ax.plot([p, p], [0, 0.13], color=INK, linewidth=1)
        ax.text(p, 0.19, str(value), ha="center", fontsize=10)
    l2, l3 = log10(2), log10(3)
    arrow(ax, (0, -0.15), (l2, -0.15), BLUE, width=2.2)
    ax.text(l2 / 2, -0.32, r"$\log_{10} 2$", ha="center", color=BLUE, fontsize=11)
    arrow(ax, (l2, -0.15), (l2 + l3, -0.15), ORANGE, width=2.2)
    ax.text(l2 + l3 / 2, -0.32, r"$\log_{10} 3$", ha="center", color=ORANGE, fontsize=11)
    ax.plot([log10(6)] * 2, [-0.5, 0.13], color=ORANGE, linestyle="--", linewidth=1.3)
    ax.text(log10(6), -0.6, "lands on 6 = 2 × 3", ha="center", color=ORANGE, fontsize=10.5)
    for k in range(3):
        arrow(ax, (k * l2, -0.85), ((k + 1) * l2, -0.85), AQUA, width=2)
    ax.text(1.5 * l2, -1.02, r"three copies of $\log_{10} 2$ reach $2^3 = 8$", ha="center", color=AQUA, fontsize=10.5)
    ax.plot([log10(8)] * 2, [-0.85, 0.13], color=AQUA, linestyle=":", linewidth=1.3)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-1.15, 0.35)
    ax.set_title("On a log ruler, multiplying is adding lengths")

    # --- product law vs false sum law
    ax = fig.add_subplot(gs[0, 1])
    xv = np.linspace(0.05, 10, 400)
    ax.plot(xv, np.log(4 * xv), color=BLUE, linewidth=5, alpha=0.45, label=r"$\ln(4x)$")
    ax.plot(xv, np.log(xv) + np.log(4), color=INK, linestyle="--", linewidth=1.6, label=r"$\ln x + \ln 4$")
    ax.plot(xv, np.log(xv + 4), color=ORANGE, linewidth=2.2, label=r"$\ln(x + 4)$  (not equal)")
    ax.plot(4 / 3, np.log(16 / 3), "o", color=ORANGE, markersize=7)
    ax.annotate("only crossing: x = 4/3\n(where 4x = x + 4)", (4 / 3, np.log(16 / 3)),
                textcoords="offset points", xytext=(14, -40), fontsize=9.5, color=INK2)
    ax.set_xlabel("x (linear scale)")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_title("The log of a product splits; of a sum, not")

    # --- power law on log-log
    ax = fig.add_subplot(gs[0, 2])
    xv = np.logspace(-1, 2, 200)
    for k, color in [(-1, PURPLE), (0.5, AQUA), (1, BLUE), (2, ORANGE), (3, INK)]:
        ax.plot(xv, xv ** k, color=color, linewidth=2, label=f"$x^{{{k:g}}}$: slope {k:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("x (log scale)")
    ax.set_ylabel(r"$x^k$ (log scale)")
    ax.grid(True, which="major")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.set_title(r"$\log_{10}(x^k) = k \log_{10} x$: powers are lines")

    # --- C(100, 50) as a sum of 50 logs
    ax = fig.add_subplot(gs[1, 0])
    ks = np.arange(1, 51)
    ax.bar(ks, pair_logs, width=0.8, color=BLUE_LIGHT, zorder=2, label=r"$\log_{10}\frac{50 + k}{k}$")
    ax.set_xlabel("k")
    ax.set_ylabel(r"one term $\log_{10}((50 + k)/k)$")
    ax.set_ylim(0, 2.4)
    ax.grid(axis="y", zorder=0)
    twin = ax.twinx()
    twin.plot(ks, np.cumsum(pair_logs), color=ORANGE, linewidth=2.4, label="running sum")
    twin.set_ylim(0, 32)
    twin.set_ylabel("running sum", color=ORANGE)
    twin.spines["right"].set_visible(True)
    twin.text(0.97, 0.42, f"total {log_binomial:.4f}: 30 digits,\n" + r"$\binom{100}{50} \approx 1.0089 \times 10^{29}$",
              transform=twin.transAxes, ha="right", fontsize=10, color=INK2)
    ax.plot([], [], color=ORANGE, linewidth=2.4, label="running sum")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title(r"$\binom{100}{50} = \prod (50+k)/k$: a sum of 50 logs")

    panel(fig.add_subplot(gs[1, 1]), "Three hopeless questions, solved", [
        r"$2^{1000}$ vs $3^{630}$ (power law):",
        f"   {log_two_power:.2f}  vs  {log_three_power:.2f}   →  $2^{{1000}}$ is {ratio:.2f}× bigger",
        r"Digits of $\binom{100}{50}$ (product + quotient laws):",
        f"   sum of 100 logs = {log_binomial:.4f}  →  30 digits  (exact: 30)",
        r"Doubling at 5% a year:  $1.05^n = 2$",
        r"   $n = \log_{10} 2 \,/\, \log_{10} 1.05$" + f" = {years:.4f} years",
        "",
        r"Domain: $\log_b(x^2) = 2\log_b|x|$, not $2\log_b x$;",
        "condensing can admit extraneous roots.",
    ], fontsize=12, spacing=0.09)

    lines = [f"`{a:<11}{va:.4f}  vs  {b:<14}{vb:.4f}`" for a, va, b, vb in false_laws]
    panel(fig.add_subplot(gs[1, 2]), "Not laws (x = 1000, y = 10, base 10)", lines + [
        "",
        "The only honest move with a sum is to factor:",
        r"$\log_b(x + y) = \log_b x + \log_b(1 + y/x)$",
    ], fontsize=11.5, spacing=0.09)
    save_00(fig, name="03_laws_of_logarithms.png")


# ================================================================ 00/04
def sheet_00_04_change_of_base():
    def log_base(x, base):
        return math.log(x) / math.log(base)

    assert f"{log_base(5_000_000, 2):.10f}" == "22.2534966642" and math.ceil(math.log2(5_000_000)) == 23
    assert f"{log_base(2, 1.07):.10f}" == "10.2447683511"
    assert f"{log_base(200, 5):.10f}" == "3.2920296742"
    assert math.log(1000, 10) == 2.9999999999999996 and math.log10(1000) == 3.0
    assert math.log(125, 5) == 3.0000000000000004
    failures = [n for n in range(1, 10 ** 6 + 1) if math.floor(math.log(n, 10)) + 1 != len(str(n))]
    assert failures == [1000, 1000000]
    mantissa_rows = []
    for x in [3720, 3.72, 0.0372]:
        value = math.log10(x)
        characteristic = math.floor(value)
        mantissa_rows.append((x, value, characteristic, value - characteristic))
    assert [f"{v:.5f}" for _, v, _, _ in mantissa_rows] == ["3.57054", "0.57054", "-1.42946"]
    assert [c for _, _, c, _ in mantissa_rows] == [3, 0, -2]
    doubling = {r: math.log(2) / math.log1p(r) for r in [0.01, 0.05, 0.07, 0.10, 0.20]}
    assert [f"{v:.2f}" for v in doubling.values()] == ["69.66", "14.21", "10.24", "7.27", "3.80"]

    fig = new_figure(r"One logarithm is enough: every $\log_b$ is $\ln$ rescaled",
                     r"$\log_b x = \log_a x \,/\, \log_a b = \ln x \,/\, \ln b$      "
                     r"$\log_b a \cdot \log_a b = 1$      base 10 for decimals, $e$ for calculus, 2 for bits")
    gs = grid(fig, top=0.845)

    curves = [(r"$\log_2 x$", np.log2, BLUE, 1 / math.log(2)), (r"$\ln x$", np.log, AQUA, 1.0),
              (r"$\log_{10} x$", np.log10, ORANGE, 1 / math.log(10))]
    ax = fig.add_subplot(gs[0, 0])
    xv = np.linspace(0.05, 20, 500)
    for label, f, color, _ in curves:
        ax.plot(xv, f(xv), color=color, linewidth=2.2, label=label)
    ax.plot(1, 0, "o", color=INK, markersize=6)
    axes_cross(ax)
    ax.set_ylim(-4, 5)
    ax.set_xlim(0, 20)
    ax.set_xlabel("x (linear scale)")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title("Three bases, one shape through (1, 0)")

    ax = fig.add_subplot(gs[0, 1])
    rx = xv[np.abs(xv - 1) > 0.05]
    for label, f, color, factor in curves:
        ax.plot(rx, f(rx) / np.log(rx), color=color, linewidth=2.2)
        ax.text(19.5, factor + 0.05, f"{label} / $\\ln x$ = {factor:.4f}", color=color, ha="right", fontsize=10)
    ax.set_ylim(0, 1.8)
    ax.set_xlim(0, 20)
    ax.set_xlabel("x (linear scale)")
    ax.set_ylabel(r"ratio to $\ln x$")
    ax.grid(True)
    ax.set_title(r"The ratios are constants: $1/\ln b$")

    ax = fig.add_subplot(gs[0, 2])
    bare(ax)
    top = math.log(1100)
    for row, (name, color) in enumerate([("bits", BLUE), ("nats", AQUA), ("digits", ORANGE)]):
        yl = 2 - row
        number_line(ax, yl, 0, top)
        ax.text(-0.25, yl, name, ha="right", va="center", color=color, fontsize=10.5)
    for p in range(0, 11):
        pos = p * math.log(2)
        ax.plot([pos, pos], [2, 2.15], color=BLUE)
        if p % 2 == 0 or p == 10:
            ax.text(pos, 2.22, str(2 ** p), ha="center", fontsize=9, color=BLUE)
    for nats in range(0, 7):
        ax.plot([nats, nats], [1, 1.15], color=AQUA)
        ax.text(nats, 1.22, f"$e^{{{nats}}}$", ha="center", fontsize=9.5, color=AQUA)
    for p in range(0, 4):
        pos = p * math.log(10)
        ax.plot([pos, pos], [0, 0.15], color=ORANGE)
        ax.text(pos, 0.22, str(10 ** p), ha="center", fontsize=9, color=ORANGE)
    ax.plot([10 * math.log(2)] * 2, [-0.2, 2.15], color=INK2, linestyle=":", linewidth=1)
    ax.text(10 * math.log(2), -0.42, r"$2^{10} = 1024 \approx 10^3$:  $10\log_{10} 2 = 3.0103$",
            ha="right", fontsize=10, color=INK2)
    ax.set_xlim(-1.6, top + 0.3)
    ax.set_ylim(-0.6, 2.55)
    ax.set_title("One log axis, three units")

    ax = fig.add_subplot(gs[1, 0])
    rates = np.logspace(np.log10(0.005), 0, 300)
    ax.plot(rates * 100, np.log(2) / np.log1p(rates), color=BLUE, linewidth=2.2,
            label=r"exact: $\log_{1+r} 2 = \ln 2 / \ln(1+r)$")
    ax.plot(rates * 100, 72 / (100 * rates), color=ORANGE, linestyle="--", linewidth=1.8, label="rule of 72: 72 / (100 r)")
    for r, v in doubling.items():
        ax.plot(r * 100, v, "o", color=INK, markersize=5)
        ax.annotate(f"{r:.0%}: {v:.2f} y", (r * 100, v), textcoords="offset points", xytext=(7, 3),
                    ha="left", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("annual growth rate in % (log scale)")
    ax.set_ylabel("years to double (log scale)")
    ax.grid(True, which="major")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Doubling time is a change of base")

    lines = []
    for x, value, characteristic, mantissa in mantissa_rows:
        lines.append(f"`{x:>7g}  {value:>9.5f} = {characteristic:>2} + {mantissa:.5f}`")
    panel(fig.add_subplot(gs[1, 1]), r"Base 10: characteristic + mantissa", [
        r"$\log_{10}(m \times 10^k) = k + \log_{10} m$,  $1 \leq m < 10$",
        "`      x     log10 x   char + mantissa`",
    ] + lines + [
        "The mantissa fixes the digits 3.72, the integer",
        "part only the decimal point: one decade of",
        "tables covered every number (Briggs 1617).",
        r"Bits:  $\lfloor \log_2 n \rfloor + 1$ = n.bit_length()",
    ], fontsize=12, spacing=0.088)

    panel(fig.add_subplot(gs[1, 2]), "Using it in Python", [
        r"$\log_2 5{,}000{,}000 = \ln(5\times10^6)/\ln 2$" + f" = {log_base(5_000_000, 2):.4f}",
        "   → 23 bits for a byte offset",
        r"$\log_{1.07} 2$" + f" = {log_base(2, 1.07):.4f} years to double at 7%",
        "",
        "`math.log(1000, 10) = 2.9999999999999996`",
        "`math.log10(1000)   = 3.0`",
        "floor(math.log(n, 10)) + 1 miscounts digits for",
        f"n = {failures[0]:,} and {failures[1]:,} (and no other n ≤ 10⁶).",
        "Prefer math.log10 / math.log2; use a tolerance.",
    ], fontsize=11.5, spacing=0.09)
    save_00(fig, name="04_change_of_base_and_common_bases.png")


# ================================================================ 01/01
def sheet_01_01_compound_interest_and_e():
    frequencies = {"yearly": 1, "half-yearly": 2, "quarterly": 4, "monthly": 12, "weekly": 52,
                   "daily": 365, "hourly": 365 * 24, "every minute": 365 * 24 * 60}
    balances = {name: (1 + 1 / n) ** n for name, n in frequencies.items()}
    printed = ["2.0000000000", "2.2500000000", "2.4414062500", "2.6130352902", "2.6925969544",
               "2.7145674820", "2.7181266916", "2.7182792427"]
    assert [f"{v:.10f}" for v in balances.values()] == printed

    naive_1e16 = (1 + 1 / 1e16) ** 1e16
    stable_1e16 = math.exp(1e16 * math.log1p(1 / 1e16))
    assert naive_1e16 == 1.0 and f"{stable_1e16:.15f}" == "2.718281828459045"
    assert f"{(1 + 1 / 1e15) ** 1e15:.15f}" == "3.035035206549262"

    partial_sum, term = 0.0, 1.0
    series = []
    for k in range(16):
        if k > 0:
            term /= k
        partial_sum += term
        series.append(partial_sum)
    assert f"{series[9]:.16f}" == "2.7182815255731922" and f"{series[15]:.16f}" == "2.7182818284589949"

    finals = {name: 1000 * (1 + 0.05 / n) ** (n * 10) for name, n in
              [("yearly", 1), ("quarterly", 4), ("monthly", 12), ("daily", 365)]}
    continuous = 1000 * math.exp(0.5)
    assert [f"{v:.4f}" for v in finals.values()] == ["1628.8946", "1643.6195", "1647.0095", "1648.6648"]
    assert f"{continuous:.4f}" == "1648.7213"

    fig = new_figure(r"Compounding more and more often converges to $e$",
                     r"$e = \lim_{n\to\infty}\left(1 + \frac{1}{n}\right)^n = \sum_{k \geq 0} \frac{1}{k!} = 2.718281828\ldots$"
                     r"        rate $r$ for time $t$:  $G = e^{rt} \;\Leftrightarrow\; rt = \ln G$")
    gs = grid(fig, top=0.845)

    ax = fig.add_subplot(gs[0, 0])
    t = np.linspace(0, 1, 2001)
    for n, color in [(1, MUTED), (2, BLUE_LIGHT), (4, AQUA), (12, BLUE)]:
        balance = (1 + 1 / n) ** np.floor(t * n + 1e-12)
        ax.step(t, balance, where="post", color=color, linewidth=2, label=f"n = {n}: ends at {(1 + 1 / n) ** n:.4f}")
    ax.plot(t, np.exp(t), color=ORANGE, linewidth=2.4, label=r"continuous $e^t$: ends at $e$")
    ax.set_xlabel("time t (years), 100% interest")
    ax.set_ylabel("balance (deposit 1)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Compounding n times a year: staircases")

    ax = fig.add_subplot(gs[0, 1])
    nv = np.arange(1, 31)
    sums = np.cumsum([1 / math.factorial(k) for k in range(31)])[1:]
    ax.plot(nv, (1 + 1 / nv) ** nv, "o-", color=BLUE, markersize=4, label=r"$(1 + 1/n)^n$  (compounding)")
    ax.plot(nv, sums, "s-", color=ORANGE, markersize=4, label=r"$\sum_{k=0}^{n} 1/k!$  (series)")
    ax.axhline(math.e, color=INK, linestyle="--", linewidth=1.2, label=r"$e = 2.71828\ldots$")
    ax.axhline(3, color=MUTED, linestyle=":", linewidth=1.2, label="Bernoulli's bound 3")
    ax.set_ylim(1.9, 3.1)
    ax.set_xlabel("n (periods / terms)")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_title("Increasing, bounded by 3: two roads to e")

    lines = [f"`{name:<13}{frequencies[name]:>7}  {value:.10f}`" for name, value in balances.items()]
    panel(fig.add_subplot(gs[0, 2]), "Bernoulli's bank, 100% a year", [
        "`compounding        n  (1 + 1/n)^n`"] + lines + [
        r"error $e - a_n \approx e/(2n)$: 1440× more periods,",
        "about three more digits.",
    ], fontsize=12.5, spacing=0.078)

    ax = fig.add_subplot(gs[1, 0])
    nv = np.logspace(0, 15, 61)
    naive = np.abs(math.e - (1 + 1 / nv) ** nv)
    stable = np.abs(math.e - np.exp(nv * np.log1p(1 / nv)))
    stable[stable == 0] = np.nan
    ax.loglog(nv, naive, "o", color=ORANGE, markersize=4, label="naive (1 + 1/n)**n")
    ax.loglog(nv, stable, "s", color=BLUE, markersize=3.5, label="exp(n · log1p(1/n))")
    ax.loglog(nv, math.e / (2 * nv), color=INK, linewidth=1.3, label=r"$e/(2n)$, slope −1")
    ax.set_xlabel("n (log scale)")
    ax.set_ylabel(r"$|e - a_n|$ (log scale)")
    ax.grid(True, which="major")
    ax.legend(loc="lower left", fontsize=9.5)
    ax.set_title("Error ~ 1/n, until rounding takes over")

    ax = fig.add_subplot(gs[1, 1])
    terms = np.arange(1, 19)
    partial = np.array([sum(1 / math.factorial(k) for k in range(m + 1)) for m in terms])
    serr = np.abs(math.e - partial)
    serr[serr == 0] = np.nan
    bound = np.array([1 / (m * math.factorial(m)) for m in terms])
    ax.semilogy(terms, serr, "o-", color=ORANGE, linewidth=2, label=r"$e - \sum_{k \leq n} 1/k!$")
    ax.semilogy(terms, bound, "--", color=INK2, label=r"bound $1/(n \cdot n!)$")
    ax.axhline(np.finfo(float).eps * math.e, color=MUTED, linestyle=":", label="float resolution near e")
    ax.annotate(f"10 terms: {series[9]:.7f}", (9, abs(math.e - series[9])), textcoords="offset points",
                xytext=(8, 6), fontsize=9.5, color=INK2)
    ax.set_xlabel("n (last term 1/n!)")
    ax.set_ylabel("error (log scale)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("The factorial series: faster than geometric")

    panel(fig.add_subplot(gs[1, 2]), r"Any rate: $(1 + r/n)^{nt} \to e^{rt}$", [
        "1000 at 5% for 10 years:",
    ] + [f"`  {name:<10}{v:10.4f}`" for name, v in finals.items()] + [
        f"`  continuous{continuous:10.4f}`",
        r"Doubling needs $rt = \ln 2 \approx 0.693$:",
        r"   13.86 years at 5%  (the rule of 70).",
        r"Floating point: $1 + 10^{-16}$ rounds to 1, so",
        r"(1 + 1/n)**n = 1 at $n = 10^{16}$; log1p fixes it.",
    ], fontsize=12, spacing=0.082)
    save_01(fig, name="01_compound_interest_and_e.png")


# ================================================================ 01/02
def sheet_01_02_ln_as_area():
    def riemann_sums(a, b, strips):
        edges = np.linspace(a, b, strips + 1)
        width = (b - a) / strips
        heights = 1 / edges
        left = width * heights[:-1].sum()
        right = width * heights[1:].sum()
        midpoint = width * (1 / (edges[:-1] + width / 2)).sum()
        return left, right, midpoint, (left + right) / 2

    printed = {4: ["0.759524", "0.634524", "0.691220", "0.697024"],
               10: ["0.718771", "0.668771", "0.692835", "0.693771"],
               100: ["0.695653", "0.690653", "0.693144", "0.693153"]}
    for strips, values in printed.items():
        assert [f"{v:.6f}" for v in riemann_sums(1, 2, strips)] == values
    assert Fraction(1, 4) * (1 + Fraction(4, 5) + Fraction(2, 3) + Fraction(4, 7)) == Fraction(319, 420)
    assert Fraction(1, 4) * (Fraction(4, 5) + Fraction(2, 3) + Fraction(4, 7) + Fraction(1, 2)) == Fraction(533, 840)

    def area_from_one(x, points=400_001):
        t = np.linspace(1, x, points)
        y = 1 / t
        return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(t)))

    low, high = 2.0, 3.0
    for _ in range(40):
        middle = (low + high) / 2
        if area_from_one(middle) < 1:
            low = middle
        else:
            high = middle
    assert f"{low:.10f}" == "2.7182818285"

    strips_list = np.unique(np.logspace(0.3, 3, 25).astype(int))
    errors = np.array([[abs(v - math.log(2)) for v in riemann_sums(1, 2, s)] for s in strips_list])

    fig = new_figure(r"$\ln x$ is the area under the hyperbola $1/t$ from 1 to $x$",
                     r"$\ln x = \int_1^x dt/t$      $\int_a^{ab} dt/t = \int_1^{b} dt/t$"
                     r"$\;\Rightarrow\; \ln(ab) = \ln a + \ln b$      the area reaches 1 at $x = e$")
    gs = grid(fig, top=0.845)

    ax = fig.add_subplot(gs[0, 0])
    t = np.linspace(0.5, 9, 600)
    ax.plot(t, 1 / t, color=INK, linewidth=2, label="y = 1/t")
    for k, color in enumerate([BLUE, ORANGE, AQUA]):
        a, b = 2 ** k, 2 ** (k + 1)
        strip = np.linspace(a, b, 200)
        ax.fill_between(strip, 1 / strip, color=color, alpha=0.5,
                        label=f"[{a}, {b}]: area = ln 2 = {math.log(2):.3f}")
    ax.set_xlim(0.5, 9)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("t")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Saint-Vincent: doubling intervals, equal areas")

    ax = fig.add_subplot(gs[0, 1])
    a, b = 3, 2.5
    t = np.linspace(0.4, 9, 600)
    region = np.linspace(1, b, 200)
    ax.plot(t, 1 / t, color=INK, linewidth=1.8)
    ax.fill_between(region, 1 / region, color=BLUE, alpha=0.55, label=f"over [1, {b}]: ln {b} = {math.log(b):.4f}")
    ax.fill_between(a * region, 1 / (a * region), color=ORANGE, alpha=0.55,
                    label=f"(t, y) → (3t, y/3): over [3, 7.5], same area")
    for tt in [1, b]:
        arrow(ax, (tt, 1 / tt), (a * tt, 1 / (a * tt)), INK2, width=1.2, style="->")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("t")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Stretch by a, squash by 1/a: same hyperbola")

    ax = fig.add_subplot(gs[0, 2])
    strips = 4
    edges = np.linspace(1, 2, strips + 1)
    upper, lower = riemann_sums(1, 2, strips)[:2]
    ax.bar(edges[:-1], 1 / edges[:-1], width=0.25, align="edge", color=ORANGE_LIGHT,
           edgecolor=ORANGE, label=f"upper (left ends) = 319/420 = {upper:.4f}")
    ax.bar(edges[:-1], 1 / edges[1:], width=0.25, align="edge", color=BLUE, edgecolor=SURFACE,
           label=f"lower (right ends) = 533/840 = {lower:.4f}")
    tt = np.linspace(1, 2, 300)
    ax.plot(tt, 1 / tt, color=INK, linewidth=2)
    ax.set_xlim(0.95, 2.05)
    ax.set_ylim(0, 1.45)
    ax.set_xlabel("t")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Four strips: {lower:.4f} < ln 2 < {upper:.4f}")

    ax = fig.add_subplot(gs[1, 0])
    names = ["left", "right", "midpoint", "trapezoid"]
    colors = [ORANGE, BLUE, AQUA, PURPLE]
    for j, (name, color) in enumerate(zip(names, colors)):
        ax.loglog(strips_list, errors[:, j], "o-" if j < 2 else "s-", color=color, markersize=3.5,
                  linewidth=1.5, label=name)
    ax.set_xlabel("strips n (log scale)")
    ax.set_ylabel("|sum − ln 2| (log scale)")
    ax.grid(True, which="major")
    ax.legend(loc="lower left", fontsize=9.5)
    ax.text(30, 3e-2, "slope −1: left, right ≈ 1/(4n)", fontsize=9.5, color=INK2)
    ax.text(1.15, 1.2e-5, r"slope −2: ∝ $1/n^2$", fontsize=9.5, color=INK2)
    ax.set_title(r"Riemann sums for $\ln 2 = 0.693147\ldots$")

    ax = fig.add_subplot(gs[1, 1])
    xv = np.logspace(-1, 1, 300)
    ax.plot(xv, np.log(xv), color=PURPLE, linewidth=2.4, label=r"$L(x) = \int_1^x dt/t$")
    ax.plot(xv, 1 - 1 / xv, color=BLUE, linestyle="--", linewidth=1.3, label=r"$1 - 1/x$")
    ax.plot(xv, xv - 1, color=ORANGE, linestyle="--", linewidth=1.3, label=r"$x - 1$")
    ax.axhline(1, color=MUTED, linestyle=":")
    ax.plot(math.e, 1, "o", color=INK, markersize=7)
    ax.annotate("(e, 1): area 1", (math.e, 1), textcoords="offset points", xytext=(-10, 10), ha="right", fontsize=10)
    ax.set_xscale("log")
    ax.set_ylim(-3, 3)
    ax.set_xlabel("x (log scale)")
    ax.set_ylabel("area")
    ax.grid(True, which="major")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title(r"On a log axis $\ln x$ is a straight line")

    panel(fig.add_subplot(gs[1, 2]), "Finding the base", [
        "L is continuous, increasing, L(ab) = L(a) + L(b),",
        r"so $L = \log_B$ where $L(B) = 1$. Squeeze [1, 1 + 1/n]:",
        r"$\frac{n}{n+1} \leq L((1 + 1/n)^n) \leq 1 \;\Rightarrow\; L(e) = 1$",
        "Bisection on the area alone (no exponentials):",
        f"   area = 1 at x = {low:.10f}",
        "",
        r"Rectangles give $1 - \frac{1}{x} \leq \ln x \leq x - 1$:",
        r"   x = 1.1:  0.0909 ≤ 0.0953 ≤ 0.1",
    ], fontsize=12, spacing=0.085)
    save_01(fig, name="02_ln_as_area_under_the_hyperbola.png")


# ================================================================ 01/03
def sheet_01_03_derivatives():
    quotients = {}
    for base in [2, math.e, 3, 10]:
        quotients[base] = [(base ** h - 1) / h for h in [0.1, 0.001, 1e-6]]
    assert [f"{q:.6f}" for q in quotients[2][:2]] == ["0.717735", "0.693387"]
    assert f"{quotients[2][2]:.8f}" == "0.69314742" and f"{quotients[10][2]:.8f}" == "2.30258774"
    assert f"{quotients[3][2]:.8f}" == "1.09861289" and f"{quotients[math.e][2]:.8f}" == "1.00000050"

    def euler(steps):
        y = 1.0
        for _ in range(steps):
            y = y + y / steps
        return y

    euler_values = {n: euler(n) for n in [1, 4, 12, 365, 10_000]}
    assert [f"{v:.8f}" for v in euler_values.values()] == ["2.00000000", "2.44140625", "2.61303529",
                                                           "2.71456748", "2.71814593"]
    assert f"{2 ** 10 * math.log(2):.2f}" == "709.78"

    fig = new_figure(r"$\ln b$ is the growth rate of $b^x$; base $e$ makes it exactly 1",
                     r"$(b^x)' = b^x \ln b$      $(e^x)' = e^x$      "
                     r"$(\ln x)' = 1/x$      $(\log_b x)' = 1/(x \ln b)$      "
                     r"$b = e^k \;\Leftrightarrow\; k = \ln b$")
    gs = grid(fig, top=0.845)

    ax = fig.add_subplot(gs[0, 0])
    xv = np.linspace(-2, 1.6, 400)
    for base, color, name in [(2, BLUE, "2"), (math.e, ORANGE, "e"), (3, AQUA, "3")]:
        slope = math.log(base)
        ax.plot(xv, base ** xv, color=color, linewidth=2.3, label=f"${name}^x$: slope at 0 = {slope:.3f}")
        ax.plot(xv, 1 + slope * xv, color=color, linestyle="--", linewidth=1.2)
    ax.plot(0, 1, "o", color=INK, markersize=6)
    ax.set_xlim(-2, 1.6)
    ax.set_ylim(-1, 5)
    axes_cross(ax)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title(r"Tangents at (0, 1) have slope $\ln b$")

    ax = fig.add_subplot(gs[0, 1])
    ex = np.linspace(-3, 1.9, 300)
    lx = np.linspace(0.05, 6.5, 300)
    ax.plot(ex, np.exp(ex), color=ORANGE, linewidth=2.3, label=r"$y = e^x$")
    ax.plot(lx, np.log(lx), color=PURPLE, linewidth=2.3, label=r"$y = \ln x$")
    ax.plot([-3, 6.5], [-3, 6.5], color=MUTED, linestyle=":", label="y = x")
    for a, color in [(0.0, AQUA), (math.log(4), BLUE)]:
        h = math.exp(a)
        span = np.array([-0.9, 0.9])
        ax.plot(a + span, h + h * span, color=color, linewidth=1.4)
        ax.plot(h + span, a + span / h, color=color, linewidth=1.4)
        ax.plot([a, h], [h, a], "o", color=color, markersize=6)
        ax.annotate(f"slope {h:g}", (a, h), textcoords="offset points", xytext=(-62, 6), fontsize=9.5, color=color)
        ax.annotate("slope 1" if h == 1 else f"slope 1/{h:g}", (h, a), textcoords="offset points",
                    xytext=(8, -16), fontsize=9.5, color=color)
    ax.set_xlim(-3, 6.5)
    ax.set_ylim(-3, 6.5)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Mirror points have reciprocal slopes")

    ax = fig.add_subplot(gs[0, 2])
    bases = np.array([0.1, 0.2, 0.5, 1, 2, math.e, 3, 5, 10])
    measured = (bases ** 1e-6 - 1) / 1e-6
    bc = np.logspace(-1.1, 1.1, 300)
    ax.plot(bc, np.log(bc), color=PURPLE, linewidth=2.2, label=r"$\ln b$")
    ax.plot(bases, measured, "o", color=INK, markersize=5, label=r"measured $(b^h - 1)/h$, $h = 10^{-6}$")
    ax.axhline(1, color=MUTED, linestyle="--", linewidth=1)
    ax.axvline(math.e, color=MUTED, linestyle=":", linewidth=1)
    ax.annotate("b = e: rate exactly 1", (math.e, 1), textcoords="offset points", xytext=(-10, 14),
                ha="right", fontsize=9.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("base b (log scale)")
    ax.set_ylabel(r"growth constant $c_b$")
    ax.grid(True, which="major")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("On a log b-axis the rate is a straight line")

    ax = fig.add_subplot(gs[1, 0])
    xv = np.linspace(0.2, 6, 400)
    ax.plot(xv, np.log(xv), color=INK, linewidth=2.3, label=r"$y = \ln x$")
    ax.plot(xv, 1 / xv, color=MUTED, linewidth=1.4, linestyle="--", label=r"$y = 1/x$")
    for p, color in zip([0.5, 1, 2, 4], [ORANGE, AQUA, BLUE, PURPLE]):
        span = np.array([-0.6, 0.6]) * p
        ax.plot(p + span, math.log(p) + span / p, color=color, linewidth=1.6)
        ax.plot(p, math.log(p), "o", color=color, markersize=6)
        ax.plot([p, p], [0, 1 / p], color=color, linewidth=2.2)
        ax.plot(p, 1 / p, "s", color=color, markersize=5)
        ax.plot([], [], "o-", color=color, label=f"x = {p:g}: slope {1 / p:g} = height {1 / p:g}")
    axes_cross(ax)
    ax.set_ylim(-1.6, 2.2)
    ax.set_xlim(0, 6)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(r"The slope of $\ln$ is the height of $1/x$")

    lines = [f"`{('e' if b == math.e else str(b)):>4}  {q[0]:.6f}  {q[1]:.6f}  {q[2]:.8f}  {math.log(b):.8f}`"
             for b, q in quotients.items()]
    panel(fig.add_subplot(gs[1, 1]), r"Measuring $c_b = \lim (b^h - 1)/h$", [
        "`   b   h=0.1     h=0.001   h=1e-6      ln b`"] + lines + [
        "",
        r"$b^x = e^{x \ln b}$, so the constant is $\ln b$:",
        r"doubling each hour = continuous rate $\ln 2$ = 69.3%/h,",
        r"$5^x = e^{1.6094 x}$;  slope of $2^x$ at 10: 1024·ln 2 = 709.78.",
    ], fontsize=12, spacing=0.088)

    panel(fig.add_subplot(gs[1, 2]), r"Why $e^x$ is its own derivative", [
        r"From the area: a strip of width h, height ≈ 1/x.",
        r"Inverse function: $(e^x)' = 1 / (1/e^x) = e^x$.",
        r"Unique: $f' = f$, $f(0) = 1$  ⟹  $f e^{-x}$ constant.",
        "",
        r"Euler's method for $y' = y$: $y \leftarrow (1 + 1/n)\,y$",
        "is Bernoulli's compounding:",
    ] + [f"`   n = {n:>6}:  y(1) ≈ {v:.8f}`" for n, v in euler_values.items() if n in (4, 12, 365, 10_000)],
        fontsize=12, spacing=0.088)
    save_01(fig, name="03_derivatives_of_ln_and_exp.png")


# ================================================================ 01/04
def sheet_01_04_series():
    def mercator(x, terms):
        total, power = 0.0, 1.0
        for k in range(1, terms + 1):
            power *= x
            total += (-1) ** (k - 1) * power / k
        return total

    def artanh_log(x, terms):
        y = (x - 1) / (x + 1)
        total, power = 0.0, y
        for j in range(terms):
            total += 2 * power / (2 * j + 1)
            power *= y * y
        return total

    assert f"{abs(mercator(0.1, 10) - math.log1p(0.1)):.2e}" == "8.33e-13"
    assert f"{abs(mercator(1.0, 1000) - math.log(2)):.2e}" == "5.00e-04"
    assert f"{artanh_log(2, 10):.12f}" == "0.693147180550"
    assert f"{abs(artanh_log(2, 10) - math.log(2)):.2e}" == "1.01e-11"
    terms_ln2 = [2 / ((2 * j + 1) * 3 ** (2 * j + 1)) for j in range(10)]
    exact = sum(Fraction(2, (2 * j + 1) * 3 ** (2 * j + 1)) for j in range(10))
    assert exact == Fraction(1302374561632216, 1878929321474205)
    partials_x1 = [mercator(1, n) for n in range(1, 11)]
    assert [f"{p:.4f}" for p in partials_x1[:8]] == ["1.0000", "0.5000", "0.8333", "0.5833", "0.7833",
                                                    "0.6167", "0.7595", "0.6345"]
    assert f"{partials_x1[9]:.4f}" == "0.6456"
    ln10 = 3 * artanh_log(2, 12) + artanh_log(1.25, 6)
    assert f"{ln10:.10f}" == f"{math.log(10):.10f}" == "2.3025850930"

    fig = new_figure("Series for the logarithm: a small argument makes them fast",
                     r"Mercator 1668:  $\ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots$  ($-1 < x \leq 1$)"
                     r"        $\ln\frac{1+y}{1-y} = 2\left(y + \frac{y^3}{3} + \frac{y^5}{5} + \cdots\right)$,  "
                     r"$y = \frac{x-1}{x+1}$")
    gs = grid(fig, top=0.845)

    ax = fig.add_subplot(gs[0, 0])
    xv = np.linspace(-0.95, 1.8, 500)
    ax.plot(xv, np.log1p(xv), color=INK, linewidth=2.8, label=r"$\ln(1+x)$")
    for n, color in [(1, MUTED), (2, ORANGE_LIGHT), (5, AQUA), (10, BLUE), (25, ORANGE)]:
        ps = np.zeros_like(xv)
        for k in range(1, n + 1):
            ps += (-1) ** (k - 1) * xv ** k / k
        ax.plot(xv, ps, color=color, linewidth=1.4, label=f"N = {n}")
    ax.axvline(-1, color=MUTED, linestyle="--")
    ax.axvline(1, color=MUTED, linestyle="--")
    ax.axvspan(1, 1.8, color=ORANGE_LIGHT, alpha=0.25, zorder=0, label="x > 1: diverges")
    ax.set_ylim(-3, 2)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.set_title("Partial sums converge only on (−1, 1]")

    ax = fig.add_subplot(gs[0, 1])
    term_counts = np.arange(1, 41)
    for x, color, label in [(1.0, ORANGE, "Mercator x = 1 (ln 2)"), (0.9, AQUA, "Mercator x = 0.9"),
                            (0.5, BLUE, "Mercator x = 0.5")]:
        ax.semilogy(term_counts, [abs(mercator(x, n) - math.log1p(x)) for n in term_counts], "o-",
                    color=color, markersize=3, linewidth=1.4, label=label)
    art = np.array([abs(artanh_log(2, n) - math.log(2)) for n in term_counts])
    art[art == 0] = np.nan
    ax.semilogy(term_counts, art, "s-", color=PURPLE, markersize=3, linewidth=1.4, label="artanh, y = 1/3 (ln 2)")
    ax.axhline(0.5e-10, color=INK2, linestyle=":", linewidth=1.2, label="ten-decimal target")
    ax.set_xlabel("number of terms N")
    ax.set_ylabel("absolute error (log scale)")
    ax.set_ylim(1e-20, 3)
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=8.5, ncol=2)
    ax.set_title("Digits per term are set by |x| or y²")

    ax = fig.add_subplot(gs[0, 2])
    tc = np.unique(np.logspace(0, 6, 40).astype(int))
    cumulative = np.cumsum((-1.0) ** np.arange(0, 10 ** 6) / np.arange(1, 10 ** 6 + 1))
    errs = np.abs(cumulative[tc - 1] - math.log(2))
    ext = np.logspace(0, 10.5, 100)
    ax.loglog(tc, errs, "o", color=ORANGE, markersize=4, label="Mercator error at x = 1")
    ax.loglog(ext, 1 / (2 * ext), "--", color=INK, linewidth=1.2, label="1/(2N)")
    ax.axhline(0.5e-10, color=INK2, linestyle=":")
    ax.axvline(1e10, color=INK2, linestyle=":")
    ax.annotate(r"ten decimals: $N \approx 10^{10}$", (1e10, 0.5e-10), textcoords="offset points",
                xytext=(-8, 10), ha="right", fontsize=9.5, color=INK2)
    ax.set_xlabel("terms N (log scale)")
    ax.set_ylabel("absolute error (log scale)")
    ax.grid(True, which="major")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title(r"$1 - \frac{1}{2} + \frac{1}{3} - \cdots$: one digit per 10× work")

    ax = fig.add_subplot(gs[1, 0])
    xv = np.logspace(-2, 2, 400)
    ax.plot(xv, (xv - 1) / (xv + 1), color=AQUA, linewidth=2.3, label=r"$y = (x-1)/(x+1)$")
    for mx, off in [(0.5, (8, -6)), (2, (8, -14)), (3, (8, -14)), (10, (8, -16))]:
        my = (mx - 1) / (mx + 1)
        ax.plot(mx, my, "o", color=INK, markersize=5)
        ax.annotate(f"x = {mx:g}: y = {Fraction(mx).limit_denominator(20) - 1}/{Fraction(mx).limit_denominator(20) + 1}"
                    if False else f"x = {mx:g}: y = {my:.3f}", (mx, my), textcoords="offset points",
                    xytext=off, ha="left" if off[0] > 0 else "right", fontsize=9.5, color=INK2)
    ax.axhline(1, color=MUTED, linestyle="--")
    ax.axhline(-1, color=MUTED, linestyle="--")
    ax.set_xscale("log")
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("x (log scale)")
    ax.set_ylabel("y")
    ax.grid(True, which="major")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Every x > 0 lands strictly inside (−1, 1)")

    panel(fig.add_subplot(gs[1, 1]), r"$\ln 2$ to ten digits by hand (y = 1/3)", [
        r"terms $\frac{2}{(2j+1)\,3^{2j+1}}$, each about 1/9 of the last:",
        "`" + ", ".join(f"{t:.10f}" for t in terms_ln2[:3]) + ",`",
        "`" + ", ".join(f"{t:.10f}" for t in terms_ln2[3:6]) + ",`",
        "`" + ", ".join(f"{t:.10f}" for t in terms_ln2[6:9]) + ",`",
        "`" + f"{terms_ln2[9]:.10f}" + "`",
        f"sum of 10 terms = {float(exact):.12f}",
        f"ln 2            = {math.log(2):.12f}",
        "",
        r"$\ln 10 = 3\ln 2 + \ln\frac{5}{4}$ (y = 1/9): " + f"{ln10:.10f}",
    ], fontsize=12, spacing=0.085)

    panel(fig.add_subplot(gs[1, 2]), "Why x = 1 is so slow", [
        "Partial sums of 1 − 1/2 + 1/3 − … zigzag:",
        "`" + "  ".join(f"{p:.4f}" for p in partials_x1[:5]) + "`",
        "`" + "  ".join(f"{p:.4f}" for p in partials_x1[5:8]) + "  …  " + f"{partials_x1[9]:.4f}`",
        "S₇ and S₈ are the 4-strip Riemann sums of ln 2.",
        r"Error after N terms ≈ 1/(2N): $10^{10}$ terms for ten",
        "decimals. Radius 1 = distance to the blow-up of",
        r"$\ln(1 + x)$ at x = −1, so x = 2 diverges too.",
        "Fix: make the argument small — y = 1/3 gives",
        "ten digits in ten terms.",
    ], fontsize=12, spacing=0.088)
    save_01(fig, name="04_series_for_the_logarithm.png")


# ================================================================ 01/05
def sheet_01_05_harmonic_numbers():
    euler_gamma = float(np.euler_gamma)
    exact_h = [sum(Fraction(1, k) for k in range(1, n + 1)) for n in range(1, 11)]
    assert [str(h) for h in exact_h[:6]] == ["1", "3/2", "11/6", "25/12", "137/60", "49/20"]
    assert exact_h[3] / 2 == Fraction(25, 24)

    targets, books = [1, 2, 3, 4, 5], []
    harmonic, n = 0.0, 0
    for overhang in targets:
        while harmonic < 2 * overhang:
            n += 1
            harmonic += 1 / n
        books.append(n)
    estimates = [math.exp(2 * d - euler_gamma) for d in targets]
    assert books == [4, 31, 227, 1674, 12367]
    assert [f"{e:.2f}" for e in estimates] == ["4.15", "30.65", "226.51", "1673.69", "12366.97"]

    h100 = float(np.sum(1 / np.arange(1, 101)))
    assert f"{h100:.9f}" == "5.187377518"
    h10 = float(exact_h[9])
    gamma_from_10 = h10 - math.log(10) - 1 / 20 + 1 / 1200
    assert f"{gamma_from_10:.7f}" == "0.5772165"

    # 4.5 coupon collector, same rng and draw order as the notebook
    rng = np.random.default_rng(seed=1683)

    def boxes_until_complete(number_of_toys):
        seen = np.zeros(number_of_toys, dtype=bool)
        boxes = distinct = 0
        while distinct < number_of_toys:
            toy = rng.integers(number_of_toys)
            boxes += 1
            if not seen[toy]:
                seen[toy] = True
                distinct += 1
        return boxes

    coupon = []
    for toys in [6, 20, 50]:
        mean = np.mean([boxes_until_complete(toys) for _ in range(2000)])
        exact_expectation = toys * np.sum(1 / np.arange(1, toys + 1))
        log_estimate = toys * (math.log(toys) + euler_gamma)
        coupon.append((toys, mean, exact_expectation, log_estimate))
    assert [f"{c[1]:.2f}" for c in coupon] == ["14.63", "72.04", "224.94"]
    assert [f"{c[2]:.2f}" for c in coupon] == ["14.70", "71.95", "224.96"]
    assert [f"{c[3]:.2f}" for c in coupon] == ["14.21", "71.46", "224.46"]

    fig = new_figure(r"$H_n$ grows like $\ln n$; the gap settles at $\gamma = 0.5772\ldots$",
                     r"$\ln(n+1) \leq H_n = 1 + \frac{1}{2} + \cdots + \frac{1}{n} \leq 1 + \ln n$"
                     r"      $H_n = \ln n + \gamma + \frac{1}{2n} - \cdots$      $H \approx \ln n + \gamma \;\Leftrightarrow\; n \approx e^{H - \gamma}$")
    gs = grid(fig, top=0.845)

    # --- book stack
    ax = fig.add_subplot(gs[0, 0])
    bare(ax)
    nb, bh = 8, 0.18
    right_edges = [sum(1 / (2 * j) for j in range(k + 1, nb + 1)) + 1 / (2 * k) for k in range(1, nb + 1)]
    ax.add_patch(Rectangle((-2.0, -0.5), 2.0, 0.5, color=MUTED, alpha=0.5))
    for k, re in enumerate(right_edges, start=1):
        level = nb - k
        ax.add_patch(Rectangle((re - 1, level * bh), 1, bh, facecolor=BLUE, edgecolor=SURFACE, linewidth=1.5))
        ax.text(re + 0.04, level * bh + bh / 2, f"+1/{2 * k}", va="center", fontsize=8.5, color=INK2)
    ax.plot([0, 0], [-0.5, nb * bh + 0.15], color=INK, linestyle=":", linewidth=1)
    top = right_edges[0]
    arrow(ax, (0, nb * bh + 0.12), (top, nb * bh + 0.12), ORANGE)
    ax.text(top / 2, nb * bh + 0.2, f"overhang $H_8/2$ = {top:.3f}", ha="center", fontsize=10.5, color=ORANGE)
    ax.text(-1.0, -0.3, "table", ha="center", fontsize=10.5, color=SURFACE)
    ax.set_xlim(-2.05, 1.9)
    ax.set_ylim(-0.55, nb * bh + 0.45)
    ax.set_aspect("equal")
    ax.set_title("Book k from the top sticks out 1/(2k)")

    # --- staircase
    ax = fig.add_subplot(gs[0, 1])
    nn = 8
    xv = np.linspace(0.6, nn + 1.3, 500)
    for k in range(1, nn + 1):
        ax.bar(k, 1 / k, width=1, align="edge", color=BLUE_LIGHT, edgecolor=BLUE, linewidth=0.8)
        strip = np.linspace(k, k + 1, 50)
        ax.fill_between(strip, 1 / strip, 1 / k, color=ORANGE, alpha=0.85)
    ax.plot(xv, 1 / xv, color=INK, linewidth=2, label="y = 1/x")
    ax.bar([], [], color=BLUE_LIGHT, label=r"bars $1/k$ on $[k, k+1]$: total $H_8$")
    ax.fill_between([], [], color=ORANGE, label=r"curvy triangles → $\gamma$")
    ax.set_xlim(0.6, nn + 1.3)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("x")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title(r"Staircase vs hyperbola: $H_n \geq \ln(n+1)$")

    # --- H_n on a log axis
    ax = fig.add_subplot(gs[0, 2])
    nv = np.arange(1, 10 ** 5 + 1)
    hn = np.cumsum(1 / nv)
    ax.fill_between(nv, np.log(nv + 1), 1 + np.log(nv), color=GRID, alpha=0.9, label=r"sandwich $[\ln(n+1),\, 1 + \ln n]$")
    ax.plot(nv, hn, color=BLUE, linewidth=2.6, label=r"$H_n$")
    ax.plot(nv, np.log(nv) + euler_gamma, color=ORANGE, linestyle="--", linewidth=1.6, label=r"$\ln n + \gamma$")
    ax.plot(nv, np.log(nv), color=INK, linestyle=":", linewidth=1.4, label=r"$\ln n$")
    ax.set_xscale("log")
    ax.set_xlabel("n (log scale)")
    ax.grid(True, which="major")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title(r"On a log axis $H_n$ is a straight line")

    # --- squeezing gamma
    ax = fig.add_subplot(gs[1, 0])
    above = hn - np.log(nv) - euler_gamma
    below = euler_gamma - (hn - np.log(nv + 1))
    ax.loglog(nv, above, color=BLUE, linewidth=2, label=r"$H_n - \ln n - \gamma$  (from above)")
    ax.loglog(nv, below, color=AQUA, linewidth=2, label=r"$\gamma - (H_n - \ln(n+1))$  (from below)")
    ax.loglog(nv, 1 / (2 * nv), color=INK, linestyle=":", linewidth=1.5, label="1/(2n)")
    ax.set_xlabel("n (log scale)")
    ax.set_ylabel(r"distance from $\gamma$ (log scale)")
    ax.grid(True, which="major")
    ax.legend(loc="lower left", fontsize=9.5)
    ax.set_title(r"Two monotone sequences squeeze $\gamma$")

    # --- coupon collector
    ax = fig.add_subplot(gs[1, 1])
    pos = np.arange(3)
    width = 0.26
    for j, (label, color) in enumerate([("simulated mean (2000 runs, seed 1683)", BLUE),
                                        (r"$n H_n$", ORANGE), (r"$n(\ln n + \gamma)$", AQUA)]):
        vals = [c[j + 1] for c in coupon]
        ax.bar(pos + (j - 1) * width, vals, width=width - 0.03, color=color, label=label, zorder=2)
        for p, v in zip(pos, vals):
            ax.text(p + (j - 1) * width, v * 1.08, f"{v:.1f}", ha="center", fontsize=8.5, color=INK, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(5, 2000)
    ax.set_xticks(pos, [f"n = {c[0]} toys" for c in coupon])
    ax.set_ylabel("boxes to complete the set (log scale)")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(r"Coupon collector: expected boxes $n H_n$")

    rows = [f"`{d:>4}  {b:>7,}  {e:>10,.2f}`" for d, b, e in zip(targets, books, estimates)]
    panel(fig.add_subplot(gs[1, 2]), "How many books for overhang d?", [
        r"smallest n with $H_n \geq 2d$, against $e^{2d - \gamma}$:",
        "`   d    books  e^(2d − γ)`"] + rows + [
        r"Each extra unit of $H_n$ costs a factor $e$ more terms:",
        r"overhang 10 needs about $e^{20-\gamma} \approx 2.7 \times 10^8$ books.",
        r"$\gamma$ from ten terms:",
        r"   $H_{10} - \ln 10 - 1/20 + 1/1200$" + f" = {gamma_from_10:.7f}",
    ], fontsize=12, spacing=0.077)
    save_01(fig, name="05_harmonic_numbers_and_euler_mascheroni.png")


if __name__ == "__main__":
    sheet_00_01_why_logarithms_exist()
    sheet_00_02_logarithm_as_inverse()
    sheet_00_03_laws_of_logarithms()
    sheet_00_04_change_of_base()
    sheet_01_01_compound_interest_and_e()
    sheet_01_02_ln_as_area()
    sheet_01_03_derivatives()
    sheet_01_04_series()
    sheet_01_05_harmonic_numbers()
