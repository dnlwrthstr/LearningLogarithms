#!/usr/bin/env python3
"""Concept sheets for modules 04 (logarithms in calculus) and 05 (information and probability).

One page per notebook that summarizes its central idea. Every number is computed here from the
same functions, parameters, sample texts and random seeds (in the same draw order) the notebook
uses, and is checked against the notebook's printed outputs with asserts.
Images are written to modules/04_logarithms_in_calculus/img/ and
modules/05_information_and_probability/img/ (one PNG per notebook, named after it).

Run from anywhere:  /opt/anaconda3/bin/python scripts/make_modules_04_05_concept_images.py
"""
import collections
import math
import zlib
from functools import partial

import numpy as np
from matplotlib.patches import Rectangle

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, CARD, BLUE, BLUE_LIGHT, ORANGE, AQUA, GRID,
                       new_figure, grid, trapezoid)
from viz_style import save as _save

MODULE_04 = REPO / "modules" / "04_logarithms_in_calculus"
MODULE_05 = REPO / "modules" / "05_information_and_probability"
save_04 = partial(_save, folder=MODULE_04 / "img")
save_05 = partial(_save, folder=MODULE_05 / "img")

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#a9e3cd"
PURPLE = "#7b5cc4"


# ================================================================ helpers
def panel(ax, heading, lines, fontsize=12.5, top=1.0, spacing=0.085):
    """Text card: bold heading, one entry per line (mathtext allowed). Lines starting with a
    backtick are set in monospace; a line holding \\frac gets display size and extra height."""
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


def log_grid(ax):
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.35, linewidth=0.6)


def close(a, b, tol):
    return abs(a - b) <= tol


def entropy_bits(probabilities):
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]
    return float(np.sum(p * -np.log2(p)))


# ================================================================ 04 / 01
def sheet_04_01_logarithmic_differentiation():
    # 4.2 finite differences against x^x (ln x + 1) and the two naive rules
    def central_difference(function, x, h=1e-6):
        return (function(x + h) - function(x - h)) / (2 * h)

    table = {}
    for x in [0.5, 1.0, 2.0, 3.0]:
        numeric = central_difference(lambda t: t**t, x)
        formula = x**x * (math.log(x) + 1)
        assert math.isclose(numeric, formula, rel_tol=1e-6)
        table[x] = (formula, x**x, x**x * math.log(x))
    assert [f"{table[x][0]:.5f}" for x in table] == ["0.21698", "1.00000", "6.77259", "56.66253"]
    assert f"{table[2.0][1]:.4f}" == "4.0000" and f"{table[2.0][2]:.4f}" == "2.7726"

    # 4.3 the messy quotient at x = 1
    def g(x):
        return x**2 * (x + 1)**3 / math.sqrt(x**2 + 1)

    log_rate_at_1 = 2 / 1 + 3 / 2 - 1 / 2
    g_prime_1 = g(1.0) * log_rate_at_1
    assert log_rate_at_1 == 3 and f"{g(1.0):.6f}" == "5.656854" and f"{g_prime_1:.5f}" == "16.97056"
    assert close((g(1 + 1e-6) - g(1 - 1e-6)) / 2e-6, g_prime_1, 1e-6)

    # 4.5 the minimum of x^x
    grid_x = np.linspace(0.01, 2, 200_001)
    values = grid_x**grid_x
    i_min = np.argmin(values)
    assert round(grid_x[i_min], 5) == 0.36788 and round(values[i_min], 5) == 0.6922
    assert f"{math.exp(-1 / math.e):.4f}" == "0.6922"

    # 4.6 relative rates of a product add
    t = np.linspace(0, 20, 2001)
    revenue = 10 * np.exp(0.03 * t) * 500 * np.exp(0.02 * t)
    assert np.allclose((np.gradient(revenue, t) / revenue)[1:-1], 0.05, atol=1e-5)
    assert round(np.polyfit(t, np.log(revenue), 1)[0], 10) == 0.05
    yearly = 1.02 / 1.005 - 1
    assert f"{yearly:.6f}" == "0.014925"

    fig = new_figure("Take ln first: products become sums, so derivatives become sums",
                     r"$f\,' = f\cdot(\ln f)\,'$.   $\ln x^x = x\ln x$  gives  "
                     r"$\frac{d}{dx}x^x = x^x(\ln x + 1)$ = power-rule part $x^x$ + exponential-rule part $x^x\ln x$")
    gs = grid(fig, top=0.83)

    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(0.005, 2, 800)
    ax.plot(x, x**x, color=BLUE, linewidth=2.4, label="$x^x$")
    x_min = 1 / math.e
    ax.plot(x_min, x_min**x_min, "o", color=ORANGE, markersize=9, zorder=5)
    ax.annotate(f"minimum at $x = 1/e$ = {x_min:.4f}\nvalue $e^{{-1/e}}$ = {x_min**x_min:.4f}",
                xy=(x_min, x_min**x_min), xytext=(0.35, 2.3), fontsize=10.5, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.axhline(1, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(0.03, 1.08, r"$x^x \to 1$ as $x \to 0^+$", fontsize=9.5, color=INK2)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("x")
    ax.set_ylabel("$x^x$")
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.set_title(r"Minimise $\ln f = x\ln x$: $\ln x + 1 = 0$")

    ax = fig.add_subplot(gs[0, 1])
    x = np.linspace(0.05, 2.2, 600)
    ax.plot(x, x**x, "--", color=BLUE, linewidth=1.8, label=r"exponent frozen: $x\cdot x^{x-1} = x^x$")
    ax.plot(x, x**x * np.log(x), "--", color=AQUA, linewidth=1.8, label=r"base frozen: $x^x\ln x$")
    ax.plot(x, x**x * (np.log(x) + 1), color=INK, linewidth=2.6, label="true derivative = sum")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    for value, color in [(table[2.0][1], BLUE), (table[2.0][2], AQUA), (table[2.0][0], INK)]:
        ax.plot(2.0, value, "o", color=color, zorder=5)
    ax.text(1.97, table[2.0][0], f"{table[2.0][0]:.4f} ", ha="right", va="center", fontsize=10)
    ax.text(0.15, 5.0, "central differences at x = 0.5, 1, 2, 3:\n" +
            ",  ".join(f"{table[k][0]:.5f}" for k in table), fontsize=9.5, color=INK2)
    ax.text(1.97, table[2.0][1] + 0.5, f"{table[2.0][1]:.0f}", ha="right", va="center", fontsize=10, color=BLUE)
    ax.text(1.97, table[2.0][2] - 0.6, f"{table[2.0][2]:.4f}", ha="right", va="center", fontsize=10, color=AQUA)
    ax.set_ylim(-1.5, 12)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Both naive rules are half the answer")

    panel(fig.add_subplot(gs[0, 2]), "The method, four steps", [
        r"1. $y = f(x)$, take $\ln$ of both sides",
        r"2. log laws: products $\to$ sums, powers $\to$ multiples",
        r"3. differentiate: the left side is $y\,'/y$",
        r"4. multiply back by $y$",
        "",
        r"Example: $g = \frac{x^2(x+1)^3}{\sqrt{x^2+1}}$",
        r"$\ln g = 2\ln x + 3\ln(x+1) - \frac{1}{2}\ln(x^2+1)$",
        rf"$g\,'/g$ at 1 = 2 + 3/2 − 1/2 = {log_rate_at_1:.0f}",
        rf"$g\,'(1) = 3\,g(1) = 12\sqrt{{2}}$ = {g_prime_1:.4f}",
    ], fontsize=11.8, spacing=0.077)

    ax = fig.add_subplot(gs[1, 0])
    x = np.linspace(0.3, 3, 400)
    terms = [(2 / x, r"$2/x$  from $x^2$", BLUE), (3 / (x + 1), r"$3/(x+1)$  from $(x+1)^3$", AQUA),
             (-x / (x**2 + 1), r"$-x/(x^2+1)$  from $1/\sqrt{x^2+1}$", ORANGE)]
    for values, label, color in terms:
        ax.plot(x, values, "--", color=color, linewidth=1.8, label=label)
    total = sum(values for values, _, _ in terms)
    ax.plot(x, total, color=INK, linewidth=2.6, label=r"$g\,'/g$ = sum")
    for value, color in [(2.0, BLUE), (1.5, AQUA), (-0.5, ORANGE), (3.0, INK)]:
        ax.plot(1.0, value, "o", color=color, zorder=5)
    ax.text(1.06, 3.15, "3", fontsize=11, weight="bold")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(1, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_ylim(-1, 8)
    ax.set_xlabel("x")
    ax.set_ylabel("logarithmic derivative")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Every factor contributes one simple term")

    ax = fig.add_subplot(gs[1, 1])
    t = np.linspace(0, 40, 400)
    for values, label, color in [(5000 * np.exp(0.05 * t), r"$5000\,e^{0.05t}$", BLUE),
                                 (50 * np.exp(0.05 * t), r"$50\,e^{0.05t}$", AQUA),
                                 (50 * np.exp(0.08 * t), r"$50\,e^{0.08t}$", ORANGE)]:
        ax.plot(t, values, color=color, linewidth=2.2, label=label)
    ax.set_yscale("log")
    log_grid(ax)
    ax.text(21, 150, "parallel: same slope 0.05", fontsize=10, color=AQUA, rotation=14)
    ax.set_xlabel("t")
    ax.set_ylabel("value (log scale)")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title(r"On a log axis the slope is $f\,'/f$")

    panel(fig.add_subplot(gs[1, 2]), r"$f\,'/f$ is the relative rate", [
        r"$(\ln f)\,' = f\,'/f$: change as a fraction of the value.",
        r"Products add:  $R = pq \Rightarrow R\,'/R = p\,'/p + q\,'/q$",
        "  price +3%, quantity +2%  →  revenue +5%",
        "  (np.gradient of R/R: 0.05 at every t)",
        r"Quotients subtract: GDP per person",
        "  grows 2.0% − 0.5% = 1.5%",
        f"  (yearly factors 1.02/1.005: {100 * yearly:.2f}%)",
        r"Powers multiply:  $A = s^2 \Rightarrow A\,'/A = 2\,s\,'/s$",
        "",
        r"Variable powers: $\frac{d}{dx}u^v = u^v(v\,'\ln u + v\,u\,'/u)$",
    ], fontsize=12, spacing=0.083)
    save_04(fig, name="01_logarithmic_differentiation.png")


# ================================================================ 04 / 02
def sheet_04_02_integrals_that_produce_logs():
    # 4.1 the gap in the power rule
    rows = {h: (2.0**h - 1) / h for h in [1, 0.1, 0.01, 0.001, 1e-4, -0.01]}
    assert f"{rows[0.1]:.10f}" == "0.7177346254" and f"{rows[0.01]:.10f}" == "0.6955550057"
    assert f"{rows[0.01] - math.log(2):.2e}" == "2.41e-03" and f"{0.01 / 2 * math.log(2)**2:.2e}" == "2.40e-03"

    # 4.3 the definite integrals, numerically
    def numeric_integral(function, a, b, points=200_001):
        xs = np.linspace(a, b, points)
        return trapezoid(function(xs), xs)

    cases = [
        (r"$\int_1^{e^3} dx/x$", lambda s: 1 / s, 1, math.e**3, 3.0),
        (r"$\int_{-3}^{-1} dx/x$", lambda s: 1 / s, -3, -1, -math.log(3)),
        (r"$\int_0^{\pi/4} \tan x\,dx$", np.tan, 0, math.pi / 4, 0.5 * math.log(2)),
        (r"$\int_0^{\pi/4} \sec x\,dx$", lambda s: 1 / np.cos(s), 0, math.pi / 4, math.log(math.sqrt(2) + 1)),
        (r"$\int_2^3 \frac{dx}{x^2-1}$", lambda s: 1 / (s**2 - 1), 2, 3, 0.5 * math.log(1.5)),
        (r"$\int_1^e \ln x\,dx$", np.log, 1, math.e, 1.0),
        (r"$\int_0^1 \frac{2x\,dx}{x^2+1}$", lambda s: 2 * s / (s**2 + 1), 0, 1, math.log(2)),
    ]
    for _, function, a, b, exact in cases:
        assert math.isclose(numeric_integral(function, a, b), exact, rel_tol=1e-8)
    assert [f"{c[4]:.6f}" for c in cases] == ["3.000000", "-1.098612", "0.346574", "0.881374",
                                              "0.202733", "1.000000", "0.693147"]

    # 4.5 integrating across the asymptote
    totals = [math.log(eps) + math.log(2) - math.log(delta)
              for eps, delta in [(1e-3, 1e-3), (1e-6, 1e-6), (1e-6, 2e-6), (1e-6, 1e-9)]]
    assert [f"{v:.4f}" for v in totals] == ["0.6931", "0.6931", "0.0000", "7.6009"]

    fig = new_figure("The logarithm fills the one gap in the power rule",
                     r"$\int x^n dx = \frac{x^{n+1}}{n+1}$ fails only at $n = -1$:   $\frac{x^h - 1}{h} \to \ln x$,   "
                     r"$\int \frac{dx}{x} = \ln|x| + C$,   $\int \frac{f\,'}{f}dx = \ln|f| + C$,   "
                     r"$\int_a^b \frac{dx}{x} = \ln\frac{b}{a}$")
    gs = grid(fig, top=0.83)

    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(0.05, 5, 500)
    colors = {1: "#1c5cab", 0.5: "#3987e5", 0.2: "#86b6ef", 0.05: BLUE_LIGHT, -0.2: ORANGE_LIGHT, -0.5: ORANGE}
    for h, color in colors.items():
        ax.plot(x, (x**h - 1) / h, color=color, linewidth=1.6, label=f"h = {h}")
    ax.plot(x, np.log(x), color=INK, linewidth=2.8, label=r"$h \to 0$:  $\ln x$")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(1, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_ylim(-4, 4)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$(x^h - 1)/h$ = area under $t^{h-1}$ from 1")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.set_title(r"Antiderivatives of $x^{h-1}$ squeeze onto $\ln x$")

    ax = fig.add_subplot(gs[0, 1])
    x = np.linspace(0.6, 22, 600)
    ax.plot(x, 1 / x, color=INK, linewidth=2.2, label="y = 1/x")
    edges = [1, math.e, math.e**2, math.e**3]
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        xs = np.linspace(a, b, 200)
        area = numeric_integral(lambda s: 1 / s, a, b)
        assert close(area, 1.0, 1e-9)
        ax.fill_between(xs, 1 / xs, color=[BLUE, AQUA, ORANGE][k], alpha=0.55, linewidth=0)
        middle = math.sqrt(a * b)
        ax.text(middle, 0.4 / middle, "1", ha="center", va="center", fontsize=13, color=INK, weight="bold")
    ax.text(8, 0.62, "1 to 1000: ln 1000 = 6.91\n1000 to 2000: ln 2 = 0.69", fontsize=10.5, color=INK2)
    ax.text(8, 0.85, "each shaded band has area 1:\n$\\int_{e^k}^{e^{k+1}} dx/x = 1$", fontsize=11, color=INK2)
    ax.set_xticks(edges, ["1", "e", "$e^2$", "$e^3$"])
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.set_title("Equal ratios of endpoints, equal areas")

    panel(fig.add_subplot(gs[0, 2]), "Closing the gap, numerically", [
        r"$x^h = e^{h\ln x}$  gives  $(x^h - 1)/h = \ln x + \frac{1}{2}h(\ln x)^2 + \dots$",
        r"At $x = 2$  ($\ln 2$ = 0.6931472):",
        "`     h      (2^h - 1)/h       error",
        *[f"`{h:>6}   {v:.10f}   {v - math.log(2):+.2e}" for h, v in rows.items() if h != 1e-4],
        r"error $\approx \frac{1}{2}h(\ln 2)^2$: linear in $h$, either sign",
    ], fontsize=12, spacing=0.075)

    ax = fig.add_subplot(gs[1, 0])
    x_pos = np.linspace(0.12, 4, 300)
    ax.plot(x_pos, 1 / x_pos, color=INK, linewidth=2.2)
    ax.plot(-x_pos, -1 / x_pos, color=INK, linewidth=2.2, label="y = 1/x")
    fill = np.linspace(1, 3, 100)
    ax.fill_between(fill, 1 / fill, color=AQUA, alpha=0.5, label=r"$\int_1^3$ = ln 3 = 1.0986")
    ax.fill_between(-fill, -1 / fill, color=ORANGE, alpha=0.5, label=r"$\int_{-3}^{-1}$ = −ln 3 = −1.0986")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title(r"$\ln|x|$ covers $x < 0$; never integrate across 0")

    ax = fig.add_subplot(gs[1, 1])
    x = np.linspace(-1.5, 4.6, 2000)
    near = np.abs(np.cos(x)) < 0.02
    ax.plot(x, np.where(near, np.nan, np.tan(x)), color=BLUE, linewidth=1.8, label=r"$\tan x$")
    ax.plot(x, np.where(near, np.nan, -np.log(np.abs(np.cos(x)))), color=ORANGE, linewidth=2.4,
            label=r"$-\ln|\cos x|$")
    shade = np.linspace(0, math.pi / 4, 100)
    ax.fill_between(shade, np.tan(shade), color=BLUE, alpha=0.3, label=r"area on $[0, \pi/4]$ = $\frac{1}{2}\ln 2$")
    ax.plot(math.pi / 4, 0.5 * math.log(2), "o", color=ORANGE, zorder=5)
    for asymptote in [-math.pi / 2, math.pi / 2, 3 * math.pi / 2]:
        ax.axvline(asymptote, color=MUTED, linestyle=":", linewidth=1)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=1, facecolor=SURFACE, frameon=True, edgecolor="none")
    ax.set_title(r"$\int \tan x\,dx = -\ln|\cos x| + C$  ($f\,'/f$, $f = \cos$)")

    panel(fig.add_subplot(gs[1, 2]), "Logarithms as answers", [
        "`integrand      interval     value",
        *[f"`{label:<13}  {interval:<11}  {exact:.6f}" for label, interval, exact in
          zip(["tan x", "sec x", "1/(x^2 - 1)", "ln x", "2x/(x^2 + 1)"],
              ["[0, pi/4]", "[0, pi/4]", "[2, 3]", "[1, e]", "[0, 1]"], [c[4] for c in cases[2:]])],
        r"sec: times $(\sec x + \tan x)$ over itself, then $f\,'/f$",
        r"partial fractions: $\frac{1}{x^2-1} = \frac{1/2}{x-1} - \frac{1/2}{x+1}$",
        r"Cut $[-\varepsilon, \delta]$ from $[-1, 2]$: $\ln 2 + \ln(\varepsilon/\delta)$",
        "   = 0.6931, 0.0000 or 7.6009: no single value",
    ], fontsize=11.5, spacing=0.074)
    save_04(fig, name="02_integrals_that_produce_logs.png")


# ================================================================ 04 / 03
def sheet_04_03_exponential_growth_and_decay():
    half_life = 5730.0
    fractions = [1.0, 0.9, 0.5, 0.30, 0.25, 0.10, 0.01, 0.002]
    ages = [half_life * math.log2(1 / f) for f in fractions]
    assert [f"{a:,.0f}" for a in ages] == ["0", "871", "5,730", "9,953", "11,460", "19,035", "38,069", "51,374"]
    k_c14 = -math.log(2) / half_life
    assert f"{k_c14:.4e}" == "-1.2097e-04"

    exact_doubling = {r: math.log(2) / math.log(1 + r / 100) for r in [1, 2, 4, 6, 8, 10, 12, 15, 20]}
    assert [f"{exact_doubling[r]:.3f}" for r in (1, 6, 8, 20)] == ["69.661", "11.896", "9.006", "3.802"]
    # where 72/R meets the exact yearly doubling time (the notebook says: near 8%)
    lo, hi = 2.0, 15.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 72 / mid - math.log(2) / math.log(1 + mid / 100) > 0:
            lo = mid
        else:
            hi = mid
    crossover = (lo + hi) / 2
    assert 7.5 < crossover < 8.1

    room, start = 20.0, 90.0
    k_coffee = math.log((start - room) / (70.0 - room)) / 10
    t50 = math.log((start - room) / (50 - room)) / k_coffee
    t25 = math.log((start - room) / (25 - room)) / k_coffee
    assert f"{k_coffee:.5f}" == "0.03365" and f"{math.log(2) / k_coffee:.2f}" == "20.60"
    assert f"{t50:.2f}" == "25.18" and f"{t25:.1f}" == "78.4"
    assert f"{room + (start - room) * math.exp(-k_coffee * 60):.2f}" == "29.30"

    rng = np.random.default_rng(seed=7)
    times = np.arange(0, 21, 2.0)
    measurements = 200.0 * np.exp(-0.15 * times) * np.exp(rng.normal(0, 0.05, size=times.size))
    slope, intercept = np.polyfit(times, np.log(measurements), 1)
    assert f"{slope:.4f}" == "-0.1493" and f"{math.exp(intercept):.2f}" == "197.22"
    assert f"{math.log(2) / -slope:.3f}" == "4.643"

    fig = new_figure("Every “how long?” question about exponential change is a logarithm",
                     r"$\frac{dy}{dt} = ky \Rightarrow \ln|y| = kt + C \Rightarrow y = y_0e^{kt}$.   "
                     r"Half-life / doubling time $T = \frac{\ln 2}{|k|}$;   elapsed time $t = T_{1/2}\,\log_2\frac{y_0}{y}$")
    gs = grid(fig, top=0.83)

    ax = fig.add_subplot(gs[0, 0])
    years = np.linspace(0, 6 * half_life, 600)
    ax.plot(years, 0.5 ** (years / half_life), color=BLUE, linewidth=2.4)
    for n in range(1, 6):
        ax.plot(n * half_life, 0.5**n, "o", color=BLUE, zorder=5)
        ax.axvline(n * half_life, color=MUTED, linestyle=":", linewidth=0.9)
    ax.plot(ages[3], 0.30, "s", color=ORANGE, markersize=9, zorder=6)
    ax.annotate(f"charcoal: 30% left\n→ {ages[3]:,.0f} years", xy=(ages[3], 0.30), xytext=(13500, 0.45),
                fontsize=10.5, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.set_yscale("log")
    ax.set_yticks([1, 0.5, 0.25, 0.125, 0.0625, 0.03125], ["1", "1/2", "1/4", "1/8", "1/16", "1/32"])
    ax.minorticks_off()
    ax.grid(True)
    ax.set_xticks([0, 5730, 11460, 17190, 22920, 28650], ["0", "5,730", "11,460", "17,190", "22,920", "28,650"],
                  fontsize=9)
    ax.set_xlabel("years since death")
    ax.set_ylabel("fraction of C-14 left (log scale)")
    ax.set_title("Each half-life is the same step on a log axis")

    ax = fig.add_subplot(gs[0, 1])
    rate = np.linspace(1, 25, 300)
    exact = np.log(2) / np.log(1 + rate / 100)
    ax.plot(rate, 72 / rate - exact, color=ORANGE, linewidth=2.4, label="rule of 72:  72/R − exact")
    ax.plot(rate, 69.3 / rate - exact, color=BLUE, linewidth=2.4, label="rule of 69.3:  69.3/R − exact")
    ax.plot(rate, 69.3 / rate + 0.35 - exact, color=AQUA, linewidth=2.0, linestyle="--",
            label="69.3/R + 0.35 − exact")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.plot(crossover, 0, "o", color=ORANGE, zorder=5)
    ax.annotate(f"72 is exact at R ≈ {crossover:.1f}%", xy=(crossover, 0), xytext=(11, 0.9),
                fontsize=10.5, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.set_xlabel("yearly rate R (%)")
    ax.set_ylabel(r"error vs exact $\ln 2/\ln(1 + r)$ (years)")
    ax.set_ylim(-0.6, 2.6)
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Doubling-time rules against the truth")

    panel(fig.add_subplot(gs[0, 2]), "Radiocarbon: how many halvings?", [
        r"$T_{1/2}$ = 5,730 years,  $k = -\ln 2/5730$",
        f"   = {k_c14:.4e} per year",
        "`fraction left      age (years)",
        *[f"`{f:>9.3f}      {a:>10,.0f}" for f, a in zip(fractions[1:], ages[1:])],
        r"1% → 0.2% (factor 5) adds only $\log_2 5$ = 2.3 half-lives",
    ], fontsize=12, spacing=0.074)

    ax = fig.add_subplot(gs[1, 0])
    minutes = np.linspace(0, 120, 400)
    gap = (start - room) * np.exp(-k_coffee * minutes)
    ax.plot(minutes, gap, color=ORANGE, linewidth=2.4, label=f"gap T − 20 °C, slope −k = −{k_coffee:.4f}")
    for t_mark, label in [(0, "90 °C"), (10, "70 °C"), (t50, f"50 °C at {t50:.1f} min"),
                          (t25, f"25 °C at {t25:.1f} min")]:
        value = (start - room) * math.exp(-k_coffee * t_mark)
        ax.plot(t_mark, value, "o", color=INK, zorder=5)
        ax.text(t_mark + 3, value * 1.12, label, fontsize=10)
    ax.set_yscale("log")
    ax.set_yticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.grid(True)
    ax.set_xlabel("minutes")
    ax.set_ylabel(r"$T - T_{env}$ (log scale)")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Newton cooling: the gap is a straight line")

    ax = fig.add_subplot(gs[1, 1])
    fine = np.linspace(0, 20, 200)
    ax.plot(times, measurements, "o", color=BLUE, markersize=7, label="noisy measurements (seed 7)")
    ax.plot(fine, np.exp(intercept + slope * fine), color=ORANGE, linewidth=2.2,
            label=f"fit: {math.exp(intercept):.2f}" + r"$\,e^{" + f"{slope:.4f}" + r"\,t}$")
    ax.plot(fine, 200 * np.exp(-0.15 * fine), "--", color=MUTED, label=r"truth: $200\,e^{-0.15t}$")
    ax.set_yscale("log")
    log_grid(ax)
    ax.set_xlabel("t")
    ax.set_ylabel("y (log scale)")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title(r"Fit a line to $\ln y$: the slope is $k$")

    panel(fig.add_subplot(gs[1, 2]), "Rates, rules and fits", [
        r"Continuous rate $r$: $T_2 = \ln 2/r \approx 69.3/R$",
        r"Yearly: $T_2 = \ln 2/\ln(1+r) \approx 69.3/R + 0.35$",
        f"   6%: exact {exact_doubling[6]:.3f},  72/6 = 12.000",
        f"   8%: exact {exact_doubling[8]:.3f},  72/8 = 9.000",
        f"   1%: exact {exact_doubling[1]:.3f},  72/1 = 72.000",
        "",
        "Coffee 90 → 70 °C in 10 min (room 20 °C):",
        f"   k = ln(70/50)/10 = {k_coffee:.5f} per min",
        f"   gap halves every {math.log(2) / k_coffee:.2f} min",
        f"Fit: k = {slope:.4f} (true −0.15)",
        f"   half-life {math.log(2) / -slope:.3f} (true 4.621)",
    ], fontsize=12, spacing=0.08)
    save_04(fig, name="03_exponential_growth_and_decay.png")


# ================================================================ 04 / 04
def sheet_04_04_stirlings_approximation():
    def stirling(n):
        return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)

    for n, text in [(20, "42.336"), (100, "363.739"), (170, "706.573"), (171, "711.715"), (1000, "5912.128")]:
        assert f"{math.lgamma(n + 1):.3f}" == text
    assert f"{float(math.factorial(170)):.4e}" == "7.2574e+306"
    try:
        float(math.factorial(171))
        raise AssertionError("171! should overflow a float")
    except OverflowError:
        pass

    differences = {n: math.lgamma(n + 1) - stirling(n) for n in [1, 10, 100, 1000]}
    assert [f"{differences[n]:.2e}" for n in differences] == ["8.11e-02", "8.33e-03", "8.33e-04", "8.33e-05"]
    digits = {n: (len(str(math.factorial(n))), stirling(n) / math.log(10)) for n in [10, 52, 100, 500, 1000]}
    assert [d for d, _ in digits.values()] == [7, 68, 158, 1135, 2568]
    assert all(math.floor(s) + 1 == d for d, s in digits.values())
    assert f"{digits[100][1]:.4f}" == "157.9696"
    crude_100 = (100 * math.log(100) - 100) / math.log(10)
    assert f"{crude_100:.2f}" == "156.57" and round(math.exp(0.5 * math.log(200 * math.pi))) == 25

    p_100 = math.comb(100, 50) / 2**100
    assert f"{p_100:.6f}" == "0.079589" and f"{1 / math.sqrt(50 * math.pi):.6f}" == "0.079788"
    stirling_10 = math.sqrt(20 * math.pi) * (10 / math.e) ** 10
    assert f"{stirling_10:.1f}" == "3598695.6" and f"{stirling_10 * (1 + 1 / 120):.1f}" == "3628684.7"
    lower, upper = 10 * math.log(10) - 9, 11 * math.log(11) - 10
    assert (f"{lower:.3f}", f"{math.lgamma(11):.3f}", f"{upper:.3f}", f"{stirling(10):.3f}") == \
           ("14.026", "15.104", "16.377", "15.096")

    fig = new_figure("Stirling: the log of a product is a sum, and a sum is almost an integral",
                     r"$\ln n! = \sum_{k=1}^n \ln k \approx \int_1^n \ln x\,dx$  $\Rightarrow$  "
                     r"$\ln n! \approx n\ln n - n + \frac{1}{2}\ln(2\pi n)$  $\Leftrightarrow$  "
                     r"$n! \approx \sqrt{2\pi n}\,(n/e)^n$,  relative error $\approx \frac{1}{12n}$")
    gs = grid(fig, top=0.83)

    ax = fig.add_subplot(gs[0, 0])
    n = 12
    k = np.arange(2, n + 1)
    x = np.linspace(1, n, 400)
    bars_total, curve_area = float(np.sum(np.log(k))), n * math.log(n) - n + 1
    assert close(bars_total, math.lgamma(13), 1e-9)
    ax.bar(k - 0.5, np.log(k), width=1.0, color=BLUE_LIGHT, edgecolor=BLUE, linewidth=1,
           label=f"bars ln k on [k−1, k]: ln 12! = {bars_total:.3f}")
    ax.plot(x, np.log(x), color=ORANGE, linewidth=2.6, label=f"ln x: area 12 ln 12 − 11 = {curve_area:.3f}")
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0, 3.4)
    ax.set_xlabel("x")
    ax.set_ylabel("ln x")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title(r"$\ln n!$ is a staircase area over $\ln x$")

    ax = fig.add_subplot(gs[0, 1])
    n_values = np.unique(np.logspace(0, 6, 120).astype(int))
    exact = np.array([math.lgamma(m + 1) for m in n_values])
    crude = n_values * np.log(n_values) - n_values
    full = crude + 0.5 * np.log(2 * np.pi * n_values)
    ax.plot(n_values, np.abs(1 - np.exp(crude - exact)), color=ORANGE, linewidth=2.2, label=r"crude $(n/e)^n$")
    ax.plot(n_values, np.abs(np.expm1(full - exact)), color=BLUE, linewidth=3,
            label=r"Stirling $\sqrt{2\pi n}(n/e)^n$")
    ax.plot(n_values, 1 / (12 * n_values), "--", color=INK, linewidth=1.4, label=r"$1/(12n)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_grid(ax)
    ax.set_xlabel("n (log scale)")
    ax.set_ylabel("relative error (log scale)")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title("Relative error: a power law of slope −1")

    panel(fig.add_subplot(gs[0, 2]), "How many digits has n!?", [
        r"digits $= \lfloor \log_{10} n! \rfloor + 1$, with $\log_{10} n! = \ln n!/\ln 10$",
        "`   n   Stirling log10   digits (exact)",
        *[f"`{m:>4}   {s:>13.4f}   {d:>6}" for m, (d, s) in digits.items()],
        f"Crude $n\\ln n - n$ alone gives {crude_100:.2f}:",
        "157 digits, off by a factor √(200π) ≈ 25.",
        "Floats stop at 170!;  math.lgamma(n + 1)",
        r"returns $\ln n!$ and never overflows.",
    ], fontsize=12, spacing=0.078)

    ax = fig.add_subplot(gs[1, 0])
    n_values = np.arange(1, 201)
    digits_factorial = np.array([math.lgamma(m + 1) / math.log(10) for m in n_values])
    ax.plot(n_values, n_values * np.log10(n_values), color=ORANGE, linewidth=2, label=r"$\log_{10} n^n$")
    ax.plot(n_values, digits_factorial, color=BLUE, linewidth=2.8, label=r"$\log_{10} n!$")
    ax.plot(n_values, n_values * math.log10(2), color=AQUA, linewidth=2, label=r"$\log_{10} 2^n$")
    ax.plot(100, digits_factorial[99], "o", color=INK, zorder=5)
    ax.annotate(f"100! ≈ 9.33 × $10^{{157}}$: 158 digits", xy=(100, digits_factorial[99]), xytext=(20, 260),
                fontsize=10.5, arrowprops=dict(arrowstyle="->", color=INK2))
    ax.text(150, 205, r"gap $\approx n\log_{10}e$" + "\n= 0.434 n digits", fontsize=10, color=INK2)
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\log_{10}$ of the number (≈ digits)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title(r"$n!$ sits just below $n^n$: the $(1/e)^n$ factor")

    ax = fig.add_subplot(gs[1, 1])
    half_values = np.unique(np.logspace(0, 5, 40).astype(int))
    exact_half = [math.exp(math.lgamma(2 * m + 1) - 2 * math.lgamma(m + 1) - 2 * m * math.log(2)) for m in half_values]
    ax.plot(2 * half_values, exact_half, "o", color=BLUE, markersize=6, label="exact P(n heads in 2n flips)")
    ax.plot(2 * half_values, 1 / np.sqrt(np.pi * half_values), color=ORANGE, linewidth=2,
            label=r"$1/\sqrt{\pi n}$")
    ax.plot(100, p_100, "s", color=ORANGE, markersize=10, zorder=5)
    ax.annotate(f"100 flips: {p_100:.4f}", xy=(100, p_100), xytext=(300, 0.15), fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_grid(ax)
    ax.set_xlabel("number of flips 2n (log scale)")
    ax.set_ylabel("probability (log scale)")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title(r"Exactly half heads: $\binom{2n}{n}/4^n \approx 1/\sqrt{\pi n}$")

    panel(fig.add_subplot(gs[1, 2]), "Squeeze, correct, check", [
        r"$\ln x$ increasing, so $\ln n!$ lies between the areas",
        r"under $\ln x$ on $[1, n]$ and on $[1, n+1]$:",
        f"n = 10:  {lower:.3f} ≤ {math.lgamma(11):.3f} ≤ {upper:.3f};  Stirling {stirling(10):.3f}",
        r"Trapezoid correction: $+\frac{1}{2}\ln n$; the constant is $\frac{1}{2}\ln 2\pi$",
        r"$\ln n!$ − Stirling ≈ $1/(12n)$:",
        "   n = 10, 100, 1000:  " + ",  ".join(f"{differences[m]:.2e}" for m in [10, 100, 1000]),
        f"10! = 3,628,800;  Stirling {stirling_10:,.0f} (−0.83%)",
        f"   × (1 + 1/120) = {stirling_10 * (1 + 1 / 120):,.0f}",
        f"100 flips, 50 heads: {p_100:.5f} vs 1/√(50π) = {1 / math.sqrt(50 * math.pi):.5f}",
    ], fontsize=11.4, spacing=0.083)
    save_04(fig, name="04_stirlings_approximation.png")


# ================================================================ 04 / 05
def sieve(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, math.isqrt(limit) + 1):
        if is_prime[p]:
            is_prime[p * p::p] = False
    return is_prime


def count_primes_segmented(limit, small_primes, block_size=10**7):
    count = 0
    for block_start in range(0, limit + 1, block_size):
        block_end = min(block_start + block_size, limit + 1)
        block = np.ones(block_end - block_start, dtype=bool)
        if block_start == 0:
            block[:2] = False
        for p in small_primes:
            p = int(p)
            first_multiple = max(p * p, ((block_start + p - 1) // p) * p)
            if first_multiple < block_end:
                block[first_multiple - block_start::p] = False
        count += int(block.sum())
    return count


def Li(x, points=200_001):
    u = np.linspace(math.log(2), math.log(x), points)
    return float(trapezoid(np.exp(u) / u, u))


def sheet_04_05_prime_number_theorem():
    primes = np.flatnonzero(sieve(10**8))

    def prime_count(n):
        return np.searchsorted(primes, n, side="right")

    true_counts = {e: int(prime_count(10**e)) for e in range(1, 9)}
    assert [true_counts[e] for e in range(1, 9)] == [4, 25, 168, 1229, 9592, 78498, 664579, 5761455]
    pi_billion = count_primes_segmented(10**9, primes[primes <= math.isqrt(10**9)])
    assert pi_billion == 50_847_534
    true_counts[9] = pi_billion
    li = {e: Li(10**e) for e in range(2, 10)}
    assert f"{li[6]:,.3f}" == "78,626.504" and f"{li[9]:,.0f}" == "50,849,234"
    ratio_simple = {e: true_counts[e] / (10**e / math.log(10**e)) for e in range(2, 10)}
    ratio_li = {e: true_counts[e] / li[e] for e in range(2, 10)}
    assert [f"{ratio_simple[e]:.4f}" for e in (4, 9)] == ["1.1320", "1.0537"]
    assert [f"{ratio_li[e]:.5f}" for e in (5, 7, 9)] == ["0.99618", "0.99949", "0.99997"]
    assert f"{10**9 / math.log(10**9):,.0f}" == "48,254,942"

    gap_rows = {}
    for e in range(3, 9):
        n = 10**e
        in_window = primes[(primes > n - n // 10) & (primes <= n)]
        gaps = np.diff(in_window)
        gap_rows[e] = (len(in_window), gaps.mean(), math.log(n), int(gaps.max()))
    assert [f"{gap_rows[e][1]:.3f}" for e in range(3, 9)] == ["6.923", "8.757", "11.378", "13.842", "16.106", "18.365"]
    assert gap_rows[8][3] == 210
    assert int(prime_count(10**6) - prime_count(999_000)) == 65

    fig = new_figure("Near n, about one number in ln n is prime",
                     r"Density of primes near $n$ $\approx \frac{1}{\ln n}$,   "
                     r"$\pi(n) \approx \mathrm{Li}(n) = \int_2^n \frac{dt}{\ln t} \sim \frac{n}{\ln n}$,   "
                     r"average gap between primes near $n$ $\approx \ln n$")
    gs = grid(fig, top=0.83)

    ax = fig.add_subplot(gs[0, 0])
    small_n = np.arange(2, 201)
    ax.step(small_n, prime_count(small_n), where="post", color=INK, linewidth=1.8, label=r"$\pi(n)$")
    ax.plot(small_n, small_n / np.log(small_n), color=ORANGE, linewidth=2.2, label=r"$n/\ln n$")
    ax.plot(small_n, [Li(m, 2001) for m in small_n], color=BLUE, linewidth=2.2, label=r"$\mathrm{Li}(n)$")
    ax.set_xlabel("n")
    ax.set_ylabel("number of primes ≤ n")
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.set_title(r"Up to 200: a staircase between two curves")

    ax = fig.add_subplot(gs[0, 1])
    n_values = np.unique(np.logspace(2, 8, 150).astype(np.int64))
    counts = prime_count(n_values)
    ax.plot(n_values, counts / (n_values / np.log(n_values)), color=ORANGE, linewidth=2.2,
            label=r"$\pi(n)\,/\,(n/\ln n)$")
    ax.plot(n_values, 1 + 1 / np.log(n_values), "--", color=ORANGE, alpha=0.6, label=r"$1 + 1/\ln n$")
    ax.plot(n_values, counts / np.array([Li(m, 20001) for m in n_values]), color=BLUE, linewidth=2.4,
            label=r"$\pi(n)\,/\,\mathrm{Li}(n)$")
    ax.plot(10**9, ratio_simple[9], "o", color=ORANGE, zorder=5)
    ax.plot(10**9, ratio_li[9], "o", color=BLUE, zorder=5)
    ax.text(10**9, ratio_simple[9] + 0.015, f"{ratio_simple[9]:.4f}", ha="right", fontsize=10, color=ORANGE)
    ax.text(10**9, ratio_li[9] + 0.015, f"{ratio_li[9]:.5f}", ha="right", fontsize=10, color=BLUE)
    ax.axhline(1, color=INK2, linewidth=0.9)
    ax.set_xscale("log")
    log_grid(ax)
    ax.set_xlabel("n (log scale)")
    ax.set_ylabel("ratio")
    ax.set_ylim(0.84, 1.24)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Both ratios tend to 1, at very different speeds")

    panel(fig.add_subplot(gs[0, 2]), "Counting them (NumPy sieve)", [
        "`  n          pi(n)        n/ln n         Li(n)",
        *[f"`10^{e}  {true_counts[e]:>12,}  {10**e / math.log(10**e):>12,.0f}  {li[e]:>12,.0f}"
          for e in (3, 5, 6, 7, 8, 9)],
        "",
        f"At $10^9$: $n/\\ln n$ is {100 * (ratio_simple[9] - 1):.1f}% low;",
        f"Li is {li[9] - pi_billion:,.0f} too high ({100 * (li[9] - pi_billion) / pi_billion:.4f}%).",
        r"Li by $t = e^u$: $\int_{\ln 2}^{\ln x} \frac{e^u}{u}du$ (trapezoid rule).",
    ], fontsize=12, spacing=0.08)

    ax = fig.add_subplot(gs[1, 0])
    block = 100_000
    block_starts = np.unique(np.logspace(np.log10(block), 8, 60).astype(np.int64) // block * block)
    block_starts = block_starts[block_starts + block <= 10**8]
    densities = (prime_count(block_starts + block) - prime_count(block_starts)) / block
    ax.plot(block_starts + block / 2, densities, "o", color=BLUE, markersize=6,
            label="share of primes in a block of 100,000")
    smooth = np.logspace(np.log10(block), 8, 300)
    ax.plot(smooth, 1 / np.log(smooth), color=ORANGE, linewidth=2.2, label=r"$1/\ln n$")
    for e in (6, 8):
        ax.annotate(f"1 in {math.log(10**e):.1f}", xy=(10**e, 1 / math.log(10**e)),
                    xytext=(1.3e6, 0.0765) if e == 6 else (2e6, 0.0560),
                    fontsize=10.5, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.set_xscale("log")
    log_grid(ax)
    ax.set_xlabel("n (log scale)")
    ax.set_ylabel("density of primes")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(r"Each factor 10 in $n$ adds $\ln 10 \approx 2.3$ to the spacing")

    ax = fig.add_subplot(gs[1, 1])
    n = 10**8
    gaps = np.diff(primes[primes > n - 10**6])
    ax.hist(gaps, bins=np.arange(1, gaps.max() + 3, 2), color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(gaps.mean(), color=INK, linewidth=2, label=f"mean gap {gaps.mean():.2f}")
    ax.axvline(math.log(n), color=ORANGE, linestyle="--", linewidth=2, label=rf"$\ln 10^8$ = {math.log(n):.2f}")
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="y")
    ax.set_xlabel("gap between consecutive primes")
    ax.set_ylabel("count (log scale)")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(r"Gaps in the last million below $10^8$")

    panel(fig.add_subplot(gs[1, 2]), r"Average gap $\approx \ln n$", [
        "`window below  primes   mean gap   ln n   max gap",
        *[f"`10^{e}       {gap_rows[e][0]:>7,}   {gap_rows[e][1]:>7.3f}  {gap_rows[e][2]:>6.3f}   {gap_rows[e][3]:>4}"
          for e in range(3, 9)],
        "Single gaps vary wildly; the mean tracks ln n.",
        "",
        r"A random 300-digit number is prime with probability",
        r"$1/(300\ln 10) \approx$ 1 in 690 (1 in 345 among odd ones).",
    ], fontsize=12, spacing=0.08)
    save_04(fig, name="05_prime_number_theorem.png")


# ================================================================ 05 / 01
def sheet_05_01_bits_digits_and_counting():
    def questions_to_find(secret, n_candidates):
        low, high, questions = 0, n_candidates, 0
        while high - low > 1:
            middle = (low + high) // 2
            questions += 1
            if secret < middle:
                high = middle
            else:
                low = middle
        return questions

    worst = {}
    for n_candidates in [2, 10, 1000, 1_000_000]:
        worst[n_candidates] = max(questions_to_find(s, n_candidates)
                                  for s in range(0, n_candidates, max(1, n_candidates // 5000)))
        assert worst[n_candidates] == math.ceil(math.log2(n_candidates))
    assert [worst[n] for n in worst] == [1, 4, 10, 20]
    world = math.log2(8.1e9)
    assert f"{world:.2f}" == "32.92" and math.ceil(world) == 33

    assert math.floor(math.log(1000, 10)) + 1 == 3 and len(str(1000)) == 4
    assert (2**64 - 1).bit_length() == 64 and math.floor(math.log2(2**64 - 1)) + 1 == 65
    assert len(str(3**50)) == 24 and (3**50).bit_length() == 80

    ratios = {d: (10**d - 1).bit_length() / d for d in [1, 3, 10, 30, 100, 1000]}
    assert [f"{r:.4f}" for r in ratios.values()] == ["4.0000", "3.3333", "3.4000", "3.3333", "3.3300", "3.3220"]

    recipes = [("4-digit PIN", 10, 4), ("8 lowercase", 26, 8), ("8 printable ASCII", 95, 8),
               ("4 words (7776)", 7776, 4), ("12 printable ASCII", 95, 12), ("6 words (7776)", 7776, 6)]
    bits = [length * math.log2(alphabet) for _, alphabet, length in recipes]
    assert [f"{b:.1f}" for b in bits] == ["13.3", "37.6", "52.6", "51.7", "78.8", "77.5"]
    seconds_per_year = 365.25 * 24 * 3600
    years = [2**b / 1e10 / seconds_per_year for b in bits]
    assert [f"{y:.2e}" for y in years] == ["3.17e-14", "6.62e-07", "2.10e-02", "1.16e-02", "1.71e+06", "7.01e+05"]
    one_day_bits = math.log2(1e10 * 86400)
    one_millennium_bits = math.log2(1e10 * 86400 * 365.25 * 1000)

    separately = math.log2(52) + math.log2(6) + math.log2(2)
    assert f"{separately:.6f}" == f"{math.log2(624):.6f}" == "9.285402"
    units = (math.log2(1e6), math.log10(1e6), math.log(1e6))
    assert [f"{u:.3f}" for u in units] == ["19.932", "6.000", "13.816"]

    fig = new_figure("Counting choices is a logarithm: N options take log₂ N yes/no questions",
                     r"$2^k \geq N \Leftrightarrow k \geq \log_2 N$,   digits of $N$ in base $b$: "
                     r"$\lfloor \log_b N \rfloor + 1$,   bits for $N$ values: $\lceil \log_2 N \rceil$,   "
                     r"password strength $L\log_2 A$ bits")
    gs = grid(fig, top=0.83)

    ax = fig.add_subplot(gs[0, 0])
    questions = np.arange(0, 21)
    remaining = np.ceil(1_000_000 / 2.0**questions)
    assert remaining[20] == 1 and remaining[19] == 2
    ax.plot(questions, remaining, "o-", color=BLUE, linewidth=2)
    ax.axhline(1, color=ORANGE, linestyle="--", linewidth=1.6)
    ax.text(0.3, 1.6, "one candidate left after 20 questions", color=ORANGE, fontsize=10.5)
    ax.set_yscale("log")
    log_grid(ax)
    ax.set_xticks(range(0, 21, 2))
    ax.set_xlabel("questions asked")
    ax.set_ylabel("candidates remaining (log scale)")
    ax.set_title("Halving a million: one factor 2 per question")

    ax = fig.add_subplot(gs[0, 1])
    n_values = np.unique(np.round(np.logspace(0, 6, 3000)).astype(np.int64))
    ax.step(n_values, [int(n).bit_length() for n in n_values], where="post", color=ORANGE, linewidth=2,
            label="binary digits (bits)")
    ax.step(n_values, [len(str(n)) for n in n_values], where="post", color=BLUE, linewidth=2,
            label="decimal digits")
    ax.plot(n_values, np.log2(n_values) + 1, ":", color=ORANGE, linewidth=1.4, label=r"$\log_2 N + 1$")
    ax.plot(n_values, np.log10(n_values) + 1, ":", color=BLUE, linewidth=1.4, label=r"$\log_{10} N + 1$")
    ax.set_xscale("log")
    log_grid(ax)
    ax.set_xlabel("N (log scale)")
    ax.set_ylabel("number of digits")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title(r"Digits $\lfloor \log_b N \rfloor + 1$: equal steps per decade")

    panel(fig.add_subplot(gs[0, 2]), "Independent choices: bits add", [
        r"$\log_2(52 \cdot 6 \cdot 2) = \log_2 52 + \log_2 6 + \log_2 2$",
        f"   = {separately:.6f} bits for card, die and coin",
        r"Colour: $\log_2 256^3 = 3 \times 8$ = 24 bits",
        "",
        "The only monotone, additive measure of",
        r"choice is $c\,\log N$; the base picks the unit:",
        f"   N = $10^6$: {units[0]:.3f} bits = {units[1]:.0f} decimal digits",
        f"                = {units[2]:.3f} nats",
        r"1 decimal digit = $\log_2 10$ = 3.32 bits",
        f"   ($10^{{1000}} - 1$ needs 3322 bits: {ratios[1000]:.4f} per digit)",
    ], fontsize=12, spacing=0.08)

    ax = fig.add_subplot(gs[1, 0])
    names = [name for name, _, _ in recipes]
    colors = [AQUA_LIGHT if b > one_millennium_bits else (BLUE_LIGHT if b > one_day_bits else CARD) for b in bits]
    ax.barh(np.arange(len(bits)), bits, color=colors, height=0.66, zorder=2)
    for y, (name, value) in enumerate(zip(names, bits)):
        ax.text(1, y, name, va="center", fontsize=10.5, color=INK, zorder=3)
        ax.text(max(value + 1, 22), y, f"{value:.1f}", va="center", fontsize=10.5)
    ax.axvline(one_day_bits, color=ORANGE, linestyle="--", linewidth=1.6)
    ax.axvline(one_millennium_bits, color=PURPLE, linestyle="--", linewidth=1.6)
    ax.text(one_day_bits - 1, -0.75, f"1 day ({one_day_bits:.0f} bits)", ha="right", va="center",
            fontsize=9.5, color=ORANGE)
    ax.text(one_millennium_bits + 1, -0.75, f"1000 years ({one_millennium_bits:.0f})", ha="left",
            va="center", fontsize=9.5, color=PURPLE)
    ax.set_ylim(-1.1, len(bits) - 0.5)
    ax.set_xlim(0, 96)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.grid(True, axis="x", zorder=0)
    ax.set_xlabel(r"strength $L\log_2 A$ in bits (each bit doubles the search)")
    ax.set_title(r"Passwords on the bit scale ($10^{10}$ guesses/s)")

    ax = fig.add_subplot(gs[1, 1])
    n_values = np.unique(np.round(np.logspace(0, 10.2, 4000)).astype(np.int64))
    needed = [math.ceil(math.log2(n)) if n > 1 else 0 for n in n_values]
    ax.step(n_values, needed, where="post", color=BLUE, linewidth=2, label=r"$\lceil \log_2 N \rceil$ questions")
    ax.plot(n_values, np.log2(n_values), ":", color=MUTED, linewidth=1.4, label=r"$\log_2 N$")
    for n, label in [(52, "52 cards: 6"), (1_000_000, "city of a million: 20"), (8.1e9, "everyone on Earth: 33")]:
        k = math.ceil(math.log2(n))
        ax.plot(n, k, "o", color=ORANGE, zorder=5)
        if n < 100:
            ax.text(n * 3, k - 2.5, label, fontsize=10.5, ha="left", color=ORANGE)
        else:
            ax.text(n / 3, k + 0.8, label, fontsize=10.5, ha="right", color=ORANGE)
    for n_candidates, k in worst.items():
        ax.plot(n_candidates, k, "s", color=INK, markersize=5, zorder=6)
    ax.plot([], [], "s", color=INK, markersize=5, label="worst case of binary search")
    ax.set_xscale("log")
    log_grid(ax)
    ax.set_ylim(0, 38)
    ax.set_xlabel("number of candidates N (log scale)")
    ax.set_ylabel("yes/no questions needed")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_title("Twenty Questions is a logarithm")

    panel(fig.add_subplot(gs[1, 2]), "Right idea, but exact tools at powers", [
        r"$b^{d-1} \leq N < b^d \Leftrightarrow d = \lfloor \log_b N \rfloor + 1$",
        f"   $3^{{50}}$: {50 * math.log10(3):.3f} → 24 decimal digits; {50 * math.log2(3):.2f} → 80 bits",
        "",
        "Floating-point logs can land just short:",
        "`math.log(1000, 10) = 2.9999999999999996",
        "   → the formula says 3 digits, 1000 has 4",
        "`2**64 - 1 rounds up to 2.0**64",
        "   → formula 65 bits, bit_length() = 64",
        "Near powers of the base use len(str(n)) and",
        "n.bit_length(), which are exact.",
    ], fontsize=12, spacing=0.08)
    save_05(fig, name="01_bits_digits_and_counting.png")


# ================================================================ 05 / 02
SAMPLE_TEXT = """
The lighthouse keeper woke before dawn, as he had done every morning for thirty years.
He climbed the narrow stairs, trimmed the wick, and watched the grey sea turn slowly to silver.
Most days nothing happened at all. A fishing boat would pass, a gull would complain about the wind,
and the tide would come in and go out again exactly when the almanac said it would.
He wrote the same three words in his logbook almost every evening: calm, clear, quiet.
Once in a long while the logbook said something else. There was the night of the great storm,
when the waves broke over the rocks and the lamp was the only thing that stood between a ship and the reef.
There was the morning a whale surfaced so close to the shore that he could hear it breathe.
Those were the pages his grandchildren wanted to read. Nobody ever asked about the calm days,
because a calm day told them nothing they did not already expect. The rare entries were the ones
that carried the news, and the keeper understood, without ever putting it into numbers,
that the less likely a message is, the more it tells you when it finally arrives.
"""


def sample_counts():
    characters = [c for c in SAMPLE_TEXT.lower().replace("\n", " ") if c.isalpha() or c == " "]
    cleaned = " ".join("".join(characters).split())
    return collections.Counter(cleaned)


def sheet_05_02_entropy_and_information():
    assert f"{-math.log2(0.99):.4f}" == "0.0145" and f"{-math.log2(0.01):.4f}" == "6.6439"
    h_desert, h_city = entropy_bits([0.99, 0.01]), entropy_bits([0.5, 0.5])
    assert f"{h_desert:.4f}" == "0.0808" and h_city == 1.0
    assert f"{365 * h_desert:.1f}" == "29.5"
    assert f"{entropy_bits([1 / 6] * 6):.3f}" == "2.585"
    abcd = entropy_bits([0.5, 0.25, 0.125, 0.125])
    assert abcd == 1.75 and f"{abcd * math.log(2):.4f}" == "1.2130"
    left = entropy_bits([1 / 2, 1 / 3, 1 / 6])
    right = entropy_bits([1 / 2, 1 / 2]) + 0.5 * entropy_bits([2 / 3, 1 / 3])
    assert f"{left:.6f}" == f"{right:.6f}" == "1.459148"

    counts = sample_counts()
    total = sum(counts.values())
    assert total == 1083 and len(counts) == 25
    entropy = kraft = average_length = 0.0
    for _, count in counts.most_common():
        p = count / total
        length = math.ceil(-math.log2(p))
        entropy -= p * math.log2(p)
        average_length += p * length
        kraft += 2.0**-length
    assert f"{entropy:.3f}" == "4.035" and f"{average_length:.3f}" == "4.588" and f"{kraft:.4f}" == "0.6929"
    assert f"{math.log2(27):.3f}" == "4.755"

    rng = np.random.default_rng(0)
    zlib_rows = []
    for p_one in [0.5, 0.1, 0.01]:
        flips = (rng.random(400_000) < p_one).astype(np.uint8)
        compressed = zlib.compress(np.packbits(flips).tobytes(), level=9)
        zlib_rows.append((p_one, entropy_bits([p_one, 1 - p_one]), 8 * len(compressed) / 400_000))
    assert [(f"{h:.3f}", f"{z:.3f}") for _, h, z in zlib_rows] == \
           [("1.000", "1.001"), ("0.469", "0.561"), ("0.081", "0.111")]

    rng = np.random.default_rng(1)
    distributions = rng.dirichlet(np.ones(4), size=20_000)
    random_entropies = -np.sum(distributions * np.log2(distributions), axis=1)
    assert random_entropies.max() < 2

    fig = new_figure("Information is surprise, −log₂ p; entropy is its average",
                     r"$I = -\log_2 p \Leftrightarrow p = 2^{-I}$:  independent events multiply $p$ and add $I$.   "
                     r"$H = -\sum_i p_i\log_2 p_i \leq \log_2 n$,  and no lossless code beats $H$ bits per symbol")
    gs = grid(fig, top=0.83)

    ax = fig.add_subplot(gs[0, 0])
    p = np.logspace(-3, 0, 400)
    ax.plot(p, -np.log2(p), color=PURPLE, linewidth=2.4)
    for k in range(0, 10):
        ax.plot(2.0**-k, k, "o", color=ORANGE, zorder=5)
    ax.annotate("rain in the desert\np = 0.01: 6.64 bits", xy=(0.01, -math.log2(0.01)), xytext=(0.0012, 2.2),
                fontsize=10.5, arrowprops=dict(arrowstyle="->", color=INK2))
    ax.annotate("fair coin: 1 bit", xy=(0.5, 1), xytext=(0.06, 0.4), fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xscale("log")
    log_grid(ax)
    ax.set_xlabel("probability p (log scale)")
    ax.set_ylabel(r"surprise $-\log_2 p$ (bits)")
    ax.set_title("Each halving of p adds exactly one bit")

    ax = fig.add_subplot(gs[0, 1])
    p = np.linspace(1e-6, 1 - 1e-6, 1000)
    ax.plot(p, -p * np.log2(p) - (1 - p) * np.log2(1 - p), color=INK, linewidth=2.6, label="h(p)")
    ax.plot(p, -p * np.log2(p), "--", color=BLUE, linewidth=1.6, label=r"$-p\log_2 p$")
    ax.plot(p, -(1 - p) * np.log2(1 - p), "--", color=ORANGE, linewidth=1.6, label=r"$-(1-p)\log_2(1-p)$")
    for p_marked, name, offset in [(0.01, "desert", (12, 10)), (0.5, "city", (-150, 12))]:
        value = entropy_bits([p_marked, 1 - p_marked])
        ax.plot(p_marked, value, "o", color=AQUA, markersize=8, zorder=5)
        ax.annotate(f"{name}: {value:.4f} bits/day", (p_marked, value), textcoords="offset points",
                    xytext=offset, fontsize=10.5)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("p = P(rain)")
    ax.set_ylabel("entropy (bits)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Binary entropy: largest for a fair coin")

    panel(fig.add_subplot(gs[0, 2]), "Worked examples", [
        r"Desert: $0.99 \times 0.0145 + 0.01 \times 6.644$",
        f"   = {h_desert:.4f} bits/day, {365 * h_desert:.1f} bits/year (city: 365)",
        r"Fair die: $\log_2 6$ = " + f"{entropy_bits([1 / 6] * 6):.3f} bits, the maximum for 6",
        r"A, B, C, D with $\frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \frac{1}{8}$: surprises 1, 2, 3, 3",
        f"   H = {abcd:.2f} bits = {abcd * math.log(2):.3f} nats",
        "   code 0, 10, 110, 111 averages exactly 1.75",
        r"Coin + die: $\log_2 12 = 1 + 2.585$ = 3.585 bits",
        r"Grouping: $H(\frac{1}{2},\frac{1}{3},\frac{1}{6}) = h(\frac{1}{2}) + \frac{1}{2}H(\frac{2}{3},\frac{1}{3})$",
        f"   = {left:.6f} bits on both sides",
    ], fontsize=11.8, spacing=0.078)

    ax = fig.add_subplot(gs[1, 0])
    symbols = [s for s, _ in counts.most_common()]
    probabilities = np.array([counts[s] / total for s in symbols])
    surprises = -np.log2(probabilities)
    labels = ["spc" if s == " " else s for s in symbols]
    positions = np.arange(len(symbols))
    ax.bar(positions, surprises, color=PURPLE, alpha=0.8, width=0.7, zorder=2, label=r"surprise $-\log_2 p$")
    ax.plot(positions, np.ceil(surprises), "_", color=ORANGE, markersize=11, markeredgewidth=2.4, zorder=3,
            label=r"codeword length $\lceil -\log_2 p \rceil$")
    ax.axhline(entropy, color=INK, linestyle="--", linewidth=1.4, label=f"entropy {entropy:.3f} bits")
    ax.set_xticks(positions, labels, fontsize=9)
    ax.set_xlim(-0.7, len(symbols) - 0.3)
    ax.set_ylim(0, 12.5)
    ax.set_xlabel("character of the sample text, most frequent first")
    ax.set_ylabel("bits")
    ax.grid(True, axis="y", zorder=0)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Frequent characters get short codewords")

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(random_entropies, bins=80, color=AQUA, zorder=2)
    ax.axvline(2, color=ORANGE, linestyle="--", linewidth=2, label=r"$\log_2 4$ = 2 bits (uniform)")
    ax.text(0.2, ax.get_ylim()[1] * 0.55, f"20,000 random distributions\non 4 outcomes (seed 1)\n"
            f"largest entropy seen: {random_entropies.max():.4f}", fontsize=10.5, color=INK2)
    ax.set_xlabel("entropy (bits)")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", zorder=0)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("No distribution beats the uniform one")

    panel(fig.add_subplot(gs[1, 2]), "The compression bound", [
        f"Sample text: {total:,} characters, {len(counts)} symbols",
        f"   H = {entropy:.3f} bits/char (27 equal: 4.755; ASCII: 8)",
        r"Shannon code $\lceil -\log_2 p_i \rceil$: Kraft sum " + f"{kraft:.4f} ≤ 1",
        f"   H = {entropy:.3f} ≤ average length {average_length:.3f} < H + 1",
        "",
        "zlib on 400,000 biased coin flips (seed 0):",
        "`P(1)    bound h(p)   zlib bits/flip",
        *[f"`{p_one:<6}  {h:>8.3f}     {z:>8.3f}" for p_one, h, z in zlib_rows],
        "A general compressor never beats H.",
    ], fontsize=12, spacing=0.078)
    save_05(fig, name="02_entropy_and_information.png")


# ================================================================ 05 / 03
def sheet_05_03_log_likelihood_and_mle():
    products = {}
    for n_tosses in [100, 1000, 1074, 1075]:
        product = 1.0
        for _ in range(n_tosses):
            product *= 0.5
        products[n_tosses] = product
    assert f"{products[1000]:.4g}" == "9.333e-302" and f"{products[1074]:.4g}" == "4.941e-324"
    assert products[1075] == 0.0

    rng = np.random.default_rng(42)
    data = rng.normal(loc=3.0, scale=2.0, size=1000)

    def gaussian_log_likelihood(mu, sigma):
        return float(np.sum(-np.log(sigma) - 0.5 * np.log(2 * np.pi) - (data - mu) ** 2 / (2 * sigma**2)))

    gaussian_ll = {mu: gaussian_log_likelihood(mu, 2.0) for mu in [2.0, 3.0, 4.0]}
    assert [f"{v:.2f}" for v in gaussian_ll.values()] == ["-2211.84", "-2101.29", "-2240.73"]
    densities = np.exp(-((data - 3.0) ** 2) / 8) / (2 * np.sqrt(2 * np.pi))
    assert np.prod(densities) == 0.0
    ratio_3_vs_2 = gaussian_ll[3.0] - gaussian_ll[2.0]
    assert f"{ratio_3_vs_2:.3f}" == "110.554" and f"{np.exp(ratio_3_vs_2):.3g}" == "1.03e+48"
    assert f"{data.mean():.4f}" == "2.9422" and f"{data.std():.4f}" == "1.9774"
    mu, log_sigma = 0.0, 0.0
    for _ in range(500):
        sigma = np.exp(log_sigma)
        residuals = data - mu
        gradient_mu = np.mean(residuals) / sigma**2
        gradient_log_sigma = -1 + np.mean(residuals**2) / sigma**2
        mu += 0.1 * gradient_mu
        log_sigma += 0.1 * gradient_log_sigma
    assert f"{mu:.4f}" == "2.9422" and f"{np.exp(log_sigma):.4f}" == "1.9774"

    heads, tosses = 620, 1000
    grid_p = np.linspace(0.40, 0.80, 4001)
    log_likelihood = heads * np.log(grid_p) + (tosses - heads) * np.log(1 - grid_p)
    assert round(grid_p[np.argmax(np.exp(log_likelihood))], 4) == round(grid_p[np.argmax(log_likelihood)], 4) == 0.62
    assert f"{np.exp(log_likelihood).max():.4e}" == "3.9867e-289" and f"{log_likelihood.max():.3f}" == "-664.064"
    gradients = {}
    for p in [0.5, 0.6]:
        dl = heads / p - (tosses - heads) / (1 - p)
        gradients[p] = (dl, np.exp(heads * np.log(p) + (tosses - heads) * np.log(1 - p)) * dl)
    assert [(f"{a:.2f}", f"{b:.3e}") for a, b in gradients.values()] == \
           [("480.00", "4.480e-299"), ("83.33", "1.437e-287")]
    ll_fair = tosses * math.log(0.5)
    assert f"{log_likelihood.max() - ll_fair:.2f}" == "29.08"   # the notebook's Example 4 rounds this to 29.0
    standard_error = math.sqrt(0.62 * 0.38 / 1000)
    assert f"{standard_error:.5f}" == "0.01535"

    fig = new_figure("Maximize ln L, not L: the same peak, with sums instead of products",
                     r"$L(\theta) = \prod_i p(x_i|\theta) \Leftrightarrow \ell(\theta) = \ln L = \sum_i \ln p(x_i|\theta)$.   "
                     r"$\ln$ is increasing, so $\arg\max\,\ell = \arg\max\,L$;   Bernoulli: $\hat p = k/n$")
    gs = grid(fig, top=0.83)

    grid_wide = np.linspace(0.3, 0.9, 1000)
    curves = []
    for n, color in [(10, BLUE_LIGHT), (100, BLUE), (1000, "#0d366b")]:
        k = round(0.62 * n)
        ll = k * np.log(grid_wide) + (n - k) * np.log(1 - grid_wide)
        curves.append((n, color, ll, ll.max()))

    ax = fig.add_subplot(gs[0, 0])
    for n, color, ll, peak in curves:
        ax.plot(grid_wide, np.exp(ll - peak), color=color, linewidth=2.2,
                label=f"n = {n}  (max L = $10^{{{peak / np.log(10):.1f}}}$)")
    ax.axvline(0.62, color=ORANGE, linestyle=":", linewidth=1.4)
    ax.set_xlabel("p")
    ax.set_ylabel("L(p) / max L")
    ax.set_ylim(0, 1.32)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Likelihood, 62% heads: rescaled to peak 1")

    ax = fig.add_subplot(gs[0, 1])
    for n, color, ll, peak in curves:
        ax.plot(grid_wide, ll - peak, color=color, linewidth=2.2, label=f"n = {n}")
    ax.axvline(0.62, color=ORANGE, linestyle=":", linewidth=1.4)
    ax.text(0.635, -18.5, r"$\hat p$ = 0.62", color=ORANGE, fontsize=10.5)
    ax.set_ylim(-20, 1)
    ax.set_xlabel("p")
    ax.set_ylabel(r"$\ell(p) - \max \ell$")
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title(r"Log-likelihood: concave, same $\arg\max$")

    panel(fig.add_subplot(gs[0, 2]), "620 heads in 1000 tosses", [
        r"$\ell(p) = 620\ln p + 380\ln(1-p)$,  $\ell\,'(p) = \frac{620}{p} - \frac{380}{1-p}$",
        f"max L = {np.exp(log_likelihood).max():.4e}: representable, just",
        f"max ℓ = {log_likelihood.max():.3f}: an ordinary number",
        "",
        "Gradients an optimizer would follow:",
        "`  p     dl/dp     dL/dp",
        *[f"`  {p}  {a:>7.2f}   {b:.3e}" for p, (a, b) in gradients.items()],
        r"$L\,' = L\cdot\ell\,'$ carries the tiny factor $L$.",
        f"$\\ell(0.62) - \\ell(0.5)$ = {log_likelihood.max() - ll_fair:.2f}:  L(0.62)/L(0.5) = {math.exp(log_likelihood.max() - ll_fair):.1e}",
    ], fontsize=11.8, spacing=0.078)

    ax = fig.add_subplot(gs[1, 0])
    n_values = np.arange(1, 1601)
    for probability, color in [(0.5, BLUE), (0.1, AQUA), (0.01, ORANGE)]:
        product = np.cumprod(np.full(n_values.size, probability))
        first_zero = int(n_values[np.argmax(product == 0)])
        if probability == 0.5:
            assert first_zero == 1075
        ax.plot(n_values, np.where(product > 0, product, np.nan), color=color, linewidth=2.2,
                label=f"$p^n$, p = {probability}: exactly 0 from n = {first_zero}")
        ax.plot(first_zero, 1e-322, "x", color=color, markersize=9, markeredgewidth=2)
    ax.axhline(2.2e-308, color=INK2, linestyle="--", linewidth=1)
    ax.text(620, 1e-300, "smallest normal float 2.2e-308", ha="center", fontsize=9.5, color=INK2)
    ax.set_xlim(-30, 1250)
    ax.set_yscale("log")
    ax.set_ylim(1e-323, 1e12)
    ax.set_yticks([10.0**e for e in range(0, -301, -50)])
    ax.minorticks_off()
    ax.grid(True)
    ax.set_xlabel("number of factors n")
    ax.set_ylabel("computed product (log scale)")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title(r"Products underflow; the sum $n\ln p$ never does")

    ax = fig.add_subplot(gs[1, 1])
    mu_grid = np.linspace(2.5, 3.5, 200)
    sigma_grid = np.linspace(1.7, 2.4, 200)
    mu_mesh, sigma_mesh = np.meshgrid(mu_grid, sigma_grid)
    n = data.size
    ssr = (data**2).sum() - 2 * mu_mesh * data.sum() + n * mu_mesh**2
    surface = -n * np.log(sigma_mesh) - 0.5 * n * np.log(2 * np.pi) - ssr / (2 * sigma_mesh**2)
    levels = surface.max() - np.array([20, 10, 5, 2, 0.5])
    contour = ax.contour(mu_mesh, sigma_mesh, surface, levels=levels,
                         colors=["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"], linewidths=1.6)
    ax.clabel(contour, fmt=lambda value: f"max − {surface.max() - value:.1f}", fontsize=8.5)
    ax.plot(data.mean(), data.std(), "*", color=ORANGE, markersize=16,
            label=f"MLE ($\\bar x$, $\\hat\\sigma$) = ({data.mean():.3f}, {data.std():.3f})")
    ax.plot(3.0, 2.0, "x", color=INK, markersize=10, markeredgewidth=2, label="true (3, 2)")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma$")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title(r"Gaussian $\ell(\mu, \sigma)$, 1000 points (seed 42)")

    panel(fig.add_subplot(gs[1, 2]), "Gaussian data: L is 0, ℓ decides", [
        r"np.prod of the 1000 densities = 0.0 for every $\mu$",
        "`mu     log-likelihood",
        *[f"`{m}    {v:>10.2f}" for m, v in gaussian_ll.items()],
        f"ℓ(3) − ℓ(2) = {ratio_3_vs_2:.3f}:  L(3)/L(2) = {np.exp(ratio_3_vs_2):.3g}",
        r"$\partial\ell/\partial\mu = 0 \Rightarrow \hat\mu = \bar x$;  "
        r"$\partial\ell/\partial\sigma = 0 \Rightarrow \hat\sigma^2 = \frac{1}{n}\sum(x_i - \bar x)^2$",
        "Gradient ascent on (μ, ln σ), 500 steps:",
        f"   μ = {mu:.4f}, σ = {np.exp(log_sigma):.4f} = closed form",
        "Least squares = Gaussian maximum likelihood.",
    ], fontsize=11.8, spacing=0.078)
    save_05(fig, name="03_log_likelihood_and_mle.png")


# ================================================================ 05 / 04
def sheet_05_04_cross_entropy_kl_and_softmax():
    def entropy_nats(p):
        p = np.asarray(p, dtype=float)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    def cross_entropy_nats(p, q):
        p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
        mask = p > 0
        return float(-np.sum(p[mask] * np.log(q[mask])))

    def kl_bits(p, q):
        return (cross_entropy_nats(p, q) - entropy_nats(p)) / math.log(2)

    p_example, q_example = [0.5, 0.25, 0.25], [0.8, 0.1, 0.1]
    h_pq = cross_entropy_nats(p_example, q_example) / math.log(2)
    assert f"{h_pq:.4f}" == "1.8219" and f"{kl_bits(p_example, q_example):.4f}" == "0.3219"
    assert f"{kl_bits(q_example, p_example):.4f}" == "0.2781"
    assert f"{-math.log(0.24):.3f}" == "1.427" and f"{-math.log(0.35):.3f}" == "1.050"

    rng = np.random.default_rng(3)
    p_random = rng.dirichlet(np.ones(5), size=100_000)
    q_random = rng.dirichlet(np.ones(5), size=100_000)
    kl_values = np.sum(p_random * np.log(p_random / q_random), axis=1)
    reverse_values = np.sum(q_random * np.log(q_random / p_random), axis=1)
    assert f"{kl_values.min():.6f}" == "0.001318" and int(np.sum(np.isclose(kl_values, reverse_values))) == 2

    def logsumexp(z):
        z = np.asarray(z, dtype=float)
        return z.max() + np.log(np.sum(np.exp(z - z.max())))

    z_small = np.array([2.0, 1.0, 0.1])
    q_small = np.exp(z_small - logsumexp(z_small))
    assert np.round(q_small, 4).tolist() == [0.659, 0.2424, 0.0986] and f"{logsumexp(z_small):.3f}" == "2.417"
    gradient = q_small - np.array([0.0, 1.0, 0.0])
    assert np.round(gradient, 6).tolist() == [0.659001, -0.757567, 0.098566]
    z_large = np.array([1000.0, 999.0, 998.0])
    with np.errstate(over="ignore", invalid="ignore"):
        assert np.all(np.isnan(np.exp(z_large) / np.exp(z_large).sum()))
    assert np.round(np.exp(z_large - logsumexp(z_large)), 4).tolist() == [0.6652, 0.2447, 0.09]
    assert f"{logsumexp(z_large):.4f}" == "1000.4076"
    extreme = np.array([0.0, -800.0, 5.0])
    assert f"{logsumexp(extreme) - extreme[1]:.4f}" == "805.0067"

    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 2.0], [2.0, -1.0], [-2.0, -1.0]])
    labels = np.repeat(np.arange(3), 100)
    points = centres[labels] + rng.normal(scale=0.9, size=(300, 2))
    one_hot = np.eye(3)[labels]
    weights, bias = np.zeros((2, 3)), np.zeros(3)
    losses, accuracies = [], []
    for epoch in range(201):
        logits = points @ weights + bias
        log_q = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        losses.append(-np.mean(np.sum(one_hot * log_q, axis=1)))
        accuracies.append(np.mean(np.argmax(logits, axis=1) == labels))
        gradient_logits = (np.exp(log_q) - one_hot) / len(points)
        weights -= 0.5 * points.T @ gradient_logits
        bias -= 0.5 * gradient_logits.sum(axis=0)
    assert [f"{losses[e]:.4f}" for e in (0, 10, 50, 200)] == ["1.0986", "0.1793", "0.1155", "0.1017"]
    assert [f"{accuracies[e]:.3f}" for e in (10, 50, 200)] == ["0.957"] * 3

    fig = new_figure("Cross-entropy: the surprise of the truth under your prediction",
                     r"$H(p,q) = -\sum_i p_i\ln q_i = H(p) + D_{KL}(p\,\Vert\,q)$,  $D_{KL} \geq 0$.   "
                     r"Softmax: $\ln q_i = z_i - \ln\sum_j e^{z_j}$;   gradient of the loss in the logits: $q - p$")
    gs = grid(fig, top=0.83)

    ax = fig.add_subplot(gs[0, 0])
    q = np.logspace(-4, 0, 400)
    ax.plot(q, -np.log(q), color=ORANGE, linewidth=2.4)
    for probability, name, y_text in [(0.24, "classifier 1", 1.6), (0.35, "classifier 2", 0.6)]:
        ax.plot(probability, -math.log(probability), "o", color=INK, zorder=5)
        ax.annotate(f"{name}: q(dog) = {probability} → loss {-math.log(probability):.3f}",
                    (probability, -math.log(probability)), xytext=(1.3e-4, y_text),
                    fontsize=10, arrowprops=dict(arrowstyle="->", color=INK2))
    for k in range(5):
        ax.plot(10.0**-k, k * math.log(10), "s", color=BLUE, markersize=5, zorder=4)
    ax.text(1.3e-4, 3.4, "each factor 10 less on the truth:\n+ln 10 = 2.30 nats", fontsize=10.5, color=BLUE)
    ax.set_xscale("log")
    log_grid(ax)
    ax.set_xlabel("predicted probability of the true class q (log scale)")
    ax.set_ylabel(r"loss $-\ln q$ (nats)")
    ax.set_title("One example's loss is a straight line on a log axis")

    ax = fig.add_subplot(gs[0, 1])
    p_bar = np.array([0.5, 0.25, 0.25])
    models = {"q = p": p_bar, "(0.4,\n0.3, 0.3)": np.array([0.4, 0.3, 0.3]), "uniform": np.full(3, 1 / 3),
              "(0.8,\n0.1, 0.1)": np.array([0.8, 0.1, 0.1]), "(0.1,\n0.1, 0.8)": np.array([0.1, 0.1, 0.8])}
    h_bits = entropy_bits(p_bar)
    kls = [kl_bits(p_bar, m) for m in models.values()]
    positions = np.arange(len(models))
    ax.bar(positions, [h_bits] * len(models), width=0.62, color=MUTED, zorder=2, label="H(p) = 1.5 bits (unavoidable)")
    ax.bar(positions, kls, bottom=h_bits, width=0.62, color=ORANGE, zorder=2, label="KL(p ‖ q): cost of the wrong belief")
    for x_pos, kl in zip(positions, kls):
        ax.text(x_pos, h_bits + kl + 0.04, f"{h_bits + kl:.3f}", ha="center", fontsize=10)
    ax.set_xticks(positions, list(models), fontsize=9.5)
    ax.set_ylim(0, 3.3)
    ax.set_ylabel("H(p, q) in bits per symbol")
    ax.grid(True, axis="y", zorder=0)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Cross-entropy for p = (0.5, 0.25, 0.25)")

    panel(fig.add_subplot(gs[0, 2]), "Scores and divergences", [
        r"Truth dog: loss $-\ln q_{dog}$ = 1.427 vs 1.050 nats",
        r"   (the second puts 1.46 × more on the truth)",
        r"p = (0.5, 0.25, 0.25), q = (0.8, 0.1, 0.1), in bits:",
        f"   H(p) = 1.5,  H(p, q) = {h_pq:.4f},  KL(p ‖ q) = {kl_bits(p_example, q_example):.4f}",
        f"   KL(q ‖ p) = {kl_bits(q_example, p_example):.4f}: not symmetric",
        "",
        r"Gibbs: $-D_{KL} = \sum p_i\ln\frac{q_i}{p_i} \leq \sum p_i(\frac{q_i}{p_i} - 1) = 0$",
        "100,000 random pairs on 5 outcomes (seed 3):",
        f"   smallest KL {kl_values.min():.6f}; symmetric for only 2",
        "Minimizing cross-entropy = maximizing likelihood.",
    ], fontsize=11.8, spacing=0.076)

    ax = fig.add_subplot(gs[1, 0])
    p_true = 0.2
    q = np.linspace(0.001, 0.999, 999)
    forward = p_true * np.log(p_true / q) + (1 - p_true) * np.log((1 - p_true) / (1 - q))
    reverse = q * np.log(q / p_true) + (1 - q) * np.log((1 - q) / (1 - p_true))
    ax.plot(q, forward, color=BLUE, linewidth=2.4, label="KL(p ‖ q): infinite at both ends")
    ax.plot(q, reverse, color=ORANGE, linewidth=2.4, label="KL(q ‖ p): finite, 0.22 and 1.61")
    ax.axvline(p_true, color=MUTED, linestyle=":", linewidth=1.4)
    ax.text(0.21, 2.75, "q = p = 0.2", fontsize=10, color=INK2)
    ax.set_ylim(0, 3)
    ax.set_xlabel("model probability q of heads (true p = 0.2)")
    ax.set_ylabel("divergence (nats)")
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=9.5, bbox_to_anchor=(0.6, 0.92))
    ax.set_title("KL is not a distance: two directions differ")

    ax = fig.add_subplot(gs[1, 1])
    epochs = np.arange(201)
    ax.plot(epochs, losses, color=BLUE, linewidth=2.4, label="average cross-entropy")
    ax.axhline(math.log(3), color=ORANGE, linestyle="--", linewidth=1.4, label="ln 3: a uniform guess")
    for e in (0, 10, 50, 200):
        ax.plot(e, losses[e], "o", color=INK, zorder=5)
        ax.text(e + 4, losses[e] + 0.05, f"{losses[e]:.4f}", fontsize=10)
    ax.text(70, 0.28, "3 clusters, 300 points (seed 0)\nlinear logits z = Wx + b\n"
            r"update uses only $q - p$" + f"\naccuracy {accuracies[200]:.3f}", fontsize=10.5, color=INK2)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (nats)")
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.84), fontsize=10)
    ax.set_title("Training a softmax classifier with gradient q − p")

    panel(fig.add_subplot(gs[1, 2]), "Softmax through the logarithm", [
        r"$z = (2, 1, 0.1)$:  LSE(z) = $\ln(e^2 + e^1 + e^{0.1})$ = " + f"{logsumexp(z_small):.3f}",
        "   q = (" + ", ".join(f"{v:.3f}" for v in q_small) + ")",
        "   loss for dog = 2.417 − 1 = 1.417 nats",
        "   q − p = (" + ", ".join(f"{v:+.3f}" for v in gradient) + ")",
        "z = (1000, 999, 998): naive exp → nan;",
        r"subtract the max: $1000 + \ln(1 + e^{-1} + e^{-2})$",
        f"   = {logsumexp(z_large):.4f},  q = (0.665, 0.245, 0.090)",
        r"Loss as $\mathrm{LSE}(z) - z_y$ never fails:",
        f"   z = (0, −800, 5), y = 2nd: {logsumexp(extreme) - extreme[1]:.4f}",
    ], fontsize=11.8, spacing=0.08)
    save_05(fig, name="04_cross_entropy_kl_and_softmax.png")


# ================================================================ 05 / 05
def first_digits(values):
    mantissas = np.mod(np.log10(np.asarray(values, dtype=float)), 1.0)
    return np.floor(10**mantissas + 1e-12).astype(int)


def sheet_05_05_lognormal_and_benfords_law():
    rng = np.random.default_rng(2024)
    final_sizes = np.prod(rng.uniform(0.75, 1.30, size=(20_000, 100)), axis=1)
    log_sizes = np.log(final_sizes)
    grid_g = np.linspace(0.75, 1.30, 200_001)
    mean_log_factor = np.mean(np.log(grid_g))
    var_log_factor = np.mean(np.log(grid_g) ** 2) - mean_log_factor**2
    predicted = (100 * mean_log_factor, np.sqrt(100 * var_log_factor))
    assert (f"{predicted[0]:.3f}", f"{predicted[1]:.3f}") == ("1.243", "1.576")
    assert (f"{log_sizes.mean():.3f}", f"{log_sizes.std():.3f}") == ("1.241", "1.583")
    skew_x = np.mean((final_sizes - final_sizes.mean()) ** 3) / final_sizes.std() ** 3
    skew_log = np.mean((log_sizes - log_sizes.mean()) ** 3) / log_sizes.std() ** 3
    assert (f"{skew_x:.2f}", f"{skew_log:.3f}") == ("16.93", "-0.027")
    assert (f"{np.median(final_sizes):.3f}", f"{final_sizes.mean():.3f}", f"{final_sizes.max():.1f}") == \
           ("3.487", "11.698", "1736.0")

    benford = np.log10(1 + 1 / np.arange(1, 10))
    powers_counts = np.zeros(9)
    power = 1
    for _ in range(10_000):
        power *= 2
        powers_counts[int(str(power)[0]) - 1] += 1
    powers_shares = powers_counts / 10_000
    assert [f"{v:.4f}" for v in powers_shares] == ["0.3010", "0.1761", "0.1249", "0.0970", "0.0791",
                                                   "0.0670", "0.0579", "0.0512", "0.0458"]
    assert f"{100 * math.log10(2):.3f}" == "30.103" and len(str(2**100)) == 31

    rng = np.random.default_rng(11)
    data_sets = {
        "product of\n100 factors": np.prod(rng.uniform(0.75, 1.30, size=(20_000, 100)), axis=1),
        "product of\n400 factors": np.prod(rng.uniform(0.75, 1.30, size=(20_000, 400)), axis=1),
        "sum of 100\nuniform steps": np.sum(rng.uniform(0, 2, size=(20_000, 100)), axis=1),
        "uniform on\n[1, 1000)": rng.uniform(1, 1000, size=20_000),
    }
    chi_squares, share_of_ones = [], []
    for values in data_sets.values():
        digits = first_digits(values)
        observed = np.array([np.sum(digits == d) for d in range(1, 10)])
        expected = len(digits) * benford
        chi_squares.append(float(np.sum((observed - expected) ** 2 / expected)))
        share_of_ones.append(observed[0] / len(digits))
    assert [f"{c:.1f}" for c in chi_squares] == ["6.2", "7.5", "89397.8", "8020.4"]

    rng = np.random.default_rng(12)
    multiplicative = np.prod(rng.uniform(0.75, 1.30, size=(20_000, 400)), axis=1)
    uniform = rng.uniform(1, 1000, size=20_000)
    scale_rows = {}
    for scale in [1.0, 0.3048]:
        shares_m = np.array([np.mean(first_digits(scale * multiplicative) == d) for d in range(1, 10)])
        shares_u = np.array([np.mean(first_digits(scale * uniform) == d) for d in range(1, 10)])
        scale_rows[scale] = (shares_m[0], shares_u[0])
    assert [f"{v:.3f}" for v in scale_rows[1.0] + scale_rows[0.3048]] == ["0.297", "0.113", "0.298", "0.361"]

    rng = np.random.default_rng(2024)
    companies = np.prod(rng.uniform(0.75, 1.30, size=(20_000, 400)), axis=1)
    company_shares = np.array([np.mean(first_digits(companies) == d) for d in range(1, 10)])

    fig = new_figure("Multiply random factors and the logarithm does the rest",
                     r"$X = X_0\prod G_i \Leftrightarrow \ln X = \ln X_0 + \sum \ln G_i \approx$ normal (log-normal $X$).   "
                     r"First digit $d \Leftrightarrow \{\log_{10} x\} \in [\log_{10} d, \log_{10}(d+1))$,  "
                     r"$P(d) = \log_{10}(1 + \frac{1}{d})$")
    gs = grid(fig, top=0.83)
    mu_hat, sigma_hat = log_sizes.mean(), log_sizes.std()

    ax = fig.add_subplot(gs[0, 0])
    bin_width = 0.1
    ax.hist(final_sizes, bins=np.arange(0, 15 + bin_width, bin_width),
            weights=np.full(final_sizes.size, 1 / (final_sizes.size * bin_width)), color=BLUE_LIGHT,
            label="companies (seed 2024)")
    x = np.linspace(0.01, 15, 1000)
    ax.plot(x, np.exp(-((np.log(x) - mu_hat) ** 2) / (2 * sigma_hat**2)) / (x * sigma_hat * np.sqrt(2 * np.pi)),
            color=ORANGE, linewidth=2.2, label="log-normal")
    for value, name, color in [(np.median(final_sizes), "median", INK), (final_sizes.mean(), "mean", AQUA)]:
        ax.axvline(value, color=color, linestyle="--", linewidth=1.4)
        ax.text(value + 0.15, 0.17, f"{name} {value:.2f}", color=color, fontsize=10)
    ax.set_xlim(0, 15)
    ax.set_xlabel("final size X after 100 years (linear scale)")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.3, 1.0), fontsize=9.5)
    ax.set_title("Skewed on a linear axis")

    ax = fig.add_subplot(gs[0, 1])
    log_bins = np.logspace(np.log10(final_sizes.min()), np.log10(final_sizes.max()), 80)
    ax.hist(final_sizes, bins=log_bins, color=BLUE_LIGHT, label="the same companies")
    centres = np.sqrt(log_bins[:-1] * log_bins[1:])
    widths = np.log(log_bins[1:]) - np.log(log_bins[:-1])
    ax.plot(centres, len(final_sizes) * widths * np.exp(-((np.log(centres) - mu_hat) ** 2) / (2 * sigma_hat**2))
            / (sigma_hat * np.sqrt(2 * np.pi)), color=ORANGE, linewidth=2.2, label=r"normal curve in $\ln X$")
    ax.set_xscale("log")
    log_grid(ax)
    ax.set_xlabel("final size X (log scale)")
    ax.set_ylabel("count per log-spaced bin")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("A symmetric bell on a log axis")

    panel(fig.add_subplot(gs[0, 2]), "Products of 100 factors U(0.75, 1.30)", [
        r"CLT for $\ln X = \sum \ln G_i$:",
        f"   predicted mean {predicted[0]:.3f}, sd {predicted[1]:.3f}",
        f"   simulated mean {log_sizes.mean():.3f}, sd {log_sizes.std():.3f}",
        f"skewness of X: {skew_x:.2f};  of ln X: {skew_log:.3f}",
        f"median {np.median(final_sizes):.3f},  mean {final_sizes.mean():.3f},  largest {final_sizes.max():.1f}",
        "",
        r"If $\ln X \sim N(\mu, \sigma^2)$:  median $e^{\mu}$,",
        r"mean $e^{\mu + \sigma^2/2}$,  mode $e^{\mu - \sigma^2}$",
        r"($\mu = 0, \sigma = 1$: 1, 1.649, 0.368)",
    ], fontsize=12, spacing=0.08)

    sub = gs[1, 0].subgridspec(2, 1, height_ratios=[1, 3.2], hspace=0.55)
    ax_ruler = fig.add_subplot(sub[0])
    ruler_colors = ["#0d366b", "#1c5cab", "#2a78d6", "#3987e5", "#86b6ef", "#b7d3f6", "#cde2fb", "#e3eefb", "#f3f7fc"]
    for d, color in zip(range(1, 10), ruler_colors):
        left, right = math.log10(d), math.log10(d + 1)
        ax_ruler.barh(0, right - left, left=left, color=color, edgecolor=SURFACE, height=1)
        ax_ruler.text((left + right) / 2, 0, str(d), ha="center", va="center", fontsize=10,
                      color="white" if d <= 4 else INK)
    ax_ruler.set_xlim(0, 1)
    ax_ruler.set_ylim(-0.5, 0.5)
    ax_ruler.set_yticks([])
    for spine in ("left", "top", "right"):
        ax_ruler.spines[spine].set_visible(False)
    ax_ruler.tick_params(labelsize=9)
    ax_ruler.set_title(r"One decade of a log ruler: digit $d$ owns $\log_{10}(1 + 1/d)$", fontsize=12)
    ax = fig.add_subplot(sub[1])
    digits = np.arange(1, 10)
    width = 0.27
    ax.bar(digits - width, benford, width, color=INK, label=r"Benford $\log_{10}(1 + 1/d)$", zorder=2)
    ax.bar(digits, powers_shares, width, color=ORANGE, label=r"$2^1 \dots 2^{10000}$", zorder=2)
    ax.bar(digits + width, company_shares, width, color=BLUE, label="companies, 400 factors", zorder=2)
    ax.text(1.0 - width, benford[0] + 0.01, "30.1%", ha="center", fontsize=9.5)
    ax.text(9.0 - width, benford[8] + 0.01, "4.6%", ha="center", fontsize=9.5)
    ax.set_xticks(digits)
    ax.set_xlabel("first digit d")
    ax.set_ylabel("share")
    ax.set_ylim(0, 0.36)
    ax.grid(True, axis="y", zorder=0)
    ax.legend(loc="upper right", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 1])
    positions = np.arange(len(chi_squares))
    ax.bar(positions, chi_squares, width=0.6, color=[AQUA, AQUA, ORANGE, ORANGE], zorder=2)
    for x_pos, value, ones in zip(positions, chi_squares, share_of_ones):
        ax.text(x_pos, max(value * 1.25, 40), f"χ² = {value:,.1f}\n1s: {ones:.3f}", ha="center", fontsize=10)
    ax.axhline(15.51, color=INK, linestyle="--", linewidth=1.4, label="5% critical value 15.51 (8 df)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_yscale("log")
    ax.set_ylim(1, 3e7)
    ax.set_xticks(positions, list(data_sets), fontsize=9.5)
    ax.grid(True, axis="y", zorder=0)
    ax.set_ylabel("chi-square vs Benford (log scale)")
    ax.set_title("Multiplicative data pass, additive and uniform fail")

    panel(fig.add_subplot(gs[1, 2]), "Reading digits off the mantissa", [
        r"$x = 10^{\lfloor \log_{10} x \rfloor} \cdot 10^{\{\log_{10} x\}}$",
        r"$\log_{10} 2^{100}$ = 30.103: 31 digits; $10^{0.103}$ = 1.27,",
        "   so the first digit is 1",
        r"$P(d \leq 3) = \log_{10}(2 \cdot \frac{3}{2} \cdot \frac{4}{3}) = \log_{10} 4$ = 0.602",
        "",
        "Scale invariance (seed 12): share of first digit 1",
        f"   ×1:      multiplicative {scale_rows[1.0][0]:.3f},  uniform {scale_rows[1.0][1]:.3f}",
        f"   ×0.3048: multiplicative {scale_rows[0.3048][0]:.3f},  uniform {scale_rows[0.3048][1]:.3f}",
        "Only a uniform mantissa survives all units.",
    ], fontsize=11.8, spacing=0.08)
    save_05(fig, name="05_lognormal_and_benfords_law.png")


if __name__ == "__main__":
    sheet_04_01_logarithmic_differentiation()
    sheet_04_02_integrals_that_produce_logs()
    sheet_04_03_exponential_growth_and_decay()
    sheet_04_04_stirlings_approximation()
    sheet_04_05_prime_number_theorem()
    sheet_05_01_bits_digits_and_counting()
    sheet_05_02_entropy_and_information()
    sheet_05_03_log_likelihood_and_mle()
    sheet_05_04_cross_entropy_kl_and_softmax()
    sheet_05_05_lognormal_and_benfords_law()
