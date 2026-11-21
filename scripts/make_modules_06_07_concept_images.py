#!/usr/bin/env python3
"""Concept sheets for modules 06 (applications) and 07 (beyond the reals).

One page per notebook that summarizes its central idea, in exponent form, in log form and as a picture.
Every number is computed here from the same formulas, parameters and random seeds (same draw order) the
notebook uses, and is checked against the notebook's printed outputs with asserts.
Images are written to modules/06_applications/img/ and modules/07_beyond_the_reals/img/.

Run from anywhere:  /opt/anaconda3/bin/python scripts/make_modules_06_07_concept_images.py
"""
import cmath
import math
import random
from fractions import Fraction
from functools import lru_cache, partial

import numpy as np
import sympy as sp
from matplotlib.patches import Arc, Rectangle

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA, GRID,
                       new_figure, grid, panel)
from viz_style import save as _save

APPLICATIONS = REPO / "modules" / "06_applications"
BEYOND = REPO / "modules" / "07_beyond_the_reals"
save_06 = partial(_save, folder=APPLICATIONS / "img")
save_07 = partial(_save, folder=BEYOND / "img")

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#a9e3cd"
GOLD = "#c9a227"


# ================================================================ helpers
def legend(ax, loc="best", fontsize=9, **kw):
    ax.legend(loc=loc, fontsize=fontsize, framealpha=0.92, frameon=True, facecolor=SURFACE,
              edgecolor="none", **kw)


def arrow(ax, start, end, color, width=2.0, zorder=4):
    ax.annotate("", xy=end, xytext=start, zorder=zorder,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=width, mutation_scale=14,
                                shrinkA=0, shrinkB=0))


def pi_ticks(ax, axis, multiples, denominator=1):
    """Ticks at k*pi/denominator labelled with mathtext."""
    values, labels = [], []
    for k in multiples:
        values.append(k * np.pi / denominator)
        f = Fraction(k, denominator)
        if f == 0:
            labels.append("0")
            continue
        sign = "-" if f < 0 else ""
        num, den = abs(f.numerator), f.denominator
        top = r"\pi" if num == 1 else rf"{num}\pi"
        labels.append(rf"${sign}{top}$" if den == 1 else rf"${sign}{top}/{den}$")
    (ax.set_xticks if axis == "x" else ax.set_yticks)(values, labels)


# ================================================================ 06 / 01 decibels, Richter, pH
def sheet_06_01_decibels_richter_and_ph():
    reference_intensity = 1e-12

    def intensity_to_db(intensity):
        return 10 * math.log10(intensity / reference_intensity)

    def combine_levels(levels):
        return 10 * math.log10(sum(10 ** (level / 10) for level in levels))

    assert [round(intensity_to_db(i), 1) for i in (1e-12, 1e-10, 1e-5, 1e-2, 1.0)] == [0, 20, 70, 100, 120]
    assert round(10 ** 0.3, 4) == 1.9953
    assert [round(combine_levels(x), 2) for x in ([70, 70], [80, 70], [60] * 10)] == [73.01, 80.41, 70.0]
    assert round(combine_levels([80, 60]), 3) == 80.043
    assert f"{20 * math.log10(0.02 / 20e-6):.2f}" == "60.00" and f"{20 * math.log10(2):.3f}" == "6.021"

    def energy_joules(magnitude):
        return 10 ** (1.5 * magnitude + 4.8)

    assert f"{10 ** 1.5:.1f}" == "31.6"
    assert f"{energy_joules(9) / energy_joules(6):,.0f}" == "31,623"
    assert [f"{energy_joules(m):.1e}" for m in (4, 6, 8)] == ["6.3e+10", "6.3e+13", "6.3e+16"]
    assert round(-math.log10(3.2e-4), 2) == 3.49 and f"{10 ** -2.5:.3e}" == "3.162e-03"
    assert round(10 ** (-0.4 * (0 - 1)), 4) == 2.5119 and round(10 ** (-0.4 * (-1 - 6)), 1) == 631.0
    gap_for_half_db = -10 * math.log10(10 ** 0.05 - 1)
    assert round(gap_for_half_db, 2) == 9.14
    mixed = (1e-2 + 1e-4) / 2
    assert round(-math.log10(mixed), 2) == 2.3
    assert f"{10 ** (0.4 * 14):,.0f}" == "398,107"

    fig = new_figure("Decibels, Richter, pH: one unit is one fixed ratio",
                     r"$L = 10\,\log_{10}(I/I_0)$ dB  $\Leftrightarrow$  $I = I_0\,10^{L/10}$,     "
                     r"$\mathrm{pH} = -\log_{10}[\mathrm{H}^+]$,     $\log_{10} E \approx 1.5\,M + 4.8$,     "
                     r"$m_1 - m_2 = -2.5\,\log_{10}(F_1/F_2)$")
    gs = grid(fig, top=0.84, wspace=0.42, hspace=0.55)

    # --- the decibel ruler
    ax = fig.add_subplot(gs[0, 0])
    sounds = [("threshold of hearing", 0), ("rustling leaves", 20), ("quiet library", 40),
              ("conversation", 60), ("busy traffic", 80), ("loud concert", 110), ("threshold of pain", 120)]
    for name, level in sounds:
        intensity = reference_intensity * 10 ** (level / 10)
        ax.plot(0, intensity, "o", color=BLUE, markersize=8, zorder=3)
        ax.text(0.12, intensity, name, va="center", fontsize=10, color=INK)
    ax.axvline(0, color=BLUE_LIGHT, linewidth=3, zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(3e-14, 3e1)
    ax.set_xlim(-0.25, 1.6)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks([10.0 ** k for k in range(-12, 1, 2)])
    ax.set_ylabel(r"intensity $I$ in W/m$^2$ (log scale)")
    right = ax.twinx()
    right.set_yscale("log")
    right.set_ylim(ax.get_ylim())
    right.set_yticks([reference_intensity * 10 ** (level / 10) for level in range(0, 121, 20)],
                     [f"{level} dB" for level in range(0, 121, 20)])
    right.minorticks_off()
    right.spines["right"].set_visible(True)
    ax.set_title(r"Sound: a factor $10^{12}$ becomes 0 to 120 dB")

    # --- adding sources
    ax = fig.add_subplot(gs[0, 1])
    level_gap = np.linspace(0, 25, 300)
    increase = 10 * np.log10(1 + 10 ** (-level_gap / 10))
    ax.plot(level_gap, increase, color=BLUE, linewidth=2.2)
    for gap in (0, 10, 20):
        value = 10 * np.log10(1 + 10 ** (-gap / 10))
        ax.plot(gap, value, "o", color=ORANGE, markersize=7, zorder=4)
        ax.annotate(f"+{value:.2f} dB", (gap, value), xytext=(8, 6), textcoords="offset points", fontsize=10)
    ax.plot([gap_for_half_db, gap_for_half_db], [0, 0.5], color=AQUA, linestyle="--", linewidth=1.4)
    ax.plot([0, gap_for_half_db], [0.5, 0.5], color=AQUA, linestyle="--", linewidth=1.4)
    ax.text(gap_for_half_db + 1.2, 1.15, f"under +0.5 dB once the\nsecond source is {gap_for_half_db:.2f} dB quieter",
            fontsize=9.5, color=AQUA)
    ax.set_ylim(-0.1, 3.5)
    ax.set_xlabel(r"how much quieter the second source is, $L_1 - L_2$ (dB)")
    ax.set_ylabel("increase of the total level (dB)")
    ax.grid(True)
    ax.set_title(r"$10\,\log_{10}(1 + 10^{-d/10})$: decibels do not add")

    panel(fig.add_subplot(gs[0, 2]), "Leave log space, add, come back", [
        r"$L_{tot} = 10\,\log_{10}(10^{L_1/10} + 10^{L_2/10})$",
        "",
        "`70 + 70 dB      -> 73.01 dB`",
        "`80 + 70 dB      -> 80.41 dB`",
        "`ten times 60 dB -> 70.0 dB`",
        "`80 + 60 dB      -> 80.043 dB`",
        "",
        "Amplitudes use 20 log₁₀ because I ∝ p²:",
        r"pressure $\times 2$: $20\,\log_{10} 2 = 10\,\log_{10} 4$ = 6.02 dB",
        "inverse square, 8 m instead of 1 m: −18.06 dB",
    ], spacing=0.083)

    # --- pH
    ax = fig.add_subplot(gs[1, 0])
    liquids = [("stomach acid", 1.5), ("lemon juice", 2.0), ("coffee", 5.0), ("pure water", 7.0),
               ("seawater", 8.1), ("household ammonia", 11.5)]
    floor = 1e-13
    for index, (name, ph) in enumerate(liquids):
        concentration = 10 ** (-ph)
        ax.barh(index, concentration - floor, left=floor, color=AQUA if ph < 7 else BLUE, height=0.6, zorder=2)
        ax.text(concentration * 2, index, f"{name}, pH {ph:g}", va="center", fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(floor, 1e7)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xticks([10.0 ** -k for k in (12, 9, 6, 3, 0)])
    ax.set_xlabel(r"$[\mathrm{H}^+]$ in mol/L (log scale)")
    ax.grid(True, axis="x", which="major")
    ax.set_title(r"pH $= -\log_{10}[\mathrm{H}^+]$: a reversed log axis")

    # --- earthquake energy
    ax = fig.add_subplot(gs[1, 1])
    magnitudes = np.linspace(2, 9.5, 100)
    ax.semilogy(magnitudes, energy_joules(magnitudes), color=ORANGE, linewidth=2.2)
    for m in (4, 6, 8):
        ax.plot(m, energy_joules(m), "o", color=INK, markersize=6, zorder=4)
        ax.annotate(f"M{m}: {energy_joules(m):.1e} J".replace("e+", "e"), (m, energy_joules(m)),
                    xytext=(-12, 10), textcoords="offset points", fontsize=9.5, ha="right")
    ax.text(6.55, energy_joules(6) * 0.05, "one unit:\namplitude ×10\nenergy ×31.6", fontsize=9.5,
            color=INK2, va="top")
    ax.set_xlabel("magnitude M (linear)")
    ax.set_ylabel("energy E in joules (log scale)")
    ax.grid(True, which="major")
    ax.set_title(r"Earthquakes: 1.5 decades of energy per unit")

    panel(fig.add_subplot(gs[1, 2]), "The common pattern", [
        "`scale        +1 unit means`",
        "`sound level  I x 1.259 (p x 1.122)`",
        "`Richter      amplitude x 10`",
        "`pH           [H+] divided by 10`",
        "`star mag.    flux divided by 2.512`",
        "",
        r"Weber–Fechner: $dS = k\,dI/I \;\Rightarrow\; S = k\,\ln(I/I_0)$",
        "M7 vs M5: amplitude ×100, energy ×1,000",
        "Mixing pH 2 and pH 4: pH 2.30, not 3",
        r"Sun vs full Moon, $\Delta m$ = 14: $\times 10^{5.6}$ ≈ 398,107",
    ], spacing=0.083)
    save_06(fig, name="01_decibels_richter_and_ph.png")


# ================================================================ 06 / 02 log returns
def sheet_06_02_log_returns():
    prices3 = np.array([100.0, 150.0, 75.0])
    log3 = np.diff(np.log(prices3))
    assert np.round(log3, 4).tolist() == [0.4055, -0.6931] and round(log3.sum(), 4) == -0.2877

    monthly = np.array([0.10, -0.05, 0.03])
    assert round(np.log1p(monthly).sum(), 5) == 0.07358 and round(np.prod(1 + monthly) - 1, 5) == 0.07635

    compounding = {periods: 1000 * (1 + 0.05 / periods) ** (periods * 10) for periods in (1, 4, 12, 365, 8760)}
    assert [f"{v:.2f}" for v in compounding.values()] == ["1628.89", "1643.62", "1647.01", "1648.66", "1648.72"]
    assert f"{1000 * math.exp(0.5):.2f}" == "1648.72"
    assert round(math.log(1.08), 5) == 0.07696 and round(math.expm1(0.08), 5) == 0.08329

    rng = np.random.default_rng(seed=4)
    trading_days = 2520
    daily_log = rng.normal(loc=0.0003, scale=0.02, size=trading_days)
    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(daily_log)]))
    daily_simple = prices[1:] / prices[:-1] - 1
    years = np.arange(trading_days + 1) / 252
    assert f"{prices[-1]:.2f}" == "222.63" and f"{daily_log.sum():.6f}" == "0.800351"
    assert f"{daily_simple.sum():.6f}" == "1.298185" and f"{prices[-1] / prices[0] - 1:.6f}" == "1.226321"
    assert f"{daily_simple.mean():.6f}" == "0.000515" and f"{daily_log.mean():.6f}" == "0.000318"
    assert f"{daily_simple.mean() - daily_simple.var() / 2:.6f}" == "0.000318"
    running_log = np.concatenate([[0.0], np.cumsum(daily_log)])
    running_simple = np.concatenate([[0.0], np.cumsum(daily_simple)])

    rng = np.random.default_rng(seed=1)
    players, rounds = 20000, 50
    factors = np.where(rng.random((players, rounds)) < 0.5, 1.5, 0.5)
    wealth = np.hstack([np.ones((players, 1)), np.cumprod(factors, axis=1)])
    expected_log_return = 0.5 * (np.log(1.5) + np.log(0.5))
    assert round(expected_log_return, 4) == -0.1438
    assert round(wealth[:, -1].mean(), 3) == 1.643 and f"{np.median(wealth[:, -1]):.2e}" == "7.53e-04"

    fig = new_figure("Log returns add up over time; simple returns multiply",
                     r"$r = \ln(P_1/P_0) = \ln(1 + R)$  $\Leftrightarrow$  $R = e^{r} - 1$,     "
                     r"$\ln(P_T/P_0) = r_1 + \cdots + r_T$,     $A = P\,e^{rt}$,     "
                     r"mean log return $\approx \mu - \sigma^2/2$")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    simple_return = np.linspace(-0.8, 1.5, 400)
    ax.plot(simple_return, simple_return, "--", color=MUTED, linewidth=1.4, label="tangent r = R")
    ax.fill_between(simple_return, np.log1p(simple_return), simple_return, color=ORANGE_LIGHT, alpha=0.7,
                    label=r"gap $R - \ln(1+R) \approx R^2/2$")
    ax.plot(simple_return, np.log1p(simple_return), color=BLUE, linewidth=2.2, label=r"$r = \ln(1 + R)$")
    for point, offset in ((0.5, (10, -22)), (-0.5, (10, -10))):
        ax.plot(point, np.log1p(point), "o", color=ORANGE, zorder=4)
        ax.annotate(f"R = {point:+.0%}\nr = {np.log1p(point):+.4f}", (point, np.log1p(point)),
                    xytext=offset, textcoords="offset points", fontsize=9.5)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("simple return R")
    ax.set_ylabel("return")
    ax.grid(True)
    legend(ax, loc="upper left")
    ax.set_title(r"$+50\%$ then $-50\%$: $0.4055 - 0.6931 = \ln 0.75$")

    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(years, prices, color=BLUE, linewidth=1)
    ax.semilogy(years, 100 * np.exp(0.0003 * np.arange(trading_days + 1)), "--", color=ORANGE, linewidth=1.6,
                label=r"$100\,e^{0.0003\,t}$: constant growth is a line")
    ax.plot(years[-1], prices[-1], "o", color=INK, zorder=4)
    ax.annotate(f"{prices[-1]:.2f}", (years[-1], prices[-1]), xytext=(-40, 10), textcoords="offset points",
                fontsize=10)
    ax.set_xlabel("years (2520 trading days)")
    ax.set_ylabel("price (log scale)")
    ax.grid(True, which="both")
    legend(ax, loc="upper left")
    ax.set_title("Simulated price (seed 4) on a log axis")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(years, np.log(prices / prices[0]), color=INK, linewidth=4, alpha=0.3, label=r"$\ln(P_t/P_0)$")
    ax.plot(years, running_log, color=BLUE, linewidth=1.1, label="running sum of log returns")
    ax.plot(years, running_simple, color=ORANGE, linewidth=1.1, label="running sum of simple returns")
    ax.text(10.1, running_log[-1], f"{running_log[-1]:.3f}", va="center", fontsize=10, color=BLUE)
    ax.text(10.1, running_simple[-1], f"{running_simple[-1]:.3f}", va="center", fontsize=10, color=ORANGE)
    ax.set_xlim(0, 11.6)
    ax.set_xlabel("years")
    ax.set_ylabel("cumulative value")
    ax.grid(True)
    legend(ax, loc="upper left")
    ax.set_title(r"Only log returns sum to $\ln(P_T/P_0)$")

    ax = fig.add_subplot(gs[1, 0])
    periods = np.unique(np.logspace(0, 6, 60).astype(int))
    ax.semilogx(periods, (1 + 1 / periods) ** periods, "o-", markersize=3, color=BLUE,
                label=r"$(1 + 1/m)^m$")
    ax.axhline(np.e, color=ORANGE, linestyle="--", label=r"$e = 2.71828\ldots$")
    for m, name, offset in ((1, "annual", (25, 15)), (12, "monthly", (30, -40)), (365, "daily", (30, -40))):
        value = (1 + 1 / m) ** m
        ax.annotate(f"{name}: {value:.4f}", (m, value), xytext=offset, textcoords="offset points", fontsize=9.5,
                    arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("compounding periods per year m (log scale)")
    ax.set_ylabel("growth factor after one year at 100%")
    ax.set_ylim(1.9, 2.85)
    ax.grid(True, which="major")
    legend(ax, loc="lower right")
    ax.set_title(r"Compounding m times a year: $(1 + 1/m)^{m} \to e$")

    ax = fig.add_subplot(gs[1, 1])
    round_index = np.arange(rounds + 1)
    for path in wealth[:30]:
        ax.semilogy(round_index, path, color=MUTED, alpha=0.35, linewidth=0.8)
    ax.semilogy(round_index, np.median(wealth, axis=0), color=BLUE, linewidth=2.2, label="median wealth")
    ax.semilogy(round_index, np.exp(expected_log_return * round_index), "--", color=ORANGE, linewidth=1.8,
                label=r"$e^{-0.1438\,n}$")
    ax.axhline(1.0, color=INK, linewidth=1.2, label="expected wealth = 1")
    ax.set_xlabel("round n")
    ax.set_ylabel("wealth (log scale)")
    ax.grid(True, which="major")
    legend(ax, loc="lower left")
    ax.set_title("Coin game ×1.5 or ×0.5: fair mean, losing median")

    panel(fig.add_subplot(gs[1, 2]), "Numbers from the notebook", [
        "Months +10%, −5%, +3%:",
        "  0.09531 − 0.05129 + 0.02956 = 0.07358",
        r"  $e^{0.07358} - 1$ = 7.635%  (naive sum: 8%)",
        "EUR 1000 at 5% for 10 years:",
        "  annual 1628.89, monthly 1647.01,",
        r"  continuous $1000\,e^{0.5}$ = 1648.72",
        "effective 8% = continuous 7.696%;",
        "continuous 8% = effective 8.329%",
        "Seed-4 path: mean simple 0.000515,",
        r"  mean log 0.000318 = mean simple $-\ \sigma^2/2$",
        "Coin game, 50 rounds: mean 1.643, median 7.53e-04",
    ], fontsize=11.3, spacing=0.075)
    save_06(fig, name="02_log_returns_and_continuous_compounding.png")


# ================================================================ 06 / 03 music
def continued_fraction(value, terms):
    coefficients = []
    for _ in range(terms):
        whole = math.floor(value)
        coefficients.append(whole)
        fractional = value - whole
        if fractional < 1e-12:
            break
        value = 1 / fractional
    return coefficients


def convergents(coefficients):
    results = []
    num_prev, num = 1, coefficients[0]
    den_prev, den = 0, 1
    results.append(Fraction(num, den))
    for a in coefficients[1:]:
        num_prev, num = num, a * num + num_prev
        den_prev, den = den, a * den + den_prev
        results.append(Fraction(num, den))
    return results


def sheet_06_03_music():
    def note_frequency(n):
        return 440 * 2 ** ((n - 69) / 12)

    assert round(2 ** (1 / 12), 6) == 1.059463
    assert [f"{note_frequency(n):.2f}" for n in (60, 62, 72)] == ["261.63", "293.66", "523.25"]
    assert round(note_frequency(21), 2) == 27.5 and round(note_frequency(108), 2) == 4186.01
    assert round(1200 * math.log2(450 / 440), 2) == 38.91
    assert round(69 + 12 * math.log2(300 / 440), 3) == 62.37
    assert round(2 ** (1 / 1200), 6) == 1.000578

    intervals = [("fifth 3/2", 3 / 2, 7), ("fourth 4/3", 4 / 3, 5), ("major third 5/4", 5 / 4, 4),
                 ("minor third 6/5", 6 / 5, 3), ("major sixth 5/3", 5 / 3, 9), ("minor sixth 8/5", 8 / 5, 8)]
    errors = [100 * s - 1200 * math.log2(ratio) for _, ratio, s in intervals]
    assert [f"{e:+.2f}" for e in errors[:5]] == ["-1.96", "+1.96", "+13.69", "-15.64", "+15.64"]

    scale_length = 650.0
    frets = np.arange(0, 20)
    positions = scale_length * (1 - 2.0 ** (-frets / 12))
    assert [f"{p:.2f}" for p in positions[[1, 7, 12]]] == ["36.48", "216.18", "325.00"]
    assert round(1 / (1 - 2 ** (-1 / 12)), 3) == 17.817

    fifth = math.log2(3 / 2)
    cf = continued_fraction(fifth, 8)
    assert cf == [0, 1, 1, 2, 2, 3, 1, 5]
    conv = convergents(cf)
    assert [str(c) for c in conv[3:7]] == ["3/5", "7/12", "24/41", "31/53"]
    comma = Fraction(3, 2) ** 12 / 2 ** 7
    comma_cents = 1200 * math.log2(comma)
    assert round(float(comma), 6) == 1.013643 and round(comma_cents, 3) == 23.46
    assert round(comma_cents / 12, 3) == 1.955

    just_fifth_cents = 1200 * fifth
    divisions = np.arange(1, 61)
    best_errors = np.array([abs(round(just_fifth_cents / (1200 / n)) * (1200 / n) - just_fifth_cents)
                            for n in divisions])
    assert [f"{best_errors[n - 1]:.2f}" for n in (5, 12, 29, 41, 53)] == ["18.04", "1.96", "1.49", "0.48", "0.07"]
    assert int(np.argmax(best_errors < 2)) + 1 == 12

    fig = new_figure("Pitch is a logarithm: equal intervals are equal frequency ratios",
                     r"interval in cents $c = 1200\,\log_2(f_2/f_1)$  $\Leftrightarrow$  $f_2/f_1 = 2^{c/1200}$,     "
                     r"$f(n) = 440 \cdot 2^{(n-69)/12}$ Hz,     frets at $d_n = L\,(1 - 2^{-n/12})$")
    gs = grid(fig, top=0.84, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    notes = np.arange(21, 109)
    freqs = 440 * 2.0 ** ((notes - 69) / 12)
    is_a = (notes - 69) % 12 == 0
    ax.plot(notes, freqs, "o", markersize=2.5, color=BLUE_LIGHT)
    ax.plot(notes[is_a], freqs[is_a], "o", markersize=7, color=ORANGE, label="the A's: 27.5, 55, …, 3520 Hz")
    ax.set_yscale("log", base=2)
    ax.set_yticks(freqs[is_a], [f"{f:g}" for f in freqs[is_a]])
    ax.minorticks_off()
    ax.set_xticks([21, 33, 45, 57, 69, 81, 93, 105])
    ax.set_xlabel("MIDI note number n  (piano keys 21 to 108)")
    ax.set_ylabel("frequency in Hz (log scale, base 2)")
    ax.grid(True)
    legend(ax, loc="upper left")
    ax.set_title(r"88 keys: $n = 69 + 12\,\log_2(f/440)$ is a line")

    sub = gs[0, 1].subgridspec(2, 1, hspace=0.9)
    ax_board = fig.add_subplot(sub[0])
    ax_board.add_patch(Rectangle((0, 0), scale_length, 1, color=ORANGE_LIGHT, alpha=0.8))
    ax_board.vlines(positions, 0, 1, color=INK2, linewidth=1.8)
    for fret in (1, 3, 5, 7, 9, 12, 15, 17, 19):
        ax_board.text(positions[fret], 1.12, str(fret), ha="center", fontsize=9)
    ax_board.set_xlim(-10, scale_length + 10)
    ax_board.set_ylim(0, 1.45)
    ax_board.set_yticks([])
    ax_board.spines["left"].set_visible(False)
    ax_board.set_xlabel("distance from the nut (mm): gaps shrink by 0.9439")
    ax_board.set_title("650 mm string: fret 12 at exactly 325 mm", pad=14)
    ax_log = fig.add_subplot(sub[1])
    octaves = -np.log2((scale_length - positions) / scale_length)
    ax_log.vlines(octaves, 0, 1, color=BLUE, linewidth=1.8)
    ax_log.set_xlim(-0.03, 1.65)
    ax_log.set_ylim(0, 1)
    ax_log.set_yticks([])
    ax_log.spines["left"].set_visible(False)
    ax_log.set_xticks([0, 0.5, 1, 1.5], ["0", "1/2", "1", "3/2"])
    ax_log.set_xlabel(r"$-\log_2$(remaining length / L) in octaves: equal steps 1/12")

    ax = fig.add_subplot(gs[0, 2])
    names = [name.replace(" ", "\n", 1) for name, _, _ in intervals]
    colours = [BLUE if abs(e) < 5 else ORANGE for e in errors]
    pos = np.arange(len(errors))
    ax.bar(pos, errors, color=colours, width=0.62, zorder=2)
    for p, e in zip(pos, errors):
        ax.text(p, e + (0.7 if e >= 0 else -0.7), f"{e:+.2f}", ha="center", va="bottom" if e >= 0 else "top",
                fontsize=10)
    ax.axhline(0, color=INK2, linewidth=0.9)
    ax.set_xticks(pos, names, fontsize=9)
    ax.set_ylim(-20, 20)
    ax.set_ylabel("12-TET minus just interval (cents)")
    ax.grid(True, axis="y", zorder=0)
    ax.set_title("Fifths almost pure, thirds 14–16 cents off")

    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(divisions, best_errors, "o-", markersize=3.5, color=BLUE_LIGHT, linewidth=1)
    for n in (5, 12, 29, 41, 53):
        ax.plot(n, best_errors[n - 1], "o", color=ORANGE, markersize=8, zorder=4)
        ax.annotate(f"N = {n}\n{best_errors[n - 1]:.2f} c", (n, best_errors[n - 1]), xytext=(0, -30),
                    textcoords="offset points", ha="center", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor=SURFACE, edgecolor="none", alpha=0.9))
    ax.set_ylim(0.01, 1500)
    ax.set_xlabel("notes per octave N")
    ax.set_ylabel("error of the best fifth (cents, log scale)")
    ax.grid(True, which="major")
    ax.set_title("Which equal division gives a good fifth?")

    panel(fig.add_subplot(gs[1, 1]), "Why 12 keys per octave", [
        r"$(3/2)^k = 2^m$ would need $\log_2(3/2) = m/k$:",
        r"impossible, $3^k$ is odd and $2^{m+k}$ even.",
        r"$\log_2(3/2) = 0.5849625\ldots = [0; 1, 1, 2, 2, 3, 1, 5]$",
        "convergents 3/5, 7/12, 24/41, 31/53",
        "",
        r"12 fifths vs 7 octaves: $(3/2)^{12} = 129.75$ vs $2^7 = 128$",
        "Pythagorean comma 531441/524288",
        r"  = $1200\,\log_2 1.013643$ = 23.46 cents",
        "12-TET shrinks each fifth by 23.46 / 12 = 1.955 c",
    ], spacing=0.085)

    panel(fig.add_subplot(gs[1, 2]), "Exponent form and log form", [
        r"semitone $2^{1/12}$ = 1.059463",
        r"cent $2^{1/1200}$ = 1.000578 (about 0.06%)",
        r"C4 = $440 \cdot 2^{-9/12}$ = 261.63 Hz (note 60)",
        r"C5 = $440 \cdot 2^{3/12}$ = 523.25 Hz",
        "440 → 450 Hz: 1200 log₂(450/440) = 38.91 cents",
        "300 Hz: n = 69 + 12 log₂(300/440) = 62.37,",
        "  37 cents above D4 (293.66 Hz)",
        "piano A0 27.5 Hz to C8 4186.01 Hz:",
        "  log₂(4186.01/27.5) = 7.25 octaves = 87/12",
        r"rule of 18: exact divisor $1/(1 - 2^{-1/12})$ = 17.817",
    ], fontsize=11.8, spacing=0.082)
    save_06(fig, name="03_music_octaves_and_cents.png")


# ================================================================ 06 / 04 algorithms
def binary_search(sorted_items, target):
    low, high = 0, len(sorted_items) - 1
    probes = 0
    while low <= high:
        middle = (low + high) // 2
        probes += 1
        if sorted_items[middle] == target:
            return middle, probes
        if sorted_items[middle] < target:
            low = middle + 1
        else:
            high = middle - 1
    return None, probes


def merge_sort_count(items):
    if len(items) <= 1:
        return items, 0
    middle = len(items) // 2
    left, a = merge_sort_count(items[:middle])
    right, b = merge_sort_count(items[middle:])
    merged, i, j, count = [], 0, 0, a + b
    while i < len(left) and j < len(right):
        count += 1
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    return merged + left[i:] + right[j:], count


def search_tree_height(keys):
    left_child, right_child = {}, {}
    root, height = keys[0], 0
    for key in keys[1:]:
        node, depth = root, 0
        while True:
            depth += 1
            children = left_child if key < node else right_child
            if node in children:
                node = children[node]
            else:
                children[node] = key
                break
        height = max(height, depth)
    return height


def balanced_order(sorted_keys):
    order, stack = [], [(0, len(sorted_keys) - 1)]
    while stack:
        low, high = stack.pop()
        if low > high:
            continue
        middle = (low + high) // 2
        order.append(sorted_keys[middle])
        stack.append((low, middle - 1))
        stack.append((middle + 1, high))
    return order


def sheet_06_04_algorithms():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    assert binary_search(primes, 23) == (8, 4) and binary_search(primes, 24) == (None, 4)
    measured_worst = {}
    for n in (10, 100, 1000, 10_000, 100_000):
        counts = [binary_search(range(n), t)[1] for t in range(n)]
        measured_worst[n] = max(counts)
        assert max(counts) == math.floor(math.log2(n)) + 1
    assert [measured_worst[n] for n in measured_worst] == [4, 7, 10, 14, 17]
    phone_book = range(0, 2 * 10 ** 9, 2)
    assert binary_search(phone_book, 1_999_999_998) == (999_999_999, 30)
    assert binary_search(phone_book, 777) == (None, 30)

    random.seed(0)
    merge_table = {}
    for n in (16, 256, 4096, 65536):
        data = [random.random() for _ in range(n)]
        merge_table[n] = merge_sort_count(data)[1]
    assert list(merge_table.values()) == [48, 1711, 44032, 965883]
    assert round(math.lgamma(4096 + 1) / math.log(2)) == 43250

    @lru_cache(maxsize=None)
    def recurrence_cost(n):
        return 0 if n <= 1 else recurrence_cost(n // 2) + recurrence_cost(n - n // 2) + n

    assert recurrence_cost(1024) == 10240 and recurrence_cost(10 ** 6) == 19951424

    def multiplications(n):
        return math.floor(math.log2(n)) + bin(n).count("1") - 1

    assert [multiplications(n) for n in (13, 45, 64, 1000, 10 ** 6, 2 ** 20 - 1)] == [5, 8, 6, 14, 25, 38]

    random.seed(1)
    heights = {}
    for n in (15, 127, 1023, 4000):
        keys = list(range(n))
        shuffled = keys[:]
        random.shuffle(shuffled)
        heights[n] = (search_tree_height(balanced_order(keys)), search_tree_height(shuffled),
                      search_tree_height(keys))
    assert heights == {15: (3, 7, 14), 127: (6, 12, 126), 1023: (9, 19, 1022), 4000: (11, 25, 3999)}

    random.seed(2)
    sizes = np.unique(np.logspace(1, 4.7, 15).astype(int))
    measured = [merge_sort_count([random.random() for _ in range(n)])[1] for n in sizes]

    fig = new_figure(r"Halving $n$ down to 1 takes $\log_2 n$ steps",
                     r"$n/2^k \leq 1 \Leftrightarrow 2^k \geq n \Leftrightarrow k \geq \log_2 n$:     "
                     r"binary search $\leq \lfloor\log_2 n\rfloor + 1$ probes,     "
                     r"$T(n) = 2\,T(n/2) + n = n\,\log_2 n$,     $x^n$ in $\leq 2\,\log_2 n$ multiplications")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    grid_sizes = np.unique(np.logspace(0, 9, 400).astype(int))
    ax.step(grid_sizes, [math.floor(math.log2(n)) + 1 for n in grid_sizes], where="post", color=BLUE,
            linewidth=2, label=r"worst case $\lfloor\log_2 n\rfloor + 1$")
    ax.plot(grid_sizes, np.log2(grid_sizes), "--", color=ORANGE, label=r"$\log_2 n$")
    ax.plot(list(measured_worst), list(measured_worst.values()), "o", color=INK, markersize=6,
            label="measured worst case", zorder=4)
    ax.plot(1e9, 30, "*", color=ORANGE, markersize=14, zorder=5)
    ax.annotate("phone book: $10^9$ names,\n30 probes", (1e9, 30), xytext=(30, 25), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xscale("log")
    ax.set_xlabel("list length n (log scale)")
    ax.set_ylabel("probes")
    ax.grid(True, which="major")
    legend(ax, loc="lower right")
    ax.set_title("Binary search: a straight line on a log axis")

    ax = fig.add_subplot(gs[0, 1])
    n, levels = 16, 5
    for level in range(levels):
        block = n // 2 ** level
        for k in range(2 ** level):
            ax.add_patch(Rectangle((k * block + 0.1, -level - 0.38), block - 0.2, 0.76,
                                   facecolor=BLUE, alpha=0.18 + 0.14 * level, edgecolor=SURFACE, linewidth=1))
            if block >= 2:
                ax.text(k * block + block / 2, -level, str(block), ha="center", va="center", fontsize=9)
        text = f"{2 ** level} × {block}: work {n}" if level < levels - 1 else "16 × 1: no work"
        ax.text(n + 0.6, -level, text, va="center", fontsize=9.5, color=INK2)
    ax.set_xlim(0, n + 8.5)
    ax.set_ylim(-levels + 0.45, 0.6)
    ax.axis("off")
    ax.set_title(r"Merge sort, n = 16: $\log_2 16 = 4$ levels × 16 = 64")

    ax = fig.add_subplot(gs[0, 2])
    ax.loglog(sizes, sizes ** 2 / 2, color=ORANGE, linewidth=2, label=r"$n^2/2$ (insertion sort)")
    ax.loglog(sizes, sizes * np.log2(sizes), color=BLUE, linewidth=2, label=r"$n\,\log_2 n$")
    ax.loglog(sizes, sizes, "--", color=MUTED, label="n")
    ax.loglog(sizes, measured, "o", color=INK, markersize=5, label="merge sort, measured (seed 2)")
    ax.set_xlabel("n (log scale)")
    ax.set_ylabel("comparisons (log scale)")
    ax.grid(True, which="major")
    legend(ax, loc="upper left")
    ax.set_title(r"Sorting: $n\,\log_2 n$ against $n^2/2$")

    ax = fig.add_subplot(gs[1, 0])
    exponents = np.arange(2, 2001)
    ax.semilogx(exponents, [multiplications(int(e)) for e in exponents], ".", markersize=2.5, color=BLUE,
                label="repeated squaring, exact count")
    ax.semilogx(exponents, np.log2(exponents), color=AQUA, linewidth=2, label=r"$\log_2 n$")
    ax.semilogx(exponents, 2 * np.log2(exponents), color=ORANGE, linewidth=2, label=r"$2\,\log_2 n$")
    ax.annotate(r"$3^{13}$: 5 multiplications, not 12", (13, 5), xytext=(30, 1.5), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("exponent n (log scale)")
    ax.set_ylabel("multiplications")
    ax.grid(True, which="major")
    legend(ax, loc="upper left")
    ax.set_title(r"$x^n$: $\lfloor\log_2 n\rfloor$ + (1-bits) − 1 multiplications")

    ax = fig.add_subplot(gs[1, 1])
    tree_ns = list(heights)
    pos = np.arange(len(tree_ns))
    width = 0.26
    for k, (name, colour) in enumerate((("balanced", AQUA), ("random (seed 1)", BLUE), ("sorted insertion", ORANGE))):
        values = [heights[n][k] for n in tree_ns]
        ax.bar(pos + (k - 1) * width, values, width=width - 0.03, color=colour, label=name, zorder=2)
        for p, v in zip(pos, values):
            ax.text(p + (k - 1) * width, v * 1.12, str(v), ha="center", va="bottom", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(1, 3e4)
    ax.set_xticks(pos, [f"n = {n}" for n in tree_ns])
    ax.set_ylabel("tree height (log scale)")
    ax.grid(True, axis="y", which="major", zorder=0)
    legend(ax, loc="upper left")
    ax.set_title(r"Search-tree height: $\lfloor\log_2 n\rfloor$ or $n - 1$")

    panel(fig.add_subplot(gs[1, 2]), "Counts from the notebook", [
        r"$\log_2 10^9 = 9\,\log_2 10 = 29.90$: 30 probes",
        "`    n  merge  n log2 n  log2(n!)`",
        "`   16     48        64        44`",
        "` 4096  44032     49152     43250`",
        "`65536 965883   1048576    954037`",
        r"lower bound $\log_2 n! \approx n\,\log_2 n - 1.44\,n$",
        r"$T(10^6)$ = 19,951,424 vs $n\,\log_2 n$ = 19,931,569",
        r"Base does not matter: $\log_b n = \log_2 n\ /\ \log_2 b$,",
        r"so ternary search costs $2\,\log_3 n = 1.26\,\log_2 n$",
    ], fontsize=11.8, spacing=0.085)
    save_06(fig, name="04_logarithmic_algorithms.png")


# ================================================================ 06 / 05 discrete logarithm
def bsgs(target, generator, prime):
    order = prime - 1
    m = math.isqrt(order)
    if m * m < order:
        m += 1
    operations = 0
    baby_steps = {}
    value = 1
    for j in range(m):
        baby_steps.setdefault(value, j)
        value = (value * generator) % prime
        operations += 1
    giant_factor = pow(generator, -m, prime)
    current = target % prime
    for i in range(m):
        if current in baby_steps:
            return (i * m + baby_steps[current]) % order, operations
        current = (current * giant_factor) % prime
        operations += 1
    return None, operations


def sheet_06_05_discrete_logarithm():
    assert pow(5, 6, 23) == 8 and pow(5, 15, 23) == 19
    assert pow(19, 6, 23) == 2 and pow(8, 15, 23) == 2
    table_11 = [pow(2, x, 11) for x in range(10)]
    assert table_11 == [1, 2, 4, 8, 5, 10, 9, 7, 3, 6]
    log_11 = {v: x for x, v in enumerate(table_11)}
    assert log_11[3] + log_11[5] == 12 and log_11[15 % 11] == 2
    assert bsgs(8, 5, 23) == (6, 6) and bsgs(pow(2, 765_432, 1_000_003), 2, 1_000_003) == (765432, 1765)
    assert sp.primitive_root(23) == 5

    cost_table = {}
    for size in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 8, 10 ** 10, 10 ** 11):
        prime = int(sp.nextprime(size))
        generator = int(sp.primitive_root(prime))
        found, ops = bsgs(pow(generator, prime - 2, prime), generator, prime)
        assert found == prime - 2
        cost_table[prime] = ops
    assert cost_table == {1009: 63, 10007: 200, 100003: 632, 1000003: 2000, 100000007: 20000,
                          10000000019: 200000, 100000000003: 632455}

    plot_primes = [int(sp.nextprime(10 ** k)) for k in range(2, 11)]
    plot_bsgs = []
    for prime in plot_primes:
        generator = int(sp.primitive_root(prime))
        plot_bsgs.append(bsgs(pow(generator, prime - 2, prime), generator, prime)[1])

    prime = int(sp.nextprime(10 ** 10))
    generator = int(sp.primitive_root(prime))
    rng = random.Random(42)
    alice_secret = rng.randrange(2, prime - 1)
    bob_secret = rng.randrange(2, prime - 1)
    alice_public, bob_public = pow(generator, alice_secret, prime), pow(generator, bob_secret, prime)
    shared = pow(bob_public, alice_secret, prime)
    assert (prime, generator, alice_public, bob_public, shared) == (10000000019, 2, 8187152157, 974135636,
                                                                    1997098755)
    assert bsgs(alice_public, generator, prime)[0] == alice_secret == 2746317215

    p_small = 1019
    g_small = int(sp.primitive_root(p_small))
    exponents = np.arange(p_small - 1)
    values = np.array([pow(g_small, int(x), p_small) for x in exponents])
    true_log = 400
    target = pow(g_small, true_log, p_small)
    value_distance = np.abs(values - target)
    exponent_distance = np.abs(exponents - true_log)
    correlation = np.corrcoef(exponent_distance, value_distance)[0, 1]
    assert round(correlation, 3) == 0.109
    years = {k: 10 ** ((k / 2 + 1) * np.log10(2) - 9) / 3.15e7 for k in (64, 128, 256)}
    assert [f"{y:.1e}" for y in years.values()] == ["2.7e-07", "1.2e+03", "2.2e+22"]

    fig = new_figure("The discrete logarithm: easy to exponentiate, hard to invert",
                     r"$g^x \equiv h \ (\mathrm{mod}\ p)$  $\Leftrightarrow$  $x = \log_g h \ (\mathrm{mod}\ p - 1)$:     "
                     r"pow(g, x, p) costs $\approx 2\,\log_2 p$ steps,  baby-step giant-step $\approx 2\sqrt{p}$,"
                     r"  brute force $\approx p$")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    x_real = np.linspace(0, 10, 400)
    ax.plot(x_real, 2.0 ** x_real, color=BLUE, linewidth=2.2, label=r"$2^x$")
    ax.axhline(600, color=ORANGE, linestyle="--", label="target h = 600")
    ax.axvline(np.log2(600), color=MUTED, linestyle=":",
               label=rf"$x = \log_2 600 = {np.log2(600):.3f}$")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$2^x$")
    ax.grid(True)
    legend(ax, loc="upper left")
    ax.set_title("Real: increasing, so bisection finds the log")

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(exponents, values, s=4, color=ORANGE)
    ax.set_xlabel("x")
    ax.set_ylabel(rf"${g_small}^x \ \mathrm{{mod}}\ {p_small}$")
    ax.grid(True)
    ax.set_title(rf"Modular: ${g_small}^x \ \mathrm{{mod}}\ {p_small}$ has no order to exploit")

    panel(fig.add_subplot(gs[0, 2]), "Toy Diffie–Hellman, p = 23, g = 5", [
        r"Alice a = 6:  $A = 5^{6}$ mod 23 = 8",
        r"Bob b = 15:  $B = 5^{15}$ mod 23 = 19",
        r"$19^{6} \equiv 2 \equiv 8^{15}$ (mod 23): shared s = 2,",
        r"because $(g^b)^a = g^{ab} = (g^a)^b$",
        "Eve needs log₅ 8 mod 23. Baby-step giant-step:",
        r"m = 5, baby steps $5^{j}$ = 1, 5, 2, 10, 4;",
        r"$5^{-5} \equiv 15$; giant steps 8, 8·15 ≡ 5 → j = 1",
        "x = 1·5 + 1 = 6 = a",
        "Log laws hold mod p − 1 (p = 11, g = 2):",
        "  log₂3 + log₂5 = 8 + 4 = 12 ≡ 2 = log₂4 (mod 10)",
    ], fontsize=11.5, spacing=0.083)

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(exponent_distance, value_distance, s=4, color=ORANGE)
    ax.set_xlabel("|x − true log|   (true log = 400)")
    ax.set_ylabel(rf"$|{g_small}^x \ \mathrm{{mod}}\ {p_small} - h|$")
    ax.grid(True)
    ax.set_title(f"Close in value says nothing (corr {correlation:.3f})")

    ax = fig.add_subplot(gs[1, 1])
    p_axis = np.logspace(2, 10, 100)
    ax.loglog(p_axis, p_axis, color=ORANGE, linewidth=2, label="brute force ~ p")
    ax.loglog(plot_primes, [p - 2 for p in plot_primes], "s", color=ORANGE, markersize=4)
    ax.loglog(p_axis, 2 * np.sqrt(p_axis), color=BLUE, linewidth=2, label=r"baby-step giant-step ~ $2\sqrt{p}$")
    ax.loglog(plot_primes, plot_bsgs, "o", color=BLUE, markersize=6, label="BSGS, measured")
    ax.loglog(p_axis, 2 * np.log2(p_axis), color=AQUA, linewidth=2, label=r"fast pow ~ $2\,\log_2 p$")
    ax.set_xlabel("prime p (log scale)")
    ax.set_ylabel("modular multiplications (log scale)")
    ax.grid(True, which="major")
    legend(ax, loc="upper left")
    ax.set_title("Easy direction versus hard direction")

    panel(fig.add_subplot(gs[1, 2]), "How the attack scales", [
        "`p                brute    BSGS`",
        "`1,009            1,007      63`",
        "`1,000,003    1,000,001   2,000`",
        "`10,000,000,019       -  200,000`",
        "p ×100 → BSGS work ×10: exponential in digits",
        "",
        "Toy DH at p = 10,000,000,019: Eve recovers",
        "a = 2,746,317,215 with BSGS in about 0.01 s",
        r"BSGS at $10^{9}$ ops/s: 64-bit p 2.7e-07 years,",
        "128-bit 1.2e+03 years, 256-bit 2.2e+22 years",
    ], fontsize=11.5, spacing=0.083)
    save_06(fig, name="05_discrete_logarithm_and_cryptography.png")


# ================================================================ 07 / 01 complex logarithms
def sheet_07_01_complex_logarithms():
    assert cmath.isclose(cmath.log(-1), 1j * math.pi)
    assert f"{cmath.log(1 + 1j):.4f}" == "0.3466+0.7854j" and f"{cmath.log(-1 - 1j):.4f}" == "0.3466-2.3562j"
    assert f"{cmath.log(-math.e):.4f}" == "1.0000+3.1416j"
    assert cmath.log(complex(-1, -0.0)).imag == -math.pi

    rng = np.random.default_rng(seed=7)
    counts = {-1: 0, 0: 0, 1: 0}
    for _ in range(10000):
        z = complex(*rng.normal(size=2))
        w = complex(*rng.normal(size=2))
        difference = cmath.log(z * w) - (cmath.log(z) + cmath.log(w))
        counts[round(difference.imag / (2 * math.pi))] += 1
    assert counts == {-1: 1278, 0: 7444, 1: 1278}
    z = -1 + 1j
    assert f"{(cmath.log(z) + cmath.log(z)).imag:.4f}" == "4.7124" and f"{cmath.log(z * z).imag:.4f}" == "-1.5708"
    i_to_i = {k: math.exp(-math.pi / 2 - 2 * math.pi * k) for k in (-1, 0, 1, 2)}
    assert [f"{v:.6g}" for v in i_to_i.values()] == ["111.318", "0.20788", "0.000388203", "7.24947e-07"]
    assert abs((1j ** 1j) - math.exp(-math.pi / 2)) < 1e-15

    fig = new_figure("The complex logarithm: log of the size, plus the angle",
                     r"$e^{w} = z$  $\Leftrightarrow$  $w = \ln|z| + i\,(\operatorname{Arg} z + 2\pi k)$,     "
                     r"$e^{i\pi} = -1$  $\Leftrightarrow$  $\ln(-1) = i\pi\,(2k + 1)$,     "
                     r"$\operatorname{Log}(zw) = \operatorname{Log} z + \operatorname{Log} w + 2\pi i\,n$")
    gs = grid(fig, top=0.84, wspace=0.3)

    radii = [0.5, 1, 2, 4, 8]
    radius_colours = [BLUE_LIGHT, "#86b6ef", BLUE, "#1c5cab", "#0d366b"]
    ray_angles = np.linspace(-np.pi, np.pi, 9)[1:]
    circle_t = np.linspace(-np.pi, np.pi, 400)
    ray_r = np.geomspace(0.4, 10, 200)

    ax_z = fig.add_subplot(gs[0, 0])
    ax_w = fig.add_subplot(gs[0, 1])
    for r, colour in zip(radii, radius_colours):
        circle = r * np.exp(1j * circle_t)
        ax_z.plot(circle.real, circle.imag, color=colour, linewidth=1.6)
        log_circle = np.log(circle)
        ax_w.plot(log_circle.real, log_circle.imag, color=colour, linewidth=2, label=f"|z| = {r:g}")
    for angle in ray_angles:
        is_cut = np.isclose(angle, np.pi)
        style = dict(color=ORANGE, linewidth=2.6, zorder=5) if is_cut else dict(color=MUTED, linewidth=0.9)
        ray = ray_r * np.exp(1j * angle)
        ax_z.plot(ray.real, ray.imag, **style)
        ax_w.plot(np.log(ray_r), np.full_like(ray_r, angle), **style)
        if is_cut:
            ax_w.plot(np.log(ray_r), np.full_like(ray_r, -np.pi), color=ORANGE, linewidth=2.6, linestyle="--")
    ax_z.set_xlim(-9, 9)
    ax_z.set_ylim(-9, 9)
    ax_z.set_aspect("equal")
    ax_z.set_xlabel("Re z")
    ax_z.set_ylabel("Im z")
    ax_z.set_title("z-plane: circles |z| = 0.5 … 8, rays (orange = cut)")
    ax_w.set_xlim(-1.2, 2.6)
    ax_w.set_ylim(-3.7, 3.7)
    ax_w.set_aspect("equal")
    pi_ticks(ax_w, "y", [-2, -1, 0, 1, 2], 2)
    ax_w.set_xticks([np.log(r) for r in radii], ["½", "1", "2", "4", "8"])
    ax_w.set_xlabel(r"Re $w = \ln|z|$ (ticks labelled by |z|)")
    ax_w.set_ylabel(r"Im $w = \operatorname{Arg} z$")
    ax_w.grid(True)
    ax_w.set_title(r"$w = \operatorname{Log} z$ unrolls the plane into a strip")

    panel(fig.add_subplot(gs[0, 2]), "Exponent form ⇔ log form", [
        r"$e^{0} = 1$  ⇔  $\ln 1 = 0 + 2\pi i k$",
        r"$e^{i\pi} = -1$  ⇔  $\ln(-1) = i\pi + 2\pi i k$",
        r"$e^{i\pi/2} = i$  ⇔  $\ln i = i\pi/2 + 2\pi i k$",
        r"$e^{1 + i\pi} = -e$  ⇔  $\ln(-e) = 1 + i\pi + 2\pi i k$",
        "",
        "`cmath.log(-1-1j) -> 0.3466-2.3562j`",
        "= ½ ln 2 − (3π/4) i,  Arg in (−π, π]",
        "Bernoulli: ln(−x) − ln x is constant, but the",
        r"constant is $\ln(-1) = i\pi$, not 0.",
    ], spacing=0.088)

    ax = fig.add_subplot(gs[1, 0])
    ax.axhspan(-np.pi, np.pi, color=GRID, alpha=0.7, label=r"principal strip $(-\pi, \pi]$")
    points = {"−1": (-1 + 0j, ORANGE), "i": (1j, BLUE), "1+i": (1 + 1j, AQUA)}
    for label, (zz, colour) in points.items():
        principal = cmath.log(zz)
        for k in range(-1, 2):
            value = principal + 2j * math.pi * k
            ax.plot(value.real, value.imag, "o" if k == 0 else "s", color=colour,
                    markersize=10 if k == 0 else 7, zorder=4)
        ax.text(principal.real + 0.07, principal.imag + 0.4, f"Log({label})", color=colour, fontsize=10)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axvline(0, color=INK2, linewidth=0.8)
    pi_ticks(ax, "y", range(-3, 4))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-10.5, 10.5)
    ax.set_xlabel(r"Re $w = \ln|z|$")
    ax.set_ylabel("Im w = angle")
    ax.grid(True)
    legend(ax, loc="upper left")
    ax.set_title(r"Each z has a column of logs, $2\pi$ apart")

    ax = fig.add_subplot(gs[1, 1])
    t = np.linspace(0, 2 * np.pi, 801)
    principal_angle = np.log(np.exp(1j * t)).imag
    ax.plot(t, t, "--", color=BLUE, linewidth=1.8, label="continuous log: Im = t")
    ax.plot(t[t <= np.pi], principal_angle[t <= np.pi], color=ORANGE, linewidth=2.6,
            label=r"principal Log: Im in $(-\pi, \pi]$")
    ax.plot(t[t > np.pi], principal_angle[t > np.pi], color=ORANGE, linewidth=2.6)
    ax.axvline(np.pi, color=MUTED, linestyle=":", label="crossing the negative real axis")
    pi_ticks(ax, "x", range(0, 5), 2)
    pi_ticks(ax, "y", [-1, 0, 1, 2])
    ax.set_xlabel(r"t on the path $z = e^{it}$")
    ax.set_ylabel("imaginary part of the logarithm")
    ax.grid(True)
    legend(ax, loc="upper left")
    ax.set_title(r"At the branch cut Log jumps by $2\pi$")

    panel(fig.add_subplot(gs[1, 2]), "The price: laws hold up to 2πi", [
        r"$\operatorname{Log}(-1) + \operatorname{Log}(-1) = 2\pi i$,  but  $\operatorname{Log} 1 = 0$",
        "z = w = −1 + i:  Log z + Log w = 0.6931 + 4.7124i,",
        "  Log(zw) = Log(−2i) = 0.6931 − 1.5708i  (n = −1)",
        "10,000 random pairs (seed 7):",
        "  n = −1: 1,278,  n = 0: 7,444,  n = +1: 1,278",
        "",
        r"$i^i = e^{i \ln i} = e^{-\pi/2 - 2\pi k}$: all real",
        "  k = 0: 0.20788,  k = −1: 111.32,  k = 1: 0.000388",
        "`cmath.log(complex(-1, -0.0)) -> -3.1416j`",
    ], fontsize=11.5, spacing=0.087)
    save_07(fig, name="01_logarithms_of_complex_numbers.png")


# ================================================================ 07 / 02 matrix exponential and logarithm
def apply_to_eigenvalues(M, scalar_function):
    eigenvalues, P = np.linalg.eig(np.asarray(M, dtype=complex))
    return (P @ np.diag(scalar_function(eigenvalues)) @ np.linalg.inv(P)).real


def expm_eig(A):
    return apply_to_eigenvalues(A, np.exp)


def logm_eig(M):
    return apply_to_eigenvalues(M, np.log)


def matrix_power(M, t):
    return expm_eig(t * logm_eig(M))


def expm_series(A, terms=30):
    result = np.eye(A.shape[0])
    term = np.eye(A.shape[0])
    for k in range(1, terms + 1):
        term = term @ A / k
        result = result + term
    return result


def rotation(theta):
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def sheet_07_02_matrix_exponential():
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    M = np.array([[2.0, 1.0], [0.0, 3.0]])
    L = logm_eig(M)
    assert np.round(L, 6).tolist() == [[0.693147, 0.405465], [0.0, 1.098612]]
    assert np.allclose(expm_eig(L), M)
    R = rotation(math.radians(30))
    log_R = logm_eig(R)
    assert np.allclose(log_R, math.pi / 6 * J) and np.allclose(log_R.T, -log_R)
    half = expm_eig(0.5 * log_R)
    assert np.allclose(half, rotation(math.radians(15))) and np.round(half, 4).tolist() == [[0.9659, -0.2588],
                                                                                          [0.2588, 0.9659]]
    roots = {}
    for k in (-1, 0, 1, 2):
        L_k = (math.pi / 6 + 2 * math.pi * k) * J
        assert np.allclose(expm_eig(L_k), R)
        root = expm_eig(0.5 * L_k)
        roots[k] = round(math.degrees(math.atan2(root[1, 0], root[0, 0])), 1)
    assert roots == {-1: -165.0, 0: 15.0, 1: -165.0, 2: 15.0}

    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert (expm_series(A) @ expm_series(B)).round(4).tolist() == [[2, 1], [1, 1]]
    assert expm_series(A + B).round(4).tolist() == [[1.5431, 1.1752], [1.1752, 1.5431]]
    rng = np.random.default_rng(seed=1)
    X = rng.normal(size=(3, 3))
    assert round(np.linalg.det(expm_series(X)), 10) == round(math.exp(np.trace(X)), 10) == 5.0304651497

    A_ode = np.array([[0.0, 1.0], [-2.0, -3.0]])
    x0 = np.array([1.0, 0.0])
    x_at_1 = expm_eig(A_ode) @ x0
    assert x_at_1.round(6).tolist() == [0.600424, -0.465088]
    assert round(2 * math.exp(-1) - math.exp(-2), 6) == 0.600424

    fig = new_figure(r"The matrix logarithm turns powers into multiples: $M^{t} = e^{t \ln M}$",
                     r"$e^{A} = \sum_k A^k/k! = P\,e^{D}P^{-1}$,     "
                     r"$\ln M = P\,\mathrm{diag}(\operatorname{Log}\mu_i)\,P^{-1}$,     "
                     r"$\det e^{A} = e^{\mathrm{tr}\,A}$,     $e^{A+B} = e^{A}e^{B}$ only if $AB = BA$")
    gs = grid(fig, top=0.84, wspace=0.3)

    shape = np.array([[0, 0], [1, 0], [1, 0.3], [0.3, 0.3], [0.3, 1], [0, 1], [0, 0]]).T
    t_values = np.linspace(0, 1, 5)
    ramp = [BLUE_LIGHT, "#86b6ef", BLUE, "#1c5cab", "#0d366b"]
    for column, (matrix, title, limits) in enumerate((
            (R, r"$R^t = e^{t\,(\pi/6)J}$: equal steps, equal angles", (-0.8, 1.3)),
            (M, r"$M^t$, $M = [[2, 1], [0, 3]]$: each step $\times M^{1/4}$", (-0.2, 3.3)))):
        ax = fig.add_subplot(gs[0, column])
        for colour, t in zip(ramp, t_values):
            image = matrix_power(matrix, t) @ shape
            ax.plot(image[0], image[1], color=colour, linewidth=2.2, label=f"t = {t:.2f}")
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal")
        ax.axhline(0, color=INK2, linewidth=0.6)
        ax.axvline(0, color=INK2, linewidth=0.6)
        ax.grid(True)
        legend(ax, loc="lower right" if column == 0 else "upper right")
        ax.set_title(title)

    panel(fig.add_subplot(gs[0, 2]), "Logs of a rotation: generators", [
        r"$J = [[0, -1], [1, 0]]$, $J^2 = -I$:  $e^{\theta J} = \cos\theta\,I + \sin\theta\,J$",
        r"$\ln R(30°) = (\pi/6)\,J$ = [[0, −0.5236], [0.5236, 0]]",
        r"$R^{1/2} = e^{(\pi/12)J}$ = R(15°)",
        "      = [[0.9659, −0.2588], [0.2588, 0.9659]]",
        r"other logs $(\pi/6 + 2\pi k)\,J$ give the root",
        "  rotating by 195° (atan2: −165°) = −R(15°)",
        "",
        "ln [[2, 1], [0, 3]] = [[0.693147, 0.405465],",
        "                         [0,          1.098612]]",
        r"$\ln\det M = \mathrm{tr}\,\ln M$:  $\ln 6 = \ln 2 + \ln 3$",
    ], fontsize=11.8, spacing=0.087)

    ax = fig.add_subplot(gs[1, 0])
    times = np.linspace(0, 12, 400)
    start_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    spiral_A = np.array([[-0.15, -1.0], [1.0, -0.15]])
    for angle in start_angles:
        start = np.array([np.cos(angle), np.sin(angle)])
        trajectory = np.array([expm_eig(A_ode * t) @ start for t in times])
        highlight = np.isclose(angle, 0)
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=ORANGE if highlight else BLUE_LIGHT,
                linewidth=2.4 if highlight else 1.4, zorder=3 if highlight else 2)
        ax.plot(*start, "o", color=INK, markersize=4)
    ax.plot(*x_at_1, "o", color=ORANGE, markersize=8, zorder=4)
    ax.annotate(f"x(1) = ({x_at_1[0]:.4f}, {x_at_1[1]:.4f})", x_at_1, xytext=(-1.3, -1.25), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(r"$\mathbf{x}' = A\mathbf{x}$, $A = [[0, 1], [-2, -3]]$: $\mathbf{x}(t) = e^{At}\mathbf{x}_0$")

    ax = fig.add_subplot(gs[1, 1])
    for angle in start_angles:
        start = np.array([np.cos(angle), np.sin(angle)])
        trajectory = np.array([expm_eig(spiral_A * t) @ start for t in times])
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=BLUE, linewidth=1.2, alpha=0.8)
        ax.plot(*start, "o", color=INK, markersize=4)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(r"$A = -0.15\,I + J$: $e^{At} = e^{-0.15t}R(t)$, a log spiral")

    panel(fig.add_subplot(gs[1, 2]), "Where the scalar laws break", [
        "A = [[0, 1], [0, 0]],  B = [[0, 0], [1, 0]],  AB ≠ BA:",
        r"$e^{A}e^{B}$ = [[2, 1], [1, 1]]",
        r"$e^{A+B}$ = [[cosh 1, sinh 1], [sinh 1, cosh 1]]",
        "       = [[1.5431, 1.1752], [1.1752, 1.5431]]",
        r"Both have det 1 = $e^{\mathrm{tr}(A+B)}$: det law always holds",
        r"3×3 X (seed 1): det $e^X$ = $e^{\mathrm{tr}\,X}$ = 5.0304651497",
        "",
        r"No real log if det M ≤ 0, since det $e^L = e^{\mathrm{tr}\,L} > 0$",
        r"Example 5: $x_1(1) = 2e^{-1} - e^{-2}$ = 0.600424",
    ], fontsize=11.5, spacing=0.087)
    save_07(fig, name="02_matrix_exponential_and_logarithm.png")


# ================================================================ 07 / 03 logarithmic spirals
def sheet_07_03_logarithmic_spirals():
    b = 0.2
    assert f"{math.degrees(math.atan(1 / b)):.4f}" == "78.6901" and f"{math.exp(2 * math.pi * b):.4f}" == "3.5136"
    shift = math.log(2) / b
    assert f"{-shift:.4f}" == "-3.4657" and f"{math.degrees(-shift):.1f}" == "-198.6"
    theta = np.linspace(-20, 20, 4001)
    spiral = np.exp((b + 1j) * theta)
    assert np.allclose(2 * spiral, np.exp((b + 1j) * (theta + shift)) * np.exp(-1j * shift))

    golden_ratio = (1 + math.sqrt(5)) / 2
    golden_b = math.log(golden_ratio) / (math.pi / 2)
    assert f"{golden_b:.4f}" == "0.3063" and f"{math.degrees(math.atan(1 / golden_b)):.2f}" == "72.97"
    assert f"{math.exp(2 * math.pi * golden_b):.3f}" == "6.854"
    golden_length = math.sqrt(1 + golden_b ** 2) / golden_b
    assert f"{golden_length:.6f}" == "3.413991"
    nautilus_b = math.log(3) / (2 * math.pi)
    assert f"{nautilus_b:.4f}" == "0.1748" and f"{math.degrees(math.atan(1 / nautilus_b)):.2f}" == "80.08"

    # falcon: the notebook's unit-speed simulation
    alpha = math.radians(40)
    r0 = 100.0
    position = np.array([r0, 0.0])
    step = 0.001
    path_length, unwrapped, previous = 0.0, 0.0, 0.0
    cos_a, sin_a = math.cos(alpha), math.sin(alpha)
    while math.hypot(*position) > 1.0:
        to_prey = -position / math.hypot(*position)
        direction = np.array([cos_a * to_prey[0] + sin_a * to_prey[1], -sin_a * to_prey[0] + cos_a * to_prey[1]])
        position = position + step * direction
        path_length += step
        angle = math.atan2(position[1], position[0])
        change = angle - previous
        if change < -math.pi:
            change += 2 * math.pi
        elif change > math.pi:
            change -= 2 * math.pi
        unwrapped += change
        previous = angle
    assert f"{path_length:.2f}" == "129.24" and f"{(r0 - 1) / cos_a:.2f}" == "129.24"
    assert f"{math.degrees(unwrapped):.1f}" == "221.4" and f"{r0 / cos_a:.2f}" == "130.54"

    fig = new_figure(r"A logarithmic spiral is a straight line in log form: $\ln r = \ln a + b\,\theta$",
                     r"$r = a\,e^{b\theta}$,   $z = a\,e^{(b+i)\theta}$,     equiangular: $\cot\psi = b$,     "
                     r"per turn $\times\,e^{2\pi b}$,     length from the pole $s = r\sqrt{1+b^2}/b = r/\cos\psi$")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    th = np.linspace(-3 * np.pi, 2.2 * np.pi, 3000)
    r = np.exp(b * th)
    ax.plot(r * np.cos(th), r * np.sin(th), color=BLUE, linewidth=2)
    for ray_angle in np.arange(0, 2 * np.pi, np.pi / 3):
        ax.plot([0, 4.5 * np.cos(ray_angle)], [0, 4.5 * np.sin(ray_angle)], color=MUTED, linewidth=0.8)
        for turn in (0, 2 * np.pi):
            t0 = ray_angle + turn - 2 * np.pi
            rr = math.exp(b * t0)
            point = np.array([rr * math.cos(t0), rr * math.sin(t0)])
            tangent = np.array([b * math.cos(t0) - math.sin(t0), b * math.sin(t0) + math.cos(t0)])
            tangent = tangent / np.linalg.norm(tangent) * 0.35 * rr
            ax.plot(*np.c_[point - tangent, point + tangent], color=ORANGE, linewidth=1.6)
    ax.plot(0, 0, "+", color=INK, markersize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.grid(True)
    ax.set_title(r"$r = e^{0.2\theta}$: every tangent meets its ray at 78.69°")

    ax = fig.add_subplot(gs[0, 1])
    theta_pos = np.linspace(0.3, 4 * np.pi, 400)
    ax.plot(theta_pos, b * theta_pos, color=BLUE, linewidth=2.2, label=r"log spiral $r = e^{0.2\theta}$: a line")
    ax.plot(theta_pos, np.log(0.25 * theta_pos), color=ORANGE, linewidth=2.2,
            label=r"Archimedean $r = 0.25\,\theta$: curved")
    ax.text(6.0, -1.9, "one turn: +2πb = +1.257\nin ln r, i.e. ×3.514 in r", fontsize=10, color=INK2)
    pi_ticks(ax, "x", range(0, 5))
    ax.set_xlabel(r"$\theta$ (linear)")
    ax.set_ylabel(r"$\ln r$")
    ax.grid(True)
    legend(ax, loc="upper left")
    ax.set_title(r"Log form: slope $b$, intercept $\ln a$")

    ax = fig.add_subplot(gs[0, 2])
    th = np.linspace(-6 * np.pi, 1.2 * np.pi, 4000)
    z = np.exp((b + 1j) * th)
    ax.plot(z.real, z.imag, color=MUTED, linewidth=1, label="spiral z")
    ax.plot((2 * z).real, (2 * z).imag, color=ORANGE, linewidth=5, alpha=0.45, label="scaled by 2")
    th_longer = np.linspace(-6 * np.pi, 1.2 * np.pi + shift, 5000)
    turned = np.exp((b + 1j) * th_longer) * np.exp(-1j * shift)
    ax.plot(turned.real, turned.imag, color=BLUE, linewidth=1.6,
            label=r"rotated by $-\ln 2/b = -198.6°$")
    ax.plot(0, 0, "+", color=INK, markersize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True)
    legend(ax, loc="lower left", fontsize=8.8)
    ax.set_title("Self-similar: scaling = rotating")

    ax = fig.add_subplot(gs[1, 0])
    fibonacci = [1, 1, 2, 3, 5, 8, 13, 21]

    def rotate_90(v):
        return np.array([-v[1], v[0]])

    point, heading = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    end_points, corners = [], []
    for radius in fibonacci:
        centre = point + radius * rotate_90(heading)
        start_angle = math.degrees(math.atan2(*(point - centre)[::-1]))
        ax.add_patch(Arc(centre, 2 * radius, 2 * radius, theta1=start_angle, theta2=start_angle + 90,
                         color=INK, linewidth=1.4))
        end = centre + radius * heading
        corner = point + end - centre
        square = np.array([centre, point, corner, end, centre])
        ax.plot(square[:, 0], square[:, 1], color=GRID, linewidth=1)
        corners.extend(square)
        end_points.append(end)
        point, heading = end, rotate_90(heading)
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    pole = np.linalg.solve(np.eye(2) - golden_ratio * rot, end_points[-1] - golden_ratio * rot @ end_points[-2])
    relative = end_points[-1] - pole
    r_end, angle_end = np.hypot(*relative), math.atan2(relative[1], relative[0])
    for name, growth, colour in (("golden: ×6.85 per turn", golden_ratio ** 4, GOLD),
                                 ("nautilus-like: ×3 per turn", 3.0, AQUA)):
        bb = math.log(growth) / (2 * math.pi)
        tt = np.linspace(angle_end - 8 * np.pi, angle_end, 4000)
        rr = r_end * np.exp(bb * (tt - angle_end))
        ax.plot(pole[0] + rr * np.cos(tt), pole[1] + rr * np.sin(tt), color=colour, linewidth=2.2, label=name)
    ax.plot([], [], color=INK, linewidth=1.4, label="Fibonacci quarter circles")
    corners = np.array(corners)
    ax.set_aspect("equal")
    ax.set_xlim(corners[:, 0].min() - 1, corners[:, 0].max() + 1)
    ax.set_ylim(corners[:, 1].min() - 1, corners[:, 1].max() + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    legend(ax, loc="lower left", fontsize=8.8)
    ax.set_title("Golden spiral vs Fibonacci arcs vs nautilus")

    ax = fig.add_subplot(gs[1, 1])
    for alpha_degrees, colour in ((0, INK), (20, BLUE_LIGHT), (40, ORANGE), (60, BLUE), (75, AQUA)):
        if alpha_degrees == 0:
            ax.plot([r0, 0], [0, 0], color=colour, linewidth=2, label="α = 0°: straight, 100 m")
            continue
        a_rad = math.radians(alpha_degrees)
        tt = np.linspace(0, 12 * np.pi, 20000)
        rr = r0 * np.exp(-tt / math.tan(a_rad))
        ax.plot(rr * np.cos(tt), rr * np.sin(tt), color=colour, linewidth=2.4 if alpha_degrees == 40 else 1.6,
                label=f"α = {alpha_degrees}°: {r0 / math.cos(a_rad):.0f} m")
    ax.plot(0, 0, "*", color=INK, markersize=13)
    ax.text(3, -9, "prey", fontsize=10)
    ax.plot(r0, 0, "o", color=INK)
    ax.set_aspect("equal")
    ax.set_xlim(-60, 110)
    ax.set_ylim(-60, 90)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True)
    legend(ax, loc="lower right", fontsize=8.8)
    ax.set_title(r"Constant-angle dive: length $r_0/\cos\alpha$")

    panel(fig.add_subplot(gs[1, 2]), "Numbers from the notebook", [
        "`             per turn   b       ψ`",
        "`golden       6.854    0.3063  72.97 deg`",
        "`nautilus ~   3.000    0.1748  80.08 deg`",
        "`e^(0.2 θ)    3.514    0.2000  78.69 deg`",
        "",
        "Golden spiral, pole to r = 1: length 3.414",
        "  (31.8 turns inside r = 1 still add up to that)",
        "Falcon, α = 40°, 100 m out: dive 130.54 m;",
        "  simulated to 1 m: 129.24 m, sweeping 221.4°",
        r"$\cot\psi = b$, and scaling by $\lambda$ = rotating by $-\ln\lambda/b$",
    ], fontsize=11.5, spacing=0.083)
    save_07(fig, name="03_logarithmic_spirals.png")


if __name__ == "__main__":
    sheet_06_01_decibels_richter_and_ph()
    sheet_06_02_log_returns()
    sheet_06_03_music()
    sheet_06_04_algorithms()
    sheet_06_05_discrete_logarithm()
    sheet_07_01_complex_logarithms()
    sheet_07_02_matrix_exponential()
    sheet_07_03_logarithmic_spirals()
