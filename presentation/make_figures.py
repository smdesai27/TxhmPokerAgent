#!/usr/bin/env python3.10
"""
Generate the portfolio figure set for the AlphaHoldem-inspired RL writeup + deck.

Every chart is sourced from a committed artifact in docs/ (no fabricated data):
  - Leduc NashConv curves      -> docs/leduc_sweep7/sw7_nfsp7_{7,42}.json
  - Fixed-reference ladder      -> docs/strong_fchpa_ladder/{new_vs_champ,oracle_new}.json
  - Behavior probe              -> docs/strong_fchpa_ladder/behavior_probe.json
  - Champion / collapse behavior, calibration anchors -> docs/WRITEUP.md (endpoint values)

Run:  python3.10 presentation/make_figures.py
Out:  docs/figures/*.png (200 dpi, for slides) and *.svg (for the web page)
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------- #
# Shared style
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
OUT = os.path.join(DOCS, "figures")
# DECK mode: suppress in-figure titles (the slide's assertion headline replaces them)
# and write to docs/figures/deck/ so the web page keeps the titled versions.
DECK = os.environ.get("DECK_FIGS") == "1"
if DECK:
    OUT = os.path.join(DOCS, "figures", "deck")
os.makedirs(OUT, exist_ok=True)


def set_title(ax, text, **kw):
    if not DECK:
        ax.set_title(text, **kw)

INK = "#15233b"
MUTED = "#5b6b82"
GRID = "#e6eaf0"
BLUE = "#2563eb"
TEAL = "#0d9488"
GREEN = "#15a34a"
RED = "#dc2626"
AMBER = "#d97706"
GRAY = "#9aa6b8"
LIGHT = "#f1f5f9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 12.5,
    "text.color": INK,
    "axes.edgecolor": "#c9d2de",
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def load(rel):
    with open(os.path.join(DOCS, rel)) as f:
        return json.load(f)


def style_ax(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"),
                    dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  wrote {name}.png / .svg")


# --------------------------------------------------------------------------- #
# Fig 2 — the metric mirage (the killer chart)
# --------------------------------------------------------------------------- #
def fig_metric_mirage():
    nvc = load("strong_fchpa_ladder/new_vs_champ.json")
    orc = load("strong_fchpa_ladder/oracle_new.json")

    rows = [
        # label, value, (lo, hi), color, is_band
        ("Training signal\nvs co-evolving league", 7.5, (3, 12), GREEN, True),
        ("Sanity check\nnew vs new (oracle)", orc["bb_per_100"],
         (orc["ci95_lower"], orc["ci95_upper"]), GRAY, False),
        ("Real strength\nvs fixed champion (12k pairs)", nvc["bb_per_100"],
         (nvc["ci95_lower"], nvc["ci95_upper"]), RED, False),
    ]
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    style_ax(ax, grid_axis="x")
    ys = [2, 1, 0]
    for y, (lab, val, (lo, hi), color, band) in zip(ys, rows):
        if band:
            ax.barh(y, hi - lo, left=lo, height=0.46, color=color, alpha=0.85,
                    zorder=3, edgecolor="white")
            ax.text(3, y + 0.33, "+3 to +12 bb  ·  positive every iter of the 90k run",
                    va="bottom", ha="left", fontsize=11.5, color=GREEN, weight="bold")
        else:
            ax.barh(y, val, height=0.46, color=color, alpha=0.92, zorder=3,
                    edgecolor="white")
            ax.errorbar(val, y, xerr=[[val - lo], [hi - val]], fmt="none",
                        ecolor=INK, elinewidth=1.6, capsize=5, zorder=4)
            tag = f"{val:+.0f} bb/100    ·    95% CI [{lo:.0f}, {hi:.0f}]"
            lab_x = val if val < 0 else val + 20
            ax.text(lab_x, y + 0.33, tag, va="bottom", ha="left",
                    fontsize=11.5, color=color, weight="bold")
    ax.axvline(0, color=INK, linewidth=1.4, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.set_xlim(-640, 230)
    ax.set_xlabel("bb / 100 hands  (heads-up win-rate)")
    set_title(ax, "The training metric stayed positive the entire run.\n"
                 "A fixed reference said the model had regressed −511 bb/100.",
                 fontsize=15, weight="bold", loc="left", pad=14)
    save(fig, "02_metric_mirage")


# --------------------------------------------------------------------------- #
# Fig 3 — behavior gates catch the collapse
# --------------------------------------------------------------------------- #
def fig_behavior_gates():
    # champion (21k, all gates pass) vs collapsed 49k "final_optimization" (gate FAIL)
    champ = {"Fold": 46, "Call": 43, "Raise": 11}   # entropy 1.44 bits
    coll = {"Fold": 62.6, "Call": 33.9, "Raise": 3.5}  # entropy collapsed
    cats = ["Fold", "Call", "Raise"]
    x = range(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    style_ax(ax, grid_axis="y")
    b1 = ax.bar([i - w / 2 for i in x], [champ[c] for c in cats], w,
                label="Champion · 21k  —  entropy 1.44 bits, all gates pass",
                color=TEAL, zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], [coll[c] for c in cats], w,
                label="Later 49k run  —  entropy collapsed, gate FAIL", color=RED,
                alpha=0.9, zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0,
                    f"{b.get_height():.0f}%", ha="center", va="bottom",
                    fontsize=11, color=INK, weight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, fontsize=12.5)
    ax.set_ylabel("action frequency (%)")
    ax.set_ylim(0, 72)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1,
              fontsize=11, loc="upper right")
    set_title(ax, "A later, longer run scored well on win-rate — but had collapsed\n"
                 "to fold-heavy play. The entropy + fold gates refused to promote it.",
                 fontsize=14.5, weight="bold", loc="left", pad=14)
    save(fig, "03_behavior_gates")


# --------------------------------------------------------------------------- #
# Fig 4 — Leduc: PPO iterate cycles, strategy-averaging is non-divergent
# --------------------------------------------------------------------------- #
def fig_leduc_curves():
    s7 = load("leduc_sweep7/sw7_nfsp7_7.json")
    s42 = load("leduc_sweep7/sw7_nfsp7_42.json")

    def xy(d):
        c = d["curve"]
        return [p["iteration"] for p in c], [p["pi_bar_nash_conv"] for p in c]

    x7, y7 = xy(s7)
    x42, y42 = xy(s42)
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    style_ax(ax, grid_axis="y")
    ax.set_yscale("log")

    # raw PPO self-play current iterate: cycles, never settles (writeup band 1.3-2.6)
    ax.axhspan(1.3, 2.6, color=RED, alpha=0.10, zorder=1)
    ax.text(20000, 1.85, "raw PPO self-play iterate\ncycles here — never settles",
            color=RED, fontsize=10.5, ha="right", va="center", weight="bold")

    ax.plot(x7, y7, color=BLUE, lw=2.4, marker="o", ms=3.5,
            label=f"NFSP-lite strategy-avg, seed 7  (final {s7['final_nash_conv']:.3f})",
            zorder=4)
    ax.plot(x42, y42, color=TEAL, lw=2.4, marker="o", ms=3.5,
            label=f"NFSP-lite strategy-avg, seed 42  (final {s42['final_nash_conv']:.3f})",
            zorder=4)

    # calibration reference lines (0.05 and 0.06 are indistinguishable on this
    # axis, so they are drawn as one "near-optimal solvers" reference)
    refs = [(4.76, "uniform random ≈ 4.76", GRAY),
            (0.055, "near-optimal solvers ≈ 0.05–0.06  (tuned NFSP · CFR+)", GREEN)]
    for val, lab, col in refs:
        ax.axhline(val, color=col, lw=1.3, ls="--", zorder=2)
        ax.text(300, val * 1.10, lab, color=col, fontsize=10, va="bottom")

    ax.set_xlim(0, 20800)
    ax.set_ylim(0.04, 6)
    ax.set_xlabel("self-play iteration")
    ax.set_ylabel("NashConv  (exact exploitability, log scale)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v/1000)}k" if v else "0"))
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1,
              fontsize=11, loc="upper right")
    set_title(ax, "Leduc poker (936 states, exploitability is exact): the raw iterate\n"
                 "cycles; only strategy-averaging is non-divergent — plateau ≈ 0.56.",
                 fontsize=14.5, weight="bold", loc="left", pad=14)
    save(fig, "04_leduc_convergence")


# --------------------------------------------------------------------------- #
# Fig 5 — calibration ladder: 0.56 is methodology, not magnitude
# --------------------------------------------------------------------------- #
def fig_calibration():
    fig, ax = plt.subplots(figsize=(10.2, 3.2))
    ax.set_xscale("log")
    ax.axhline(0, color="#c9d2de", lw=2.5, zorder=1)

    # markers on the NashConv axis (sum of both players' best-response gains).
    # CFR+ (~0.05) and tuned NFSP (~0.06) are visually indistinguishable, so they
    # are grouped as one "solved-game optimum" reference.
    for val, col in [(4.76, GRAY), (0.563, RED), (0.055, GREEN)]:
        ax.plot(val, 0, "o", ms=16, color=col, zorder=3,
                markeredgecolor="white", markeredgewidth=1.6)

    ax.annotate("uniform random\n≈ 4.76", xy=(4.76, 0), xytext=(4.76, 0.34),
                ha="center", va="bottom", fontsize=11, color=GRAY, weight="bold")
    ax.annotate("this work — NFSP-lite\n≈ 0.56", xy=(0.563, 0), xytext=(0.563, 0.34),
                ha="center", va="bottom", fontsize=12, color=RED, weight="bold")
    ax.annotate("solved-game optimum\ntabular CFR+ ≈0.05  ·  tuned NFSP* ≈0.06",
                xy=(0.055, 0), xytext=(0.055, 0.34),
                ha="center", va="bottom", fontsize=10.5, color=GREEN, weight="bold")
    ax.annotate("← Nash = 0  (off log scale)", xy=(0.03, 0), xytext=(0.03, -0.40),
                ha="left", va="top", fontsize=10.5, color=INK)

    ax.set_ylim(-0.85, 1.05)
    ax.set_xlim(0.022, 9)
    ax.set_yticks([])
    for s in ("left", "top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("NashConv  (log scale)  —  lower is closer to optimal", labelpad=8)
    set_title(ax, "Calibrated honestly: ≈8× below random, but ≈9× above tuned NFSP —\n"
                 "the contribution is the method, not the magnitude.",
                 fontsize=14, weight="bold", loc="left", pad=12)
    save(fig, "05_calibration_ladder")


# --------------------------------------------------------------------------- #
# Fig 6 — over-aggression drift diagnosis
# --------------------------------------------------------------------------- #
def fig_over_aggression():
    bp = load("strong_fchpa_ladder/behavior_probe.json")
    nvc = load("strong_fchpa_ladder/new_vs_champ.json")
    cats = ["vs random\nopponent", "vs calling station\n(always-call)", "vs the champion\n(12k pairs)"]
    new = [bp["new_90k"]["random"]["bb_per_100"],
           bp["new_90k"]["always_call"]["bb_per_100"],
           nvc["bb_per_100"]]
    champ = [bp["champion"]["random"]["bb_per_100"],
             bp["champion"]["always_call"]["bb_per_100"], None]
    x = range(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    style_ax(ax, grid_axis="y")
    for i, (n, c) in enumerate(zip(new, champ)):
        ax.bar(i - (w / 2 if c is not None else 0), n, w,
               color=(GREEN if n > 0 else RED), zorder=3,
               label="Scaled 90k model" if i == 0 else None,
               edgecolor="white")
        if c is not None:
            ax.bar(i + w / 2, c, w, color=TEAL, alpha=0.85, zorder=3,
                   label="Champion 21k" if i == 0 else None, edgecolor="white")
    for i, (n, c) in enumerate(zip(new, champ)):
        xoff = (w / 2 if c is not None else 0)
        ax.text(i - xoff, n + (25 if n > 0 else -25), f"{n:+.0f}",
                ha="center", va="bottom" if n > 0 else "top", fontsize=10.5,
                weight="bold", color=INK)
        if c is not None:
            ax.text(i + w / 2, c + (25 if c > 0 else -25), f"{c:+.0f}",
                    ha="center", va="bottom" if c > 0 else "top", fontsize=10.5,
                    weight="bold", color=INK)
    ax.axhline(0, color=INK, lw=1.3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, fontsize=11.5)
    ax.set_ylabel("bb / 100 hands")
    ax.set_ylim(-720, 1230)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    set_title(ax, "Diagnosis — over-aggression drift, not collapse: it beats random even\n"
                 "harder than the champion, but a calling station trivially punishes it.",
                 fontsize=14.5, weight="bold", loc="left", pad=14)
    save(fig, "06_over_aggression")


# --------------------------------------------------------------------------- #
# Fig 1 — the network architecture
# --------------------------------------------------------------------------- #
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11.6, 5.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, label, fc, tc="white", fs=12, sub=None):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=2.4",
            linewidth=0, facecolor=fc, zorder=3))
        if sub:
            ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                    color=tc, fontsize=fs, weight="bold", zorder=4)
            ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                    color=tc, fontsize=fs - 3, zorder=4, linespacing=1.25)
        else:
            ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                    color=tc, fontsize=fs, weight="bold", zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, lw=2,
            color="#8593a8", zorder=2, shrinkA=1, shrinkB=1))

    # inputs
    box(1, 64, 20, 13, "7 cards", LIGHT, INK, 11.5, sub="2 hole + 5 community")
    box(1, 43.5, 20, 13, "betting history", LIGHT, INK, 11.5, sub="action sequence")
    box(1, 23, 20, 13, "11 scalars", LIGHT, INK, 11.5, sub="pot · stacks · position")
    # towers
    box(29, 61.5, 23, 18, "Card tower", BLUE, "white", 13,
        sub="53-token embed\n→ 3× FC + LayerNorm")
    box(29, 39, 23, 18, "Action tower", TEAL, "white", 13,
        sub="embed → LSTM\n(packed sequence)")
    # trunk
    box(60, 37, 20, 42.5, "Shared trunk", "#334155", "white", 13,
        sub="concat →\nLinear →\n2× Residual\n(LayerNorm)")
    # heads
    box(87, 60, 12.5, 16, "Policy", AMBER, "white", 12.5, sub="masked\nlogits")
    box(87, 39, 12.5, 16, "Value", GREEN, "white", 12.5, sub="state\nvalue")

    arrow(21, 70.5, 29, 70.5)      # cards -> card tower
    arrow(21, 50, 29, 48)          # history -> action tower
    arrow(21, 29.5, 60, 43)        # scalars -> trunk (diagonal)
    arrow(52, 70.5, 60, 64)        # card tower -> trunk
    arrow(52, 48, 60, 55)          # action tower -> trunk
    arrow(80, 62, 87, 68)          # trunk -> policy
    arrow(80, 54, 87, 47)          # trunk -> value

    if not DECK:
        ax.text(50, 95, "Pseudo-siamese two-tower actor–critic   ·   ≈ 1.5M parameters",
                ha="center", va="center", fontsize=15.5, weight="bold", color=INK)
    ax.text(50, 5,
            "search-free: one masked forward pass per decision   ·   trained with PPO + GAE against a K-best self-play league",
            ha="center", va="center", fontsize=10.5, color=MUTED)
    save(fig, "01_architecture")


if __name__ == "__main__":
    print("Generating figures into docs/figures/ ...")
    fig_architecture()
    fig_metric_mirage()
    fig_behavior_gates()
    fig_leduc_curves()
    fig_calibration()
    fig_over_aggression()
    print("done.")
