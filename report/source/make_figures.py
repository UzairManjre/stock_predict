"""Figures for the printed project report (A4, 6.3 in text width, 220 dpi)."""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

INK, BODY, MUTED, GRID = "#0E1726", "#3A4556", "#5E6676", "#D9D4C7"
BLUE, BLUE_TINT, AMBER, GREY = "#2F5BD8", "#EAF0FD", "#C98A2E", "#A9A497"
plt.rcParams.update({
    "font.family": "Calibri", "font.size": 11, "text.color": BODY,
    "axes.edgecolor": MUTED, "axes.labelcolor": BODY, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 220, "savefig.dpi": 220,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})
W = 6.3


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), facecolor="white")
    plt.close(fig)
    print("saved", name)


# 1. System pipeline (snake layout, like the deck)
fig, ax = plt.subplots(figsize=(W, 3.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")
boxes = [
    (1, 31, "1  Collect", "Daily prices: 209 stocks\n+ NIFTY 50 index;\n169,343 announcements"),
    (36, 31, "2  Store and repair", "PostgreSQL database;\nremove bad rows, fix\nsplits and gaps"),
    (71, 31, "3  Build features", "70 columns per stock\nper day: trend, risk,\nactivity, news tone"),
    (71, 2, "4  Train models", "Neural network (TFT),\nLightGBM, rank\nregression, momentum"),
    (36, 2, "5  Test honestly", "Walk-forward test on\n18 months the models\nnever saw"),
    (1, 2, "6  Run every day", "18:30: new picks,\nconfidence flag, accuracy\ncheck, dashboard"),
]
bw, bh = 28, 23
for i, (x, y, title, text) in enumerate(boxes):
    last = i == len(boxes) - 1
    ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=BLUE_TINT if last else "#FDFCF9", ec=BLUE if last else GRID, lw=1.6 if last else 1))
    ax.text(x + 1.8, y + bh - 4.0, title, fontsize=11, fontweight="bold", color=INK, va="center")
    ax.text(x + 1.8, y + bh - 14.0, text, fontsize=9, color=BODY, va="center", linespacing=1.35)
arrow = dict(arrowstyle="-|>", mutation_scale=13, color=BLUE, lw=1.6)
for a, b in [((29.3, 42.5), (35.7, 42.5)), ((64.3, 42.5), (70.7, 42.5)), ((85, 30.7), (85, 25.3)), ((70.7, 13.5), (64.3, 13.5)), ((35.7, 13.5), (29.3, 13.5))]:
    ax.add_patch(FancyArrowPatch(a, b, **arrow))
save(fig, "fig1_pipeline.png")

# 2. Walk-forward testing
fig, ax = plt.subplots(figsize=(W, 2.9))
labels = ["Apr–Jun 2025", "Jul–Sep 2025", "Oct–Dec 2025", "Jan–Mar 2026", "Apr–Jun 2026", "Jul–Sep 2026"]
for i, lab in enumerate(labels):
    y = 5 - i
    train_end = 56 + 6 * i
    ax.barh(y, train_end, left=0, height=0.62, color=GRID)
    ax.barh(y, 4, left=train_end + 0.6, height=0.62, color="white", edgecolor=MUTED, linestyle="--", lw=0.9)
    ax.barh(y, 14, left=train_end + 5.2, height=0.62, color=BLUE)
    ax.text(train_end + 20.5, y, lab, va="center", fontsize=9.5, color=INK)
    ax.text(-1.5, y, f"Round {i + 1}", va="center", ha="right", fontsize=9.5, color=MUTED)
ax.set_xlim(-14, 118); ax.set_ylim(-1.6, 5.6); ax.axis("off")
for x, color, edge, text in [(0, GRID, GRID, "Train: all earlier data"),
                             (38, "white", MUTED, "Gap: 21 trading days"),
                             (74, BLUE, BLUE, "Test: next 3 months")]:
    ax.add_patch(plt.Rectangle((x, -1.45), 3.5, 0.5, fc=color, ec=edge, ls="--" if color == "white" else "-", lw=0.9))
    ax.text(x + 4.8, -1.2, text, va="center", fontsize=9, color=BODY)
save(fig, "fig2_walkforward.png")

# 3. TFT training vs validation loss (version 21, no weight decay)
epochs = [0, 1, 2, 3, 4, 5]
train = [0.021444, 0.021207, 0.020824, 0.020305, 0.019949, 0.019551]
val = [0.020772, 0.020831, 0.021396, 0.021261, 0.022100, 0.021698]
fig, ax = plt.subplots(figsize=(W, 2.9))
ax.plot(epochs, val, color=BLUE, lw=2, marker="o", ms=5.5, mec="white", mew=1.2)
ax.plot(epochs, train, color=AMBER, lw=2, marker="o", ms=5.5, mec="white", mew=1.2)
ax.text(5.12, val[-1], "Error on new data\n(validation, +4.5%)", va="center", fontsize=9.5, color=INK)
ax.text(5.12, train[-1], "Error on training data\n(−8.8%)", va="center", fontsize=9.5, color=INK)
ax.set_xlim(-0.2, 6.6); ax.set_xticks(epochs)
ax.set_xlabel("Training round (epoch)"); ax.set_ylabel("Quantile loss (lower is better)")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(["Validation", "Training"], frameon=False, loc="upper left", fontsize=9.5, ncol=2)
save(fig, "fig3_tft_loss.png")

# 4. Reliability of the daily ranking (t-statistic), 5-day horizon
names = ["LightGBM forecast", "Rank regression", "Momentum 12-1", "Stock's own history", "Average of all four"]
tvals = [1.36, 1.65, 1.86, 2.20, 3.02]
fig, ax = plt.subplots(figsize=(W, 2.5))
bars = ax.barh(names, tvals, height=0.58, color=[GREY] * 4 + [BLUE])
for b, v in zip(bars, tvals):
    ax.text(v + 0.05, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=10, color=INK,
            fontweight="bold" if v == tvals[-1] else "normal")
ax.axvline(2, color=MUTED, ls="--", lw=1)
ax.text(2.03, 4.55, "t = 2: likely real", fontsize=9, color=MUTED, va="bottom")
ax.set_xlim(0, 3.5); ax.set_xlabel("t-statistic of the daily ranking score (IC)")
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
save(fig, "fig4_tstat.png")

# 5. Growth of Rs 100 (weekly re-buy, after costs, Mar 2025 to Oct 2026), shown as gain or loss
names = ["NIFTY 50 index", "LightGBM price forecast", "All 209 stocks, equal amounts", "Rank-regression model", "Momentum, top 20"]
gains = [-3.8, -2.7, 13.2, 20.4, 27.4]
fig, ax = plt.subplots(figsize=(W, 2.5))
bars = ax.barh(names, gains, height=0.58, color=[GREY, GREY, GREY, GREY, BLUE])
for b, g in zip(bars, gains):
    y = b.get_y() + b.get_height() / 2
    label = f"₹{100 + g:.0f}  ({'+' if g > 0 else '−'}{abs(g):.1f}%)"
    ax.text(g + (0.6 if g > 0 else -0.6), y, label, va="center", ha="left" if g > 0 else "right", fontsize=10, color=INK)
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlim(-16, 40); ax.set_xticks([-10, 0, 10, 20, 30])
ax.set_xticklabels(["−10%", "0", "+10%", "+20%", "+30%"])
ax.set_xlabel("Gain or loss on ₹100 invested, end of March 2025 to October 2026")
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
save(fig, "fig5_growth.png")

# 6. Hit rate: share of days the top 20 beat the NIFTY 50 over the next 21 trading days
panels = [
    ("A. Our 20 picks grew more than the NIFTY 50",
     ["Average of all 209 stocks", "Our picks, all days", "Our picks, high-confidence days"], [63, 68, 74]),
    ("B. Our 20 picks grew more than the average stock",
     ["No skill (coin toss)", "Our picks, all days", "Our picks, high-confidence days"], [50, 65, 70]),
]
fig, axes = plt.subplots(2, 1, figsize=(W, 3.6), sharex=True)
for ax, (title, names, vals) in zip(axes, panels):
    bars = ax.barh(names, vals, height=0.56, color=[GREY, BLUE, BLUE])
    for b, v in zip(bars, vals):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v}%", va="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
axes[1].set_xlabel("Share of test days, measured over the next 21 trading days (%)")
fig.tight_layout(h_pad=1.2)
save(fig, "fig6_hitrate.png")
