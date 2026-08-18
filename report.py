"""
Soccer PoC — Report Generator
Reads match_data.json and produces a visual match report PNG.
Run AFTER analyse.py has finished.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from collections import defaultdict

DATA_FILE   = "match_data.json"
OUTPUT_FILE = "match_report.png"

with open(DATA_FILE) as f:
    data = json.load(f)

meta     = data["meta"]
poss     = data["possession"]
players  = data["player_positions"]
balls    = data["ball_positions"]
events   = data["events"]

home = meta["home_team"]
away = meta["away_team"]

goals      = [e for e in events if e["type"] == "goal"]
free_kicks = [e for e in events if e["type"] == "free_kick"]
corners    = [e for e in events if e["type"] == "corner"]

# ================================================================
# Figure layout: 2 rows, 3 cols
# ================================================================
fig = plt.figure(figsize=(18, 11), facecolor="#1a1a2e")
fig.suptitle(
    f"{home}  vs  {away}   —   AI Match Report",
    color="white", fontsize=18, fontweight="bold", y=0.97
)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.38, wspace=0.3,
                       left=0.05, right=0.97, top=0.92, bottom=0.06)

PITCH_BG    = "#2d5a1b"
TEXT_COL    = "white"
ACCENT      = "#1D9E75"
HOME_COL    = "#378ADD"
AWAY_COL    = "#E24B4A"


# ================================================================
# Helper: draw a pitch outline on an axes
# ================================================================
def draw_pitch(ax, xlim=(0,100), ylim=(0,100)):
    ax.set_facecolor(PITCH_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    kw = dict(fill=False, edgecolor="white", linewidth=1.2)
    ax.add_patch(patches.Rectangle((0,0), 100, 100, **kw))
    ax.add_patch(patches.Rectangle((0, 21), 17, 58, **kw))
    ax.add_patch(patches.Rectangle((83, 21), 17, 58, **kw))
    ax.plot([50, 50], [0, 100], color="white", lw=1.2)
    ax.add_patch(plt.Circle((50, 50), 9.15, **kw))
    ax.plot(50, 50, "wo", ms=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


# ================================================================
# 1. Player heatmap (top-left)
# ================================================================
ax1 = fig.add_subplot(gs[0, 0])
draw_pitch(ax1)

if players:
    grid = np.zeros((20, 20))
    for p in players:
        xi = min(int(p["x_pct"] / 5), 19)
        yi = min(int(p["y_pct"] / 5), 19)
        grid[yi][xi] += 1

    ax1.imshow(grid, cmap="YlOrRd", alpha=0.65,
               aspect="auto", extent=[0, 100, 100, 0], vmin=0)

ax1.set_title("Player presence heatmap", color=TEXT_COL, fontsize=11, pad=6)


# ================================================================
# 2. Ball position heatmap (top-centre)
# ================================================================
ax2 = fig.add_subplot(gs[0, 1])
draw_pitch(ax2)

if balls:
    bgrid = np.zeros((20, 20))
    for b in balls:
        xi = min(int(b["x_pct"] / 5), 19)
        yi = min(int(b["y_pct"] / 5), 19)
        bgrid[yi][xi] += 1

    ax2.imshow(bgrid, cmap="Blues", alpha=0.7,
               aspect="auto", extent=[0, 100, 100, 0], vmin=0)

ax2.set_title("Ball position heatmap", color=TEXT_COL, fontsize=11, pad=6)


# ================================================================
# 3. Events map (top-right)
# ================================================================
ax3 = fig.add_subplot(gs[0, 2])
draw_pitch(ax3)

event_styles = {
    "goal":      ("*", "#FFD700", 200, "Possible goal"),
    "free_kick": ("o", "#60B3FF", 80,  "Free kick"),
    "corner":    ("^", "#50E88A", 80,  "Corner"),
    "penalty":   ("D", "#FF6B6B", 80,  "Penalty area"),
}
legend_handles = []
for etype, (marker, colour, size, label) in event_styles.items():
    ex = [e["x_pct"] for e in events if e["type"] == etype]
    ey = [e["y_pct"] for e in events if e["type"] == etype]
    if ex:
        sc = ax3.scatter(ex, ey, marker=marker, c=colour, s=size,
                         zorder=5, edgecolors="white", linewidths=0.5,
                         label=label)
        # Annotate minute
        for e in events:
            if e["type"] == etype:
                ax3.annotate(e["minute"], (e["x_pct"], e["y_pct"]),
                             textcoords="offset points", xytext=(6, 4),
                             fontsize=7, color="white")

ax3.legend(loc="lower center", fontsize=7, facecolor="#222",
           labelcolor="white", framealpha=0.7, ncol=2,
           bbox_to_anchor=(0.5, -0.14))
ax3.set_title("Detected events", color=TEXT_COL, fontsize=11, pad=6)


# ================================================================
# 4. Possession pie (bottom-left)
# ================================================================
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor("#1a1a2e")
for spine in ax4.spines.values():
    spine.set_visible(False)

hp = poss.get(home, 50)
ap = poss.get(away, 50)

wedges, texts, autotexts = ax4.pie(
    [hp, ap],
    labels=[home, away],
    autopct="%1.0f%%",
    colors=[HOME_COL, AWAY_COL],
    startangle=90,
    wedgeprops=dict(edgecolor="#1a1a2e", linewidth=2),
    textprops=dict(color=TEXT_COL, fontsize=11)
)
for at in autotexts:
    at.set_color("white")
    at.set_fontsize(13)
    at.set_fontweight("bold")

ax4.set_title("Possession (ball location)", color=TEXT_COL, fontsize=11, pad=6)


# ================================================================
# 5. Player count over time (bottom-centre)
# ================================================================
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor("#16213e")
for spine in ax5.spines.values():
    spine.set_color("#444")

if players:
    # Bucket player counts per minute
    by_minute = defaultdict(list)
    for p in players:
        minute = int(p["second"] // 60)
        by_minute[minute].append(1)

    minutes = sorted(by_minute.keys())
    # We need raw counts — count entries per minute bucket
    from collections import Counter
    minute_counter = Counter(int(p["second"] // 60) for p in players)
    # Normalise: divide by frames per minute to get avg players per frame
    frames_per_minute = 60 / (30 / max(1, 30))  # approx
    mins_list  = sorted(minute_counter.keys())
    counts     = [minute_counter[m] for m in mins_list]

    ax5.fill_between(mins_list, counts, alpha=0.3, color=ACCENT)
    ax5.plot(mins_list, counts, color=ACCENT, lw=1.5)

    # Mark events as vertical lines
    for e in events:
        ec = event_styles.get(e["type"], (None, "white", None, None))[1]
        ax5.axvline(e["second"] / 60, color=ec, lw=1.2, alpha=0.7, linestyle="--")

ax5.set_xlabel("Minute", color=TEXT_COL, fontsize=9)
ax5.set_ylabel("Detection count", color=TEXT_COL, fontsize=9)
ax5.tick_params(colors=TEXT_COL, labelsize=8)
ax5.set_title("Player detections over time", color=TEXT_COL, fontsize=11)


# ================================================================
# 6. Stats summary card (bottom-right)
# ================================================================
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor("#16213e")
ax6.axis("off")

stats_lines = [
    ("Duration",          f"{meta['duration_mins']:.1f} mins"),
    ("Player detections", str(len(players))),
    ("Ball detections",   str(len(balls))),
    ("Ball detection %",  f"{round(len(balls)/max(1,len(players)+len(balls))*100,0):.0f}%"),
    ("",                  ""),
    ("Possible goals",    str(len(goals))),
    ("Free kicks",        str(len(free_kicks))),
    ("Corners",           str(len(corners))),
    ("",                  ""),
    (f"{home} possession", f"{hp}%"),
    (f"{away} possession", f"{ap}%"),
]

y = 0.95
for label, value in stats_lines:
    if not label:
        y -= 0.04
        continue
    ax6.text(0.05, y, label, transform=ax6.transAxes,
             color="#aaa", fontsize=10, va="top")
    ax6.text(0.95, y, value, transform=ax6.transAxes,
             color="white", fontsize=10, va="top", ha="right", fontweight="bold")
    ax6.axhline(y=y-0.03, xmin=0.03, xmax=0.97,
                color="#333", linewidth=0.5, transform=ax6.transAxes)
    y -= 0.075

ax6.set_title("Match summary", color=TEXT_COL, fontsize=11)

# Footer
fig.text(0.5, 0.01,
         f"Generated by Soccer PoC  |  {meta['processed_at'][:10]}  |  Note: goals require manual verification",
         ha="center", color="#555", fontsize=8)

plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Report saved to {OUTPUT_FILE}")
print("Open it with: open match_report.png")
