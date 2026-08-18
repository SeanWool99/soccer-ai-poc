"""
Soccer PoC — Verification Script
Plays your video back with all detections drawn on top so you can
manually judge accuracy. Run AFTER analyse.py has finished.

Mac: press SPACE to pause, Q to quit, S to save a frame as PNG
"""

import cv2
import json
import numpy as np

# ================================================================
# CONFIG
# ================================================================
VIDEO_PATH      = "game.mp4"
DATA_FILE       = "match_data.json"
OUTPUT_VIDEO    = "verified_output.mp4"
SHOW_LIVE       = True    # False = just write the file, no window
PLAYBACK_SPEED  = 1.0     # 0.5 = half speed, 2.0 = double speed
# ================================================================

# Load match data
with open(DATA_FILE) as f:
    data = json.load(f)

meta     = data["meta"]
events   = data["events"]
players  = data["player_positions"]
balls    = data["ball_positions"]

# Index by second for fast lookup
def index_by_second(records, window=0.6):
    """Group records into buckets by nearest second."""
    idx = {}
    for r in records:
        key = round(r["second"])
        idx.setdefault(key, []).append(r)
    return idx

player_idx = index_by_second(players)
ball_idx   = index_by_second(balls)
event_idx  = {}
for e in events:
    key = round(e["second"])
    event_idx.setdefault(key, []).append(e)

# Event type colours (BGR)
EVENT_COLOURS = {
    "goal":      (0,   215, 255),   # gold
    "free_kick": (255, 180,  30),   # blue
    "corner":    (30,  255, 180),   # green
    "penalty":   (50,   50, 255),   # red
}
EVENT_LABELS = {
    "goal":      "POSSIBLE GOAL",
    "free_kick": "FREE KICK",
    "corner":    "CORNER",
    "penalty":   "PENALTY AREA",
}

# Open video
cap    = cv2.VideoCapture(VIDEO_PATH)
fps    = cap.get(cv2.CAP_PROP_FPS) or 30
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Active event banner state
active_banner  = None
banner_until   = 0

frame_n  = 0
paused   = False
last_sec = -1

print("=" * 50)
print("  Verification Player")
print("=" * 50)
print("  SPACE  =  pause / play")
print("  S      =  save current frame as PNG")
print("  Q      =  quit")
print("  →      =  skip forward 5 seconds")
print("=" * 50)
print()

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

    secs = frame_n / fps
    sec  = round(secs)
    mins = int(secs // 60)
    sec_ = int(secs % 60)

    # ---- Draw player boxes ----
    for p in player_idx.get(sec, []):
        px = int((p["x_pct"] / 100) * width)
        py = int((p["y_pct"] / 100) * height)
        bw, bh = 55, 75

        x1 = max(0, px - bw // 2)
        y1 = max(0, py - bh // 2)
        x2 = min(width,  x1 + bw)
        y2 = min(height, y1 + bh)

        conf = p["confidence"]
        if conf >= 0.75:
            colour = (50, 200, 50)      # green — high confidence
        elif conf >= 0.55:
            colour = (200, 160, 30)     # blue — medium
        else:
            colour = (60, 60, 220)      # red — low

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"{int(conf*100)}%", (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1)

    # ---- Draw ball ----
    for b in ball_idx.get(sec, []):
        bx = int((b["x_pct"] / 100) * width)
        by = int((b["y_pct"] / 100) * height)
        cv2.circle(frame, (bx, by), 12, (0, 255, 255), 2)      # cyan ring
        cv2.circle(frame, (bx, by), 3,  (0, 255, 255), -1)     # centre dot
        cv2.putText(frame, f"ball {int(b['confidence']*100)}%",
                    (bx + 14, by + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

    # ---- Check for events at this second ----
    for e in event_idx.get(sec, []):
        if e["second"] > banner_until:
            active_banner = e
            banner_until  = e["second"] + 4   # show banner for 4 seconds

        # Draw a marker on pitch at event location
        ex = int((e["x_pct"] / 100) * width)
        ey = int((e["y_pct"] / 100) * height)
        ec = EVENT_COLOURS.get(e["type"], (255,255,255))
        cv2.drawMarker(frame, (ex, ey), ec,
                       cv2.MARKER_STAR, 30, 2)

    # ---- Event banner ----
    if active_banner and secs <= banner_until:
        label  = EVENT_LABELS.get(active_banner["type"], active_banner["type"].upper())
        bcolour = EVENT_COLOURS.get(active_banner["type"], (255,255,255))
        bw = 420
        bh = 54
        bx = (width - bw) // 2
        by_banner = 80

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx, by_banner), (bx+bw, by_banner+bh), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, label,
                    (bx + 20, by_banner + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, bcolour, 2)
        cv2.putText(frame, f"at {active_banner['minute']}",
                    (bx + bw - 130, by_banner + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)

    # ---- HUD (top left) ----
    n_players = len(player_idx.get(sec, []))
    n_balls   = len(ball_idx.get(sec, []))

    hud_lines = [
        f"{mins:02d}:{sec_:02d}",
        f"Players: {n_players}",
        f"Ball:    {'yes' if n_balls else 'no '}",
    ]
    cv2.rectangle(frame, (0,0), (160, 18 + 20*len(hud_lines)), (0,0,0), -1)
    for i, line in enumerate(hud_lines):
        cv2.putText(frame, line, (8, 18 + i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)

    # ---- Confidence legend (bottom left) ----
    legend = [
        ((50,200,50),  "High confidence (>75%)"),
        ((200,160,30), "Medium (55-75%)"),
        ((60,60,220),  "Low (<55%)"),
        ((0,255,255),  "Ball"),
    ]
    ly = height - 10 - len(legend)*22
    cv2.rectangle(frame, (0, ly-8), (240, height), (0,0,0), -1)
    for i, (col, label) in enumerate(legend):
        y = ly + i*22
        cv2.rectangle(frame, (8, y-10), (22, y+4), col, -1)
        cv2.putText(frame, label, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220,220,220), 1)

    writer.write(frame)

    if SHOW_LIVE:
        cv2.imshow("Verification  —  SPACE pause  |  S save  |  Q quit", frame)
        delay = max(1, int((1000 / fps) / PLAYBACK_SPEED))
        key   = cv2.waitKey(delay if not paused else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            fname = f"frame_{mins:02d}m{sec_:02d}s.png"
            cv2.imwrite(fname, frame)
            print(f"Saved screenshot: {fname}")
        elif key == 0x27:   # right arrow — skip 5 seconds
            skip = int(fps * 5)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n + skip)
            frame_n += skip
            continue

    if not paused:
        frame_n += 1

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"\nDone. Annotated video saved to: {OUTPUT_VIDEO}")
print("Open it in QuickTime to review detections at your own pace.")
