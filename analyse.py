"""
Soccer PoC — v4
- Fully local inference, zero API calls after first model download
- Ball tracking removed — focus on players, events, heatmaps
- Optimised for stationary wide angle footage
"""

import os
os.environ["QWEN_2_5_ENABLED"]              = "False"
os.environ["QWEN_3_ENABLED"]               = "False"
os.environ["CORE_MODEL_SAM_ENABLED"]       = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"]      = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"]      = "False"
os.environ["CORE_MODEL_YOLO_WORLD_ENABLED"]= "False"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CoreMLExecutionProvider,CPUExecutionProvider"

import cv2
import json
import uuid
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from datetime import datetime
from inference import get_model
import supervision as sv

# ================================================================
# CONFIG
# ================================================================
VIDEO_PATH      = "game.mp4"
API_KEY         = "YOUR_KEY_HERE"   # still needed to download model
                                    # first time only — then cached locally

PLAYER_MODEL_ID = "football-players-detection-3zvbc/19"

SAMPLE_EVERY    = 30       # every 30 frames = 1 sample/sec at 30fps
                           # wide angle stationary = less need for dense sampling
PLAYER_CONF     = 0.25     # slightly lower for wide angle (players are smaller)

HOME_TEAM       = "Home"
AWAY_TEAM       = "Away"

# ByteTrack
TRACK_THRESH    = 0.25
TRACK_BUFFER    = 45       # wider buffer — stationary camera = more consistent
MATCH_THRESH    = 0.8
# ================================================================

print("=" * 55)
print("  Soccer PoC v4 — Local Inference")
print("=" * 55)
print(f"  Video:       {VIDEO_PATH}")
print(f"  Sample rate: every {SAMPLE_EVERY} frames")
print(f"  Player conf: {PLAYER_CONF}")
print(f"  Running:     100% local, no API calls")
print()

print("Loading model (downloads once, cached after)...")
model   = get_model(model_id=PLAYER_MODEL_ID, api_key=API_KEY)
tracker = sv.ByteTrack(
    track_activation_threshold=TRACK_THRESH,
    lost_track_buffer=TRACK_BUFFER,
    minimum_matching_threshold=MATCH_THRESH,
    frame_rate=30
)
print("Model ready.\n")

cap    = cv2.VideoCapture(VIDEO_PATH)
fps    = cap.get(cv2.CAP_PROP_FPS) or 30
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dur    = total / fps / 60

print(f"Video: {width}x{height}  {fps:.0f}fps  {dur:.1f} mins")
print(f"Frames to process: ~{total // SAMPLE_EVERY}")
print()

# ================================================================
# DATA STORES
# ================================================================
player_positions = []   # every tracked position
events           = []   # free kicks, goal kicks, corners, goals

# Event detection state
prev_positions   = []   # player positions last frame
still_counter    = {}   # tracker_id -> frames_not_moved count
motion_history   = []
frame_n          = 0

def pitch_pct(px, py):
    """Pixel coords → 0-100 pitch percentage"""
    return round((px / width) * 100, 1), round((py / height) * 100, 1)

def get_zones(x, y):
    return {
        "in_corner":         (y < 10 or y > 90) and (x < 8  or x > 92),
        "in_left_goal_area": x < 8  and 35 < y < 65,
        "in_right_goal_area":x > 92 and 35 < y < 65,
        "in_left_box":       x < 18 and 20 < y < 80,
        "in_right_box":      x > 82 and 20 < y < 80,
        "in_penalty_area":   (x < 18 and 20 < y < 80) or (x > 82 and 20 < y < 80),
    }

def event_exists(etype, secs, window=12):
    return any(e["type"] == etype and abs(secs - e["second"]) < window
               for e in events)

def cluster_centre(positions):
    """Find the densest cluster of players — proxy for where the ball is"""
    if not positions:
        return None
    xs = [p["x_pct"] for p in positions]
    ys = [p["y_pct"] for p in positions]
    # Simple mean for now — good enough for zone detection
    return round(sum(xs)/len(xs), 1), round(sum(ys)/len(ys), 1)

# ================================================================
# MAIN LOOP
# ================================================================
print("Processing...")
print("-" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    secs = round(frame_n / fps, 2)
    mins = int(secs // 60)

    if frame_n % SAMPLE_EVERY == 0:

        # ---- PLAYER DETECTION ----
        result     = model.infer(frame, confidence=PLAYER_CONF)[0]
        detections = sv.Detections.from_inference(result)

        # Filter out referees
        if detections.data and "class_name" in detections.data:
            mask = np.array([
                cls.lower() not in {"referee"}
                for cls in detections.data["class_name"]
            ])
            detections = detections[mask]

        # ByteTrack — persistent player IDs
        tracked = tracker.update_with_detections(detections)

        frame_players = []
        for i in range(len(tracked)):
            box = tracked.xyxy[i]
            cx  = (box[0] + box[2]) / 2
            cy  = (box[1] + box[3]) / 2
            xp, yp = pitch_pct(cx, cy)
            tid    = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
            conf   = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

            frame_players.append({
                "second":     secs,
                "tracker_id": tid,
                "x_pct":      xp,
                "y_pct":      yp,
                "confidence": round(conf, 2)
            })

        player_positions.extend(frame_players)

        # ---- EVENT DETECTION (player-cluster based, no ball needed) ----
        # Use the cluster of players as a proxy for ball location
        cluster = cluster_centre(frame_players)

        if cluster:
            cx, cy = cluster
            zones  = get_zones(cx, cy)

            # Check if the cluster has been stationary for ~3 seconds
            # (players standing around = set piece)
            if prev_positions:
                prev_cluster = cluster_centre(prev_positions)
                if prev_cluster:
                    dist = np.sqrt(
                        (cx - prev_cluster[0])**2 +
                        (cy - prev_cluster[1])**2
                    )
                    is_still = dist < 3.0 and len(frame_players) >= 4
                else:
                    is_still = False
            else:
                is_still = False

            # Track consecutive still frames
            if not hasattr(still_counter, 'count'):
                still_counter['count'] = 0
            still_counter['count'] = still_counter['count'] + 1 \
                                     if is_still else 0

            # ~3 seconds of stillness = set piece
            secs_still = (still_counter['count'] * SAMPLE_EVERY) / fps
            if secs_still >= 3.0 and len(frame_players) >= 4:
                if zones["in_corner"]:
                    etype = "corner"
                elif zones["in_left_goal_area"] or zones["in_right_goal_area"]:
                    etype = "goal_kick"
                elif zones["in_penalty_area"]:
                    etype = "free_kick"  # could be penalty too
                else:
                    etype = "free_kick"

                if not event_exists(etype, secs):
                    events.append({
                        "id":     str(uuid.uuid4())[:8],
                        "type":   etype,
                        "second": secs,
                        "minute": f"{mins}:{int(secs%60):02d}",
                        "x_pct":  cx,
                        "y_pct":  cy,
                    })
                    label = etype.upper().replace("_", " ")
                    print(f"  [{label}] at {mins}:{int(secs%60):02d}  "
                          f"cluster at ({cx:.0f}%, {cy:.0f}%)")
                still_counter['count'] = 0  # reset after logging

            # Possible goal — high frame motion when players clustered near goal
            if zones["in_penalty_area"]:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion_history.append(gray)
                if len(motion_history) > 3:
                    motion_history.pop(0)
                if len(motion_history) == 3:
                    diff = (
                        cv2.absdiff(motion_history[0], motion_history[1]).mean() +
                        cv2.absdiff(motion_history[1], motion_history[2]).mean()
                    ) / 2
                    if diff > 12 and not event_exists("goal", secs, window=15):
                        events.append({
                            "id":     str(uuid.uuid4())[:8],
                            "type":   "goal",
                            "second": secs,
                            "minute": f"{mins}:{int(secs%60):02d}",
                            "x_pct":  cx,
                            "y_pct":  cy,
                            "motion": round(float(diff), 1)
                        })
                        print(f"  [POSSIBLE GOAL] at {mins}:{int(secs%60):02d}  "
                              f"motion={diff:.1f}")

        prev_positions = frame_players

        # Progress every 10 samples
        if frame_n % (SAMPLE_EVERY * 10) == 0:
            pct = round((frame_n / total) * 100, 1)
            print(f"  {mins:02d}:{int(secs%60):02d}  [{pct}%]  "
                  f"players={len(frame_players):2d}  "
                  f"events={len(events)}")

    frame_n += 1

cap.release()

# ================================================================
# POSSESSION — which half had more player density
# ================================================================
left_frames  = sum(1 for p in player_positions if p["x_pct"] < 50)
right_frames = sum(1 for p in player_positions if p["x_pct"] >= 50)
total_pf     = left_frames + right_frames or 1
# Home attacks right by default
home_poss = round(right_frames / total_pf * 100, 1)
away_poss = round(left_frames  / total_pf * 100, 1)

# ================================================================
# RESULTS
# ================================================================
goals      = [e for e in events if e["type"] == "goal"]
free_kicks = [e for e in events if e["type"] == "free_kick"]
corners    = [e for e in events if e["type"] == "corner"]
goal_kicks = [e for e in events if e["type"] == "goal_kick"]

print()
print("=" * 55)
print("  RESULTS")
print("=" * 55)
print(f"  Duration:            {dur:.1f} mins")
print(f"  Frames processed:    {frame_n // SAMPLE_EVERY}")
print(f"  Player detections:   {len(player_positions)}")
print(f"  Unique tracker IDs:  "
      f"{len(set(p['tracker_id'] for p in player_positions))}")
print()
print(f"  Possession (player density):")
print(f"    {HOME_TEAM}:  {home_poss}%")
print(f"    {AWAY_TEAM}:  {away_poss}%")
print()
print(f"  Events detected:")
print(f"    Possible goals:  {len(goals)}")
print(f"    Free kicks:      {len(free_kicks)}")
print(f"    Corners:         {len(corners)}")
print(f"    Goal kicks:      {len(goal_kicks)}")

if goals:
    print("\n  Goal timestamps (verify in verify.py):")
    for g in goals:
        print(f"    {g['minute']}  motion={g.get('motion','?')}")
if free_kicks:
    print("\n  Free kick timestamps:")
    for e in free_kicks:
        print(f"    {e['minute']}  at ({e['x_pct']:.0f}%, {e['y_pct']:.0f}%)")
if corners:
    print("\n  Corner timestamps:")
    for e in corners:
        print(f"    {e['minute']}  at ({e['x_pct']:.0f}%, {e['y_pct']:.0f}%)")

# ================================================================
# SAVE
# ================================================================
output = {
    "meta": {
        "video":         VIDEO_PATH,
        "processed_at":  datetime.utcnow().isoformat(),
        "fps":           fps,
        "duration_mins": round(dur, 1),
        "home_team":     HOME_TEAM,
        "away_team":     AWAY_TEAM,
        "model":         PLAYER_MODEL_ID,
        "mode":          "local_inference"
    },
    "possession":       {HOME_TEAM: home_poss, AWAY_TEAM: away_poss},
    "player_positions": player_positions,
    "ball_positions":   [],   # removed for now
    "events":           events
}

with open("match_data.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("  Saved → match_data.json")
print("  Run:   python verify.py")
print("  Run:   python report.py")
print("=" * 55)