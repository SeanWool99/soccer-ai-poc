# Soccer AI — Project Context for Claude Code

## What this project is
A local AI-powered soccer match analysis system built as a side project / PoC.
The goal is to track players, detect events (goals, free kicks, corners), generate
heatmaps, and eventually feed structured stats into a Databricks Lakehouse for
a club-wide dashboard — replacing or augmenting what Veo Analytics charges for.

The operator (Sean) is a data engineer, not an ML engineer. Keep code practical,
well-commented, and avoid unnecessary complexity.

---

## Current tech stack
- **Mac M5 Pro** — all inference runs locally via Apple MPS (Metal Performance Shaders)
- **Python 3.9** inside a venv at `~/Documents/Home/soccer_poc/venv`
- **Roboflow sports repo** — the core codebase, cloned from github.com/roboflow/sports
- **Ultralytics YOLOv8** — player, pitch, and ball detection models
- **Supervision** — detection utilities, ByteTrack tracker, video I/O
- **PyTorch** — model inference, MPS backend
- **SigLIP + UMAP + KMeans** — team classification (jersey colour clustering)

---

## Project folder structure
```
~/Documents/Home/soccer_poc/
  venv/                          ← Python virtual environment
  sports-main/
    examples/soccer/
      main.py                    ← PRIMARY FILE — all analysis modes live here
      data/
        football-player-detection.pt   ← pretrained player model (generic)
        football-pitch-detection.pt    ← pitch keypoint model
        football-ball-detection.pt     ← ball detection model
        player_id_list.json            ← generated after PLAYER_TRACKING run
        match_data.json                ← generated after FULL_ANALYSIS run
        output_*.mp4                   ← annotated output videos
    sports/
      common/
        team.py                  ← TeamClassifier (SigLIP embeddings + KMeans)
        ball.py                  ← BallTracker + BallAnnotator
        view.py                  ← ViewTransformer (homography)
      annotators/soccer.py       ← draw_pitch, draw_points_on_pitch
      configs/soccer.py          ← SoccerPitchConfiguration
```

---

## How to run — always activate venv first
```bash
cd ~/Documents/Home/soccer_poc/sports-main/examples/soccer
source ~/Documents/Home/soccer_poc/venv/bin/activate
```

## Available modes and commands

### Player detection (boxes around players, no tracking)
```bash
PYTHONPATH=~/Documents/Home/soccer_poc/sports-main python main.py \
  --source_video_path "/Users/sean/Documents/Home/game Stationary.mp4" \
  --target_video_path data/output_detection.mp4 \
  --device mps \
  --mode PLAYER_DETECTION
```

### Player tracking (ByteTrack IDs + ReID logic)
```bash
PYTHONPATH=~/Documents/Home/soccer_poc/sports-main python main.py \
  --source_video_path "/Users/sean/Documents/Home/game Stationary.mp4" \
  --target_video_path data/output_tracking.mp4 \
  --device mps \
  --mode PLAYER_TRACKING
```

### Player tracking — focus on one specific player
```bash
PYTHONPATH=~/Documents/Home/soccer_poc/sports-main python main.py \
  --source_video_path "/Users/sean/Documents/Home/game Stationary.mp4" \
  --target_video_path data/output_focus.mp4 \
  --device mps \
  --mode PLAYER_TRACKING \
  --focus_id 7
```

### Team classification (SigLIP jersey colour clustering)
```bash
PYTHONPATH=~/Documents/Home/soccer_poc/sports-main python main.py \
  --source_video_path "/Users/sean/Documents/Home/game Stationary.mp4" \
  --target_video_path data/output_teams.mp4 \
  --device mps \
  --mode TEAM_CLASSIFICATION
```

### Radar (top-down tactical map overlay)
```bash
PYTHONPATH=~/Documents/Home/soccer_poc/sports-main python main.py \
  --source_video_path "/Users/sean/Documents/Home/game Stationary.mp4" \
  --target_video_path data/output_radar.mp4 \
  --device mps \
  --mode RADAR
```

### Full analysis (tracking + radar + ball + stats JSON output)
```bash
PYTHONPATH=~/Documents/Home/soccer_poc/sports-main python main.py \
  --source_video_path "/Users/sean/Documents/Home/game Stationary.mp4" \
  --target_video_path data/output_full.mp4 \
  --device mps \
  --mode FULL_ANALYSIS
```

---

## Key constants in main.py (top of file — tune these)
```python
STRIDE = 30              # frames between crop samples (lower = more crops = better team classification)
MIN_FRAMES_TO_KEEP = 90  # tracker ID must appear for this many frames to be kept
MIN_MOVEMENT_PX    = 120 # tracker ID must move this many pixels total to be kept
PITCH_LEFT_PCT   = 5     # pitch boundary filter — detections outside these
PITCH_RIGHT_PCT  = 95    # bounds are discarded as sideline/crowd noise
PITCH_TOP_PCT    = 10
PITCH_BOTTOM_PCT = 90

# In PlayerReIDTracker class:
REID_DISTANCE_THRESHOLD = 0.12   # 12% of frame diagonal — spatial match radius
REID_WINDOW_FRAMES      = 150    # 5 seconds — how long to keep lost tracks for re-id
```

---

## Current state — what works and what doesn't

### Working well ✅
- **PLAYER_DETECTION** — detects players accurately on stationary wide-angle footage
- **PLAYER_TRACKING** — ByteTrack + custom ReID logic, runs two passes (no RAM crash)
- **RADAR** — pitch keypoint detection + homography + top-down overlay working
- **Pitch boundary filter** — removes sideline/crowd detections
- **Lifetime filters** — removes static and brief false detections
- **player_id_list.json** — saved after tracking run, ready for manual validation

### Partially working ⚠️
- **TEAM_CLASSIFICATION** — SigLIP + KMeans works but accuracy is poor on hazy/screen-recorded footage. Team split often comes out ~17 vs 4 instead of ~11 vs 11. STRIDE=30, contrast enhancement, and UMAP n_components=2 all applied.
- **Player ID stability** — currently getting ~53 valid IDs for a 2-minute clip (target ~22-25). High number IDs (770, 880, 999 etc.) are duplicates of the same players re-assigned after dropouts. ReID helps but doesn't fully solve it.
- **Ball detection** — spotty, especially on screen-recorded footage. Ball is too small in wide-angle view. Ball model exists and is wired up but not reliable.

### Not working / not attempted yet ❌
- **Event detection** (goals, free kicks, corners) — removed from pipeline pending better detection foundation
- **Stats output** — FULL_ANALYSIS mode writes match_data.json but stats are only as good as the detection quality
- **Databricks integration** — designed but not connected yet. match_data.json is the bridge.

---

## Root cause of remaining issues
The generic pretrained model (`football-player-detection.pt`) was trained on
**broadcast TV footage** — fixed high-angle cameras, professional pitches, perfect
lighting. Sean's footage is a **screen recording of Veo's follow-cam view** which
is fundamentally different. No amount of parameter tuning fully bridges this gap.

**The fix is training a custom model on Sean's own footage.**

---

## Next steps (in priority order)

### 1. Get better footage
- Sean is pushing Veo club admin for Editor/Admin access to download full game MP4
- Direct download is dramatically better quality than screen recording
- Veo only allows follow-cam download — panoramic view is online-only
- Tried Chrome DevTools to stretch panoramic view on ultrawide monitor (5120px)
  — partially working, aspect ratio issues being debugged

### 2. Train a custom model (the real fix)
Once footage is available:

```bash
# Install ultralytics if not already done
pip install ultralytics

# After downloading your Roboflow dataset export (YOLOv8 format):
yolo train \
  model=yolov8m.pt \
  data=/path/to/roboflow/data.yaml \
  epochs=50 \
  imgsz=1280 \
  device=mps \
  project=soccer_training \
  name=local_model
```

Then swap the model in:
```bash
cp ~/soccer_training/local_model/weights/best.pt \
   ~/Documents/Home/soccer_poc/sports-main/examples/soccer/data/football-player-detection.pt
```

No other code changes needed — pipeline picks it up automatically.

**Roboflow labelling tips:**
- Upload video directly to Roboflow — it extracts frames automatically
- Aim for 50-100 labelled frames spread across different game moments
- Label class: `player` only — NOT referees, NOT sideline people, NOT ball
- Include partial players at frame edges
- Use Label Assist to auto-suggest boxes, accept good ones, delete wrong ones
- Add augmentations: flip horizontal, brightness ±15%, blur up to 1px

### 3. Manual player ID validation workflow
After a PLAYER_TRACKING run, `data/player_id_list.json` is saved.
Open it in VS Code and fill in `player_name` and `team` for each valid ID:
```json
{
  "canonical_id": 3,
  "frames_seen": 847,
  "total_movement_px": 1243.5,
  "passed_filter": true,
  "player_name": "Sean",
  "team": "home"
}
```
IDs with very high numbers (500+) are almost certainly duplicates of lower-numbered
IDs for the same player — mark them as `"player_name": "DUPLICATE"`.

### 4. Databricks integration (future)
- match_data.json loads straight into Delta tables
- Schema already designed (games, players, player_positions, game_events, fines tables)
- Notebooks already built in earlier PoC work (01_setup_tables.py etc.)
- Looker Studio connects to Databricks via partner connector

---

## team.py changes already applied
```python
# UMAP — changed from n_components=3 to:
self.reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)

# Contrast enhancement in extract_features:
from PIL import ImageEnhance
img = ImageEnhance.Contrast(img).enhance(1.5)
img = ImageEnhance.Color(img).enhance(1.8)

# predict() — no tqdm loop, processes per-frame crops in one batch:
def predict(self, crops):
    # processes crops directly without create_batches loop
    # see team.py for full implementation
```

---

## Known errors and fixes

### zsh: killed (out of memory)
Caused by storing all video frames in RAM. Fixed — PLAYER_TRACKING now reads
video from disk twice (two passes) instead of storing frames in memory.

### TypeError: Object of type float32 is not JSON serializable
Fix: wrap values in int(), float(), bool() before json.dump()

### ModuleNotFoundError: No module named 'inference'
Fix: `source ~/Documents/Home/soccer_poc/venv/bin/activate` then `pip install inference`

### Embedding extraction: 1it per frame (very slow)
Caused by predict() calling extract_features per frame. Fixed in team.py —
predict() now processes crops directly without the tqdm/batching loop.

### Processing: 0it then crash
Usually means cv2.imshow() is trying to open a display window.
Fixed — removed cv2.imshow() from main loop, replaced with tqdm progress bar.

---

## The bigger vision
1. Screen record (or download) game footage
2. Run pipeline → get annotated video + player_id_list.json + match_data.json
3. Manually validate player_id_list.json (assign names + teams, merge duplicate IDs)
4. Load validated JSON into Databricks Delta tables
5. Looker Studio dashboard shows per-player stats, heatmaps, season summaries
6. Club gets free analytics platform — richer than Veo Analytics because it spans
   all teams, all seasons, and integrates with existing fines tracker Google Sheet

---

## Long term plan — dual camera fusion (after current model training is complete)

### Why dual camera

The stationary wide angle camera sees all 22 players at once but struggles with:
- Ball detection (ball is tiny in a wide shot)
- Player detection confidence drops when camera is zoomed far out
- Individual player detail is low resolution

The moving follow camera solves these problems but introduces a new one:
- Players outside the frame of action are invisible
- Cannot track team shape or players not involved in play

Using both cameras together solves all problems:
- Stationary = player positions for all 22 players across the whole pitch
- Moving = ball tracking, event detection (goals, shots, corners, free kicks)
- Combined = complete picture with everything in the same coordinate space

### How the fusion works (conceptually)

The YOLO model does NOT do the fusion — it is just a detector.
The fusion is code written around the model in main.py.

Per frame the pipeline would:
1. Read one frame from each video simultaneously (both pre-cut to same start point)
2. Run the model on the stationary frame → get all player positions
3. Run the model on the moving frame → get ball position and events
4. Use homography to transform ball pixel coordinates from moving cam
   into the stationary cam coordinate space (same maths as the radar mode)
5. Now players (from stationary) and ball (from moving) are on the same 2D pitch map
6. Render the output on top of the stationary frame — full pitch always visible

### Training the model for dual camera

Do NOT train two separate models. Train one model on frames from both cameras:
- Add stationary camera frames to Roboflow (already doing this)
- Also add moving camera frames to the same Roboflow project
- Same class labels: player, ball, goalkeeper, referee
- Model learns to detect in both wide-angle and close-up contexts
- One best.pt file covers both use cases

### Video synchronisation

Both videos must be pre-cut to the same start moment before being fed to the pipeline.
Sean will trim both videos externally (iMovie, QuickTime, ffmpeg) to start at the
same reference point — kickoff whistle is the ideal sync point.
The code assumes videos are already in sync and reads them frame by frame together.
No hardcoded frame offset constants in the code — sync is handled in pre-processing.

### New mode to implement: DUAL_CAMERA

Add to main.py alongside existing modes. Command will look like:

```bash
PYTHONPATH=~/Documents/Home/soccer_poc/sports-main python main.py \
  --source_video_path_stationary "/path/to/stationary.mp4" \
  --source_video_path_moving "/path/to/moving.mp4" \
  --target_video_path data/output_dual.mp4 \
  --device mps \
  --mode DUAL_CAMERA
```

The output video is based on the stationary frame (full pitch always visible)
with ball position projected from the moving camera on top.

### What the DUAL_CAMERA output produces

- Annotated video: stationary view with player ellipses + IDs + ball marker + radar
- match_data.json: player positions from stationary + ball positions from moving
- player_id_list.json: for manual validation as before
- Events detected from moving camera (goal, shot, corner) with timestamps

### Medallion architecture for Databricks (future)

Bronze layer: raw video files in cloud storage (two per game)
Silver layer: AI inference job using custom YOLO model stored in MLflow Model Registry
              + manual validation step (player_id_list.json filled in)
Gold layer:   clean Delta tables — player_positions, game_events, player_game_stats, fines
Dashboard:    Looker Studio connected to Gold tables

The model (.pt file) lives in MLflow Model Registry, not as a Delta table.
The Silver job pulls the registered model version and runs inference.
This allows model versioning — you can compare stats from model v1 vs v2 over same footage.