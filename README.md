# Soccer PoC — Setup & Run Guide (Mac)

## One-time setup (copy/paste these into Terminal)

```bash
# 1. Create your project folder
mkdir ~/soccer_poc
cd ~/soccer_poc

# 2. Create a Python virtual environment (keeps things tidy)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install inference-sdk opencv-python matplotlib numpy scipy

# 4. Copy the 4 Python scripts into this folder:
#    analyse.py  verify.py  report.py  (+ this README)

# 5. Put your video in the folder and name it game.mp4
#    Or edit VIDEO_PATH in analyse.py to match your filename
```

---

## Get your Roboflow key

1. Go to roboflow.com → sign up free
2. Profile (top right) → Settings → Roboflow API → copy your key
3. Go to Universe → search "football players detection"
   Recommended model: football-players-detection-3zvbc  (version 8)
   Also search "football ball detection" for the ball model
4. Paste your key into the API_KEY line at the top of analyse.py

---

## Every time you want to analyse a game

```bash
# Navigate to your folder and activate the environment
cd ~/soccer_poc
source venv/bin/activate

# Step 1: Run the main analysis (takes 10-30 mins per game)
python analyse.py

# Step 2: Watch the annotated video to check accuracy
python verify.py

# Step 3: Generate the visual report
python report.py

# View the report
open match_report.png
```

---

## What each script does

| Script | What it produces | How long |
|---|---|---|
| analyse.py | match_data.json with all detections | 10-30 mins |
| verify.py | verified_output.mp4 with boxes drawn on video | 2-5 mins |
| report.py | match_report.png — charts and heatmaps | Seconds |

---

## What the model tracks

| Stat | Method | Reliability |
|---|---|---|
| Player positions | Roboflow detection per frame | Good if camera is wide |
| Ball position | Roboflow ball model | Medium — ball is small |
| Possession | Which half ball spends more time in | Rough but useful |
| Free kicks | Ball stationary 2+ secs with players nearby | Good |
| Corners | Ball stationary in corner zone | Good |
| Possible goals | High motion near goal area | Needs manual verification |
| Player movement | Aggregate position over time | Zone-level only |

---

## The moving camera problem

Your camera auto-tracks the ball which means:
- When play is in the centre, you'll get detections across most of the pitch
- When camera pans to follow a run, some players drop off screen
- This is NORMAL and expected — the data will just be sparser in those moments
- The heatmap will reflect what the camera actually captured, not the full pitch

To compensate: use SAMPLE_EVERY = 15 (2 samples/sec) for more data points.
This uses more Roboflow API credits but gives better coverage.

---

## Tuning tips

Too many false detections (referee, flags, etc.)?
→ Raise CONFIDENCE to 0.55 in analyse.py

Missing too many real players?
→ Lower CONFIDENCE to 0.35

Too slow?
→ Raise SAMPLE_EVERY to 60 (one sample per 2 seconds)

Want more data?
→ Lower SAMPLE_EVERY to 15

Free kicks not being detected?
→ Lower the still_frames threshold in analyse.py from 2 to 1

---

## Common errors and fixes

"No module named inference"
→ Make sure you ran: source venv/bin/activate

"Cannot open video"
→ Check the VIDEO_PATH in analyse.py matches your actual filename

"0 detections on everything"
→ Check your API key is correct and you have Roboflow credits remaining

Video window won't open on Mac
→ This is a known Mac/OpenCV issue. Set SHOW_LIVE = False in verify.py
   and just watch verified_output.mp4 in QuickTime afterwards

---

## Moving to Databricks later

When you're ready, match_data.json loads straight into a Delta table:

```python
import json
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

with open("match_data.json") as f:
    data = json.load(f)

spark.createDataFrame(data["player_positions"]).write.saveAsTable("soccer_ai.player_positions")
spark.createDataFrame(data["ball_positions"]).write.saveAsTable("soccer_ai.ball_positions")
spark.createDataFrame(data["events"]).write.saveAsTable("soccer_ai.game_events")
```

That's it. Everything you've built locally ports directly across.
