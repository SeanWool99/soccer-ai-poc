"""Roboflow hosted detection, returned in the same shape as our local model.

Why this exists: the hosted `football-players-detection-3zvbc/20` model detects
this footage far better than any weights we have locally. Measured on the 12_08
moving view, where a dozen players are plainly visible:

    stock (broadcast)  6 detections at conf 0.10, 3 above 0.25
    round1 (game.mp4)  0
    round2 (ultrawide) 2
    roboflow v20       ~20-30, confidences 0.4-0.9, and it finds the ball

It is the same DATASET our stock weights came from — but our copy is an early
version and the hosted model is at v20, nineteen rounds of additions later.

Its class ids already match main.py (ball=0, goalkeeper=1, player=2,
referee=3), so detections drop straight into the existing pipeline.

The API key is read from the environment, or from ~/.config/soccer_poc/env.
Never hardcode it — it should not land in the repo.
"""
import os
from typing import List, Optional

import numpy as np
import supervision as sv

MODEL_ID = 'football-players-detection-3zvbc/20'
API_URL = 'https://serverless.roboflow.com'
ENV_FILE = os.path.expanduser('~/.config/soccer_poc/env')


def _api_key() -> str:
    key = os.environ.get('ROBOFLOW_API_KEY')
    if key:
        return key
    if os.path.exists(ENV_FILE):
        for line in open(ENV_FILE):
            if line.startswith('ROBOFLOW_API_KEY='):
                return line.split('=', 1)[1].strip()
    raise RuntimeError(
        'No Roboflow API key. Set ROBOFLOW_API_KEY, or put '
        f'ROBOFLOW_API_KEY=... in {ENV_FILE}')


_client = None


def client():
    global _client
    if _client is None:
        from inference_sdk import InferenceHTTPClient
        _client = InferenceHTTPClient(api_url=API_URL, api_key=_api_key())
    return _client


def to_detections(payload) -> sv.Detections:
    """Convert a Roboflow response into sv.Detections.

    Roboflow gives box CENTRES plus width/height; supervision wants corners.
    The response is sometimes wrapped in an extra 'predictions' layer, so
    unwrap defensively rather than assuming one shape.
    """
    preds = payload
    while isinstance(preds, dict) and 'predictions' in preds:
        preds = preds['predictions']
    if not isinstance(preds, list):
        return sv.Detections.empty()

    xyxy, conf, cls = [], [], []
    for p in preds:
        w, h = float(p['width']), float(p['height'])
        cx, cy = float(p['x']), float(p['y'])
        xyxy.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        conf.append(float(p.get('confidence', 0.0)))
        cls.append(int(p.get('class_id', 2)))
    if not xyxy:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(conf, dtype=np.float32),
        class_id=np.array(cls, dtype=int),
    )


def infer_image(path: str) -> sv.Detections:
    """Run the hosted model on an image file."""
    return to_detections(client().infer(path, model_id=MODEL_ID))


def infer_frame(frame: np.ndarray, tmp_dir: Optional[str] = None) -> sv.Detections:
    """Run the hosted model on an in-memory BGR frame.

    Each call is a network round trip, so this suits labelling a few hundred
    frames — not a 45-minute match at ~150,000 frames.
    """
    import tempfile
    import cv2
    d = tmp_dir or tempfile.gettempdir()
    tmp = os.path.join(d, '_rf_frame.jpg')
    cv2.imwrite(tmp, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return infer_image(tmp)
