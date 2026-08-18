"""Place the far touchline from data, using motion rather than my eye.

The rebuilt polygon fixed the real bug (near players deleted) but admits ~20
crowd detections along the far side on the right. Eyeballing the touchline is
unreliable there because spectators stand only ~30px above it.

Better signal: CROWD IS STATIC, PLAYERS MOVE. Sample frames across the clip and
histogram detection feet. A cell that is occupied in most frames is furniture —
a spectator, a parked car, a clubhouse door. A cell occupied occasionally is
pitch that players pass through. The touchline sits below the static band.
"""
import warnings, sys, collections
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/sean/Documents/Home/soccer_poc/sports-main/examples/soccer')
import cv2, supervision as sv, rfdetr_onnx as rf, main

ST='data/base_datasets/14_09/Stationary_Camera_14_08.mp4'
info=sv.VideoInfo.from_video_path(ST); fps=info.fps
W,H=info.width,info.height
main.PITCH_POLYGON=None; main.FAR_TOUCHLINE=None

CELL=64
occ=collections.Counter()
# ONE sequential pass, sampling as we go. video_frames(start_frame=N) grabs
# frame-by-frame from zero, so calling it once per sample re-grabs 26k frames
# every time — that is a 10-minute timeout, not a measurement.
NF=40
STEP=int(3*fps)
n=0; taken=0
for f in main.video_frames(ST, start_frame=int(477*fps)):
    if n % STEP == 0:
        d=rf.detect(f, conf=0.20)
        if len(d) and d.class_id is not None:
            d=d[np.isin(d.class_id,[1,2,3])]
            for (x1,y1,x2,y2) in d.xyxy:
                occ[(int((x1+x2)/2)//CELL, int(y2)//CELL)] += 1
        taken+=1
        if taken>=NF: break
    n+=1

XB=W//CELL+1
print(f'{NF} frames sampled, cell {CELL}px')
print(f'{"x range":>14}{"static band (>=60% of frames)":>32}{"touchline y":>13}')
edge={}
for xb in range(XB):
    col=[(yb,c) for (x,yb),c in occ.items() if x==xb]
    if not col: continue
    static=[yb for yb,c in col if c >= 0.60*NF]
    # touchline sits just below the lowest static row; if nothing is static in
    # this column there is no crowd here and the pitch runs to the frame top
    y = (max(static)+1)*CELL if static else 0
    edge[xb]=y
    if xb % 8 == 0 or static:
        print(f'{xb*CELL:>6}-{(xb+1)*CELL:<7}'
              f'{str(sorted(set(yb*CELL for yb in static))):>32}{y:>13}')

# Smooth into a monotone-ish boundary and emit polygon points
xs=sorted(edge)
ys=[edge[x] for x in xs]
sm=[int(max(ys[max(0,i-2):i+3])) for i in range(len(ys))]
pts=[(xs[i]*CELL, min(sm[i], 400)) for i in range(0,len(xs),4)]
pts=[(0,pts[0][1])]+pts+[(W, pts[-1][1])]
poly=np.array(pts+[(W,H),(0,H)], dtype=np.int32)
np.save('data/pitch_polygon_14_09_v3.npy', poly)
print(f'\nsaved v3: {len(poly)} points, area '
      f'{cv2.contourArea(poly)/(W*H):.1%} of frame')
print('top edge:', [(int(a),int(b)) for a,b in pts])
