"""Is the detector missing the near players, or is our own filter deleting them?

The visual comparison showed boxes clustered along the FAR touchline while the
large foreground players went unboxed. That is the opposite of a small-object
problem, so before doing anything about preprocessing, find out whether these
detections exist and are being thrown away by clean_detections() — the pitch
polygon being the obvious suspect, since it is the filter that knows about
position.
"""
import warnings, sys
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/sean/Documents/Home/soccer_poc/sports-main/examples/soccer')
import cv2, supervision as sv, rfdetr_onnx as rf, main

ST='data/base_datasets/14_09/Stationary_Camera_14_08.mp4'
info=sv.VideoInfo.from_video_path(ST); fps=info.fps
POLY=np.load('data/pitch_polygon_14_09.npy').astype(np.int32)

frame=None
for f in main.video_frames(ST, start_frame=int((477+30)*fps)):
    frame=f; break
H,W=frame.shape[:2]
print(f'frame {W}x{H}')
print(f'polygon x range {POLY[:,0].min()}-{POLY[:,0].max()}, '
      f'y range {POLY[:,1].min()}-{POLY[:,1].max()}')
print(f'polygon area {cv2.contourArea(POLY)/(W*H):.1%} of frame')

raw = rf.detect(frame, conf=0.20)
def people(d):
    return d[np.isin(d.class_id,[1,2,3])] if len(d) and d.class_id is not None else d
raw_p = people(raw)

main.PITCH_POLYGON=POLY; main.FAR_TOUCHLINE=None
cleaned = people(main.clean_detections(rf.detect(frame, conf=0.20), W, H))
main.PITCH_POLYGON=None
nopoly  = people(main.clean_detections(rf.detect(frame, conf=0.20), W, H))

print(f'\nraw detections (people)            {len(raw_p)}')
print(f'after clean_detections, NO polygon  {len(nopoly)}')
print(f'after clean_detections, WITH polygon {len(cleaned)}')

# where are the dropped ones?
feet=np.stack([(raw_p.xyxy[:,0]+raw_p.xyxy[:,2])/2, raw_p.xyxy[:,3]],axis=1)
inside=np.array([cv2.pointPolygonTest(POLY,(float(x),float(y)),False)>=0 for x,y in feet])
print(f'\nraw feet inside polygon {inside.sum()} / {len(feet)}')
h=raw_p.xyxy[:,3]-raw_p.xyxy[:,1]
if (~inside).any():
    print(f'  dropped boxes: median height {np.median(h[~inside]):.0f}px, '
          f'feet y range {feet[~inside,1].min():.0f}-{feet[~inside,1].max():.0f}')
if inside.any():
    print(f'  kept boxes   : median height {np.median(h[inside]):.0f}px, '
          f'feet y range {feet[inside,1].min():.0f}-{feet[inside,1].max():.0f}')

vis=frame.copy()
cv2.polylines(vis,[POLY],True,(0,255,255),4)
for (x1,y1,x2,y2),ins in zip(raw_p.xyxy.astype(int), inside):
    cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0) if ins else (0,0,255),3)
cv2.putText(vis,'green=kept  red=dropped by polygon  yellow=polygon',
            (20,50),cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,255,255),3)
cv2.imwrite(f'{sys.argv[1]}/polygon_check.png',
            cv2.resize(vis,None,fx=0.42,fy=0.42))
print('\n-> polygon_check.png')
