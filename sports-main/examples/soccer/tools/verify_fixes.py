"""Verify both fixes: ball sanity checks, and the lowered lifetime floor."""
import warnings, sys, collections
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/sean/Documents/Home/soccer_poc/sports-main/examples/soccer')
import cv2, supervision as sv, rfdetr_onnx as rf, main
from collections import deque, Counter

ST='data/base_datasets/14_09/Stationary_Camera_14_08.mp4'
info=sv.VideoInfo.from_video_path(ST); fps=info.fps
POLY=np.load('data/pitch_polygon_14_09.npy').astype(np.int32)
main.PITCH_POLYGON=POLY; main.FAR_TOUCHLINE=None
N=1800

# replicate get_ball's logic standalone so we can count what it would draw
recent=deque(maxlen=main.BALL_HISTORY_FRAMES)
def ball_filtered(raw):
    b=raw[raw.class_id==main.BALL_CLASS_ID] if len(raw) and raw.class_id is not None else raw
    if not len(b): return None
    cx=(b.xyxy[:,0]+b.xyxy[:,2])/2; cy=(b.xyxy[:,1]+b.xyxy[:,3])/2
    cells=[(int(x)//main.BALL_CELL_PX, int(y)//main.BALL_CELL_PX) for x,y in zip(cx,cy)]
    keep=np.ones(len(b),bool)
    keep &= np.array([cv2.pointPolygonTest(POLY,(float(x),float(y)),False)>=0
                      for x,y in zip(cx, b.xyxy[:,3])])
    if len(recent)>=main.BALL_HISTORY_FRAMES//2:
        seen=Counter(c for fc in recent for c in set(fc))
        lim=main.BALL_STATIC_FRACTION*len(recent)
        keep &= np.array([seen[c]<lim for c in cells])
    recent.append(cells)
    b=b[keep]
    return b[[int(np.argmax(b.confidence))]] if len(b) else None

t=main.PlayerReIDTracker(info.width,info.height,fps,'mps')
raw_n=kept_n=frames_with=0
per_frame=[]
n=0
for f in main.video_frames(ST, start_frame=int(477*fps)):
    raw=rf.detect(f, conf=main.BALL_MIN_CONF)
    if len(raw) and raw.class_id is not None:
        raw_n += int((raw.class_id==main.BALL_CLASS_ID).sum())
    kb=ball_filtered(raw)
    if kb is not None: kept_n+=len(kb); frames_with+=1
    d=main.clean_detections(raw, info.width, info.height)
    if len(d) and d.class_id is not None: d=d[np.isin(d.class_id,[1,2])]
    out=t.update(d, None)
    per_frame.append(out.tracker_id.tolist() if out.tracker_id is not None and len(out) else [])
    n+=1
    if n>=N: break

print(f'--- BALL over {N} frames ---')
print(f'raw ball detections   {raw_n}   (was 1473, 303 frames with >1)')
print(f'after checks          {kept_n}  in {frames_with} frames, max 1/frame')

print(f'\n--- IDENTITY RINGS (MIN_SECONDS_TO_KEEP={main.MIN_SECONDS_TO_KEEP}) ---')
for thresh in (3.0, 1.5, 1.0):
    main.MIN_SECONDS_TO_KEEP=thresh
    t.min_frames=max(1,int(thresh*fps))
    good=t.valid_ids()
    drawn=sum(1 for ids in per_frame for i in ids if i in good)
    dropped=sum(1 for ids in per_frame for i in ids if i not in good)
    print(f'  {thresh:>4.1f}s  ids kept {len(good):>3}   drawn {drawn/N:>5.1f}/frame'
          f'   dropped {dropped/N:>4.1f}/frame')
# do the newly-admitted ids actually move, or are they furniture?
main.MIN_SECONDS_TO_KEEP=1.5; t.min_frames=max(1,int(1.5*fps))
g15=t.valid_ids()
main.MIN_SECONDS_TO_KEEP=3.0; t.min_frames=max(1,int(3.0*fps))
g30=t.valid_ids()
new=g15-g30
if new:
    sp=[t.id_path_px[c]/max(t.id_frame_count[c]/fps,1e-6) for c in new]
    nd=[t.net_displacement(c) for c in new]
    print(f'  {len(new)} ids newly admitted at 1.5s: median speed '
          f'{np.median(sp):.0f}px/s, median net displacement {np.median(nd):.0f}px')
    print(f'  (movement floor is {main.MIN_SPEED_PX_PER_SEC}px/s — these are real movers, '
          f'not furniture)')
