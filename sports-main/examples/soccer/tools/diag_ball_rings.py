"""Two reported defects, measured before fixing.

1. BALL near the goalpost. get_ball() keeps every ball-class detection above
   0.30 with no size, position or temporal test — and there is only one ball on
   a pitch. Question: are the false ones separable by size, position or
   confidence, or do they look exactly like the real ball?

2. MISSING IDENTITY RINGS on 1-2 visible players. Pass 2 draws only ids in
   good_ids, so a player whose fragment is shorter than MIN_SECONDS_TO_KEEP
   (3.0s) is detected but never drawn. Question: how many detections per frame
   are dropped this way, and how long do the dropped ids actually live?
"""
import warnings, sys, collections
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/sean/Documents/Home/soccer_poc/sports-main/examples/soccer')
import cv2, supervision as sv, rfdetr_onnx as rf, main

ST='data/base_datasets/14_09/Stationary_Camera_14_08.mp4'
info=sv.VideoInfo.from_video_path(ST); fps=info.fps
POLY=np.load('data/pitch_polygon_14_09.npy').astype(np.int32)
main.PITCH_POLYGON=POLY; main.FAR_TOUCHLINE=None
N=1800

t=main.PlayerReIDTracker(info.width,info.height,fps,'mps')
balls=[]; per_frame=[]
n=0
for f in main.video_frames(ST, start_frame=int(477*fps)):
    raw=rf.detect(f, conf=0.20)
    if len(raw) and raw.class_id is not None:
        b=raw[raw.class_id==main.BALL_CLASS_ID]
        for (x1,y1,x2,y2),c in zip(b.xyxy, b.confidence):
            if c>=0.30:
                balls.append((n, float((x1+x2)/2), float(y2), float(x2-x1),
                              float(y2-y1), float(c)))
    d=main.clean_detections(raw, info.width, info.height)
    if len(d) and d.class_id is not None:
        d=d[np.isin(d.class_id,[1,2])]
    out=t.update(d, None)
    per_frame.append(out.tracker_id.tolist()
                     if out.tracker_id is not None and len(out) else [])
    n+=1
    if n>=N: break

# ---------- BALL ----------
B=np.array(balls) if balls else np.zeros((0,6))
print(f'--- BALL over {N} frames ({N/fps:.0f}s) ---')
print(f'{len(B)} ball detections above 0.30 in {len(set(B[:,0].astype(int)))} frames'
      if len(B) else 'no ball detections')
if len(B):
    mult=collections.Counter(B[:,0].astype(int))
    print(f'frames with >1 ball: {sum(1 for v in mult.values() if v>1)}')
    print(f'width  px: p10={np.percentile(B[:,3],10):.0f} '
          f'median={np.median(B[:,3]):.0f} p90={np.percentile(B[:,3],90):.0f} '
          f'max={B[:,3].max():.0f}')
    print(f'conf     : p10={np.percentile(B[:,5],10):.2f} '
          f'median={np.median(B[:,5]):.2f}')
    inside=np.array([cv2.pointPolygonTest(POLY,(float(x),float(y)),False)>=0
                     for x,y in B[:,1:3]])
    print(f'inside pitch polygon: {inside.sum()}/{len(B)}')
    # cluster x positions to spot a persistent false source (a goalpost)
    hist,edges=np.histogram(B[:,1], bins=16, range=(0,info.width))
    hot=[(int(edges[i]),int(edges[i+1]),int(hist[i])) for i in np.argsort(hist)[::-1][:4]]
    print(f'busiest x bands (a static hotspot = furniture): {hot}')

# ---------- RINGS ----------
good=t.valid_ids()
life={}
for cid,cnt in t.id_frame_count.items(): life[cid]=cnt
drawn=dropped=0
for ids in per_frame:
    for i in ids:
        if i in good: drawn+=1
        else: dropped+=1
print(f'\n--- IDENTITY RINGS ---')
print(f'detections drawn   {drawn/N:.1f}/frame')
print(f'detections dropped {dropped/N:.1f}/frame  (id not in good_ids)')
bad=[c for c in t.id_frame_count if c not in good]
if bad:
    secs=np.array([life[c]/fps for c in bad])
    print(f'{len(bad)} rejected ids: lifetime median {np.median(secs):.1f}s, '
          f'p90 {np.percentile(secs,90):.1f}s')
    under=sum(1 for s in secs if s < main.MIN_SECONDS_TO_KEEP)
    print(f'  {under}/{len(bad)} rejected purely for living < '
          f'{main.MIN_SECONDS_TO_KEEP}s (MIN_SECONDS_TO_KEEP)')
    slow=[c for c in bad if life[c]/fps >= main.MIN_SECONDS_TO_KEEP]
    print(f'  {len(slow)} lived long enough but failed the movement test')
