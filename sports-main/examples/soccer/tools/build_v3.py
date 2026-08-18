"""Build the final polygon: static-crowd boundary, gaps filled, then verify.

The per-column static band is the right signal but sparse — a column with no
static detection is a gap in DETECTION, not a gap in crowd, since spectators
line the touchline continuously. So fill empty columns from their neighbours
rather than letting the boundary drop to the frame top.
"""
import warnings, sys, collections, pickle
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/sean/Documents/Home/soccer_poc/sports-main/examples/soccer')
import cv2, supervision as sv, rfdetr_onnx as rf, main

ST='data/base_datasets/14_09/Stationary_Camera_14_08.mp4'
info=sv.VideoInfo.from_video_path(ST); fps=info.fps
W,H=info.width,info.height
main.PITCH_POLYGON=None; main.FAR_TOUCHLINE=None
CELL=64; NF=40; STEP=int(3*fps)

occ=collections.Counter(); keep=[]
n=taken=0
for f in main.video_frames(ST, start_frame=int(477*fps)):
    if n % STEP == 0:
        d=rf.detect(f, conf=0.20)
        if len(d) and d.class_id is not None:
            d=d[np.isin(d.class_id,[1,2,3])]
            for (x1,y1,x2,y2) in d.xyxy:
                occ[(int((x1+x2)/2)//CELL, int(y2)//CELL)] += 1
        if taken<6: keep.append(f)
        taken+=1
        if taken>=NF: break
    n+=1

XB=W//CELL+1
raw={}
for xb in range(XB):
    static=[yb for (x,yb),c in occ.items() if x==xb and c>=0.60*NF]
    raw[xb]=(max(static)+1)*CELL if static else None

# fill gaps from neighbours, then take a local max so the boundary never dips
# below a neighbouring column's crowd
filled=[]
for xb in range(XB):
    if raw[xb] is not None:
        filled.append(raw[xb]); continue
    near=[raw[x] for x in range(max(0,xb-6), min(XB,xb+7)) if raw[x] is not None]
    filled.append(max(near) if near else 0)
sm=[int(max(filled[max(0,i-2):i+3])) for i in range(XB)]

pts=[(xb*CELL, min(sm[xb], 420)) for xb in range(0, XB, 3)]
pts=[(0,pts[0][1])]+pts+[(W, sm[-1])]
poly=np.array(pts+[(W,H),(0,H)], dtype=np.int32)
np.save('data/pitch_polygon_14_09_v3.npy', poly)
print(f'v3: {len(poly)} points, area {cv2.contourArea(poly)/(W*H):.1%} of frame')
print('top edge y by x:', [(int(a),int(b)) for a,b in pts[::3]])

# --- verify against v2 and the original ---
OLD=np.load('data/pitch_polygon_14_09.npy').astype(np.int32)
V2 =np.load('data/pitch_polygon_14_09_v2.npy').astype(np.int32)
def count(frame, p):
    main.PITCH_POLYGON=p; main.FAR_TOUCHLINE=None
    d=main.clean_detections(rf.detect(frame, conf=0.20), W, H)
    if not len(d) or d.class_id is None: return 0
    return int(np.isin(d.class_id,[1,2]).sum())
print(f'\n{"frame":>6}{"original":>10}{"v2 (loose)":>12}{"v3 (static)":>13}')
a=b=c=0
for i,f in enumerate(keep):
    x,y,z=count(f,OLD),count(f,V2),count(f,poly); a+=x;b+=y;c+=z
    print(f'{i*3:>4}s{x:>10}{y:>12}{z:>13}', flush=True)
k=len(keep)
print(f'{"mean":>6}{a/k:>10.1f}{b/k:>12.1f}{c/k:>13.1f}   (~24 on the pitch)')

f=keep[len(keep)//2]
main.PITCH_POLYGON=poly; main.FAR_TOUCHLINE=None
d=main.clean_detections(rf.detect(f, conf=0.20), W, H)
d=d[np.isin(d.class_id,[1,2])]
vis=f.copy(); cv2.polylines(vis,[poly],True,(0,255,255),4)
for (x1,y1,x2,y2) in d.xyxy.astype(int):
    cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),3)
cv2.putText(vis,f'v3 STATIC-DERIVED: {len(d)} kept',(20,70),
            cv2.FONT_HERSHEY_SIMPLEX,1.7,(0,255,255),4)
cv2.imwrite(f'{sys.argv[1]}/poly_v3.png', cv2.resize(vis,None,fx=0.45,fy=0.45))
print('-> poly_v3.png')
