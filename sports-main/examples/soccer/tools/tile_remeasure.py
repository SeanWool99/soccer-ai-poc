"""Re-measure the tiling gain now the pitch polygon is correct.

The earlier +29% was measured through a polygon that deleted the near half of
the pitch, so it counted only what survived a broken filter. Redo it properly,
and time it — tiling costs 4x inference, which only pays if the gain is real.
"""
import warnings, sys, time
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/sean/Documents/Home/soccer_poc/sports-main/examples/soccer')
import cv2, supervision as sv, rfdetr_onnx as rf, main

ST='data/base_datasets/14_09/Stationary_Camera_14_08.mp4'
info=sv.VideoInfo.from_video_path(ST); fps=info.fps
W,H=info.width,info.height
main.PITCH_POLYGON=np.load('data/pitch_polygon_14_09.npy').astype(np.int32)
main.FAR_TOUCHLINE=None

def tiled(frame, conf, n_tiles, overlap=0.18):
    h,w=frame.shape[:2]; step=w/n_tiles; tw=step*(1+overlap); out=[]
    for i in range(n_tiles):
        x0=max(0,int(i*step-(tw-step)/2)); x1=min(w,int(x0+tw))
        d=rf.detect(frame[:, x0:x1], conf=conf)
        if len(d):
            d.xyxy[:, [0,2]] += x0; out.append(d)
    if not out: return sv.Detections.empty()
    m=sv.Detections(xyxy=np.vstack([d.xyxy for d in out]),
                    confidence=np.concatenate([d.confidence for d in out]),
                    class_id=np.concatenate([d.class_id for d in out]))
    return main.suppress_contained_boxes(m.with_nms(threshold=0.55, class_agnostic=True))

frames=[]
n=0
for f in main.video_frames(ST, start_frame=int(477*fps)):
    if n % int(5*fps) == 0: frames.append(f)
    n+=1
    if len(frames)>=8: break

def people(d):
    if not len(d) or d.class_id is None: return 0
    return int(np.isin(d.class_id,[1,2]).sum())

print(f'{len(frames)} frames, correct polygon, ~24 people on the pitch\n')
print(f'{"config":<14}{"players/frame":>15}{"sec/frame":>12}')
for name, fn in (('whole frame', lambda f: rf.detect(f, conf=0.20)),
                 ('2 tiles',     lambda f: tiled(f, 0.20, 2)),
                 ('3 tiles',     lambda f: tiled(f, 0.20, 3)),
                 ('4 tiles',     lambda f: tiled(f, 0.20, 4))):
    t0=time.time(); tot=0
    for f in frames:
        tot += people(main.clean_detections(fn(f), W, H))
    dt=(time.time()-t0)/len(frames)
    print(f'{name:<14}{tot/len(frames):>15.1f}{dt:>12.2f}', flush=True)
