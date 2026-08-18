"""Is the stationary camera losing players to RF-DETR's stretch preprocessing?

RF-DETR takes a fixed 576x576 input and STRETCHES to it. The stationary view is
4096x1152 (aspect 3.56), so x is scaled 0.141 and y 0.500 — players arrive
squashed 3.6x horizontally, roughly 5px wide. The moving view is 1920x1080
(aspect 1.78) and suffers far less. That would explain players being missed on
the stationary view wherever they stand, while the moving view finds everyone.

Test: same frames, whole-frame versus square TILES. A 1152x1152 tile stretched
to 576 scales both axes by 0.5 — no distortion at all. If the stretch is the
problem, tiling should find substantially more players.
"""
import warnings, sys
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/sean/Documents/Home/soccer_poc/sports-main/examples/soccer')
import cv2, supervision as sv, rfdetr_onnx as rf, main

ST='data/base_datasets/14_09/Stationary_Camera_14_08.mp4'
MV='data/base_datasets/14_09/Moving_Camera_14_08.mp4'
info=sv.VideoInfo.from_video_path(ST); fps=info.fps

def tiled(frame, conf, n_tiles=4, overlap=0.18):
    """Run the detector on square-ish tiles and merge, undoing distortion."""
    h,w=frame.shape[:2]
    step=w/n_tiles
    tw=step*(1+overlap)
    out=[]
    for i in range(n_tiles):
        x0=max(0, int(i*step - (tw-step)/2)); x1=min(w, int(x0+tw))
        t=frame[:, x0:x1]
        d=rf.detect(t, conf=conf)
        if len(d):
            d.xyxy[:, [0,2]] += x0
            out.append(d)
    if not out: return sv.Detections.empty()
    m=sv.Detections(xyxy=np.vstack([d.xyxy for d in out]),
                    confidence=np.concatenate([d.confidence for d in out]),
                    class_id=np.concatenate([d.class_id for d in out]))
    return main.suppress_contained_boxes(m.with_nms(threshold=0.55, class_agnostic=True))

def people(d):
    if not len(d) or d.class_id is None: return 0
    return int(np.isin(d.class_id, [1,2,3]).sum())

main.PITCH_POLYGON=np.load('data/pitch_polygon_14_09.npy').astype(np.int32)
main.FAR_TOUCHLINE=None

print(f'{"":<8}{"frame":>7}{"whole 4096x1152":>18}{"4 square tiles":>16}{"gain":>7}')
tot_w=tot_t=0
for secs in (477, 492, 507, 522, 537):
    got=None
    for f in main.video_frames(ST, start_frame=int(secs*fps)):
        got=f; break
    dw=main.clean_detections(rf.detect(got, conf=0.20), info.width, info.height)
    dt=main.clean_detections(tiled(got, 0.20), info.width, info.height)
    tot_w+=people(dw); tot_t+=people(dt)
    print(f'{"stat":<8}{secs-477:>7}{people(dw):>18}{people(dt):>16}'
          f'{people(dt)-people(dw):>+7}', flush=True)
print(f'{"":<8}{"mean":>7}{tot_w/5:>18.1f}{tot_t/5:>16.1f}{(tot_t-tot_w)/5:>+7.1f}')

# the moving view, for reference — the one that works
mi=sv.VideoInfo.from_video_path(MV)
print(f'\nmoving view {mi.width}x{mi.height} (aspect {mi.width/mi.height:.2f}):')
tw_=tt_=0
for secs in (477, 492, 507):
    got=None
    for f in main.video_frames(MV, start_frame=int(secs*mi.fps)):
        got=f; break
    dw=rf.detect(got, conf=0.20); dt=tiled(got, 0.20, n_tiles=2)
    tw_+=people(dw); tt_+=people(dt)
    print(f'  t={secs-477:>3}s  whole {people(dw):>3}   2 tiles {people(dt):>3}')
print(f'  mean   whole {tw_/3:.1f}   tiled {tt_/3:.1f}')
