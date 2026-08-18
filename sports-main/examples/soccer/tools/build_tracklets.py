"""Detect + track 2000 frames once, cache tracklets for stitcher experiments.

Detection is the expensive part (~6 min), and the stitcher work needs many
runs over the same fragments, so this is cached to disk.
"""
import warnings, sys, pickle, collections
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/sean/Documents/Home/soccer_poc/sports-main/examples/soccer')
import supervision as sv, rfdetr_onnx as rf, main

OUT=sys.argv[1]
ST='data/base_datasets/14_09/Stationary_Camera_14_08.mp4'
main.PITCH_POLYGON=np.load('data/pitch_polygon_14_09.npy').astype(np.int32)
main.FAR_TOUCHLINE=None
info=sv.VideoInfo.from_video_path(ST); fps=info.fps
N=2000

t=main.PlayerReIDTracker(info.width,info.height,fps,'mps')
tracks=collections.defaultdict(lambda: {'frames':[], 'xy':[], 'cls':[]})
n=0
for f in main.video_frames(ST, start_frame=int(477*fps)):
    d=rf.detect(f, conf=0.20)
    d=main.clean_detections(d, info.width, info.height)
    if len(d) and d.class_id is not None:
        d=d[np.isin(d.class_id,[1,2])]
    out=t.update(d, None)
    if out.tracker_id is not None and len(out):
        anch=out.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        for tid,xy,cl in zip(out.tracker_id, anch, out.class_id):
            tracks[int(tid)]['frames'].append(n)
            tracks[int(tid)]['xy'].append([float(xy[0]),float(xy[1])])
            tracks[int(tid)]['cls'].append(int(cl))
    n+=1
    if n>=N: break
    if n%500==0: print(f'  {n}/{N}', flush=True)

pickle.dump({'fps':float(fps),'n_frames':N,'tracks':dict(tracks)}, open(OUT,'wb'))
print(f'cached {len(tracks)} raw tracks over {n} frames -> {OUT}')
