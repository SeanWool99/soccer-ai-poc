"""Is team classification good enough to constrain stitching?

Cluster balance (the old 17-v-4 symptom) is a weak test — it says nothing
about whether individual crops are labelled correctly. This uses a stronger
signal that costs no labelling: WITHIN-TRACKLET CONSISTENCY. Every crop drawn
from one tracklet is the same player by construction, so all of them must get
the same team. Chance is ~50%; usable is ~95%+.

Also tests a suspected bug. TeamClassifier.fit() embeds crops through
extract_features(), which applies contrast x1.5 and colour x1.8. predict()
inlines its own embedding path and applies NEITHER. Fit and predict therefore
see differently-processed images and the UMAP projection learnt by one does not
correspond to the other — which would degrade clustering regardless of how
separable the kits are.
"""
import warnings, sys, collections, pickle
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/sean/Documents/Home/soccer_poc/sports-main/examples/soccer')
sys.path.insert(0,'/Users/sean/Documents/Home/soccer_poc/sports-main')
import cv2, supervision as sv, rfdetr_onnx as rf, main
from sports.common.team import TeamClassifier
from sklearn.metrics import silhouette_score

ST='data/base_datasets/14_09/Stationary_Camera_14_08.mp4'
main.PITCH_POLYGON=np.load('data/pitch_polygon_14_09.npy').astype(np.int32)
main.FAR_TOUCHLINE=None
info=sv.VideoInfo.from_video_path(ST); fps=info.fps
N=1200
PER_TRACK=12

t=main.PlayerReIDTracker(info.width,info.height,fps,'mps')
crops=collections.defaultdict(list); heights=[]
n=0
for f in main.video_frames(ST, start_frame=int(477*fps)):
    d=rf.detect(f, conf=0.20)
    d=main.clean_detections(d, info.width, info.height)
    if len(d) and d.class_id is not None:
        d=d[np.isin(d.class_id,[1,2])]
    out=t.update(d, None)
    if out.tracker_id is not None and len(out) and n%6==0:
        for tid,box in zip(out.tracker_id, out.xyxy):
            if len(crops[int(tid)])>=PER_TRACK: continue
            x1,y1,x2,y2=[int(v) for v in box]
            x1,y1=max(0,x1),max(0,y1); x2,y2=min(info.width,x2),min(info.height,y2)
            if x2-x1<8 or y2-y1<16: continue
            crops[int(tid)].append(f[y1:y2, x1:x2].copy()); heights.append(y2-y1)
    n+=1
    if n>=N: break

crops={k:v for k,v in crops.items() if len(v)>=6}
allc=[c for v in crops.values() for c in v]
owner=[k for k,v in crops.items() for _ in v]
print(f'{len(crops)} tracklets, {len(allc)} crops, median player height '
      f'{np.median(heights):.0f}px', flush=True)

clf=TeamClassifier(device='mps')
clf.fit(allc)

def consistency(labels):
    per=collections.defaultdict(list)
    for o,l in zip(owner,labels): per[o].append(l)
    fr=[collections.Counter(v).most_common(1)[0][1]/len(v) for v in per.values()]
    maj=[collections.Counter(v).most_common(1)[0][0] for v in per.values()]
    return np.mean(fr), collections.Counter(maj)

# a) predict() as it currently stands — no enhancement
lab_now = clf.predict(allc)
c_now, bal_now = consistency(lab_now)

# b) same pipeline as fit() — enhancement applied
emb = clf.extract_features(allc)
proj = clf.reducer.transform(emb)
lab_fix = clf.cluster_model.predict(proj)
c_fix, bal_fix = consistency(lab_fix)
sil = silhouette_score(proj, lab_fix)

print(f'\n{"path":<34}{"consistency":>13}{"team split":>14}')
print(f'{"predict() as shipped":<34}{c_now:>12.0%}   {sorted(bal_now.values(), reverse=True)}')
print(f'{"predict() with fit preprocessing":<34}{c_fix:>12.0%}   {sorted(bal_fix.values(), reverse=True)}')
print(f'\nsilhouette on UMAP projection: {sil:.3f}  (>0.5 = well separated)')
print(f'crops agreeing between the two paths: {np.mean(lab_now==lab_fix):.0%}')
pickle.dump({'labels':lab_fix,'owner':owner}, open(f'{sys.argv[1]}/team_labels.pkl','wb'))
