"""Separate NAVY from SKY BLUE — the axis these two kits actually differ on.

Looking at the crops settled it: both teams wear blue. The previous torso test
clustered on hue and split 27-25 with clusters at 227deg and 34deg, but 34deg
is orange-brown — the dry pitch. It was separating tightly-framed crops from
ones containing a lot of ground, not one team from the other, and per-tracklet
framing consistency is why it still scored 89%.

So: restrict to blue-ish pixels only, and cluster on BRIGHTNESS within them.
Navy is dark blue, sky is light blue, and pitch/skin/shadow are excluded by the
hue gate rather than by a saturation threshold that brown passes.
"""
import warnings, sys, collections, pickle
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/sean/Documents/Home/soccer_poc/sports-main/examples/soccer')
import cv2, supervision as sv, rfdetr_onnx as rf, main
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ST='data/base_datasets/14_09/Stationary_Camera_14_08.mp4'
main.PITCH_POLYGON=np.load('data/pitch_polygon_14_09.npy').astype(np.int32)
main.FAR_TOUCHLINE=None
info=sv.VideoInfo.from_video_path(ST); fps=info.fps

t=main.PlayerReIDTracker(info.width,info.height,fps,'mps')
crops=collections.defaultdict(list)
n=0
for f in main.video_frames(ST, start_frame=int(477*fps)):
    d=rf.detect(f, conf=0.20)
    d=main.clean_detections(d, info.width, info.height)
    if len(d) and d.class_id is not None:
        d=d[np.isin(d.class_id,[1,2])]
    out=t.update(d, None)
    if out.tracker_id is not None and len(out) and n%8==0:
        for tid,box in zip(out.tracker_id, out.xyxy):
            if len(crops[int(tid)])>=10: continue
            x1,y1,x2,y2=[int(v) for v in box]
            x1,y1=max(0,x1),max(0,y1); x2,y2=min(info.width,x2),min(info.height,y2)
            if x2-x1<8 or y2-y1<16: continue
            crops[int(tid)].append(f[y1:y2, x1:x2].copy())
    n+=1
    if n>=900: break
crops={k:v for k,v in crops.items() if len(v)>=4}

def blue_feat(c):
    """Brightness and saturation of the BLUE pixels in the torso region."""
    h,w=c.shape[:2]
    roi=c[int(h*0.15):int(h*0.55), int(w*0.25):int(w*0.75)]
    if roi.size==0: return None
    hsv=cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H,S,V = hsv[...,0].astype(float), hsv[...,1].astype(float), hsv[...,2].astype(float)
    blue = (H>=95)&(H<=140)&(S>50)          # OpenCV hue: 95-140 ~ 190-280 deg
    if blue.sum() < 15: return None          # not enough shirt visible
    return np.array([V[blue].mean()/255, S[blue].mean()/255, blue.mean()])

feats=[]; owner=[]
for tid,v in crops.items():
    for c in v:
        f_=blue_feat(c)
        if f_ is not None: feats.append(f_); owner.append(tid)
X=np.array(feats)
kept=len(set(owner))
print(f'{len(crops)} tracklets, {kept} with enough blue pixels, {len(X)} crops')

km=KMeans(n_clusters=2, n_init=10, random_state=0).fit(X[:, :2])
per=collections.defaultdict(list)
for o,l in zip(owner, km.labels_): per[o].append(l)
maj={o:collections.Counter(v).most_common(1)[0][0] for o,v in per.items()}
cons=np.mean([collections.Counter(v).most_common(1)[0][1]/len(v) for v in per.values()])
split=sorted(collections.Counter(maj.values()).values(), reverse=True)
print(f'\nblue-brightness KMeans: split {split}  '
      f'within-tracklet consistency {cons:.0%}  '
      f'silhouette {silhouette_score(X[:, :2], km.labels_):.3f}')
for lab in (0,1):
    sel=X[km.labels_==lab]
    print(f'  cluster {lab}: n={len(sel):>4}  brightness {sel[:,0].mean():.2f}  '
          f'saturation {sel[:,1].mean():.2f}')

# montage ordered by cluster, so the split can be checked by eye
CW,CH=48,96
order=sorted(maj, key=lambda o:(maj[o],o))
tiles=[cv2.resize(crops[o][len(crops[o])//2],(CW,CH)) for o in order]
cols=13; rows=(len(tiles)+cols-1)//cols
sheet=np.zeros((rows*CH, cols*CW,3), np.uint8)
for i,tl in enumerate(tiles):
    r,c=divmod(i,cols); sheet[r*CH:(r+1)*CH, c*CW:(c+1)*CW]=tl
n0=sum(1 for o in order if maj[o]==0)
cv2.imwrite(f'{sys.argv[1]}/kits_split.png',
            cv2.resize(sheet,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
print(f'\nmontage -> kits_split.png : first {n0} tiles are cluster 0, rest cluster 1')
