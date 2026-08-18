"""Look at the crops, and test kit colour directly.

SigLIP embeds pose, background, lighting and kit together, and KMeans split
60-v-2 on this footage — so whatever dominates that embedding is not the shirt.
Before theorising, two concrete things:

  1. a montage of one crop per tracklet, to see what the kits actually look like
  2. clustering on TORSO COLOUR alone — the mean hue/saturation of the upper
     middle of each crop, which is shirt and little else

If the teams are separable at all, a targeted feature should find them where a
general-purpose embedding does not. If torso colour cannot separate them
either, the kits genuinely do not differ enough at this resolution and the
whole team-constraint idea needs rethinking.
"""
import warnings, sys, collections
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
            if len(crops[int(tid)])>=8: continue
            x1,y1,x2,y2=[int(v) for v in box]
            x1,y1=max(0,x1),max(0,y1); x2,y2=min(info.width,x2),min(info.height,y2)
            if x2-x1<8 or y2-y1<16: continue
            crops[int(tid)].append(f[y1:y2, x1:x2].copy())
    n+=1
    if n>=900: break
crops={k:v for k,v in crops.items() if len(v)>=4}
print(f'{len(crops)} tracklets')

# --- montage: one representative crop per tracklet, upscaled ---
CW,CH=48,96
tiles=[cv2.resize(v[len(v)//2],(CW,CH)) for v in crops.values()]
cols=min(16,len(tiles)); rows=(len(tiles)+cols-1)//cols
sheet=np.zeros((rows*CH, cols*CW, 3), np.uint8)
for i,tl in enumerate(tiles):
    r,c=divmod(i,cols); sheet[r*CH:(r+1)*CH, c*CW:(c+1)*CW]=tl
sheet=cv2.resize(sheet,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
cv2.imwrite(f'{sys.argv[1]}/kits.png', sheet)
print(f'montage -> kits.png  ({len(tiles)} tracklets)')

# --- torso colour only ---
def torso(c):
    h,w=c.shape[:2]
    roi=c[int(h*0.15):int(h*0.55), int(w*0.2):int(w*0.8)]
    if roi.size==0: return None
    hsv=cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m=hsv[...,1]>40                      # ignore washed-out/grey pixels
    if m.sum()<20: m=np.ones(hsv.shape[:2],bool)
    hh=hsv[...,0][m].astype(np.float32)*2*np.pi/180
    return np.array([np.cos(hh).mean(), np.sin(hh).mean(),
                     hsv[...,1][m].mean()/255, hsv[...,2][m].mean()/255])

feats=[]; owner=[]
for tid,v in crops.items():
    for c in v:
        f_=torso(c)
        if f_ is not None: feats.append(f_); owner.append(tid)
X=np.array(feats)
km=KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
per=collections.defaultdict(list)
for o,l in zip(owner,km.labels_): per[o].append(l)
maj={o:collections.Counter(v).most_common(1)[0][0] for o,v in per.items()}
cons=np.mean([collections.Counter(v).most_common(1)[0][1]/len(v) for v in per.values()])
print(f'\ntorso-colour KMeans: split {sorted(collections.Counter(maj.values()).values(), reverse=True)}'
      f'  within-tracklet consistency {cons:.0%}'
      f'  silhouette {silhouette_score(X, km.labels_):.3f}')
for lab in (0,1):
    sel=X[km.labels_==lab]
    hue=(np.degrees(np.arctan2(sel[:,1].mean(), sel[:,0].mean()))%360)
    print(f'  cluster {lab}: n={len(sel):>4}  mean hue {hue:>5.0f}deg  '
          f'sat {sel[:,2].mean():.2f}  val {sel[:,3].mean():.2f}')
