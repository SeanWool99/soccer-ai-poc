"""Does the colour team term actually improve stitching, or just look sensible?

Two questions, and the second is the one that matters:
  1. does the CIELAB torso classifier separate these navy/sky-blue kits, and how
     consistent is it within a tracklet (free ground truth — same player)
  2. does adding it as a link-cost penalty improve link PRECISION on the
     manufactured-cut benchmark, or merely change the number of links
"""
import warnings, sys, collections
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/sean/Documents/Home/soccer_poc/sports-main/examples/soccer')
import cv2, supervision as sv, rfdetr_onnx as rf, main, stitch_tracks as st
from team_colour import TeamColourClassifier

ST='data/base_datasets/14_09/Stationary_Camera_14_08.mp4'
info=sv.VideoInfo.from_video_path(ST); fps=info.fps
main.PITCH_POLYGON=np.load('data/pitch_polygon_14_09.npy').astype(np.int32)
main.FAR_TOUCHLINE=None
N=1800

t=main.PlayerReIDTracker(info.width,info.height,fps,'mps')
tracks=collections.defaultdict(lambda:{'frames':[],'xy':[],'cls':[]})
crops=collections.defaultdict(list)
n=0
for f in main.video_frames(ST, start_frame=int(477*fps)):
    d=rf.detect(f, conf=0.20)
    d=main.clean_detections(d, info.width, info.height)
    if len(d) and d.class_id is not None:
        d=d[np.isin(d.class_id,[1,2])]
    out=t.update(d, None)
    if out.tracker_id is not None and len(out):
        anch=out.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        for tid,xy,cl,box in zip(out.tracker_id, anch, out.class_id, out.xyxy):
            tid=int(tid)
            tracks[tid]['frames'].append(n); tracks[tid]['cls'].append(int(cl))
            tracks[tid]['xy'].append([float(xy[0]),float(xy[1])])
            if len(crops[tid])<10 and n%8==0:
                x1,y1,x2,y2=[int(v) for v in box]
                x1,y1=max(0,x1),max(0,y1)
                x2,y2=min(info.width,x2),min(info.height,y2)
                if x2-x1>=8 and y2-y1>=16: crops[tid].append(f[y1:y2,x1:x2].copy())
    n+=1
    if n>=N: break

allc=[c for v in crops.values() for c in v]
clf=TeamColourClassifier().fit(allc)
print(f'{len(crops)} ids, {len(allc)} crops')
print(f'separation {clf.separation:.2f}  (>=1.0 usable)')
print(f'cluster centres (CIELAB L,a,b): '
      f'{np.round(clf.centres[0],1)} vs {np.round(clf.centres[1],1)}')

# per-crop vs per-tracklet consistency
per=collections.defaultdict(list)
for tid,v in crops.items():
    for c in v:
        p=clf.predict_crop(c)
        if p is not None: per[tid].append(p)
cons=[collections.Counter(v).most_common(1)[0][1]/len(v) for v in per.values() if v]
teams={tid: clf.predict_tracklet(v) for tid,v in crops.items()}
teams={k:v for k,v in teams.items() if v is not None}
print(f'per-crop consistency within a tracklet: {np.mean(cons):.0%}')
print(f'team split across ids: {dict(collections.Counter(teams.values()))}')

# --- does it help linking? manufactured cuts, with and without the penalty ---
BASE=[]
for tid,rec in tracks.items():
    fr=rec['frames']; xy=np.array(rec['xy'],dtype=np.float32)
    if len(fr)<2: continue
    breaks=[0]+[i for i in range(1,len(fr)) if fr[i]-fr[i-1]>fps]+[len(fr)]
    for a,b in zip(breaks[:-1],breaks[1:]):
        if b-a>=2:
            BASE.append((tid, fr[a:b], xy[a:b],
                         collections.Counter(rec['cls'][a:b]).most_common(1)[0][0]))

def build(gap_s, with_team):
    gap=int(gap_s*fps); pool=[]; truth={}; k=0
    for tid,fr,xy,cls in BASE:
        tm = teams.get(tid) if with_team else None
        dur=fr[-1]-fr[0]+1
        if dur < gap+int(2.0*fps):
            pool.append(st.Tracklet(tid,fr,xy,cls,team=tm)); continue
        mid=len(fr)//2; lo,hi=mid-gap//2, mid+gap//2
        if lo<int(0.8*fps) or len(fr)-hi<int(0.8*fps):
            pool.append(st.Tracklet(tid,fr,xy,cls,team=tm)); continue
        pool += [st.Tracklet(10000+k,fr[:lo],xy[:lo],cls,team=tm),
                 st.Tracklet(20000+k,fr[hi:],xy[hi:],cls,team=tm)]
        truth[10000+k]=20000+k; k+=1
    return pool,truth

print(f'\n{"gap":>5}{"team term":>11}{"links":>7}{"recall":>9}{"precision":>11}')
for gap_s in (1.0, 2.0):
    for wt in (False, True):
        pool,truth = build(gap_s, wt)
        _,links = st.stitch_global(pool, fps)
        made={l['from']:l['to'] for l in links}
        hit=sum(1 for a,b in truth.items() if made.get(a)==b)
        fc=[a for a in made if a in truth]
        r=hit/len(truth) if truth else 0
        p=hit/len(fc) if fc else float('nan')
        print(f'{gap_s:>5.1f}{"yes" if wt else "no":>11}{len(links):>7}'
              f'{r:>9.0%}{p:>11.0%}', flush=True)
