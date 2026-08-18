"""Precision/recall curves — is global assignment actually better than greedy?

The first comparison was unfair. Greedy made 53 links and global 97, so global
looked worse on precision simply by being less conservative. Both methods have
a knob that trades recall for precision (greedy: ambiguity_ratio; global: the
no-link price), so the honest question is whether global's curve sits ABOVE
greedy's — better precision at the same recall — not how each does at whatever
default it happened to ship with.
"""
import sys, pickle, collections
import numpy as np
sys.path.insert(0,'/Users/sean/Documents/Home/soccer_poc/sports-main/examples/soccer')
import stitch_tracks as st

d=pickle.load(open(sys.argv[1],'rb')); fps=d['fps']; N=d['n_frames']
BASE=[]
for tid,rec in d['tracks'].items():
    fr=rec['frames']; xy=np.array(rec['xy'],dtype=np.float32)
    if len(fr)<2: continue
    breaks=[0]+[i for i in range(1,len(fr)) if fr[i]-fr[i-1]>fps]+[len(fr)]
    for a,b in zip(breaks[:-1],breaks[1:]):
        if b-a>=2:
            BASE.append(st.Tracklet(tid, fr[a:b], xy[a:b],
                        collections.Counter(rec['cls'][a:b]).most_common(1)[0][0]))

def make_cuts(gap_s):
    gap=int(gap_s*fps); pool=[]; truth={}; k=0
    for t in BASE:
        if t.duration < gap + int(2.0*fps): pool.append(t); continue
        mid=len(t.frames)//2; lo,hi = mid-gap//2, mid+gap//2
        if lo<int(0.8*fps) or len(t.frames)-hi<int(0.8*fps): pool.append(t); continue
        pool += [st.Tracklet(10000+k, t.frames[:lo], t.xy[:lo], t.cls),
                 st.Tracklet(20000+k, t.frames[hi:], t.xy[hi:], t.cls)]
        truth[10000+k]=20000+k; k+=1
    return pool, truth

def score(made, truth):
    hit=sum(1 for a,b in truth.items() if made.get(a)==b)
    fc=[a for a in made if a in truth]
    return (hit/len(truth) if truth else 0.0,
            hit/len(fc) if fc else float('nan'), len(fc))

GAP=1.0
pool, truth = make_cuts(GAP)
cost = st.build_cost_matrix(pool, fps)
print(f'gap {GAP}s, {len(truth)} known pairs among {len(pool)} tracklets\n')

print(f'{"GLOBAL":<12}{"no_link":>9}{"links":>7}{"recall":>9}{"precision":>11}')
best=[]
for nl in (0.10,0.15,0.20,0.25,0.30,0.40,0.55,0.70,0.90):
    ident, links = st.stitch_global(pool, fps, nl, cost=cost)
    made={l['from']:l['to'] for l in links}
    r,p,n = score(made, truth)
    best.append((r,p,nl))
    print(f'{"":<12}{nl:>9.2f}{len(links):>7}{r:>9.0%}{p:>11.0%}')

print(f'\n{"GREEDY":<12}{"ambig":>9}{"links":>7}{"recall":>9}{"precision":>11}')
for ar in (0.50,0.65,0.75,0.85,0.95,1.00):
    gident,_ = st.stitch_greedy(pool, fps, ar)
    made={}
    for ch in gident:
        for x,y in zip(ch[:-1],ch[1:]): made[pool[x].id]=pool[y].id
    r,p,n = score(made, truth)
    nl=sum(len(c)-1 for c in gident)
    print(f'{"":<12}{ar:>9.2f}{nl:>7}{r:>9.0%}{p:>11.0%}')

# Thin-margin flagging: does it actually identify the wrong links?
print()
ident, links = st.stitch_global(pool, fps, 0.55, cost=cost)
for group in (True, False):
    sel=[l for l in links if l['thin']==group and l['from'] in truth]
    ok=sum(1 for l in sel if truth[l['from']]==l['to'])
    tag='thin (flagged)' if group else 'confident'
    if sel:
        print(f'{tag:<18}{len(sel):>4} links from cut pieces, {ok/len(sel):>4.0%} correct')
