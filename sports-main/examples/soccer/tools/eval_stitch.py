"""Measure whether stitched links are CORRECT, not just how many were made.

Identity count alone cannot judge a stitcher: linking everything to everything
lands on ~22 identities and is worthless. We need ground truth, and we have no
hand-labelled player identities.

So manufacture it. Take real tracklets, cut each long one in two and delete a
gap in the middle. The true answer is known by construction — piece A continues
into piece B — while every other tracklet stays in the pool as a distractor, so
the stitcher faces the same crowded field it faces for real.

  recall    of the known pairs, how many were rejoined
  precision of the links made FROM a cut piece, how many went to the right one

Precision is the number that matters. A wrong link welds two players into one
identity and corrupts every statistic derived from it — worse than leaving them
apart, because a fragment is visibly incomplete while a bad merge looks fine.
"""
import sys, pickle, collections
import numpy as np
sys.path.insert(0,'/Users/sean/Documents/Home/soccer_poc/sports-main/examples/soccer')
import stitch_tracks as st

d=pickle.load(open(sys.argv[1],'rb'))
fps=d['fps']; N=d['n_frames']

def base_tracklets():
    out=[]
    for tid,rec in d['tracks'].items():
        fr=rec['frames']; xy=np.array(rec['xy'],dtype=np.float32)
        if len(fr)<2: continue
        breaks=[0]+[i for i in range(1,len(fr)) if fr[i]-fr[i-1]>fps]+[len(fr)]
        for a,b in zip(breaks[:-1],breaks[1:]):
            if b-a>=2:
                cls=collections.Counter(rec['cls'][a:b]).most_common(1)[0][0]
                out.append(st.Tracklet(tid, fr[a:b], xy[a:b], cls))
    return out

BASE=base_tracklets()
print(f'{len(BASE)} tracklets from {N} frames (~{N/fps:.0f}s), '
      f'median duration {np.median([t.duration for t in BASE])/fps:.1f}s')

def make_cuts(gap_s):
    """Cut long tracklets in half with a gap; return pool + truth map."""
    gap=int(gap_s*fps)
    pool=[]; truth={}; k=0
    for t in BASE:
        # need enough either side of the gap to extrapolate from
        if t.duration < gap + int(2.0*fps):
            pool.append(t); continue
        mid=len(t.frames)//2
        lo, hi = mid-gap//2, mid+gap//2
        if lo<int(0.8*fps) or len(t.frames)-hi<int(0.8*fps):
            pool.append(t); continue
        a=st.Tracklet(10000+k, t.frames[:lo], t.xy[:lo], t.cls)
        b=st.Tracklet(20000+k, t.frames[hi:], t.xy[hi:], t.cls)
        pool += [a,b]; truth[10000+k]=20000+k; k+=1
    return pool, truth

def score(links, truth):
    made={l['from']:l['to'] for l in links}
    hit=sum(1 for a,b in truth.items() if made.get(a)==b)
    from_cut=[a for a in made if a in truth]
    wrong=[a for a in from_cut if made[a]!=truth[a]]
    rec = hit/len(truth) if truth else 0.0
    prec= hit/len(from_cut) if from_cut else float('nan')
    return rec, prec, len(truth), len(wrong)

print(f'\n{"gap":>5}{"pairs":>7}  {"method":<22}{"recall":>8}{"precision":>11}{"wrong":>7}')
for gap_s in (0.5, 1.0, 2.0, 3.0):
    pool, truth = make_cuts(gap_s)
    if not truth:
        print(f'{gap_s:>5.1f}      0  (no tracklet long enough)'); continue

    ident,links = st.stitch_global(pool, fps)
    r,p,n,w = score(links, truth)
    print(f'{gap_s:>5.1f}{n:>7}  {"global":<22}{r:>8.0%}{p:>11.0%}{w:>7}')

    gident,_ = st.stitch_greedy(pool, fps)
    glinks=[]
    for ch in gident:
        for x,y in zip(ch[:-1],ch[1:]):
            glinks.append({'from':pool[x].id,'to':pool[y].id})
    r,p,n,w = score(glinks, truth)
    print(f'{"":>5}{"":>7}  {"greedy (old)":<22}{r:>8.0%}{p:>11.0%}{w:>7}', flush=True)

# What it does on the real fragments, with no cuts
print()
for label, fn in (('global @ default', lambda: st.stitch_global(BASE, fps)),
                  ('global -> roster 22', lambda: st.stitch_to_roster(BASE, fps, 22)[:2]),
                  ('greedy (old)', lambda: st.stitch_greedy(BASE, fps))):
    ident, links = fn()
    spans=[(BASE[ch[-1]].end-BASE[ch[0]].start+1)/N for ch in ident]
    spans=np.array(spans)
    nlinks = len(links) if isinstance(links, list) and links and isinstance(links[0], dict) \
             else sum(len(c)-1 for c in ident)
    thin = sum(1 for l in links if isinstance(l, dict) and l['thin'])
    print(f'{label:<22} identities {len(ident):>4}  links {nlinks:>4}  '
          f'thin {thin:>4}  span>=50% {int((spans>=0.5).sum()):>3}  '
          f'span>=75% {int((spans>=0.75).sum()):>3}')
