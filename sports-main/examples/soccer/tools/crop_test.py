"""Is the moving camera a CROP of the stationary one, or a separate camera?

This decides how hard ball fusion is:
  crop/pan of the same source -> ONE homography maps its ball onto the
      stationary view, needing no pitch landmarks. Easy.
  genuinely separate camera   -> two transforms through pitch space, which is
      blocked because the pitch model finds zero landmarks here.

Redoing a test I ran earlier and over-read. That one used
estimateAffinePartial2D — a 4-DOF similarity (scale, rotation, translation) —
and I reported it as ruling out a relationship. It does not: a different
viewpoint of a plane is a full 8-DOF homography, which a similarity cannot
represent even when the two views are perfectly related.

So: SIFT matches, then a full homography with RANSAC, judged on inlier count
and reprojection error rather than on whether a similarity happened to fit.
"""
import warnings, sys
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/sean/Documents/Home/soccer_poc/sports-main/examples/soccer')
import cv2, supervision as sv, main

ST='data/base_datasets/14_09/Stationary_Camera_14_08.mp4'
MV='data/base_datasets/14_09/Moving_Camera_14_08.mp4'
si=sv.VideoInfo.from_video_path(ST); mi=sv.VideoInfo.from_video_path(MV)
print(f'stationary {si.width}x{si.height} @ {si.fps:.2f}fps, {si.total_frames} frames')
print(f'moving     {mi.width}x{mi.height} @ {mi.fps:.2f}fps, {mi.total_frames} frames')

def grab(path, sec, fps):
    for f in main.video_frames(path, start_frame=int(sec*fps)):
        return f
    return None

sift=cv2.SIFT_create(nfeatures=6000)
bf=cv2.BFMatcher()
print(f'\n{"t(s)":>5}{"kp stat":>9}{"kp mov":>8}{"matches":>9}'
      f'{"inliers":>9}{"inlier%":>9}{"reproj px":>11}')
for sec in (480, 500, 520, 540):
    a=grab(ST, sec, si.fps); b=grab(MV, sec, mi.fps)
    if a is None or b is None:
        print(f'{sec-477:>5}  could not read'); continue
    ga=cv2.cvtColor(a, cv2.COLOR_BGR2GRAY); gb=cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ka,da=sift.detectAndCompute(ga,None); kb,db=sift.detectAndCompute(gb,None)
    if da is None or db is None or len(ka)<10 or len(kb)<10:
        print(f'{sec-477:>5}  too few keypoints'); continue
    # Lowe ratio test
    good=[m for m,n in bf.knnMatch(db, da, k=2) if m.distance < 0.75*n.distance]
    if len(good)<12:
        print(f'{sec-477:>5}{len(ka):>9}{len(kb):>8}{len(good):>9}   too few matches')
        continue
    src=np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst=np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
    if H is None:
        print(f'{sec-477:>5}{len(ka):>9}{len(kb):>8}{len(good):>9}   no homography')
        continue
    inl=mask.ravel().astype(bool)
    proj=cv2.perspectiveTransform(src[inl], H)
    err=np.linalg.norm(proj-dst[inl], axis=2).ravel()
    print(f'{sec-477:>5}{len(ka):>9}{len(kb):>8}{len(good):>9}'
          f'{int(inl.sum()):>9}{inl.mean():>8.0%}{np.median(err):>11.2f}', flush=True)
    if sec==500:
        np.save(f'{sys.argv[1]}/H_moving_to_stationary.npy', H)
        h,w=a.shape[:2]
        warp=cv2.warpPerspective(b, H, (w,h))
        blend=cv2.addWeighted(a,0.5,warp,0.5,0)
        cv2.imwrite(f'{sys.argv[1]}/crop_overlay.png',
                    cv2.resize(blend,None,fx=0.42,fy=0.42))
        print('       -> crop_overlay.png (moving warped onto stationary, 50/50)')
