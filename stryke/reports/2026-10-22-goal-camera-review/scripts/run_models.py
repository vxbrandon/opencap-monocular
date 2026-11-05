"""Research-only author-checkpoint comparison. No application model changes."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

R = Path(__file__).resolve().parent
G = R.parent / '2026-10-21-goal-trajectory'
torch.set_num_threads(4)
device = 'mps'
p = argparse.ArgumentParser()
p.add_argument('model', choices=['kicky', 'nbjw', 'pnl', 'pnl_wp'])
p.add_argument('--video-sweep',action='store_true')
args = p.parse_args()
frames = json.loads((R/'frame-pack.json').read_text())
frames.append({'id':'author-messi-sanity','path':str(R/'PnLCalib/examples/messi_sample.png')})
arrays={}
if args.video_sweep:
    frames=[]
    cap=cv2.VideoCapture('/Users/tbg/orca/workspaces/stryke/basslet/apps/fe-mobile/assets/practice/jk-curl-private/shot.mp4')
    images=[]
    while True:
        ok,im=cap.read()
        if not ok:break
        images.append(im)
    cap.release();arrays['jk']=images
    for view in 'AB':arrays[view]=np.load(R.parent/'2026-10-20-multishot-stereo/platform-curling-05'/f'{view}-frames.npy',mmap_mode='r')
    for key,arr in arrays.items():
        for i in range(len(arr)):frames.append({'id':f'{key}-{i:04}','array':key,'frame':i})
started = time.perf_counter()
if args.model == 'kicky':
    from rfdetr import RFDETRSegSmall
    weights = G/'models/kicky-rfdetr-seg.pth'
    with torch.serialization.safe_globals([argparse.Namespace]):
        model = RFDETRSegSmall(pretrain_weights=str(weights), num_classes=3, device=device)
    paths = [weights]
else:
    import yaml
    import torchvision.transforms.functional as F
    repo = G/'no-bells-research' if args.model == 'nbjw' else R/'PnLCalib'
    sys.path.insert(0, str(repo))
    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_lines
    from utils.utils_heatmap import (get_keypoints_from_heatmap_batch_maxpool,
        get_keypoints_from_heatmap_batch_maxpool_l, coords_to_dict, complete_keypoints)
    paths = ([G/'models/MV_kp', G/'models/MV_lines'] if args.model == 'nbjw'
             else [R/'models/PnL_WP_kp', R/'models/PnL_WP_lines'] if args.model == 'pnl_wp'
             else [R/'models/PnL_SV_kp', R/'models/PnL_SV_lines'])
    models = []
    for path, factory, config in zip(paths, [get_cls_net, get_lines], ['hrnetv2_w48.yaml', 'hrnetv2_w48_l.yaml']):
        m = factory(yaml.safe_load((repo/'config'/config).read_text()))
        m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        models.append(m.eval().to(device))
    kp_threshold, line_threshold = (.1486, .388) if args.model == 'nbjw' else (.3434, .7867)
load_s = time.perf_counter()-started
rows=[]
for frame in frames:
    im = arrays[frame['array']][frame['frame']] if 'array' in frame else cv2.imread(frame['path']); h,w = im.shape[:2]
    torch.mps.synchronize(); before=time.perf_counter()
    row={'id':frame['id'], 'image_size':[w,h]}
    if args.model == 'kicky':
        d=model.predict(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), threshold=.25)
        goals=[]
        for i in np.flatnonzero(d.class_id==2):
            contours,_=cv2.findContours(d.mask[i].astype(np.uint8), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            outline=cv2.approxPolyDP(max(contours,key=cv2.contourArea),2,True).reshape(-1,2)
            goals.append({'bbox':d.xyxy[i].tolist(),'confidence':float(d.confidence[i]),'outline':outline.tolist()})
        row['goals']=goals;row['front_corners']=None
    else:
        x=F.resize(F.to_tensor(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))[None],(540,960)).to(device)
        with torch.inference_mode():
            heat=models[0](x); heat_l=models[1](x)
        coords=get_keypoints_from_heatmap_batch_maxpool(heat[:,:-1])
        coords_l=get_keypoints_from_heatmap_batch_maxpool_l(heat_l[:,:-1])
        kp=coords_to_dict(coords,threshold=kp_threshold)[0]
        lines=coords_to_dict(coords_l,threshold=line_threshold)[0]
        raw_kp=json.loads(json.dumps(kp));raw_lines=json.loads(json.dumps(lines))
        try:
            if args.model == 'nbjw':
                completed=complete_keypoints([kp],[lines],w=960,h=540,normalize=False)[0]
            else:
                completed,completed_lines=complete_keypoints(kp,lines,w=960,h=540,normalize=False)
        except Exception as e:
            completed=kp;row['completion_error']=str(e)
        def rescale(points):
            for pt in points.values():
                for key in list(pt):
                    if key in ['x','x_1','x_2']:pt[key]*=w/960
                    elif key in ['y','y_1','y_2']:pt[key]*=h/540
            return points
        row['keypoints']=rescale(raw_kp);row['lines']=rescale(raw_lines)
        row['completed_keypoints']=rescale(completed)
        row['thresholds']={'point':kp_threshold,'line':line_threshold}
    torch.mps.synchronize();row['seconds']=time.perf_counter()-before
    rows.append(row)
    print(args.model,frame['id'],round(row['seconds'],3),len(row.get('keypoints',row.get('goals',[]))),flush=True)
result={'model':args.model,'device':device,'load_s':load_s,'checkpoint_sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
        'timing_scope':'RGB conversion, author resize, network, decode, postprocess and MPS sync; excludes source read and load', 'rows':rows}
(R/f'{args.model}{"-sweep" if args.video_sweep else ""}-results.json').write_text(json.dumps(result,indent=2))
