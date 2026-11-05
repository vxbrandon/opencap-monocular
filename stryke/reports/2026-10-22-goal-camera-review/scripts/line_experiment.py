import json
from pathlib import Path
import cv2
import numpy as np
from geometry_experiment import solve, project, BALLS

R=Path(__file__).resolve().parent
LINES={
 'jk-0060':[
   {'name':'goal-area front, assumed 5.5 m','world':[[-9.16,0,5.5],[9.16,0,5.5]],'pixels':[[72,253],[842,216]]},
   {'name':'penalty-area front, assumed 16.5 m','world':[[-20.16,0,16.5],[20.16,0,16.5]],'pixels':[[140,314],[794,273]]},
 ],
 'platform-curling-05-A':[
   {'name':'candidate penalty-area front; identity not surveyed','world':[[-20.16,0,16.5],[20.16,0,16.5]],'pixels':[[70,548],[1800,620]]},
 ],
 'platform-curling-05-B':[
   {'name':'candidate penalty-area front; identity not surveyed','world':[[-20.16,0,16.5],[20.16,0,16.5]],'pixels':[[70,747],[1800,625]]},
 ]}

HOLDOUT={
 'jk-0060':[
   {'name':'left goal-area corner at goal line','world':[-9.16,0,0],'pixel':[127,232]},
   {'name':'left goal-area front corner','world':[-9.16,0,5.5],'pixel':[72,253]},
   {'name':'left penalty arc intersection','world':[-7.3125,0,16.5],'pixel':[118,315]},
 ]}

def check(fit,name):
    K=np.array(fit['K']);rot=np.array(fit['R_world_to_camera']);C=np.array(fit['camera_m'])
    hold=[]
    for p in HOLDOUT.get(name,[]):
        pred,_=project([p['world']],K,rot,C)
        hold.append({**p,'predicted':pred[0].tolist(),'error_px':float(np.linalg.norm(pred[0]-p['pixel']))})
    checks=[]
    for line in LINES[name]:
        uv,_=project(line['world'],K,rot,C);a=np.cross(np.r_[line['pixels'][0],1],np.r_[line['pixels'][1],1]);e=np.c_[uv,np.ones(len(uv))]@a/np.linalg.norm(a[:2])
        checks.append({'name':line['name'],'rms_px':float(np.sqrt(np.mean(e*e)))})
    fit['line_checks']=checks;fit['holdout_points']=hold
    return fit

out=[]
for name in LINES:
    base=solve(name);check(base,name);base['variant']='goal + ball'
    one=solve(name,extra=LINES[name][:1]);check(one,name);one['variant']='goal + ball + first field line'
    variants=[base,one]
    if len(LINES[name])>1:
        both=solve(name,extra=LINES[name]);check(both,name);both['variant']='goal + ball + both field lines';variants.append(both)
    out.append({'frame':name,'variants':variants})
    for f in variants:
        print(name,f['variant'],'camera',np.round(f['camera_m'],2),'ball',np.round(f['ball_m'],2),'f',round(f['focal_px']),'goal',round(f['goal_corner_rms_px'],2),'checks',f['line_checks'],'holdout',[round(p['error_px'],1) for p in f['holdout_points']],flush=True)
(R/'geometry-lines.json').write_text(json.dumps(out,indent=2))
(R/'line-references.json').write_text(json.dumps({'lines':LINES,'holdout_points':HOLDOUT,'source':'visually selected painted lines, no automatic semantic line acceptance'},indent=2))
