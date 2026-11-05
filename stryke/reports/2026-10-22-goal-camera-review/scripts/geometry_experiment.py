"""Perspective camera + known goal + known spherical ball at rest.

This is a conditional measurement experiment, not an automatic 3D flight system.
World: X across goal; Y up; Z from goal towards shooter; all in metres.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.optimize import least_squares

R=Path(__file__).resolve().parent
GOAL=np.array([[-3.66,2.44,0],[3.66,2.44,0],[3.66,0,0],[-3.66,0,0]],float)
BALLS={
 'jk-0030':{'center':[201,626], 'axes':[71,70]},
 'jk-0060':{'center':[441,608], 'axes':[66,66]},
 'platform-curling-05-A':{'center':[880,588], 'axes':[75,79]},
 'platform-curling-05-B':{'center':[1165,744], 'axes':[75,75]},
}

def unpack(p,w,h,offset=(0,0)):
    f=np.exp(p[0]);C=np.array([p[4],np.exp(p[5]),p[6]])
    rot=cv2.Rodrigues(p[1:4])[0]
    K=np.array([[f,0,w/2+offset[0]],[0,f,h/2+offset[1]],[0,0,1]])
    return K,rot,C

def project(pts,K,rot,C):
    q=(np.asarray(pts)-C)@rot.T
    uv=q@K.T
    return uv[:,:2]/uv[:,2:],q[:,2]

def contour(ball):
    a=np.linspace(0,2*np.pi,16,endpoint=False)
    return np.array(ball['center'])+np.array(ball['axes'])/2*np.array([np.cos(a),np.sin(a)]).T

def sphere_residual(pixel_contour,B,K,rot,C,radius=.11):
    bc=rot@(B-C); f=K[0,0]
    n=np.c_[(pixel_contour-K[:2,2])/f,np.ones(len(pixel_contour))]
    dot=n@bc;ss=bc@bc-radius**2
    val=dot**2-ss*np.sum(n*n,axis=1)
    grad=2*(dot[:,None]*bc[:2]-ss*n[:,:2])/f
    return val/np.maximum(np.linalg.norm(grad,axis=1),1e-8)

def look_at(C,target):
    z=np.array(target)-C;z/=np.linalg.norm(z)
    x=np.cross(z,[0,1,0]);x/=np.linalg.norm(x)
    y=np.cross(z,x)
    return np.array([x,y,z])

def solve(name,extra=None,seed=None,noise=None,radius=.11, quick=False,pp_offset=(0,0),goal_override=None):
    im=cv2.imread(str(R/'frames'/f'{name}.jpg'));h,w=im.shape[:2]
    refs=json.loads((R/'references.json').read_text())['corners']
    goal=np.array(refs[name] if goal_override is None else goal_override,float)
    bc=contour(BALLS[name])
    if noise is not None:goal=goal+noise[0];bc=bc+noise[1]
    extra=extra or []
    def residual(p):
        K,rot,C=unpack(p,w,h,pp_offset)
        pred,depth=project(GOAL,K,rot,C)
        res=list(((pred-goal)/2).ravel())
        if len(p)>7:
            B=np.array([p[7],radius,p[8]])
            res.extend(sphere_residual(bc,B,K,rot,C,radius)/1.5)
        for item in extra:
            pix,d=project(item['world'],K,rot,C)
            # Manually reviewed painted-line points: perpendicular distance only.
            line=np.cross(np.r_[item['pixels'][0],1],np.r_[item['pixels'][1],1])
            res.extend((np.c_[pix,np.ones(len(pix))]@line)/np.linalg.norm(line[:2])/2)
        # Enforce physical visibility without assuming an exact camera height.
        res.extend(np.minimum(depth,0)*100)
        return np.array(res)
    starts=[]
    if seed is not None:starts=[np.array(seed)]
    else:
        for fm in ([.7,1.3,2.3] if not quick else [1.1]):
            f=w*fm; K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]])
            ok,rv,tv=cv2.solvePnP(GOAL,goal,K,None,flags=cv2.SOLVEPNP_ITERATIVE)
            rot=cv2.Rodrigues(rv)[0];C=-(rot.T@tv).ravel()
            for height in ([.2,.8,1.6] if not quick else [.8]):
                C[1]=height
                Bcam=np.linalg.inv(K)@np.r_[BALLS[name]['center'],1.]
                Bcam*=f*(2*radius)/np.mean(BALLS[name]['axes'])
                B=C+rot.T@Bcam
                starts.append(np.r_[np.log(f),rv.ravel(),C[0],np.log(height),max(C[2],3),B[0],max(B[2],1)])
    lo=[np.log(w*.25),-7,-7,-7,-100,np.log(.025),2,-100,-5]
    hi=[np.log(w*5),7,7,7,100,np.log(5),100,100,100]
    fits=[]
    for p in starts:
        p=np.clip(p,np.array(lo)+1e-6,np.array(hi)-1e-6)
        fit=least_squares(residual,p,bounds=(lo,hi),max_nfev=1600,loss='linear',xtol=1e-10,ftol=1e-10,gtol=1e-10)
        fits.append(fit)
    fit=min(fits,key=lambda x:np.sum(x.fun*x.fun))
    K,rot,C=unpack(fit.x,w,h,pp_offset);B=np.array([fit.x[7],radius,fit.x[8]])
    pred,depth=project(GOAL,K,rot,C)
    return {'frame':name,'K':K.tolist(),'R_world_to_camera':rot.tolist(),'camera_m':C.tolist(),'ball_m':B.tolist(),
       'ball_camera_distance_m':float(np.linalg.norm(B-C)), 'focal_px':float(K[0,0]),
       'goal_corner_rms_px':float(np.sqrt(np.mean(np.sum((pred-goal)**2,axis=1)))),
       'sphere_outline_rms_px':float(np.sqrt(np.mean(sphere_residual(bc,B,K,rot,C,radius)**2))),
       'goal_projected':pred.tolist(),'weighted_sum_squared':float(fit.fun@fit.fun),'parameters':fit.x.tolist(),
       'optimizer_success':bool(fit.success),'at_bound':bool(np.any(fit.active_mask)),
       'extra_lines':extra,'assumptions':['regulation inner goal dimensions 7.32 x 2.44 m','nominal ball diameter 0.22 m','stationary ball touches flat ground','square pixels; centred principal point; zero lens distortion','manual front corners and ball silhouette; no metric ground truth']}

if __name__=='__main__':
    results=[]
    for name in BALLS:
        result=solve(name);results.append(result)
        print(name,'camera',result['camera_m'],'ball',result['ball_m'],'f',result['focal_px'],'goalRMS',result['goal_corner_rms_px'],'ballRMS',result['sphere_outline_rms_px'],flush=True)
    (R/'geometry-initial.json').write_text(json.dumps(results,indent=2))
