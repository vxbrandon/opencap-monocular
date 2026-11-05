from pathlib import Path
import json,cv2
import numpy as np
from geometry_experiment import solve,contour,BALLS

R=Path(__file__).resolve().parent
geometry=json.loads((R/'geometry-lines.json').read_text())
rng=np.random.default_rng(20261022)
rows=[]
for group in geometry:
    name=group['frame']
    # JK line fit fails independent corner checks, so use goal + ball there.
    selected=group['variants'][0] if name.startswith('jk') else group['variants'][1]
    im=cv2.imread(str(R/'frames'/f'{name}.jpg'));h,w=im.shape[:2]
    samples=[]
    a=np.linspace(0,2*np.pi,16,endpoint=False)
    for i in range(60):
        # Sensitivity scenarios, NOT a calibrated probability/accuracy interval.
        corner_noise=rng.normal(0,2,(4,2))
        ctr=rng.normal(0,1,2);axes=rng.normal(0,1.5,2)
        ball_noise=ctr+axes/2*np.array([np.cos(a),np.sin(a)]).T
        diameter=rng.uniform(.68/np.pi,.70/np.pi)
        offset=rng.uniform(-.02,.02,2)*[w,h]
        extra=json.loads(json.dumps(selected['extra_lines']))
        for line in extra:line['pixels']=(np.array(line['pixels'])+rng.normal(0,2,(2,2))).tolist()
        fit=solve(name,extra=extra,seed=selected['parameters'],noise=(corner_noise,ball_noise),radius=diameter/2,pp_offset=offset)
        samples.append({k:fit[k] for k in ['camera_m','ball_m','focal_px','at_bound','goal_corner_rms_px']})
    q=lambda key:np.percentile([s[key] for s in samples],[5,50,95],axis=0).tolist()
    row={'frame':name,'variant':selected['variant'],'scenario_count':len(samples),'camera_p5_p50_p95_m':q('camera_m'),'ball_p5_p50_p95_m':q('ball_m'),'focal_p5_p50_p95_px':q('focal_px'),'samples_at_bound':sum(s['at_bound'] for s in samples),'samples':samples}
    rows.append(row);print(name,row['camera_p5_p50_p95_m'],row['ball_p5_p50_p95_m'],flush=True)
(R/'sensitivity.json').write_text(json.dumps({'meaning':'Input perturbation sensitivity only. Not confidence intervals or measured 3D accuracy. 60 scenarios per view, goal corners sigma 2 px, ball centre sigma 1 px, full axes sigma 1.5 px, line endpoint sigma 2 px, principal point +/-2% image dimensions, regulation ball circumference 68–70 cm. Zero distortion and assumed line identities remain fixed.','rows':rows},indent=2))
