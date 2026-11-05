"""Build a self-contained, reviewable experiment report from saved measurements."""
import json, shutil, sys
from pathlib import Path
import cv2
import numpy as np
import supervision as sv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from geometry_experiment import GOAL, BALLS, contour, project
from field_line_review import publish_field_lines, training_comparison

R=Path(__file__).resolve().parent
D=Path('/Users/tbg/orca/workspaces/stryke/basslet/docs/reports/2026-10-22-goal-camera-review')
D.mkdir(parents=True,exist_ok=True)
for sub in ['evidence','frames','overlays','scripts']:(D/sub).mkdir(exist_ok=True)
files=['frame-pack.json','references.json','goal-area-references.json','evaluation.json','geometry-initial.json','geometry-lines.json','geometry-auto-jk.json','line-references.json','sensitivity.json']
for tag in ['kicky','nbjw','pnl','pnl_wp']:files.append(f'{tag}-results.json')
for tag in ['kicky','nbjw']:files.append(f'{tag}-sweep-results.json')
for name in files:shutil.copy2(R/name,D/'evidence'/name)
for name in ['run_models.py','evaluate.py','geometry_experiment.py','line_experiment.py','sensitivity.py','field_line_review.py','publish.py']:shutil.copy2(R/name,D/'scripts'/name)
frames=json.loads((R/'frame-pack.json').read_text())
refs=json.loads((R/'references.json').read_text())['corners']
models={tag:json.loads((R/f'{tag}-results.json').read_text()) for tag in ['kicky','nbjw','pnl','pnl_wp']}
evaluation=json.loads((R/'evaluation.json').read_text())
geometry=json.loads((R/'geometry-lines.json').read_text())
auto=json.loads((R/'geometry-auto-jk.json').read_text())
sensitivity=json.loads((R/'sensitivity.json').read_text())
green=sv.Color.from_hex('#28dd8d');gold=sv.Color.from_hex('#ffce55');blue=sv.Color.from_hex('#4cb8ff')

def vertices(im,pts,color,radius=4,closed=False):
    if not len(pts):return im
    kp=sv.KeyPoints(xy=np.array([pts],dtype=np.float32))
    if closed:im=sv.EdgeAnnotator(color=color,thickness=2,edges=[(i+1,(i+1)%len(pts)+1) for i in range(len(pts))]).annotate(im,kp)
    return sv.VertexAnnotator(color=color,radius=radius).annotate(im,kp)

for f in frames:
    name=f['id'];src=cv2.imread(f['path']);shutil.copy2(f['path'],D/'frames'/f'{name}.jpg')
    ref=[p for p in refs[name] if p is not None]
    for tag,data in models.items():
        im=src.copy();row=next(x for x in data['rows'] if x['id']==name)
        if tag=='kicky':
            goals=row['goals']
            if goals:
                masks=[]
                for g in goals:
                    mask=np.zeros(src.shape[:2],np.uint8);cv2.fillPoly(mask,[np.array(g['outline'],np.int32)],1);masks.append(mask.astype(bool))
                det=sv.Detections(xyxy=np.array([g['bbox'] for g in goals]),mask=np.array(masks),class_id=np.zeros(len(goals),int),confidence=np.array([g['confidence'] for g in goals]))
                im=sv.MaskAnnotator(color=gold,opacity=.2).annotate(im,det)
                im=sv.BoxAnnotator(color=gold,thickness=2).annotate(im,det)
        else:
            pts=row['keypoints'];im=vertices(im,[[p['x'],p['y']] for k,p in pts.items() if int(k) in [12,13,14,15,16,17,18,19]],gold)
            for line in row['lines'].values():
                kp=sv.KeyPoints(xy=np.array([[[line['x_1'],line['y_1']],[line['x_2'],line['y_2']]]],np.float32))
                im=sv.EdgeAnnotator(color=gold,thickness=2,edges=[(1,2)]).annotate(im,kp)
        im=vertices(im,ref,green,radius=2,closed=len(ref)==4)
        cv2.imwrite(str(D/'overlays'/f'{tag}-{name}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,88])
        box=json.loads((R/'goal-area-references.json').read_text())['boxes_xyxy'][name]
        x,y,x2,y2=box;pad=20;x=max(0,x-pad);y=max(0,y-pad);x2=min(src.shape[1],x2+pad);y2=min(src.shape[0],y2+pad)
        cv2.imwrite(str(D/'overlays'/f'{tag}-{name}-goal.jpg'),cv2.resize(im[y:y2,x:x2],None,fx=2,fy=2),[cv2.IMWRITE_JPEG_QUALITY,90])

# Data fidelity overlays for all geometry variants. Ground-line fits have their own outputs.
for group in geometry:
    for i,fit in enumerate(group['variants']):
        name=group['frame'];im=cv2.imread(str(R/'frames'/f'{name}.jpg'))
        im=vertices(im,refs[name],green,radius=2,closed=True)
        im=vertices(im,fit['goal_projected'],blue,radius=3,closed=True)
        im=vertices(im,contour(BALLS[name]),gold,radius=1,closed=True)
        for line in fit['extra_lines']:
            kp=sv.KeyPoints(xy=np.array([line['pixels']],np.float32));im=sv.EdgeAnnotator(color=gold,thickness=3,edges=[(1,2)]).annotate(im,kp)
        cv2.imwrite(str(D/'overlays'/f'geometry-{name}-{i}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,91])

def scene(stem,title,fits,show_perturbations=True):
    colors=['#2672d6','#bf3f9c','#9399a8']
    fig=plt.figure(figsize=(15,6.7),layout='constrained');ax=fig.add_subplot(121,projection='3d');top=fig.add_subplot(122)
    interactive=go.Figure()
    def line(pts,color,label,width=2,dash=False):
        p=np.array(pts);ax.plot(p[:,0],p[:,2],p[:,1],color=color,linewidth=width,linestyle='--' if dash else '-',label=label)
        top.plot(p[:,0],p[:,2],color=color,linewidth=width,linestyle='--' if dash else '-')
        interactive.add_trace(go.Scatter3d(x=p[:,0],y=p[:,2],z=p[:,1],mode='lines',line=dict(color=color,width=width*2,dash='dash' if dash else 'solid'),name=label,showlegend=bool(label)))
    line([GOAL[3],GOAL[0],GOAL[1],GOAL[2]],'#153a2b','Goal 7.32 x 2.44 m',4)
    line([[-3.66,0,0],[3.66,0,0]],'#153a2b','',2)
    line([[-9.16,0,0],[-9.16,0,5.5],[9.16,0,5.5],[9.16,0,0]],'#b6bfba','Assumed goal area',1)
    line([[-20.16,0,0],[-20.16,0,16.5],[20.16,0,16.5],[20.16,0,0]],'#87978e','Assumed penalty area',1)
    for i,(label,fit) in enumerate(fits):
        C=np.array(fit['camera_m']);B=np.array(fit['ball_m']);rot=np.array(fit['R_world_to_camera']);K=np.array(fit['K']);color=colors[i]
        ax.scatter(C[0],C[2],C[1],s=75,c=color,marker='^');ax.text(C[0],C[2],C[1]+.4,f'Camera {label}',fontsize=9)
        ax.scatter(B[0],B[2],B[1],s=55,c=color,marker='o');ax.text(B[0]+.4,B[2],B[1]+.25,f'Ball {label}',fontsize=9)
        top.scatter(C[0],C[2],c=color,s=75,marker='^');top.scatter(B[0],B[2],c=color,s=60)
        top.annotate(f'Camera {label}',(C[0],C[2]),xytext=(12,16 if i==0 else -24),textcoords='offset points',fontsize=9,color=color)
        top.annotate(f'Ball {label}',(B[0],B[2]),xytext=(8,-20-i*14),textcoords='offset points',fontsize=9,color=color)
        line([C,B],color,f'Viewing ray {label}',1,True)
        line([C,C+rot.T@np.array([0,0,3.0])],color,'',2)
        im=cv2.imread(str(R/'frames'/f'{fit["frame"]}.jpg'));h,w=im.shape[:2]
        rays=np.c_[[[0,0],[w,0],[w,h],[0,h]],np.ones(4)]@np.linalg.inv(K).T
        far=C+(rays*1.5)@rot
        for corner in far:line([C,corner],color,'',.6)
        line([*far,far[0]],color,'',.6)
        interactive.add_trace(go.Scatter3d(x=[C[0],B[0]],y=[C[2],B[2]],z=[C[1],B[1]],mode='markers+text',marker=dict(size=[7,5],color=color),text=[f'Camera {label}',f'Ball {label}'],textposition='top center',name=label,customdata=[C,B],hovertemplate='%{text}<br>X,Y,Z: %{customdata}<extra></extra>'))
        s=next((s for s in sensitivity['rows'] if s['frame']==fit['frame']),None)
        if s and show_perturbations:
            sample=np.array([x['ball_m'] for x in s['samples']]);top.scatter(sample[:,0],sample[:,2],s=10,c=color,alpha=.2)
    ax.set(xlabel='X across goal (m)',ylabel='Z into field (m)',zlabel='Y height (m)',title='3D view; equal metre scale; ball markers enlarged')
    ax.view_init(elev=23,azim=-48);ax.set_zlim(0,5)
    ax.set_box_aspect((np.diff(ax.get_xlim())[0],np.diff(ax.get_ylim())[0],5))
    top.set(xlabel='X across goal (m)',ylabel='Z into field (m)',title='Top view; dots show input sensitivity');top.axis('equal');top.grid(alpha=.15)
    coordinate_text='X, Y, Z (metres)\n'+'\n'.join(f'{kind} {label}: '+', '.join(f'{v:.1f}' for v in fit[key]) for label,fit in fits for kind,key in [('Camera','camera_m'),('Ball','ball_m')])
    top.text(.98,.98,coordinate_text,transform=top.transAxes,ha='right',va='top',fontsize=8.5,bbox=dict(facecolor='white',alpha=.85,edgecolor='#dddddd',pad=7))
    ax.legend(loc='upper left',fontsize=8);fig.suptitle(title+'\nConditional geometry. Positions are not verified measurements.',fontsize=13)
    fig.savefig(D/f'{stem}.png',dpi=150);plt.close(fig)
    interactive.update_layout(title=title+' — conditional estimate',height=540,margin=dict(l=0,r=0,b=0,t=60),scene=dict(xaxis_title='X across goal (m)',yaxis_title='Z into field (m)',zaxis_title='Y height (m)',aspectmode='data',camera=dict(eye=dict(x=1.3,y=1.4,z=.7))),legend=dict(orientation='h',y=0))
    interactive.write_html(D/f'{stem}.html',include_plotlyjs='directory',config={'displaylogo':False,'displayModeBar':False,'responsive':True})

scene('jk-camera-ball','JK frame 60: goal + stationary ball',[('reviewed',geometry[0]['variants'][0])])
scene('platform-camera-ball','Platform curling 5: independent A/B estimates + candidate penalty line',[('A',geometry[1]['variants'][1]),('B',geometry[2]['variants'][1])])
scene('jk-auto-camera-ball','JK frame 60: automatic goal corners + reviewed ball silhouette',[('auto',auto)],show_perturbations=False)

def source_counts():
    out=[]
    for tag in ['nbjw','kicky']:
        d=json.loads((R/f'{tag}-sweep-results.json').read_text())
        for pre in ['jk','A','B']:
            rows=[r for r in d['rows'] if r['id'].startswith(pre+'-')]
            n=(sum(bool(r['goals']) for r in rows) if tag=='kicky' else sum(any(all(str(i) in r['keypoints'] for i in ids) for ids in [[12,13,16,17],[14,15,18,19]]) for r in rows))
            out.append({'model':tag,'source':pre,'frames':len(rows),'proposal_frames':n,'meaning':'presence of proposals, not accuracy'})
    return out
sweep=source_counts();(D/'evidence/sweep-summary.json').write_text(json.dumps(sweep,indent=2))
initialA=np.array(geometry[1]['variants'][0]['ball_m']);initialB=np.array(geometry[2]['variants'][0]['ball_m'])
lineA=np.array(geometry[1]['variants'][1]['ball_m']);lineB=np.array(geometry[2]['variants'][1]['ball_m'])
gaps={'without_line_m':float(np.linalg.norm(initialA-initialB)),'with_line_m':float(np.linalg.norm(lineA-lineB)),'meaning':'agreement of independently fit rest positions under shared field-size assumptions; not stereo accuracy'}
(D/'evidence/consistency.json').write_text(json.dumps(gaps,indent=2))
table='\n'.join('| '+ ' | '.join(row)+' |' for row in [
 ['Kicky RF-DETR Seg Small','goal + net box','14/14 at IoU ≥ 0.50','no front corners','46 ms'],
 ['No Bells MV','front-corner candidates','JK 16/16; platform 0/38','JK 4/4; platform 0/10','293 ms'],
 ['PnLCalib SV','front-corner candidates','JK 8/16; platform 2/38','0/14','287 ms'],
 ['PnLCalib WorldPose FT','front-corner candidates','0/54','0/14','288 ms'],
])
summary=f'''# Goal detection and camera/ball position — 22 October 2026

We ran four released model configurations on the JK clip and 플랫폼 자료.mov. No STRYKE training or fine-tuning ran. No app model was changed.

1. Kicky finds the goal area in these test images. It includes the net and rear frame. It does not find the four front corners.
2. No Bells Just Whistles is the best tested front-corner model for JK. It passes all four reviewed JK frames. It fails on the ten platform views. There is no tested winner for both sources.
3. We made 3D camera-and-ball scenes from the real images. These show the ball at rest. They do not show measured flight or speed.
4. We tested field lines. A candidate penalty line makes the two platform estimates more consistent. The JK line fit fails other field-corner checks. We do not accept that JK refinement.

## What was run

The scored set has 14 fixed original-raster frames. Four come from JK at indices 30, 60, 90 and 120. Ten are the first frames of five platform A/B pairs: curling 1, 4 and 5; knuckleball 7; driven 4. One platform goal is partly hidden. There are 54 visible reference corners. The references were reviewed on enlarged pixel grids. Their estimated selection uncertainty is about 4 pixels. These are development labels made by the agent. They are not independent ground truth. This is not held-out accuracy, recall or mAP.

Corner hits use an 8-original-pixel tolerance. We also report 16 pixels in the JSON. Matching uses the best one-to-one spatial match among semantic goal-point candidates. It ignores incorrect left/right goal class names. This is a generous diagnostic. A complete pass needs all four visible front corners within tolerance. Whole-goal box overlap is a different task.

| Model | Output | Reviewed image result | Complete front-corner pass | Warm time/frame |
|---|---|---|---|---|
{table}

Kicky's median whole-goal box IoU is 0.892. This is not front-corner accuracy. No Bells' 16-pixel totals are the same as its 8-pixel totals. PnLCalib SV rises from 10/54 to 11/54 at 16 pixels, with no complete frame.

We also ran Kicky and No Bells on every source frame of the 161-frame JK clip and both 176-frame views of platform curling 5. There was no frame skipping: 513 frames per model. Kicky emits a goal-area proposal in 513/513 frames. No Bells emits a complete semantic four-point group in 161/161 JK frames and 0/352 platform frames. These proposal counts are not accuracy scores. Only the 14 reviewed frames have scored labels.

All three point/line checkpoints detect 17–20 points in the author's broadcast example. The adapter passes this input sanity check. None emits a semantic field-line segment at its author threshold in the 14 target frames. Therefore the field-line experiment below uses reviewed line pixels. It is not automatic field-line calibration.

## How the 3D scene is made

The world origin is the centre of the front goal line. X runs across the goal. Y is height. Z runs into the field towards the shooter. Positions use metres.

The goal is assumed to have an inner width of 7.32 m and a height of 2.44 m. The ball has a nominal diameter of 0.22 m. IFAB gives a ball circumference of 68–70 cm. This gives a diameter range of about 0.216–0.223 m. These are assumptions for the recorded objects, not tape measurements. Sources: [IFAB goal and field dimensions](https://www.theifab.com/laws/latest/the-field-of-play/) and [IFAB ball measurements](https://www.theifab.com/laws/latest/the-ball/).

The solver fits the known goal rectangle and a spherical ball silhouette. The ball is stationary and touches a flat ground plane. Its centre height is thus its radius. This extra ground constraint helps estimate the unknown focal length. It must not be used for a flying ball.

The clips have no original camera calibration sidecars. The fit assumes a centred principal point, square pixels and zero lens distortion. It estimates focal length and camera pose. The ball silhouette comes from a reviewed ellipse, not a detector box. Exact perspective sphere tangency is used in the fit. The ball diameter provides distance along the viewing ray.

A homography maps points on one plane. The goal face and the grass are different planes. An airborne ball is on neither plane. Its position needs a 3D camera model and a further distance constraint. A ground homography alone cannot recover flight height.

## Camera and stationary ball estimates

| Scene | Camera X, Y, Z (m) | Ball X, Y, Z (m) | Focal length (px) | Goal fit RMS (px) |
|---|---|---|---|---|
'''
for label,fit in [('JK 60, reviewed corners',geometry[0]['variants'][0]),('JK 60, No Bells corners',auto),('Platform 5 A + candidate line',geometry[1]['variants'][1]),('Platform 5 B + candidate line',geometry[2]['variants'][1])]:
    summary+=f"| {label} | {', '.join(f'{x:.1f}' for x in fit['camera_m'])} | {', '.join(f'{x:.2f}' for x in fit['ball_m'])} | {fit['focal_px']:.0f} | {fit['goal_corner_rms_px']:.2f} |\n"
summary+=f'''
The main JK scene uses reviewed front corners and the reviewed ball silhouette. A separate JK scene uses the raw No Bells goal points with the same ball silhouette. Moving image corners by a few pixels changes the camera position by metres. A small fit residual does not prove metric accuracy.

The platform scene shows two independent view estimates. Their ball positions differ by {gaps['with_line_m']:.2f} m after the candidate line constraint, compared with {gaps['without_line_m']:.2f} m before it. We did not constrain the two balls to coincide. This is a useful consistency result. It is not a measured 3D error. The edited montage does not prove that the views are synchronized captures of the same physical kick.

## Field-line experiment

For JK frame 60, we test the assumed goal-area front at 5.5 m and the penalty-area front at 16.5 m. We fit perpendicular line distance, not invented endpoint correspondences. The goal-plus-ball fit misses the second line by 17.6 px. Adding only the first line leaves a 15.1 px error on that second line. Adding both reduces their fit errors to 1.6 and 0.6 px. However, three excluded junctions still have errors of 90.3, 55.4 and 12.8 px. The mean is 52.8 px. We reject this refinement for the main scene. The goal/field layout assumption or reviewed line identity needs correction. We did not prove which cause is responsible.

For platform A/B, the visible straight line is tested as a candidate penalty-area front. A's line discrepancy drops from 119.6 to 0.1 px, and B's from 3.5 to 0.3 px. These are fitted-line residuals. No independent surveyed platform line or metric camera position is available. The improved A/B agreement supports further testing. It does not establish accurate ball speed.

## Input sensitivity

We ran 60 perturbation scenarios per selected view. We changed goal corners, ball silhouette, line pixels, principal point and regulation ball size. The JSON contains every scenario. The ranges below are the 5th–95th percentiles of these input scenarios. They are not confidence intervals or measured accuracy. Unmodelled distortion and wrong line identity can add more error.

'''
for row in sensitivity['rows']:
    a,b=row['camera_p5_p50_p95_m'][0],row['camera_p5_p50_p95_m'][2];u,v=row['ball_p5_p50_p95_m'][0],row['ball_p5_p50_p95_m'][2]
    summary+=f"- {row['frame']}: camera Z {a[2]:.1f}–{b[2]:.1f} m; ball Z {u[2]:.1f}–{v[2]:.1f} m.\n"
summary+='''
The JK range is wide. Its single-view depth is weakly constrained. The platform ranges depend on the candidate line being the correct standard-size line. Do not use these scene values as validated speed inputs.

## Runtime, cost and reuse

Timing was measured on this Mac M4 Pro with MPS, batch size 1. Point/line models resize to 960×540. Their time includes both networks. Kicky uses its 384-pixel model input. Warm time includes colour conversion, input resize, network work, decoding, postprocessing and device synchronization. It excludes reading the source frame, loading weights and rendering. These small runs are not phone benchmarks. Raw first-frame time, load time and warm p95 are in evaluation.json.

The runs use downloaded local weights. No paid inference API was called. Incremental API cost is $0. Electricity, hardware cost and engineering time were not measured.

Kicky's author model card lists Apache-2.0. The No Bells and PnLCalib repositories use GPL-2.0. This report keeps them as research tools. No code or weights from these repositories were added to the app. No STRYKE fine-tuning ran. The WorldPose model was fine-tuned by its authors, not by us.

## Next implementation step

1. Keep Kicky as a goal-area proposal model. Keep No Bells as the tested JK front-corner baseline.
2. Label front corners in low-angle platform frames. Include the front upright bases. Exclude the rear net frame. Fine-tune and hold out complete takes.
3. Add semantic goal-area and penalty-area lines. Check them with image points excluded from the fit. Resolve the JK layout mismatch first.
4. Use native lens calibration for new phone recordings. Then test camera and ball distance against measured positions. Validate ball speed against a radar or another independent instrument.

## Sources and evidence

- [Kicky author checkpoint and training description](https://huggingface.co/dcrey7/kicky-ai-rfdetr-seg)
- [No Bells Just Whistles author repository](https://github.com/mguti97/No-Bells-Just-Whistles)
- [PnLCalib author repository and released weights](https://github.com/mguti97/PnLCalib)
- [Scored frame details](evidence/evaluation.json), [review labels](evidence/references.json), [all-frame proposal summary](evidence/sweep-summary.json)
- [Camera fits and line checks](evidence/geometry-lines.json), [automatic JK corner fit](evidence/geometry-auto-jk.json), [sensitivity scenarios](evidence/sensitivity.json)
- Model checkpoints remain under /Users/tbg/stryke-data/research. Their SHA-256 values and exact model settings are stored in each model results JSON. The external repository commits and source paths are in provenance.json.
'''
line_section,line_review=publish_field_lines(D)
training_section,training_note=training_comparison(D)
summary+='''
## Clarification: reference points and field-line images

The green corners were selected for BOTH JK and platform by this assistant's image-understanding capability. I inspected enlarged source frames and pixel-coordinate grids, then entered coordinates in JSON. No mouse labeling tool was used. No separate downloaded detector or separate API model generated these labels. No independent human reviewer checked them. These are approximate AI-assisted reference labels, not human ground truth.

The HTML now shows original frames, saved learned point/line outputs, a new automatic OpenCV LSD candidate detector, and AI-selected reference guides as separate images. The new classical detector receives the full image only. It filters line segments using whiteness and adjacent green pixels. It does not receive manual corners, guide lines or a crop. It does not identify which segment is a penalty-area line.

On JK 60, classical candidates cover both selected guide segments at the diagnostic 5px/5-degree tolerance. On platform 5 A/B, coverage is only 17%/26%. These are approximate guide-overlap diagnostics on three development frames, not recall or semantic accuracy. The classical output also includes false positives, such as goal/frame edges. It is not used to replace the selected lines in the existing 3D fits. The learned models' original line results remain unchanged.
'''+training_note
summary=summary.replace('None emits a semantic field-line segment at its author threshold in the 14 target frames.',
  'None emits a painted pitch-line segment at its author threshold in the 14 target frames. PnLCalib SV does emit the goal crossbar (line ID 7) in JK frames 30 and 60; this is a goal-frame segment, not a painted pitch line.')
(D/'SUMMARY.md').write_text(summary)
names={'kicky':'Kicky RF-DETR','nbjw':'No Bells MV','pnl':'PnLCalib SV','pnl_wp':'PnLCalib WorldPose'}
trs=''.join(f"<tr><td>{names[m['totals']['model']]}</td><td>{'14/14 goal boxes' if m['totals']['model']=='kicky' else str(m['totals']['hits_8px'])+'/54 corners'}</td><td>{'not an output' if m['totals']['model']=='kicky' else str(m['totals']['complete_8px'])+'/14'}</td><td>{m['totals']['warm_median_ms']:.0f} ms</td></tr>" for m in evaluation['models'])
html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Goal, camera and ball — STRYKE research</title>
<style>body{font:16px/1.5 system-ui,sans-serif;color:#20352e;background:#f2f5f3;margin:0}main{max-width:1200px;margin:32px auto;padding:0 24px}h1{font-size:34px;line-height:1.15}h2{font-size:23px;margin-top:38px}a{color:#185fa9}section{background:white;border:1px solid #dce5df;border-radius:12px;padding:22px;margin:20px 0}.tag{font-size:13px;letter-spacing:.08em;color:#557268}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #dce5df}iframe{border:0;width:100%;height:570px}img{max-width:100%;display:block}select{padding:9px;font-size:15px;margin:8px 10px 16px 0;max-width:100%}.note{background:#fff5d6;padding:14px;border-left:4px solid #e0b536}.muted{color:#60756a}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:700px){.grid{grid-template-columns:1fr}td,th{font-size:13px;padding:6px}main{padding:0 14px}iframe{height:460px}}</style><main>
<p class="tag">STRYKE • MEASURED MODEL RUNS • 22 OCT 2026</p><h1>Goal detection, camera position and ball position</h1>
<p>Four released model configurations. Real JK and 플랫폼 images. 3D camera-and-ball views from assumed official sizes.</p>
<p class="note"><strong>These scenes show a stationary ball.</strong> They are conditional geometry estimates. We have not verified camera position, flight or ball speed against measured ground truth.</p>
<p><a href="SUMMARY.md">Read the full report</a> · <a href="evidence/evaluation.json">View scores</a> · <a href="provenance.json">View source and model details</a></p>
<section><h2>Goal detector results</h2><table><thead><tr><th>Model</th><th>Development check</th><th>Complete front corners</th><th>Warm time/frame</th></tr></thead><tbody>'''+trs+'''</tbody></table>
<p>14 fixed frames; 54 visible reference corners; 8-original-pixel corner tolerance. Agent-reviewed labels, about 4 px selection uncertainty. This is not held-out accuracy or mAP. Goal-box overlap is a separate task.</p><p>No Bells passes 4/4 JK frames and 0/10 platform views. Kicky finds the goal area in 14/14 frames but does not return front corners. There is no tested front-corner winner for both sources.</p>
<p class="muted">All-frame check: JK 161 frames + platform curling 5 A/B 352 frames, with no skipping. Kicky emits goal areas in 513/513. No Bells emits four-point groups in JK 161/161 and platform 0/352. Proposal presence does not establish accuracy.</p></section>
<section><h2>Inspect the image evidence</h2><p><span style="color:#11854e"><strong>Green: front goal corners selected by hand.</strong></span> These are reference labels, not model predictions. <span style="color:#a27300"><strong>Gold: automatic model output.</strong></span> For Kicky, the gold rectangle and shaded mask cover the goal, net and rear frame. They do not identify the four front corners. The other models show gold corner points. The green perspective shape marks the front goal opening used for calibration.</p>
<select id="model">'''+''.join(f'<option value="{k}">{v}</option>' for k,v in names.items())+'''</select><select id="frame">'''+''.join(f'<option value="{f["id"]}" {"selected" if f["id"]=="jk-0060" else ""}>{f["id"]}</option>' for f in frames)+'''</select>
<a id="full" target="_blank"><img id="overlay" alt="Original goal crop with reviewed corners and model outputs"></a><p><a id="original" target="_blank">Open original frame</a> · Click the crop to open the full image overlay.</p></section>
<section><h2>JK camera and stationary ball</h2><p>The main view uses reviewed front corners and the ball outline. The goal is 7.32 × 2.44 m. The nominal ball diameter is 0.22 m. The solver estimates lens focal length and camera pose. The ball touches flat ground.</p><iframe src="jk-camera-ball.html" title="JK 3D camera and ball view"></iframe><p><a href="jk-camera-ball.png">Open 3D and top-view image</a> · <a href="jk-auto-camera-ball.html">View the fit from automatic No Bells goal points</a> · <a href="overlays/geometry-jk-0060-0.jpg">Check the image fit</a></p><p class="note">JK depth is weakly constrained. Input perturbations move camera Z from about 22 to 64 m. A small image fit error does not prove metre accuracy.</p></section>
<section><h2>Platform camera and stationary ball</h2><p>Each camera and ball position is fitted separately. We add the visible straight line as a candidate penalty-area front line, 16.5 m from the goal. The two ball estimates move from 6.2 m apart to about 0.5 m apart. They are not forced to coincide.</p><iframe src="platform-camera-ball.html" title="Platform A and B independent camera and ball estimates"></iframe><p><a href="platform-camera-ball.png">Open 3D and top-view image</a> · <a href="overlays/geometry-platform-curling-05-A-1.jpg">Check A image fit</a> · <a href="overlays/geometry-platform-curling-05-B-1.jpg">Check B image fit</a></p><p class="note">Agreement is not measured accuracy. The line identity and field dimensions need confirmation. The edited source does not prove synchronized captures of the same kick.</p></section>
<section><h2>What happened with the field lines?</h2><p>The tested point/line models return no accepted semantic line segment in the target frames at their author thresholds. The line experiment therefore uses reviewed image lines.</p><p>For platform, the candidate line improves A/B consistency. For JK, adding the two assumed box lines fits those lines, but excluded corner errors remain 90, 55 and 13 px. That JK refinement is rejected for the main scene.</p><p><a href="evidence/geometry-lines.json">Inspect every variant and excluded-point check</a> · <a href="overlays/geometry-jk-0060-2.jpg">Inspect the rejected JK line fit</a></p><p>The next step is to label low-angle front corners and field lines, fine-tune on these views, then test complete held-out takes. Native camera calibration and measured distance checks are needed before claiming accurate speed.</p></section>
<p class="muted">Local Mac M4 Pro / MPS runs. No paid inference API. No STRYKE fine-tuning. No app model change. Point/line times include both networks. Ball markers are enlarged for visibility.</p>
</main><script>const model=document.querySelector('#model'),frame=document.querySelector('#frame');function update(){const p='overlays/'+model.value+'-'+frame.value;document.querySelector('#overlay').src=p+'-goal.jpg';document.querySelector('#full').href=p+'.jpg';document.querySelector('#original').href='frames/'+frame.value+'.jpg'}model.onchange=frame.onchange=update;update()</script></html>'''
html=html.replace('<section><h2>What happened with the field lines?</h2>',line_section+'<section><h2>What happened with the field lines?</h2>')
html=html.replace('<p class="muted">Local Mac M4 Pro / MPS runs.',training_section+'<p class="muted">Local Mac M4 Pro / MPS runs.')
html=html.replace('Green: front goal corners selected by hand.','Green: front goal corners entered after AI visual inspection.')
html=html.replace('These are reference labels, not model predictions.','These are approximate AI-assisted reference labels. They were entered for both JK and platform. No independent human reviewer verified them.')
html=html.replace('The tested point/line models return no accepted semantic line segment in the target frames at their author thresholds.',
 'The tested point/line models return no painted grass-line segment in the reviewed target frames at their author thresholds. PnLCalib SV does detect the JK crossbar in two frames. That is a goal-frame segment, not a painted pitch line.')
(D/'index.html').write_text(html)
print(D)
