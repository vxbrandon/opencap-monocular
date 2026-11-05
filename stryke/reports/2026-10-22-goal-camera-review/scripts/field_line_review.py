"""Show learned field-line outputs, classical candidates and reviewed guides separately."""
from pathlib import Path
import json,shutil,time
import cv2
import numpy as np
import supervision as sv

R=Path(__file__).resolve().parent
CASES=['jk-0060','platform-curling-05-A','platform-curling-05-B']
NAMES={'nbjw':'No Bells MV','pnl':'PnLCalib SV','pnl_wp':'PnLCalib WorldPose'}
GOLD=sv.Color.from_hex('#ffce55');GREEN=sv.Color.from_hex('#28dd8d');PURPLE=sv.Color.from_hex('#df70ff')

def segments(im,rows,color,width=2):
    for pair in rows:
        keypoints=sv.KeyPoints(xy=np.array([pair],np.float32))
        im=sv.EdgeAnnotator(color=color,thickness=width,edges=[(1,2)]).annotate(im,keypoints)
    return im

def classical_candidates(im):
    """Full-frame LSD, filtered by white paint and adjacent green pixels.

    No goal corners, hand-picked ROI, reference lines or line identities are inputs.
    These are unlabelled segment candidates, not a learned semantic field detector.
    """
    start=time.perf_counter();h,w=im.shape[:2]
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    green=cv2.inRange(hsv,(23,40,25),(95,255,255))>0
    white=cv2.inRange(hsv,(0,0,140),(179,90,255))>0
    found=cv2.createLineSegmentDetector().detect(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
    raw=found[0] if isinstance(found,tuple) else found
    raw=np.array(raw).reshape(-1,4) if raw is not None else np.empty((0,4))
    accepted=[]
    for a in raw:
        p=a[:2];q=a[2:];delta=q-p;length=float(np.linalg.norm(delta))
        if length<max(35,w*.025):continue
        normal=np.array([-delta[1],delta[0]])/length
        sample=p+(q-p)*np.linspace(.08,.92,32)[:,None]
        support=[]
        for offset in [-7,-4,4,7]:
            xy=np.round(sample+offset*normal).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1)
            support.append(green[xy[:,1],xy[:,0]])
        white_support=[]
        for offset in [-2,0,2]:
            xy=np.round(sample+offset*normal).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1)
            white_support.append(white[xy[:,1],xy[:,0]])
        green_fraction=float(np.mean(support));white_fraction=float(np.mean(np.any(white_support,axis=0)))
        if green_fraction>=.55 and white_fraction>=.45:
            accepted.append({'pixels':[p.tolist(),q.tolist()],'length_px':length,'adjacent_green_fraction':green_fraction,'white_fraction':white_fraction})
    return {'seconds':time.perf_counter()-start,'raw_lsd_segments':len(raw),'candidates':accepted}

def diagnostic_coverage(reference,candidates):
    a,b=np.array(reference,float);vec=b-a;length=np.linalg.norm(vec);points=a+np.linspace(0,1,200)[:,None]*vec
    best=np.full(len(points),np.inf)
    for c in candidates:
        p,q=np.array(c['pixels']);v=q-p
        cosine=abs(np.dot(vec,v)/(length*np.linalg.norm(v)))
        if cosine<np.cos(np.deg2rad(5)):continue
        t=np.clip(((points-p)@v)/(v@v),0,1);nearest=p+t[:,None]*v
        best=np.minimum(best,np.linalg.norm(points-nearest,axis=1))
    return float(np.mean(best<=5))

def publish_field_lines(D):
    out=D/'field-lines';out.mkdir(exist_ok=True)
    refs=json.loads((R/'line-references.json').read_text())
    corners=json.loads((R/'references.json').read_text())['corners']
    models={m:json.loads((R/f'{m}-results.json').read_text()) for m in NAMES}
    result={'scope':'Three existing original-raster development frames. Learned outputs use saved author-threshold results. No new learned model or fine-tuning ran.',
      'reference_method':'AI visual inspection of enlarged pixel grids, followed by entering coordinates. No mouse labeling tool and no independent human annotation. References are approximate development labels.',
      'classical_method':'OpenCV Line Segment Detector on full grayscale image, length >= max(35px,2.5% width); >=55% adjacent-green support; >=45% white support. No manual ROI or reference input. Candidate line identities remain unknown.',
      'coverage_definition':'Fraction of 200 samples on an AI-selected guide within 5 original pixels of a classical candidate with orientation within 5 degrees. Tunable development diagnostic, not recall or semantic accuracy.', 'frames':{}}
    for name in CASES:
        im=cv2.imread(str(R/'frames'/f'{name}.jpg'))
        cv2.imwrite(str(out/f'{name}-original.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,91])
        row={'models':{},'manual_corners':corners[name],'guides':refs['lines'][name]}
        for tag,data in models.items():
            saved=next(x for x in data['rows'] if x['id']==name);image=im.copy()
            points=saved['keypoints']
            if points:
                kp=sv.KeyPoints(xy=np.array([[[p['x'],p['y']] for p in points.values()]],np.float32))
                image=sv.VertexAnnotator(color=GOLD,radius=5).annotate(image,kp)
            lines=[[[p['x_1'],p['y_1']],[p['x_2'],p['y_2']]] for p in saved['lines'].values()]
            image=segments(image,lines,GOLD,3)
            cv2.imwrite(str(out/f'{tag}-{name}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,91])
            row['models'][tag]={'points':len(points),'line_segments':len(lines),
              'painted_field_line_segments':sum(int(k) not in range(7,13) for k in saved['lines']),
              'goal_frame_line_segments':sum(int(k) in range(7,13) for k in saved['lines']),
              'thresholds':saved['thresholds']}
        classic=classical_candidates(im);row['classical']=classic
        candidates=segments(im.copy(),[c['pixels'] for c in classic['candidates']],PURPLE,3)
        cv2.imwrite(str(out/f'classical-{name}.jpg'),candidates,[cv2.IMWRITE_JPEG_QUALITY,91])
        guides=segments(im.copy(),[g['pixels'] for g in row['guides']],GREEN,3)
        cv2.imwrite(str(out/f'guides-{name}.jpg'),guides,[cv2.IMWRITE_JPEG_QUALITY,91])
        for guide in row['guides']:guide['classical_coverage_5px']=diagnostic_coverage(guide['pixels'],classic['candidates'])
        grid=R/f'{name}-junctions.png'
        if grid.exists():shutil.copy2(grid,out/f'{name}-corner-pixel-grid.png')
        result['frames'][name]=row
    (out/'results.json').write_text(json.dumps(result,indent=2))
    section='''<section id="field-line-images"><h2>Can the field lines be detected?</h2>
<p>Inspect the actual full-frame outputs below. <strong>Gold is the learned model output. Purple is an OpenCV line candidate. Green is an AI-selected reference guide.</strong> A line candidate has no penalty-area or goal-area identity.</p>
<label for="line-frame">Video frame </label><select id="line-frame">'''+''.join(f'<option value="{n}" {"selected" if n.endswith("05-B") else ""}>{n}</option>' for n in CASES)+'''</select>
<label for="line-model">Learned model </label><select id="line-model">'''+''.join(f'<option value="{k}">{v}</option>' for k,v in NAMES.items())+'''</select>
<p id="line-status" role="status"></p>
<div class="grid"><div><h3>Original frame</h3><a id="line-raw-link" target="_blank"><img id="line-raw" alt="Unannotated source frame showing the field markings"></a></div>
<div><h3>Learned points and lines</h3><a id="line-learned-link" target="_blank"><img id="line-learned" alt="Saved learned field points and line segments in gold"></a><p>Gold dots are points. They do not count as line segments. A detected crossbar is a goal-frame segment, not a painted grass line. PnLCalib detects that JK crossbar in two scored frames but no grass-line segment in the scored set.</p></div>
<div><h3>Automatic OpenCV candidates</h3><a id="line-classical-link" target="_blank"><img id="line-classical" alt="Automatic classical line segment candidates in purple"></a><p>Full-frame LSD with white-paint and adjacent-green filters. No manual crop or guide is an input. False positives and broken segments remain. This is not semantic field calibration.</p></div>
<div><h3>Reference guides used in the geometry test</h3><a id="line-guides-link" target="_blank"><img id="line-guides" alt="AI-selected field-line reference guides in green"></a><p>The green lines were entered after visual inspection. They are not automatic detections. Their physical line identities are assumptions.</p></div></div>
<p id="line-coverage"></p><details><summary>How were the green goal corners selected?</summary>
<p>I inspected enlarged source images and pixel-coordinate grids, identified the front crossbar/post junctions and upright bases, and entered their coordinates. This was done for both JK and platform. No mouse annotation tool was used. No independent human reviewer verified these labels. Estimated selection uncertainty is about 4 px; errors can be larger where front and rear frame parts are ambiguous.</p>
<p id="line-corner-coordinates"></p><a id="line-grid-link" target="_blank"><img id="line-corner-grid" style="max-width:680px" alt="Enlarged corner patches with original pixel coordinate labels"></a></details>
<p><a href="field-lines/results.json">Open full line outputs, thresholds and diagnostic counts</a></p></section>
<script>const fieldLineReview='''+json.dumps(result)+''';
function updateFieldLineReview(){const f=document.querySelector('#line-frame').value,m=document.querySelector('#line-model').value,r=fieldLineReview.frames[f],v=r.models[m];
const paths={raw:f+'-original.jpg',learned:m+'-'+f+'.jpg',classical:'classical-'+f+'.jpg',guides:'guides-'+f+'.jpg'};
for(const [k,p] of Object.entries(paths)){document.querySelector('#line-'+k).src='field-lines/'+p;document.querySelector('#line-'+k+'-link').href='field-lines/'+p;}
document.querySelector('#line-status').textContent='Learned model: '+v.points+' point candidates; '+v.painted_field_line_segments+' painted field-line segments; '+v.goal_frame_line_segments+' goal-frame segments. OpenCV: '+r.classical.candidates.length+' unlabelled line candidates ('+(r.classical.seconds*1000).toFixed(0)+' ms on this Mac).';
document.querySelector('#line-coverage').textContent='Classical candidate overlap with selected guides: '+r.guides.map(g=>g.name+' '+Math.round(g.classical_coverage_5px*100)+'%').join('; ')+'. This is a development diagnostic, not semantic detection accuracy.';
document.querySelector('#line-corner-coordinates').textContent='Entered goal points, TL/TR/BR/BL, in original pixels: '+JSON.stringify(r.manual_corners);
document.querySelector('#line-corner-grid').src='field-lines/'+f+'-corner-pixel-grid.png';document.querySelector('#line-grid-link').href='field-lines/'+f+'-corner-pixel-grid.png';}
document.querySelector('#line-frame').onchange=updateFieldLineReview;document.querySelector('#line-model').onchange=updateFieldLineReview;updateFieldLineReview();</script>'''
    return section,result

def training_comparison(D):
    pricing={'checked_date':'2026-10-22','currency':'USD','scope':'Public list prices, not an account quote. Training hours below are illustrative, not measured time to train these models.',
      'roboflow':{'pricing_source':'https://roboflow.com/pricing','credit_source':'https://roboflow.com/credits','free_credits_per_month':10,'core_monthly_usd':39,'core_total_credits_per_month':20,'gpu_minutes_per_credit':30,'illustrative_gpu_hours':12,'illustrative_training_credits':24,'example_pack_usd_month':89,'example_pack_total_credits':35},
      'custom_rented_gpu':{'source':'https://www.runpod.io/pricing','example_gpu':'RTX 4090','listed_usd_per_hour':.89,'illustrative_gpu_hours':12,'illustrative_gpu_cost_usd':10.68},
      'excluded_costs':['labeling and human QA','storage and transfer','idle allocated GPU time','engineering and maintenance','app export/integration','commercial model licensing where applicable'],
      'caution':'Equal GPU hours on different services are not equal completed training work. Benchmark actual runtime and held-out quality before comparing total cost.'}
    (D/'evidence/training-options.json').write_text(json.dumps(pricing,indent=2))
    section='''<section id="training-options"><h2>YOLO, RF-DETR and training options</h2>
<p><strong>The suffix selects the task.</strong> The letter n means nano, a model size. It does not mean keypoints.</p>
<table><thead><tr><th>Model name</th><th>Output</th><th>Goal training needed?</th></tr></thead><tbody>
<tr><td>yolo11n.pt / yolo11s.pt</td><td>Bounding boxes</td><td>Yes, for a new goal class; box labels do not teach corners.</td></tr>
<tr><td>yolo11n-pose.pt / yolo11s-pose.pt</td><td>Boxes plus keypoints</td><td>Yes. Default weights predict human joints. We need front-goal-corner labels.</td></tr>
<tr><td>RF-DETR detection</td><td>Bounding boxes</td><td>Custom class labels for a new object.</td></tr>
<tr><td>RF-DETR Seg / Kicky</td><td>Boxes and object masks</td><td>Kicky already has a goal class. Its mask does not label front corners.</td></tr>
<tr><td>RF-DETR Keypoint Preview</td><td>Boxes and keypoints</td><td>Yes. Supplied weights predict human joints. This is a separate early-access keypoint variant.</td></tr>
</tbody></table><p><a href="https://docs.ultralytics.com/models/yolo11/">Official YOLO11 task names</a> · <a href="https://rfdetr.roboflow.com/latest/learn/run/keypoints/">Official RF-DETR keypoint documentation</a></p>
<p><strong>Recommended approach: Roboflow for labeling and dataset versions, with our own evaluation and app geometry.</strong> Use a managed YOLO11n/s-pose baseline first. Export the dataset and use small custom training scripts when testing RF-DETR preview, special losses or temporal video models. This does not require writing a neural network from scratch.</p>
<table><thead><tr><th>Option</th><th>Cost structure</th><th>Management</th><th>Control</th></tr></thead><tbody>
<tr><td>Custom library-based training</td><td>Existing hardware or rented GPU time, plus storage and engineering.</td><td>We own labels, versions, environments, jobs and experiment records.</td><td>Full control over models, sequence inputs, losses and exports.</td></tr>
<tr><td>Roboflow managed training</td><td>Plan/credit usage for data and training; inference is a separate deployment decision.</td><td>Shared annotation UI, frozen dataset versions and managed training reduce routine work.</td><td>Restricted to supported tasks/settings. It does not replace calibration or ball physics.</td></tr>
<tr><td>Hybrid</td><td>Dataset-management usage plus our compute when needed.</td><td>Shared labeling and version history; thin training/evaluation scripts under version control.</td><td>Keep a fixed whole-shot test split and app pipeline independent of the training provider.</td></tr>
</tbody></table>
<p><strong>Public prices checked 22 October 2026:</strong> Roboflow lists a free tier with 10 credits/month and Core at $39/month with 20 total credits. One credit buys 30 GPU training minutes. Storage, uploads and other features also consume credits. See <a href="https://roboflow.com/pricing">current Roboflow pricing</a> and <a href="https://roboflow.com/credits">credit rates</a>.</p>
<p><strong>Billing example, not a training-time forecast:</strong> 12 GPU hours require 24 Roboflow training credits; the listed $89/month pack provides 35 total credits before other usage. Runpod lists an RTX 4090 at $0.89/hour, so 12 allocated hours cost $10.68 for GPU time. See <a href="https://www.runpod.io/pricing">current Runpod pricing</a>. Different GPUs can take different times to complete the same job. These examples exclude labeling, engineering, storage, idle time and any commercial model license costs.</p>
<p>Training platform choice does not guarantee better accuracy. Correct corner labels, low-angle examples and a test split held out by complete take matter more. Evaluate original-pixel corner errors, excluded-point camera checks and measured distance/speed. A standard image-model training run does not make the ball detector temporal; a learned temporal detector needs sequence data and training.</p>
<p>For the first round, compare YOLO11n-pose and YOLO11s-pose. Add RF-DETR Keypoint as a challenger. Keep No Bells as the JK baseline. No new training ran in this update. We did not upload footage or start a paid job.</p></section>'''
    note='''
## Model names and training-provider comparison

`yolo11n.pt` is a bounding-box detector. `yolo11n-pose.pt` is the keypoint variant. The n means nano size. The s size also has both detection and pose variants. Generic pose weights predict human joints. Goal corners need a custom four-keypoint dataset and fine-tuning.

RF-DETR is a model family: detection predicts boxes, Seg adds masks, and Keypoint Preview predicts keypoints. Kicky uses Seg, not the keypoint model. The supplied Keypoint Preview predicts human joints and must be trained for goal corners.

Recommendation: use Roboflow to manage labels and frozen dataset versions. Run a managed YOLO11n/s-pose baseline. Keep whole-shot splits, geometry-specific evaluation and app integration in our own code. Use custom library-based training for architecture changes, special losses, RF-DETR preview experiments or learned temporal ball sequences. This is a hybrid approach, not a new neural network implementation from scratch.

| Approach | Cash cost | Management | Flexibility |
|---|---|---|---|
| Custom scripts with existing libraries | Owned hardware or GPU rental; engineering time | We maintain data versions, runs, environments and jobs | Full control over models, sequence data and losses |
| Roboflow managed training | Plan/credits for data and training | Shared labeling, frozen versions and managed jobs reduce routine work | Supported tasks/models/settings only |
| Hybrid | Dataset-management usage plus owned/rented compute as needed | Shared data workflow with thin reproducible training scripts | Keep evaluation and deployment portable |

Public prices checked on 22 October 2026: [Roboflow pricing](https://roboflow.com/pricing) lists 10 free credits/month and $39/month Core with 20 total credits. [Credit rates](https://roboflow.com/credits) specify 30 training GPU minutes per credit. Data storage, upload and other actions use credits too. A hypothetical 12 GPU hours therefore needs 24 training credits; the listed $89/month pack has 35 total credits before other usage. This is a billing example, not a training-time prediction.

[Runpod pricing](https://www.runpod.io/pricing) lists an RTX 4090 at $0.89/hour. Twelve allocated hours would cost $10.68 for compute. Different services' GPUs can need different hours for the same training job, so compare cost per completed, equally evaluated run. These examples exclude labeling, engineering, storage/transfer, idle time and model licensing where applicable. Public list prices do not establish the user's account price or export entitlement.

No training provider guarantees accuracy. Keep complete takes out of the training split. Measure original-pixel corner errors and independent camera checks, then measured metric distance and speed. Standard image-model training does not add temporal video inputs. A learned temporal ball model requires labeled sequences and an appropriate training loop.

Sources: [YOLO11 tasks](https://docs.ultralytics.com/models/yolo11/), [RF-DETR keypoints](https://rfdetr.roboflow.com/latest/learn/run/keypoints/), [Roboflow YOLO training options](https://roboflow.com/blog/yolo-training).
'''
    return section,note

if __name__=='__main__':
    D=Path('/Users/tbg/orca/workspaces/stryke/basslet/docs/reports/2026-10-22-goal-camera-review')
    section,result=publish_field_lines(D);(R/'field-line-section.html').write_text(section)
    for k,v in result['frames'].items():print(k, len(v['classical']['candidates']),[round(g['classical_coverage_5px'],2) for g in v['guides']],flush=True)
