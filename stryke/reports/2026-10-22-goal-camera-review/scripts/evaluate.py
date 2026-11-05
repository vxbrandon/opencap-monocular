from pathlib import Path
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

R=Path(__file__).resolve().parent
refs=json.loads((R/'references.json').read_text())
boxes={
 'jk-0030':[62,91,448,226], 'jk-0060':[301,93,654,225],
 'jk-0090':[548,135,891,260], 'jk-0120':[545,167,885,292],
 'platform-curling-04-A':[884,402,1306,576],
 'platform-curling-05-A':[884,399,1306,580],
 'platform-knuckleball-07-A':[884,405,1307,586],
 'platform-driven-04-A':[884,416,1308,594],
 'platform-curling-01-A':[884,395,1308,570],
 'platform-curling-04-B':[405,455,829,628],
 'platform-curling-05-B':[404,453,829,628],
 'platform-knuckleball-07-B':[406,455,829,628],
 'platform-driven-04-B':[406,454,832,630],
 'platform-curling-01-B':[404,452,832,627],
}
(R/'goal-area-references.json').write_text(json.dumps({'boxes_xyxy':boxes,'kind':'approximate agent reviewed whole goal + net extent; hidden extents retained for partial occlusion'},indent=2))

def iou(a,b):
    a,b=np.array(a),np.array(b)
    inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    return float(inter/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter))

summary=[]
for tag in ['kicky','nbjw','pnl','pnl_wp']:
    data=json.loads((R/f'{tag}-results.json').read_text());rows=[]
    for r in data['rows']:
        if r['id'] not in refs['corners']:continue
        ref=refs['corners'][r['id']];visible=np.array([v is not None for v in ref]);ref=np.array([v for v in ref if v is not None],float)
        row={'id':r['id'],'visible_corners':int(visible.sum()),'seconds':r['seconds']}
        if tag=='kicky':
            row['best_goal_box_iou']=max([iou(g['bbox'],boxes[r['id']]) for g in r['goals']],default=0)
            row['goal_count']=len(r['goals'])
        else:
            pts=r['completed_keypoints'];ids=[k for k in pts if int(k) in [12,13,14,15,16,17,18,19]]
            candidates=np.array([[pts[k]['x'],pts[k]['y']] for k in ids])
            errors=np.full(len(ref),np.inf);assigned=[None]*len(ref)
            if len(ids):
                d=np.linalg.norm(ref[:,None]-candidates[None],axis=2)
                a,b=linear_sum_assignment(d)
                for ai,bi in zip(a,b):errors[ai]=d[ai,bi];assigned[ai]=ids[bi]
            row.update(candidate_goal_points=len(ids),errors_px=[float(x) if np.isfinite(x) else None for x in errors],assigned_ids=assigned,
                       hits_8px=int((errors<=8).sum()),hits_16px=int((errors<=16).sum()),
                       complete_8px=bool(len(ref)==4 and np.all(errors<=8)), complete_16px=bool(len(ref)==4 and np.all(errors<=16)))
        rows.append(row)
    timing=np.array([r['seconds'] for r in rows[1:]])
    totals={'model':tag,'load_s':data['load_s'],'first_frame_ms':rows[0]['seconds']*1000,'warm_median_ms':float(np.median(timing)*1000),'warm_p95_ms':float(np.percentile(timing,95)*1000)}
    if tag=='kicky':
        totals.update(goal_box_hits_iou50=sum(r['best_goal_box_iou']>=.5 for r in rows),goal_box_median_iou=float(np.median([r['best_goal_box_iou'] for r in rows])),complete_front_quads='not an output')
    else:
        totals.update(hits_8px=sum(r['hits_8px'] for r in rows),hits_16px=sum(r['hits_16px'] for r in rows),
                      visible_corner_count=sum(r['visible_corners'] for r in rows),
                      complete_8px=sum(r['complete_8px'] for r in rows),complete_16px=sum(r['complete_16px'] for r in rows))
        for source in ['jk','platform']:
            group=[r for r in rows if r['id'].startswith(source)]
            totals[source]={'frames':len(group),'hits8':sum(r['hits_8px'] for r in group),'hits16':sum(r['hits_16px'] for r in group),'visible':sum(r['visible_corners'] for r in group),'complete8':sum(r['complete_8px'] for r in group),'complete16':sum(r['complete_16px'] for r in group)}
    summary.append({'totals':totals,'rows':rows})
(R/'evaluation.json').write_text(json.dumps({'protocol':'14 fixed development frames; 54 visible corners; agent-reviewed labels ±4px. One-to-one best spatial assignment among semantic goal points, ignores left/right class errors: generous diagnostic, not held-out accuracy or mAP. Both 8px and 16px tolerances reported. All misses remain in denominator. Whole goal IoU is a different task from front corner localization.','models':summary},indent=2))
for s in summary:print(json.dumps(s['totals']))
