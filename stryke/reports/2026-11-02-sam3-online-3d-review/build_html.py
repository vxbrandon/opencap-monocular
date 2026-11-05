"""Build the offline HTML review from measured results and saved overlays."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path('/Users/tbg/stryke-research/runs/gpu-20261102-onlinehmr/results/final')
LABELS = {
    'indoor-dark': 'Indoor · dark shirt',
    'indoor-white': 'Indoor · white shirt',
    'ronaldo-screen-20261020': 'Ronaldo · red shirt',
    'ronaldo-screen-white-20261020': 'Ronaldo · white shirt',
    'jk-messi-style-20261020': 'JK · Messi-style kick',
}
SAMPLES = [
    ('ronaldo-screen-20261020', 11, 'Head and foot projections visibly miss the player during the kick.'),
    ('ronaldo-screen-white-20261020', 24, 'Broad running-pose alignment, with visible extremity offsets.'),
    ('jk-messi-style-20261020', 36, 'The lifted foot is misplaced despite using the author’s crop dilation.'),
]


def main():
    report = json.loads((ROOT / 'onlinehmr-results.json').read_text())
    cleanup = json.loads((ROOT / 'cloud-cleanup.json').read_text())
    assert cleanup['cleanup_complete']
    assert all(state == 'stopped' for state in cleanup['instance_states'].values())
    runs = report['runs']
    rows = []
    for clip, label in LABELS.items():
        pair = sorted((r for r in runs if r['clip'] == clip), key=lambda r: r['repeat'])
        assert len(pair) == 2
        rows.append(f'<tr><th scope="row">{html.escape(label)}</th>'
                    f'<td>{pair[0]["frames"]}</td><td>{pair[0]["processing_fps"]:.2f}</td>'
                    f'<td>{pair[1]["processing_fps"]:.2f}</td>'
                    f'<td>{sum(r["estimated_frames"] for r in pair)}/{sum(r["frames"] for r in pair)}</td></tr>')
    stage_rows = []
    for run in runs:
        stages = run['stage_milliseconds']
        stage_rows.append(f'<tr><th scope="row">{html.escape(LABELS[run["clip"]])} · {run["repeat"] + 1}</th>'
                          f'<td>{stages["detector_seconds"]["median"]:.1f}</td>'
                          f'<td>{stages["crop_and_body_seconds"]["median"]:.1f}</td>'
                          f'<td>{stages["total_seconds"]["median"]:.1f}</td>'
                          f'<td>{stages["total_seconds"]["p95"]:.1f}</td></tr>')
    figures = []
    for clip, frame, caption in SAMPLES:
        image = ARTIFACTS / f'{clip}-1' / f'overlay-{frame:04d}.jpg'
        encoded = base64.b64encode(image.read_bytes()).decode('ascii')
        figures.append(f'<figure><img loading="lazy" src="data:image/jpeg;base64,{encoded}" '
                       f'alt="Projected OnlineHMR joints on {html.escape(LABELS[clip])}, frame {frame}">'
                       f'<figcaption><strong>{html.escape(LABELS[clip])} · frame {frame}</strong>'
                       f'<br>{html.escape(caption)}</figcaption></figure>')
    football = [r for r in runs if not r['clip'].startswith('indoor')]
    football_fps = sum(r['frames'] for r in football) / sum(r['seconds'] for r in football)
    document = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>STRYKE · OnlineHMR candidate results</title>
<style>
:root{color-scheme:light;--ink:#17251f;--muted:#53635a;--line:#dbe3dd;--green:#146447;--paper:#f6f8f5}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,sans-serif}
main{max-width:1100px;margin:auto;padding:48px 24px 70px}header{max-width:850px;margin-bottom:28px}
.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--green);font-weight:750}
h1{font-size:clamp(30px,5vw,48px);line-height:1.12;letter-spacing:-.035em;margin:12px 0 20px}
h2{font-size:24px;line-height:1.25;margin:0 0 15px}h3{font-size:18px;margin:0 0 8px}
p{margin:10px 0 16px}a{color:var(--green);text-underline-offset:3px}a:focus-visible,summary:focus-visible{outline:3px solid #df8a26;outline-offset:4px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}
.metric,section{border:1px solid var(--line);background:white;border-radius:14px;padding:22px}
.metric strong{display:block;font-size:30px;line-height:1.25}.metric span,.muted{color:var(--muted);font-size:14px}
section{margin:18px 0}.note{border-left:4px solid #b97425;background:#fff6e8;padding:14px 18px;border-radius:4px}
.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:12px 14px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child{text-align:left}
thead{background:#f0f5f1}tbody th{font-weight:550}tbody tr:last-child>*{border-bottom:0}
figure{margin:22px 0 30px}figure:last-child{margin-bottom:0}img{display:block;width:100%;height:auto;border-radius:8px;background:#eee}
figcaption{font-size:14px;color:var(--muted);padding-top:10px}figcaption strong{color:var(--ink)}
summary{font-weight:650;cursor:pointer;padding:8px 0}code{font-size:.85em;overflow-wrap:anywhere}
.status{display:grid;grid-template-columns:1fr 1fr;gap:18px}.status>div{background:#f2f5f2;border-radius:8px;padding:18px}
footer{font-size:13px;color:var(--muted);margin-top:28px}@media(max-width:650px){main{padding:26px 16px 50px}.metrics{grid-template-columns:1fr 1fr}.status{grid-template-columns:1fr}section{padding:18px}.metric{padding:16px}.metric strong{font-size:26px}}
@media print{body{background:white}main{padding:0}section,figure{break-inside:avoid}.metrics{grid-template-columns:repeat(4,1fr)}}
</style></head><body><main>
<header><div class="eyebrow">STRYKE research · measured 2 November 2026</div>
<h1>OnlineHMR runs frame by frame.<br>Kick accuracy still needs work.</h1>
<p>The causal body branch produced finite outputs for every evaluated frame. Football replay throughput was
<strong>5.62–8.09 FPS on Tesla T4</strong>. Saved overlays show visible head and foot projection errors during kicks.</p>
<p class="muted">Offline replay with YOLO11n person detection and the author’s body model. World-motion SLAM and phone delivery are outside this measurement.</p></header>
<div class="metrics">
<div class="metric"><strong>526 / 526</strong><span>frame outputs · 263 unique frames, two repeats</span></div>
<div class="metric"><strong>FOOTBALL_FPS FPS</strong><span>combined football throughput</span></div>
<div class="metric"><strong>9.20 s</strong><span>model / detector initialization</span></div>
<div class="metric"><strong>3.86 GiB</strong><span>peak PyTorch-allocated GPU memory</span></div></div>
<section><h2>What ran</h2><p>Five clips, two passes each, with every source frame retained. The body model receives one current frame at a time and carries a bounded past-frame KV cache. Crops use the author’s 1.2 dilation.</p>
<div class="table-wrap"><table><thead><tr><th scope="col">Clip</th><th scope="col">Frames / pass</th><th scope="col">Pass 1 FPS</th><th scope="col">Pass 2 FPS</th><th scope="col">Output coverage</th></tr></thead><tbody>RESULT_ROWS</tbody></table></div>
<p class="note">Coverage measures whether the model returned finite numbers. Anatomical accuracy and metric 3D error have not been independently measured. Two short repeats do not establish stable production throughput.</p>
<details><summary>Inspect stage timings (milliseconds)</summary><div class="table-wrap"><table><thead><tr><th scope="col">Clip · pass</th><th scope="col">Detector p50</th><th scope="col">Crop + body p50</th><th scope="col">Frame p50</th><th scope="col">Frame p95</th></tr></thead><tbody>STAGE_ROWS</tbody></table></div>
<p class="muted">Crop + body includes CPU preprocessing and GPU inference. Frame times are CUDA-synchronized and include decode, detection, projection and sampled exports. Input hashing, ffprobe, aggregate artifact writes, upload and display are excluded.</p></details></section>
<section><h2>Inspect the saved projections</h2><p>Yellow points are projected model joints. These are the middle saved frames from pass 2 with the corrected crop setting. They are visual diagnostics; no reference joint labels were scored.</p>FIGURES</section>
<section><h2>Camera assumptions</h2><p>Indoor clips use intrinsics supplied in their manifests. The imported football clips use the author’s image-diagonal focal heuristic and centred principal point. This body benchmark did not calibrate those cameras.</p>
<p>For the earlier two-view ball experiment, focal length was <strong>estimated from the scene</strong>; original lens-calibration sidecars were unavailable. The fit assumed zero lens distortion, square pixels and a centred principal point. Those choices make the result conditional. Known goal dimensions supply useful constraints, while a full camera/distortion model needs additional evidence.</p>
<p><a href="../2026-10-22-goal-camera-review/index.html">Open the goal / camera geometry report</a> · <a href="../2026-10-21-ball-trajectory-comparison/index.html">Open the ball-detector comparison</a></p></section>
<section><h2>Candidate status at the 2 November run</h2><div class="status"><div><h3>OnlineHMR</h3><p>Body branch executed. Full world-coordinate SLAM remains untested in this run. The experiment substituted YOLO11n for the paper’s detector.</p></div>
<div><h3>SAM 3 / Fast SAM 3D Body</h3><p>No inference ran. Official model downloads returned gated-access errors. SAM 3 ball video tracking and Fast SAM 3D Body are separate candidates requiring their respective approved weights.</p></div></div></section>
<section><h2>Evidence and execution details</h2><p>Verified checkpoint keys and hashes, unchanged input hashes, complete time-ordered frame indices, finite output arrays and bounded caches. Both existing GPU instances were independently confirmed stopped; the temporary SSH rule was removed.</p>
<p><a href="onlinehmr-results.json">Measured results JSON</a> · <a href="onlinehmr-run.md">Full written report</a> · <a href="cloud-cleanup.json">Cloud cleanup receipt</a></p>
<details><summary>Reproduction</summary><p>Python 3.11 · PyTorch 2.7.1+cu126 · NumPy 1.26.4 · g4dn.xlarge / Tesla T4 · two PyTorch CPU threads. Existing licensed neutral SMPL asset; exact official checkpoint keys loaded.</p>
<p>Author commit: <code>COMMIT</code><br>Checkpoint SHA-256: <code>CHECKPOINT</code></p>
<p>Runner: <a href="../../../tools/benchmark_onlinehmr.py">tools/benchmark_onlinehmr.py</a>. Original videos, complete outputs and model weights remain outside the repository. Three derived overlays are embedded in this offline HTML.</p></details></section>
<footer>Generated from the saved 2 November measurements. HTML prepared 5 November 2026. No inference was rerun to build this page.</footer>
</main></body></html>'''
    replacements = {
        'FOOTBALL_FPS': f'{football_fps:.2f}', 'RESULT_ROWS': ''.join(rows),
        'STAGE_ROWS': ''.join(stage_rows), 'FIGURES': ''.join(figures),
        'COMMIT': html.escape(report['upstream_commit']),
        'CHECKPOINT': html.escape(report['checkpoint_sha256']),
    }
    for key, value in replacements.items():
        document = document.replace(key, value)
    target = ROOT / 'index.html'
    target.write_text(document)
    print(f'Wrote {target} ({target.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
