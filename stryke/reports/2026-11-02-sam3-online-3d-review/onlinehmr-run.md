# OnlineHMR candidate run — 2 November 2026

OnlineHMR's causal body branch ran successfully on the existing AWS Tesla T4.
All 263 unique frames from five clips produced finite outputs in two repeats
(526 frame instances). Football throughput was 5.62–8.09 FPS. Sampled kick
overlays show visible head/foot projection errors, so this does not establish
fitness for technique feedback. No independently labelled 3D accuracy was measured.

SAM 3 ball tracking and Fast SAM 3D Body remain blocked on Hugging Face access.
The official checkpoint endpoints returned HTTP 401 `GatedRepo`, and no standard
local credential or cached checkpoint was available. The user needs access to
[SAM 3](https://huggingface.co/facebook/sam3) and, for Fast SAM 3D Body,
[SAM 3D Body DINOv3](https://huggingface.co/facebook/sam-3d-body-dinov3), followed
by local authentication with `uvx --from huggingface_hub hf auth login`.
Existing footage and AWS GPU instances are available.

## Measured configuration

- Author's [OnlineHMR code](https://github.com/Tsukasane/Video-OnlineHMR), commit
  `5236d2206068027e1b73480e34903092728f84f5`, official public checkpoint, and
  existing licensed neutral SMPL asset. All checkpoint keys matched the model.
- Python 3.11, PyTorch 2.7.1+cu126, NumPy 1.26.4, Tesla T4 on g4dn.xlarge,
  two PyTorch CPU threads. No compilation or TensorRT optimization.
- YOLO11n COCO person detection at 640, confidence 0.5, substituted for the
  paper's detector. Indoor clips require exactly one detected person. Football
  clips use previously recorded player boxes, then causal previous-box IoU
  matching; ambiguous/lost subjects cannot silently switch to another person.
- Author's crop/normalization helpers, crop dilation 1.2 as in
  `inference_chunk_ar`, 256×256 crops, author model autocast and single-frame
  `inference_forward` calls carrying the KV cache. Cache resets at clip boundaries;
  observed cache sizes never exceeded 384 tokens per layer (two frames of patches).
- Known manifest intrinsics for indoor clips. Football uses the author's
  image-diagonal focal heuristic and image-centre principal point. These are not
  calibrated football camera intrinsics.
- Output: SMPL rotations, shape, camera parameters, translation, 49 model joints,
  source-raster projections, frame indices and timestamps. The saved `joints`
  are in SMPL coordinates; adding `trans_full` gives camera coordinates.

This is an offline replay with causal image inputs. **World-motion SLAM, metric
field registration, phone transport and phone display are not included.** It
does not reproduce the paper's complete world-coordinate system or measure live
camera-to-screen latency. It is not directly comparable to the earlier L4 GVHMR
full-pipeline or FastHMR batch timings.

## Results

| Clip | Frames per repeat | Repeat 1 FPS | Repeat 2 FPS | Estimated frames |
|---|---:|---:|---:|---:|
| Indoor dark | 60 | 3.61 | 3.90 | 120/120 |
| Indoor white | 60 | 3.32 | 3.13 | 120/120 |
| Ronaldo screen, red | 23 | 8.09 | 5.62 | 46/46 |
| Ronaldo screen, white | 48 | 5.76 | 5.91 | 96/96 |
| JK Messi-style | 72 | 6.12 | 6.04 | 144/144 |

Combined across both repeats: 6.08 FPS for football, 3.46 FPS for indoor footage.
Two short repeats are a smoke benchmark, not a stable production performance
distribution. All clips are 30 FPS derived inputs; screen-recording playback
timestamps do not establish physical ball-impact timing.

Model/detector initialization took 9.20 s, excluding imports, downloads and
checkpoint hashing. Peak PyTorch-allocated GPU memory was 3.86 GiB; this excludes
CUDA context/reserved memory. FPS includes frame decode, detection, cropping,
body inference, projection, and sampled exports. It excludes input hash/ffprobe
preflight and final aggregate artifact writes. Stage timing percentiles and
source/checkpoint hashes are in [onlinehmr-results.json](onlinehmr-results.json).

The crop/body stage includes both CPU preprocessing and GPU inference; these
timings cannot attribute that whole stage to the neural model. Different input
resolutions and short-run variation also limit comparisons between clips.

## Quality and verification

The sampled red-shirt frame 11 has misplaced head and foot projections. JK kick
frame 36 still misplaces the lifted foot after matching the author's crop
dilation. The white-shirt running pose is broadly aligned but has visible
extremity offsets. These are qualitative observations, not error measurements.
Complete output coverage means the model returned numbers on every frame.

Verified the downloaded artifact archive SHA-256, local/remote checkpoint hash
agreement, exact checkpoint key coverage, all 526 output arrays finite, all
frame indices present, strictly increasing timestamps, unchanged input hashes,
and bounded temporal caches. The runner passes Ruff and Python compilation;
six subject-selection cases cover unique matches, ambiguity, and missing targets.

Initial exploratory runs used dilation 1.0. They are retained under `run02/`
and `football01/`; the table above uses the corrected 1.2 run under `final/`.
Those preliminary 4–7.5 FPS observations must not be mixed into this table.

Compatibility changes only scoped legacy Chumpy Python/NumPy aliases during
SMPL loading and allowed NumPy scalar checkpoint metadata in the weights-only
loader. The upstream patch makes unused Detectron2/SAM imports lazy. No learned
forward-pass code was changed.

## Reproduction and artifacts

Runner: [tools/benchmark_onlinehmr.py](benchmark_onlinehmr.py).
It requires a prepared upstream checkout, checkpoint, YOLO weights, licensed
SMPL assets and an input manifest; all model/media assets stay outside the repo.

Local full outputs, timing rows, original-frame samples, joint overlays,
executed runner snapshot, logs and manifests:

`/Users/tbg/stryke-research/runs/gpu-20261102-onlinehmr/results/final/`

Archive: `results/final-artifacts.tgz` under the same run root. SHA-256:
`e1e0e7c16890bd13af8fed469c7447f877a30cc3b3c4beebc3cd654d5537e75f`.
The [HTML review](index.html) embeds three derived projection overlays. Original
videos, complete output arrays and model checkpoints remain outside the repo.

The existing T4 was started with a 30-minute shutdown limit. Results were
downloaded and verified before requesting stop. Temporary SSH access was
restricted to the current client's single IPv4 address and subsequently revoked.
The controller's 150-second stop verification window expired while AWS still
reported `stopping`; subsequent independent AWS queries confirmed both existing
GPU instances `stopped`. Final cleanup is recorded in
[cloud-cleanup.json](cloud-cleanup.json).
