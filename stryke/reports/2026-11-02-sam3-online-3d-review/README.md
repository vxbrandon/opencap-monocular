# SAM 3 ball tracking and online body feedback review

Reviewed 2 November 2026. Includes a source review, SAM checkpoint preflight,
and an executed [OnlineHMR body-branch benchmark](onlinehmr-run.md). SAM 3 and
Fast SAM 3D Body have not run: both require gated checkpoint access. No model
selection in the app was changed.

Open the [HTML results and projection overlays](index.html). Built from the saved
2 November outputs with `python3 build_html.py`; no inference was rerun.

## Existing ball comparison

The [detector and flight report](../2026-10-21-ball-trajectory-comparison/index.html)
contains YOLO11s, SoccerNet YOLO, YOLOv9 soccer, YOLO26s soccer, FootAndBall,
BallSeg, TrackNetV2, ResTrackNetV2, MonoTrack, WASB and BlurBall. Several already
use multi-frame inputs. YOLO11s remains the selected app detector. Its 47/51
centre hits use development references proposed by YOLO, not independently
labelled recall or field accuracy.

## SAM 3 evaluation status and protocol

[Meta SAM 3](https://github.com/facebookresearch/sam3) combines a detector with
a SAM 2-derived temporal video tracker. It supplies masks and object identities;
camera calibration and metric 3D reconstruction remain downstream operations.
[SAM 3.1](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
adds shared-memory multi-object tracking. Its reported many-object speedup does
not establish a single-football speedup.

Checkpoint preflight on this machine found no standard Hugging Face credential
or cached SAM model. HEAD requests to the official SAM 3 and SAM 3.1 checkpoint
repositories returned HTTP 401, `GatedRepo`. No checkpoint was downloaded and
no SAM inference ran. Meta documents checkpoint approval/authentication and a
CUDA runtime. Required next input is an authorized local checkpoint or locally
configured access; credentials should not be pasted into a conversation.

The benchmark should preserve the original clips, frame indices and timestamps:

1. Run the same four paired-shot examples and four original phone clips used
   by the existing comparison, retaining every source frame.
2. Keep text-prompted detection and first-frame visually seeded tracking as
   separate conditions. Record the seed frame and annotation effort; a seeded
   tracker must not be represented as automatic detection.
3. Record masks, IDs, missing frames, prompt/refinement events and original-raster
   centres. Keep a propagated prediction distinguishable from independently
   reviewed visible-ball support. A visible-mask centroid is not necessarily
   the physical centre of a blurred or partially occluded ball.
4. Score independently reviewed centres, visible coverage, false positives,
   identity switches, reacquisition and full-track continuity. Retain the older
   YOLO-proposed 51-point diagnostic separately for historical comparison.
5. Measure initialization, decode, inference, peak memory and full-clip latency
   on named hardware. Report any lookahead/buffering; forward propagation alone
   is not proof of an end-to-end causal live implementation.
6. Feed only accepted observations into the existing physics fit. Report
   changes in usable track length and refusals without claiming validated speed.

## Body feedback bottleneck and online candidates

For the historical warm L4 250-frame / 8.33-second replay, preparation and
subject checks took 11.40 s; the GVHMR observation/motion pipeline 11.18 s;
geometry export 4.18 s; overlay drawing 7.96 s; final encoding 2.75 s; other
work 2.39 s: 39.86 s total before delivery. In the separate body-only profile,
tracking, 2D pose and image features consumed about 90%; motion recovery was
0.34 s. These are different profiling boundaries.

The implemented joint-first path publishes approximately 187 KB before the
33.45 MB mesh. Its warm T4 bridge observation still took 85.74 s to joints,
excluding phone upload, API polling and display. See the
[timing evidence](../2026-10-20-aws-mesh-benchmark/joints-first-implementation.md).

- [OnlineHMR](https://arxiv.org/html/2603.17355v1) is explicitly causal, using
  past/current frames and incremental SLAM. The authors report 3.3 FPS and
  0.30 s average delay on RTX 6000 Ada for their complete world-coordinate
  comparison. Online operation does not establish 30/60 FPS operation.
- [Fast SAM 3D Body](https://arxiv.org/html/2603.15603v1) supports per-image
  body recovery and reports a 65 ms/frame robot-control demonstration on
  RTX 5090. Other automatic benchmark configurations are substantially slower.
  This is a separate model from SAM 3 video object segmentation. STRYKE has
  not established its phone latency, football joint accuracy or calibrated
  body-to-ball registration.
- FastHMR was already timed locally in the AWS comparison: 11.32 s for 250
  frames on L4, about 22.08 processing FPS. That clip/batch result does not
  establish causal online inference. Its SMPL output is not a drop-in
  replacement for the current SMPL-X transport.

## Current trajectory fitter and goal models

The app calls the sibling `stryke-pipeline/pipeline/ball.py` through
`stryke_vision.ball_lane`. YOLO11s raw image centres and temporal identity gates
feed STRYKE's numerical flight model. Robust bounded SciPy `least_squares`
(`soft_l1`) fits reprojection, supported radius/goal constraints and priors.
The current entry-point defaults select speed-dependent drag (`crisis`), fitted
effective gravity and lateral curvature (`magnus=lateral`). The fitted lateral
term is not measured spin RPM. Launch time, speed/direction and camera
height/tilt are fitted with bounds and quality checks.

The separate stereo research code triangulates calibrated, clock-aligned
observations and fits gravity, quadratic drag or drag plus effective curvature.
The comparative Ball3D study ran seven published forward simulators through a
matched SciPy adapter: basic parabola, fitg, kinetic and angular; MuJoCo kinetic,
angular and ellipsoid. It did not reproduce the full paper protocol, and
Ball3D has not replaced the app solver.

The app's learned goal-area proposal model is **Kicky RF-DETR Seg Small**
(`dcrey7/kicky-ai-rfdetr-seg`). Its mask includes net/rear structure, so it is
not used as four front calibration corners. The separate front-corner study
tested **No Bells Just Whistles MV**, **PnLCalib SV**, and **PnLCalib WorldPose
FT**. No Bells passed four reviewed JK frames but none of ten platform views;
it was not adopted as a general automatic calibration solution. See the
[goal/camera report](../2026-10-22-goal-camera-review/SUMMARY.md).
