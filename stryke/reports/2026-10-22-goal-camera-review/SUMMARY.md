# Goal detection and camera/ball position — 22 October 2026

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
| Kicky RF-DETR Seg Small | goal + net box | 14/14 at IoU ≥ 0.50 | no front corners | 46 ms |
| No Bells MV | front-corner candidates | JK 16/16; platform 0/38 | JK 4/4; platform 0/10 | 293 ms |
| PnLCalib SV | front-corner candidates | JK 8/16; platform 2/38 | 0/14 | 287 ms |
| PnLCalib WorldPose FT | front-corner candidates | 0/54 | 0/14 | 288 ms |

Kicky's median whole-goal box IoU is 0.892. This is not front-corner accuracy. No Bells' 16-pixel totals are the same as its 8-pixel totals. PnLCalib SV rises from 10/54 to 11/54 at 16 pixels, with no complete frame.

We also ran Kicky and No Bells on every source frame of the 161-frame JK clip and both 176-frame views of platform curling 5. There was no frame skipping: 513 frames per model. Kicky emits a goal-area proposal in 513/513 frames. No Bells emits a complete semantic four-point group in 161/161 JK frames and 0/352 platform frames. These proposal counts are not accuracy scores. Only the 14 reviewed frames have scored labels.

All three point/line checkpoints detect 17–20 points in the author's broadcast example. The adapter passes this input sanity check. None emits a painted pitch-line segment at its author threshold in the 14 target frames. PnLCalib SV does emit the goal crossbar (line ID 7) in JK frames 30 and 60; this is a goal-frame segment, not a painted pitch line. Therefore the field-line experiment below uses reviewed line pixels. It is not automatic field-line calibration.

## How the 3D scene is made

The world origin is the centre of the front goal line. X runs across the goal. Y is height. Z runs into the field towards the shooter. Positions use metres.

The goal is assumed to have an inner width of 7.32 m and a height of 2.44 m. The ball has a nominal diameter of 0.22 m. IFAB gives a ball circumference of 68–70 cm. This gives a diameter range of about 0.216–0.223 m. These are assumptions for the recorded objects, not tape measurements. Sources: [IFAB goal and field dimensions](https://www.theifab.com/laws/latest/the-field-of-play/) and [IFAB ball measurements](https://www.theifab.com/laws/latest/the-ball/).

The solver fits the known goal rectangle and a spherical ball silhouette. The ball is stationary and touches a flat ground plane. Its centre height is thus its radius. This extra ground constraint helps estimate the unknown focal length. It must not be used for a flying ball.

The clips have no original camera calibration sidecars. The fit assumes a centred principal point, square pixels and zero lens distortion. It estimates focal length and camera pose. The ball silhouette comes from a reviewed ellipse, not a detector box. Exact perspective sphere tangency is used in the fit. The ball diameter provides distance along the viewing ray.

A homography maps points on one plane. The goal face and the grass are different planes. An airborne ball is on neither plane. Its position needs a 3D camera model and a further distance constraint. A ground homography alone cannot recover flight height.

## Camera and stationary ball estimates

| Scene | Camera X, Y, Z (m) | Ball X, Y, Z (m) | Focal length (px) | Goal fit RMS (px) |
|---|---|---|---|---|
| JK 60, reviewed corners | -3.3, 1.7, 32.2 | -3.00, 0.11, 27.43 | 1459 | 0.81 |
| JK 60, No Bells corners | -6.5, 1.7, 34.4 | -5.71, 0.11, 29.16 | 1621 | 1.50 |
| Platform 5 A + candidate line | -14.6, 0.3, 24.9 | -12.94, 0.11, 20.65 | 1611 | 4.10 |
| Platform 5 B + candidate line | -17.0, 0.6, 23.1 | -13.04, 0.11, 20.12 | 1660 | 0.79 |

The main JK scene uses reviewed front corners and the reviewed ball silhouette. A separate JK scene uses the raw No Bells goal points with the same ball silhouette. Moving image corners by a few pixels changes the camera position by metres. A small fit residual does not prove metric accuracy.

The platform scene shows two independent view estimates. Their ball positions differ by 0.54 m after the candidate line constraint, compared with 6.24 m before it. We did not constrain the two balls to coincide. This is a useful consistency result. It is not a measured 3D error. The edited montage does not prove that the views are synchronized captures of the same physical kick.

## Field-line experiment

For JK frame 60, we test the assumed goal-area front at 5.5 m and the penalty-area front at 16.5 m. We fit perpendicular line distance, not invented endpoint correspondences. The goal-plus-ball fit misses the second line by 17.6 px. Adding only the first line leaves a 15.1 px error on that second line. Adding both reduces their fit errors to 1.6 and 0.6 px. However, three excluded junctions still have errors of 90.3, 55.4 and 12.8 px. The mean is 52.8 px. We reject this refinement for the main scene. The goal/field layout assumption or reviewed line identity needs correction. We did not prove which cause is responsible.

For platform A/B, the visible straight line is tested as a candidate penalty-area front. A's line discrepancy drops from 119.6 to 0.1 px, and B's from 3.5 to 0.3 px. These are fitted-line residuals. No independent surveyed platform line or metric camera position is available. The improved A/B agreement supports further testing. It does not establish accurate ball speed.

## Input sensitivity

We ran 60 perturbation scenarios per selected view. We changed goal corners, ball silhouette, line pixels, principal point and regulation ball size. The JSON contains every scenario. The ranges below are the 5th–95th percentiles of these input scenarios. They are not confidence intervals or measured accuracy. Unmodelled distortion and wrong line identity can add more error.

- jk-0060: camera Z 22.3–64.3 m; ball Z 19.1–54.4 m.
- platform-curling-05-A: camera Z 21.3–25.9 m; ball Z 20.0–21.4 m.
- platform-curling-05-B: camera Z 22.7–23.4 m; ball Z 19.9–20.4 m.

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

## Clarification: reference points and field-line images

The green corners were selected for BOTH JK and platform by this assistant's image-understanding capability. I inspected enlarged source frames and pixel-coordinate grids, then entered coordinates in JSON. No mouse labeling tool was used. No separate downloaded detector or separate API model generated these labels. No independent human reviewer checked them. These are approximate AI-assisted reference labels, not human ground truth.

The HTML now shows original frames, saved learned point/line outputs, a new automatic OpenCV LSD candidate detector, and AI-selected reference guides as separate images. The new classical detector receives the full image only. It filters line segments using whiteness and adjacent green pixels. It does not receive manual corners, guide lines or a crop. It does not identify which segment is a penalty-area line.

On JK 60, classical candidates cover both selected guide segments at the diagnostic 5px/5-degree tolerance. On platform 5 A/B, coverage is only 17%/26%. These are approximate guide-overlap diagnostics on three development frames, not recall or semantic accuracy. The classical output also includes false positives, such as goal/frame edges. It is not used to replace the selected lines in the existing 3D fits. The learned models' original line results remain unchanged.

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
