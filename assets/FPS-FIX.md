# Video frame-rate correction — 2026-08-31

The pipeline's own timestamps (`mono.json` `time[]`, `joint_series.json` `t[]`) give the true capture
rate: 202 samples over 3.7925 s = **53.0 fps**.

`1_incam.mp4` and `2_global.mp4` were written with a 30 fps header, so their 264 frames played over
8.80 s — **1.77x slower than real time**. That is the "slow video" effect; the model was never slow.

Fixed by re-timing the container only (demux to raw H.264, remux at 53 fps): every frame kept, no
re-encode, 8.80 s -> 4.92 s. `wham_vs_gvhmr.mp4` carried a nonsense 1e6 fps header and was rebuilt at
53 fps. Originals preserved in `assets-original-fps/` (untracked).

Correct as shipped: `mesh_skeleton_3d.mp4`, `skeleton_3d.mp4` (53), `keypoints_overlay.mp4`,
`stryke_hologram.mp4`, `stryke_mesh.mp4`, `mesh_overlay_wham.mp4` (~52.6).
