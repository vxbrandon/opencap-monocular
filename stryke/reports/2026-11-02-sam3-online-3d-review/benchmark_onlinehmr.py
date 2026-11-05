"""Causal camera-coordinate OnlineHMR smoke with current-frame YOLO boxes.

This measures a substituted detector + the author's body branch, not its full
world-coordinate SLAM system, phone latency, or independently labelled accuracy.
Requires a prepared upstream checkout, licensed SMPL assets and local weights.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def legacy_smpl_pickle_compatibility(np):
    """Scope Chumpy's removed Python/NumPy aliases to licensed asset loading."""
    aliases = {'bool': np.bool_, 'int': np.int_, 'float': np.float64,
               'complex': np.complex128, 'object': np.object_,
               'str': np.str_, 'unicode': np.str_}
    added = [name for name in aliases if name not in np.__dict__]
    added_inspect = 'getargspec' not in inspect.__dict__
    try:
        for name in added:
            setattr(np, name, aliases[name])
        if added_inspect:
            inspect.getargspec = inspect.getfullargspec
        yield
    finally:
        for name in added:
            delattr(np, name)
        if added_inspect:
            delattr(inspect, 'getargspec')


def select_subject(boxes, previous):
    """Require a unique overlap match to a supplied seed or preceding box."""
    if previous is None:
        return 0 if len(boxes) == 1 else None
    overlaps = []
    for index, box in enumerate(boxes):
        intersection = max(0, min(box[2], previous[2]) - max(box[0], previous[0])) * max(
            0, min(box[3], previous[3]) - max(box[1], previous[1]))
        area = (box[2] - box[0]) * (box[3] - box[1])
        previous_area = (previous[2] - previous[0]) * (previous[3] - previous[1])
        overlaps.append((intersection / max(area + previous_area - intersection, 1e-9), index))
    overlaps.sort(reverse=True)
    if not overlaps or overlaps[0][0] < .25:
        return None
    if len(overlaps) > 1 and overlaps[1][0] >= .5 * overlaps[0][0]:
        return None
    return overlaps[0][1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    for key in ('repo', 'checkpoint', 'detector', 'manifest', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    if not 1 <= args.repeats <= 5:
        parser.error('repeats must be between one and five')
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))

    import cv2
    import numpy as np
    import torch
    from lib.models import get_default_config
    from lib.models.hmr_vimo import HMR_VIMO
    from lib.utils.imutils import boxes_2_cs, crop, get_normalization
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError('This measured configuration requires CUDA')
    torch.set_num_threads(2)
    manifest = json.loads(args.manifest.read_text())
    report = {
        'model': 'OnlineHMR', 'scope': 'causal_camera_coordinate_body_with_yolo11n',
        'world_slam_included': False, 'detector_substituted': True,
        'independent_accuracy_measured': False, 'phone_latency_measured': False,
        'crop_dilation': 1.2, 'body_batch_size': 1, 'future_image_frames_used': 0,
        'joints_coordinate_convention': 'SMPL coordinates; add trans_full for camera coordinates',
        'source_sha256': digest(__file__), 'checkpoint_sha256': digest(args.checkpoint),
        'detector_sha256': digest(args.detector), 'manifest_sha256': digest(args.manifest),
        'upstream_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'upstream_patch': subprocess.check_output(['git', 'diff'], text=True),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'runs': [],
        'clock': 'CUDA-synchronized wall time; model loading separate; decode included in frame total',
    }
    started = time.perf_counter()
    with legacy_smpl_pickle_compatibility(np):
        model = HMR_VIMO(get_default_config()).eval().cuda()
    # The official checkpoint includes NumPy scalar training metadata.
    with torch.serialization.safe_globals([
        np.core.multiarray.scalar, np.dtype, type(np.dtype(np.float64)),
    ]):
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    mismatch = model.load_state_dict(checkpoint['model'], strict=False)
    if mismatch.missing_keys or mismatch.unexpected_keys:
        raise RuntimeError(f'Checkpoint mismatch: {mismatch}')
    del checkpoint
    detector = YOLO(str(args.detector))
    if detector.names.get(0) != 'person':
        raise ValueError('Expected COCO person detector')
    normalize = get_normalization()
    torch.cuda.synchronize()
    report['initialization_seconds'] = time.perf_counter() - started

    for repeat in range(args.repeats):
        for clip in manifest['clips']:
            source = Path(clip['path'])
            source_hash = digest(source)
            run_dir = args.output / f"{clip['id']}-{repeat}"
            run_dir.mkdir()
            probe = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(source),
            ], text=True))
            pts = [float(row['best_effort_timestamp_time']) for row in probe['frames']]
            if any(b <= a for a, b in itertools.pairwise(pts)):
                raise ValueError('Non-increasing original timestamps')
            cap = cv2.VideoCapture(str(source))
            cache, index, rows, estimates = None, 0, [], []
            seed = clip.get('initial_player_bbox_xyxy')
            previous_box, subject_lost = seed, False
            torch.cuda.reset_peak_memory_stats()
            run_start = time.perf_counter()
            with torch.inference_mode():
                while True:
                    frame_start = time.perf_counter()
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if index >= len(pts):
                        raise ValueError('Decoded frame count exceeds timestamp count')
                    row = {'frame': index, 'source_pts_s': pts[index]}
                    t = time.perf_counter()
                    boxes = detector.predict(frame, classes=[0], conf=.5, imgsz=640,
                                             device=0, verbose=False)[0].boxes.xyxy.cpu().numpy()
                    torch.cuda.synchronize()
                    row['detector_seconds'] = time.perf_counter() - t
                    selected = None if subject_lost else select_subject(boxes, previous_box)
                    row['person_detections'] = len(boxes)
                    if selected is None:
                        row['status'] = 'missing_or_ambiguous_person'
                        cache = None
                        # Once a seeded subject is lost, do not silently choose a replacement.
                        subject_lost = seed is not None
                    else:
                        selected_box = boxes[selected]
                        if seed is not None:
                            previous_box = selected_box
                        centers, scales = boxes_2_cs(boxes[selected:selected + 1])
                        # Match the author's inference_chunk_ar -> TrackDataset preprocessing.
                        center, scale = centers[0], scales[0] * 1.2
                        t = time.perf_counter()
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = normalize(crop(rgb, center, scale, [256, 256], rot=0).astype('uint8'))
                        height, width = frame.shape[:2]
                        intr = clip.get('metadata', {}).get('intrinsics', {})
                        focal = float(intr.get('fx', np.hypot(height, width)))
                        principal = np.array([intr.get('cx', width / 2), intr.get('cy', height / 2)])
                        batch = {'img': image[None].cuda(),
                                 'center': torch.tensor(center)[None].float().cuda(),
                                 'scale': torch.tensor([scale]).float().cuda(),
                                 'img_focal': torch.tensor([focal]).float().cuda(),
                                 'img_center': torch.tensor(principal)[None].float().cuda()}
                        out, predictions, cache = model.inference_forward(batch, cache=cache)
                        torch.cuda.synchronize()
                        row['crop_and_body_seconds'] = time.perf_counter() - t
                        row['cache_tokens_per_layer'] = [layer['mem_k'].shape[1]
                                                        for layer in cache['layers']]
                        arrays = {name: out[name].detach().float().cpu().numpy()[0]
                                  for name in ('pred_rotmat', 'pred_shape', 'pred_cam', 'trans_full')}
                        arrays['joints'] = predictions[3][-1].float().cpu().numpy()[0]
                        arrays['projected_joints_crop_pixels'] = predictions[4][-1].float().cpu().numpy()[0]
                        full_projection, _ = model.project(
                            predictions[3][-1], out['pred_cam'], batch['center'],
                            batch['scale'], batch['img_focal'], batch['img_center'],
                            return_full=True)
                        arrays['projected_joints_source_pixels'] = full_projection.float().cpu().numpy()[0]
                        if not all(np.isfinite(value).all() for value in arrays.values()):
                            raise ValueError('Non-finite fitted geometry')
                        arrays['frame'] = index
                        arrays['source_pts_s'] = pts[index]
                        estimates.append(arrays)
                        row.update(status='estimated', bbox=selected_box.tolist())
                        if index in (0, len(pts) // 2, len(pts) - 1):
                            np.savez_compressed(run_dir / f'frame-{index:04d}.npz', **arrays)
                            cv2.imwrite(str(run_dir / f'frame-{index:04d}.jpg'), frame)
                            overlay = frame.copy()
                            for x, y in arrays['projected_joints_source_pixels']:
                                if 0 <= x < width and 0 <= y < height:
                                    cv2.circle(overlay, (int(x), int(y)), 4, (0, 255, 255), -1)
                            cv2.imwrite(str(run_dir / f'overlay-{index:04d}.jpg'), overlay)
                    row['total_seconds'] = time.perf_counter() - frame_start
                    rows.append(row)
                    index += 1
            cap.release()
            elapsed = time.perf_counter() - run_start
            if index != len(pts) or digest(source) != source_hash:
                raise ValueError('Incomplete decode or changed source')
            if estimates:
                np.savez_compressed(run_dir / 'body.npz', **{
                    key: np.stack([item[key] for item in estimates]) for key in estimates[0]
                })
            summary = {'clip': clip['id'], 'repeat': repeat, 'source_sha256': source_hash,
                       'frames': index, 'estimated_frames': len(estimates), 'seconds': elapsed,
                       'processing_fps': index / elapsed, 'first_frame_seconds': rows[0]['total_seconds'],
                       'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'subject_selection': 'seeded_previous_box_iou' if seed else 'exactly_one_person',
                       'intrinsics_source': 'manifest' if clip.get('metadata', {}).get('intrinsics')
                       else 'image_diagonal_focal_heuristic',
                       'input_manifest_entry': clip, 'rows': rows}
            (run_dir / 'timing.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
            report['runs'].append({k: v for k, v in summary.items() if k != 'rows'})
            (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
            print(json.dumps({k: v for k, v in summary.items() if k not in ('rows', 'input_manifest_entry')}), flush=True)


if __name__ == '__main__':
    main()
