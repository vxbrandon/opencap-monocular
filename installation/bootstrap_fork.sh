#!/bin/bash
# Applies the four post-INSTALL_SLIM fixes documented in FORK_SETUP_NOTES.md.
# Run from the repo root, inside the conda env, after `pip install -e WHAM/third-party/ViTPose`.
set -euo pipefail

# 1. Relax ViTPose's mmcv version gate (INSTALL_SLIM uses mmcv-full 1.7.2)
sed -i "s/mmcv_maximum_version = .*/mmcv_maximum_version = '1.8.0'/" \
  WHAM/third-party/ViTPose/mmpose/__init__.py

# 2. Wholebody ViTPose+ huge checkpoint (loaded by WHAM/demo.py, absent from fetch_demo_data.sh)
mkdir -p WHAM/checkpoints/vitpose+_huge_wholebody
[ -s WHAM/checkpoints/vitpose+_huge_wholebody/wholebody.pth ] || \
  curl -L "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/wholebody/vitpose-h-wholebody.pth" \
    -o WHAM/checkpoints/vitpose+_huge_wholebody/wholebody.pth

# 3. SMPL body models (non-commercial research use only — see SMPL license)
mkdir -p WHAM/dataset/body_models/smpl
for m in NEUTRAL MALE FEMALE; do
  [ -s "WHAM/dataset/body_models/smpl/SMPL_${m}.pkl" ] || \
    curl -L "https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL_${m}.pkl" \
      -o "WHAM/dataset/body_models/smpl/SMPL_${m}.pkl"
done

# 4. Viewer-video recorder (missing from requirements_slim.txt)
pip install opencap-visualizer
playwright install chromium

# 5. ffprobe is required by utils/utilsCameraPy3.py before WHAM runs
command -v ffprobe >/dev/null || sudo apt-get install -y ffmpeg

echo "bootstrap_fork.sh: all fixes applied."
