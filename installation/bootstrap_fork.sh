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

# 5b (optional). pytorch3d for WHAM's mesh-overlay renderer (visualize=True).
#    Needs a CUDA toolchain matching torch (cu118); the conda one works even when
#    the system nvcc is newer. ~10 min build on 8 cores. Tag is lowercase v0.7.5.
#    conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
#    CUDA_HOME=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH MAX_JOBS=8 FORCE_CUDA=1 \
#      TORCH_CUDA_ARCH_LIST="8.6" pip install --no-build-isolation \
#      "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"

# 6. utils/opensim/defaults.py expects the SMPLX->SMPL transfer matrix at
#    WHAM/dataset/model_transfer/smplx_to_smpl.pkl; fetch_demo_data.sh delivers the
#    same file as WHAM/dataset/body_models/smplx2smpl.pkl (dict with 'matrix' (6890,10475)).
mkdir -p WHAM/dataset/model_transfer
[ -e WHAM/dataset/model_transfer/smplx_to_smpl.pkl ] || \
  ln -s "$(pwd)/WHAM/dataset/body_models/smplx2smpl.pkl" WHAM/dataset/model_transfer/smplx_to_smpl.pkl

echo "bootstrap_fork.sh: all fixes applied."
