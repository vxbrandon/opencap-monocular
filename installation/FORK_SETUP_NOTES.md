# Fork setup notes (vxbrandon/opencap-monocular)

Verified end-to-end on AWS **g5.2xlarge** (NVIDIA A10G, Ubuntu 22.04 Deep Learning Base GPU AMI)
on 2026-08-12, following [INSTALL_SLIM.md](INSTALL_SLIM.md). Four gaps had to be fixed to get a
fresh clone running — recorded here so the next setup is turn-key.

## 1. ViTPose mmcv version gate rejects mmcv-full 1.7.2

`INSTALL_SLIM.md` installs `mmcv-full==1.7.2`, but the pinned ViTPose submodule asserts
`mmcv <= 1.5.0` at import:

```
AssertionError: MMCV==1.7.2 is used but incompatible. Please install mmcv>=1.3.8, <=1.5.0.
```

Fix (after `pip install -v -e WHAM/third-party/ViTPose`):

```bash
sed -i "s/mmcv_maximum_version = .*/mmcv_maximum_version = '1.8.0'/" \
  WHAM/third-party/ViTPose/mmpose/__init__.py
```

## 2. Wholebody ViTPose checkpoint is not in the fetch script

`WHAM/demo.py::initialize_wham` loads
`checkpoints/vitpose+_huge_wholebody/wholebody.pth` with the COCO-WholeBody (133 kpt) config,
but `WHAM/fetch_demo_data.sh` only downloads `vitpose-h-multi-coco.pth`. Without it the API/worker
crashes at startup with `FileNotFoundError`.

Working mirror (original ViTPose+ huge wholebody weights, 2.5 GB):

```bash
mkdir -p WHAM/checkpoints/vitpose+_huge_wholebody
curl -L "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/wholebody/vitpose-h-wholebody.pth" \
  -o WHAM/checkpoints/vitpose+_huge_wholebody/wholebody.pth
```

## 3. SMPL body models

`WHAM/fetch_demo_data.sh` requires interactive MPI credentials (smpl.is.tue.mpg.de /
smplify.is.tue.mpg.de). For non-interactive setup the SMPL 1.0 pkls are mirrored on
HuggingFace (non-commercial research use only, per the SMPL license):

```bash
mkdir -p WHAM/dataset/body_models/smpl
for m in NEUTRAL MALE FEMALE; do
  curl -L "https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL_${m}.pkl" \
    -o WHAM/dataset/body_models/smpl/SMPL_${m}.pkl
done
```

## 4. `opencap-visualizer` missing from requirements_slim.txt

`visualization/automation.py` imports `opencap_visualizer` to record `viewer_mono.webm`,
but it is not in `installation/requirements_slim.txt`. It also needs a Playwright browser:

```bash
pip install opencap-visualizer
playwright install chromium
```

## 5. ffmpeg/ffprobe is a hard system prerequisite

`utils/utilsCameraPy3.py::getVideoRotation` shells out to `ffprobe` before WHAM runs; minimal
Ubuntu images (incl. the AWS Deep Learning Base GPU AMI) don't ship it:

```bash
sudo apt-get install -y ffmpeg
```

## 6. SMPLX→SMPL transfer matrix path mismatch

`utils/opensim/defaults.py` reads `WHAM/dataset/model_transfer/smplx_to_smpl.pkl`, but
`fetch_demo_data.sh`'s auxiliary `body_models.tar.gz` delivers the identical file (dict with
`matrix` of shape `(6890, 10475)`) as `WHAM/dataset/body_models/smplx2smpl.pkl`. Without the
link, `smpl_to_trc` crashes after WHAM completes:

```bash
mkdir -p WHAM/dataset/model_transfer
ln -s "$(pwd)/WHAM/dataset/body_models/smplx2smpl.pkl" \
      WHAM/dataset/model_transfer/smplx_to_smpl.pkl
```

---

`installation/bootstrap_fork.sh` applies all of these in one shot after the standard
INSTALL_SLIM steps.
