<p align="center">
  <h1 align="center"> RGB-Only Gaussian Splatting SLAM for Unbounded Outdoor Scenes
  </h1>
  <p align="center">
    <a ><strong>Sicheng Yu*</strong></a>
    ·
    <a ><strong>Chong Cheng*</strong></a>
    ·
    <a ><strong>Yifan Zhou</strong></a>
    ·
    <a ><strong>Xiaojun Yang</strong></a>
    ·
    <a href="https://wanghao.tech//"><strong>Hao Wang✉</strong></a>
  </p>
  <p align="center">The Hong Kong University of Science and Technology (GuangZhou)</p>
  <p align="center">(* Equal Contribution)</p>

  <h3 align="center"> ICRA 2025</h3>

[[Project page](https://3dagentworld.github.io/opengs-slam/)],[[arxiv](https://arxiv.org/abs/2502.15633)]  

⭐Please also check out our futher work[[ICCV 2025] S3PO-GS](https://github.com/3DAgentWorld/S3PO-GS)，which also focus on RGB-Only outdoor SLAM with better performance.

# Getting Started

## Installation

1. Clone OpenGS-SLAM.

```bash
git clone https://github.com/nhatsonle/opengs_SLAM.git --recursive
cd OpenGS-SLAM
```

2. Setup the environment.

```bash
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 build-essential git
pip install numpy==1.24.4 scipy matplotlib pandas networkx tqdm pyyaml opencv-python ffmpeg \
            pycolmap einops evo plyfile open3d trimesh roma numpy-quaternion \
            lpips tensorboard wandb pyopengl pyrender huggingface-hub cvxpy munch glfw imgviz pyglm==2.7.1 torchmetrics==1.4.0.post0
```

3. Compile submodules for Gaussian splatting

```bash
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
```

4. Compile the cuda kernels for RoPE (as in CroCo v2 and DUSt3R).

```bash
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

Our test setup was:

- Ubuntu 20.04: `pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cudatoolkit=11.8`
- NVIDIA RTX A6000

## Checkpoints

You can download the *'<u>DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth</u>'* checkpoint from the [DUSt3R](https://github.com/naver/dust3r) code repository, and save it to the 'checkpoints' folder.

Alternatively, download it directly using the following method:

```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
```

Please note that you must agree to the DUSt3R license when using it.

## Downloading Datasets

The processed data for the 9 Waymo segments can be downloaded via [baidu](https://pan.baidu.com/s/1I1rnB6B8k2d4wzcRMT6gjA?pwd=omcg ) or [google](https://drive.google.com/drive/folders/1xUyNuNzUtsvZIV_q5Qz9zIXMGoMbLuCr?usp=sharing).

Save data under the `datasets/waymo` directory.

## Run

```bash
## Taking 100613 as an example
CUDA_VISIBLE_DEVICES=0 python slam.py --config configs/mono/waymo/100613.yaml

## All 9 Waymo scenes
bash run_waymo.sh
```

## Depth-Aided Tracking Variant

This fork adds a depth-aided tracking path for the processed Waymo data. Instead of running DUSt3R on every consecutive RGB pair, the frontend now initializes each non-keyframe pose with a constant-velocity prior and refines it with the existing photometric tracking loss. Waymo depth maps are back-projected into a small render-only local Gaussian buffer to support tracking, while non-keyframe depth Gaussians are not committed to the global map.

DUSt3R is still used for keyframes. When overlap becomes low enough to create a keyframe, the keyframe RGB pair is passed through DUSt3R, and the resulting pointmap is inserted into the global Gaussian map. If DUSt3R keyframe insertion fails, the backend falls back to the existing RGB-D/depth back-projection path.

The default behavior is controlled in `configs/mono/waymo/base_config.yaml`:

```yaml
Tracking:
  pose_init: "constant_velocity"
  use_dust3r_every_frame: False
  wait_for_keyframe_backend: True
  depth_tracking_buffer: 3
  use_depth_local_map: True
  depth_min: 0.1
  depth_max: 80.0
```

Quick validation commands:

```bash
python -m py_compile slam.py utils/slam_frontend.py utils/slam_backend.py gaussian_splatting/scene/gaussian_model.py

CUDA_VISIBLE_DEVICES=0 python slam.py --config configs/mono/waymo/405841.yaml
```

For a faster smoke run, temporarily reduce `init_itr_num`, `tracking_itr_num`, and `mapping_itr_num`, and disable `eval_rendering` and `color_refinement` in the config.

See `docs/depth_aided_architecture.md` for the implementation details and the updated system architecture.

## Demo

- If you want to view the real-time interactive SLAM window, please change `Results-use_gui` in `base_config.yaml` to True.

- When running on an Ubuntu system, a GUI window will pop up.

## Run on other dataset

- Please organize your data format and modify the code in `utils/dataset.py`.

- The Waymo path now uses the retained depth input interface for depth-aided tracking and depth fallback mapping. For new datasets, provide metric depth maps through `utils/dataset.py` if you want to use this variant.

# Acknowledgement

- This work is built on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting),  [MonoGS](https://github.com/muskie82/MonoGS),  and [DUSt3R](https://github.com/naver/dust3r), thanks for these great works.

- For more details about Demo, please refer to [MonoGS](https://github.com/muskie82/MonoGS), as we are using its visualization code.

# Citation

If you found our code/work to be useful in your own research, please considering citing the following:

```bibtex
@article{yu2025rgb,
  title={Rgb-only gaussian splatting slam for unbounded outdoor scenes},
  author={Yu, Sicheng and Cheng, Chong and Zhou, Yifan and Yang, Xiaojun and Wang, Hao},
  journal={arXiv preprint arXiv:2502.15633},
  year={2025}
}

@article{cheng2025outdoor,
  title={Outdoor Monocular SLAM with Global Scale-Consistent 3D Gaussian Pointmaps},
  author={Cheng, Chong and Yu, Sicheng and Wang, Zijian and Zhou, Yifan and Wang, Hao},
  journal={arXiv preprint arXiv:2507.03737},
  year={2025}
}
```
