# 6D Object Pose Estimation: Accurate RGB-D Multi-modal Optimization and Refinement

ARMOR - 6D pose estimation framework for LINEMOD dataset using YOLO detection, ResNet rotation/translation and ICP refinement.

## Pipelines

- **RGB**: YOLO → ResNet (Rotation/Translation)
- **RGBD**: YOLO → ResNet RGBD (Rotation/Translation)
- **ICP**: RGBD → Point Cloud Registration Refinement

Supports 13 LINEMOD objects with ADD/ADD-S metrics (symmetric handling).

## Project Structure

```
├── pipeline_*.py                 # Inference pipelines (RGB, RGBD, ICP, Depth)
├── augmentations/                # Transform classes for training/inference
├── dataset/                      # Dataset loaders & LINEMOD preprocessing
├── geometry/                     # Pinhole camera model, quaternion conversion
├── metrics/                      # ADD/ADD-S metric computation
├── models/
│   ├── yolo/                     # YOLO detection (load.py, train.py)
│   ├── rgb/
│   │   ├── resnetRotation/       # Quaternion prediction
│   │   └── resnetTranslation/    # 3D translation
│   └── rgbd/
│       ├── depthCNN/             # Depth processing
│       ├── resnetRotationRGBD/   # RGBD rotation
│       └── resnetTranslationRGBD/# RGBD translation
├── scripts/                      # Validation utilities
├── utils/                        # Metrics, downloads, CAD loading
├── checkpoints/yolo/             # Pretrained YOLO weights
└── results/                      # Evaluation outputs
```

## Dependencies

```bash
pip install -r requirements.txt
```

Core: torch, torchvision, ultralytics (YOLO), opencv-python, open3d, trimesh, PyYAML, tqdm, numpy, Pillow, matplotlib, plotly, scikit-learn, gdown

## Setup

```bash
# Clone & install
git clone <repo> && cd 6D-Pose-Estimation-main
pip install -r requirements.txt

# Download dataset
python utils/download_linemod.py

# Preprocess for YOLO
python dataset/preprocessing_linemod_to_yolo.py \
    --linemod_root data/linemod/Linemod_preprocessed \
    --output_dir data/linemod_yolo
```

## Training

```bash
# YOLO detector
python models/yolo/train.py --dataset_yaml data/linemod_yolo/data.yml --epochs 50 --batch 16

# RGB Rotation
python models/rgb/resnetRotation/train.py --dataset_path data/linemod_yolo --epochs 30 --batch 64 --freeze_backbone

# RGB Translation  
python models/rgb/resnetTranslation/train.py --dataset_path data/linemod_yolo --epochs 30 --batch 64

# RGBD Rotation
python models/rgbd/resnetRotationRGBD/train.py --dataset_path data/linemod_yolo --epochs 30 --batch 64

# RGBD Translation
python models/rgbd/resnetTranslationRGBD/train.py --dataset_path data/linemod_yolo --epochs 30 --batch 64
```

## Inference

```bash
# RGB pipeline
python pipeline_rgb.py --dataset_root data/linemod_yolo --split val \
    --yolo_weights checkpoints/yolo/weights/best.pt \
    --resnet_rot_weights models/rgb/resnetRotation/weights.pth \
    --resnet_tra_weights models/rgb/resnetTranslation/weights.pth \
    --output_file results_rgb.json

# RGBD pipeline
python pipeline_rgbd.py --dataset_root data/linemod_yolo --split val \
    --yolo_weights checkpoints/yolo/weights/best.pt \
    --resnet_rot_weights models/rgbd/resnetRotationRGBD/weights.pth \
    --resnet_tra_weights models/rgbd/resnetTranslationRGBD/weights.pth \
    --output_file results_rgbd.json

# ICP refinement
python pipeline_icp.py --dataset_root data/linemod_yolo --split val \
    --yolo_weights checkpoints/yolo/weights/best.pt \
    --resnet_rot_weights models/rgbd/resnetRotationRGBD/weights.pth \
    --resnet_tra_weights models/rgbd/resnetTranslationRGBD/weights.pth \
    --icp_max_iter 50 --output_file results_icp.json

# Compute metrics
python utils/compute_pipeline_metrics.py --results results_rgb.json \
    --models_info data/linemod_yolo/models/models_info.yml
```

## LINEMOD Objects (13 Total)

| ID | Object | Type |
|----|--------|------|
| 0/1 | ape | asymmetric |
| 1/2 | benchvise | asymmetric |
| 2/4 | camera | asymmetric |
| 3/5 | can | asymmetric |
| 4/6 | cat | asymmetric |
| 5/8 | driller | asymmetric |
| 6/9 | duck | asymmetric |
| 7/10 | eggbox | **symmetric** |
| 8/11 | glue | **symmetric** |
| 9/12 | holepuncher | asymmetric |
| 10/13 | iron | asymmetric |
| 11/14 | lamp | asymmetric |
| 12/15 | phone | asymmetric |

*First ID = YOLO class, Second ID = LINEMOD object ID*

## Pipeline Architecture

**RGB**: Image → YOLO (BBox) → ResNet Rotation (Quat) → ResNet Translation (XYZ) → Pose + Metrics

**RGBD**: RGB + Depth → YOLO (BBox) → ResNet RGBD Rotation → ResNet RGBD Translation → Pose + Metrics

**ICP**: RGBD Pose → CAD Projection → Depth PointCloud → ICP Refinement → Refined Pose + Metrics

## Key Features

- **Rotation**: Quaternion with geodesic loss + multi-object conditioning
- **Translation**: Pinhole camera model + contextual RGBD fusion
- **Symmetric Handling**: ADD-S metric for eggbox & glue
- **ICP Refinement**: Optional point cloud registration
- **Metrics**: ADD, ADD-S, T/R error analysis per-object

## Output Format

```json
[
  {
    "image_id": "01_0000",
    "obj_id": 1,
    "pred_R": [[...], [...], [...]],
    "pred_t": [x, y, z],
    "gt_R": [[...], [...], [...]],
    "gt_t": [x, y, z],
    "add_error": 12.5,
    "add_metric": 1.0,
    "t_error": 5.2,
    "r_error": 3.8
  }
]
```

Results saved in JSON format. Metrics computed in results/ directory.

## Citation & License

If using this code, cite:
```bibtex
@software{ARMOR,
  title={6D Object Pose Estimation: Accurate RGB-D Multi-modal Optimization and Refinement},
  author={Paduano, Elziario and Pavanati, Marco and Qureshi, Sadaf and Zeqaj, Klejsi},
  year={2024}
}
```

See LICENSE file for details.

## Acknowledgments

- LINEMOD dataset
- Ultralytics (YOLO)
- Open3D (point cloud processing)

- PyTorch & torchvision
