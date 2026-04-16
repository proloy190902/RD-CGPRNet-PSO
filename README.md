# RD-CGPRNet-PSO

**RD-CGPRNet-PSO: RGB-D Cross-Scale Graph Prototype Refinement Network with PSO-based Heterogeneous Modal Fusion** 

## 🧠 Framework Overview

The overall architecture of **RD-CGPRNet-PSO** is illustrated below. The model integrates RGB-D feature extraction, modality reliability estimation, cross-modal graph prototype learning, PSO-based adaptive fusion, and multi-scale matching for robust segmentation.

![Architecture](https://github.com/proloy190902/RD-CGPRNet-PSO/blob/52f32c310564884ff10f1ec1e9e05f3b650509e7/RD-CGPRNet_PSO.png)

## Repository Structure

```bash
rd-cgprnet-pso/
├── rdcgprnet/
│   ├── __init__.py
│   ├── dataset.py
│   ├── encoders.py
│   ├── metrics.py
│   ├── model.py
│   ├── modules.py
│   ├── trainer.py
│   └── utils.py
├── scripts/
│   └── train.py
├── configs/
│   └── nyu_depth_v2_example.sh
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
git clone <your-repo-url>
cd rd-cgprnet-pso
pip install -r requirements.txt
```

## Dataset Layout

Expected NYU Depth V2 layout:

```bash
nyu_data/
├── nyu2_train.csv
├── nyu2_test.csv
└── data/
    ├── nyu2_train/
    └── nyu2_test/
```

## Training

```bash
python scripts/train.py   --data_root /path/to/nyu_data   --shot 1   --backbone resnet50   --epochs 50   --batch_size 8   --save_dir ./checkpoints
```

## Evaluation

```bash
python scripts/train.py   --mode eval   --data_root /path/to/nyu_data   --checkpoint ./checkpoints/best_resnet50_1shot.pth
```

## Ablation

```bash
python scripts/train.py   --mode ablation   --data_root /path/to/nyu_data   --checkpoint ./checkpoints/best_resnet50_1shot.pth
```

## Suggested Git Commands

```bash
git init
git add .
git commit -m "Initial commit: organized RD-CGPRNet-PSO codebase"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```



## Original Single-File Source

The original pasted implementation is preserved here for reference:
- `original_source.txt`
