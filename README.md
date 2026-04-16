# RD-CGPRNet-PSO

GitHub-ready organized implementation of **RD-CGPRNet-PSO: RGB-D Cross-Scale Graph Prototype Refinement Network with PSO-based Heterogeneous Modal Fusion** for NYU Depth V2 few-shot RGB-D segmentation.

This repository was organized from the user-provided single-file implementation. Source basis: fileciteturn0file0

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

## Notes

- I kept the architecture and logic close to the original one-file code.
- I reorganized the code into reusable modules for easier maintenance and GitHub presentation.
- Before public release, you may want to add:
  - sample outputs
  - training logs
  - qualitative figures
  - license
  - citation block

## Original Single-File Source

The original pasted implementation is preserved here for reference:
- `original_source.txt`
