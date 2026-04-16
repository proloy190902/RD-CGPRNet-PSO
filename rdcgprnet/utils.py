import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auto_detect_data_root() -> str:
    candidates = [
        '/kaggle/input/datasets/soumikrakshit/nyu-depth-v2/nyu_data',
        '/kaggle/input/nyu-depth-v2/nyu_data',
        '/content/nyu_data',
        './data/nyu_data',
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    for base in ['/kaggle/input', '/content', '.']:
        if not os.path.exists(base):
            continue
        for root, dirs, files in os.walk(base):
            if any(f.endswith('.csv') and 'nyu2' in f for f in files):
                return root
            if root.count(os.sep) - base.count(os.sep) > 4:
                dirs.clear()
    return './data'


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='RD-CGPRNet-PSO | NYU Depth V2')
    p.add_argument('--data_root', default=None)
    p.add_argument('--shot', type=int, default=1)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--depth_max', type=float, default=10.0)
    p.add_argument('--depth_fg_thr', type=float, default=0.4)
    p.add_argument('--backbone', default='resnet50', choices=['resnet50', 'resnet101', 'vgg16'])
    p.add_argument('--embed_dim', type=int, default=256)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_iter', type=int, default=2)
    p.add_argument('--use_pso', action='store_true', default=True)
    p.add_argument('--pso_particles', type=int, default=30)
    p.add_argument('--pso_iters', type=int, default=40)
    p.add_argument('--mode', default='train', choices=['train', 'eval', 'ablation'])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_dir', default='./checkpoints')
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--seed', type=int, default=42)
    return p
