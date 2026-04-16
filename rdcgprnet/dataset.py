import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class NYUDepthV2Dataset(Dataset):
    RGB_MEAN = [0.485, 0.456, 0.406]
    RGB_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str,
        shot: int = 1,
        img_size: int = 224,
        split: str = 'train',
        depth_max: float = 10.0,
        depth_fg_threshold: float = 0.4,
        min_images_per_class: int = 5,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.shot = shot
        self.img_size = img_size
        self.split = split
        self.depth_max = depth_max
        self.depth_fg_thr = depth_fg_threshold
        self.min_img_per_class = min_images_per_class

        self.rgb_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.RGB_MEAN, self.RGB_STD),
        ])

        csv_name = f'nyu2_{split}.csv'
        csv_path = self.root / csv_name
        if not csv_path.exists():
            csv_path = self.root / 'data' / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f'CSV not found: {csv_path}')

        self.pairs = self._load_csv(csv_path)
        self.class_map = self._build_class_map()
        self.episodes = self._build_episodes()

    def _load_csv(self, csv_path: Path) -> List[Tuple[Path, Path]]:
        pairs = []
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                rgb_rel = row[0].strip()
                depth_rel = row[1].strip()
                rgb_path = self.root / rgb_rel
                depth_path = self.root / depth_rel
                if not rgb_path.exists():
                    rgb_path = self.root / 'data' / rgb_rel
                    depth_path = self.root / 'data' / depth_rel
                if rgb_path.exists():
                    pairs.append((rgb_path, depth_path))
        return pairs

    def _get_class(self, rgb_path: Path) -> str:
        scene = rgb_path.parent.name
        parts = scene.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return scene

    def _build_class_map(self) -> Dict[str, List[int]]:
        cls_map: Dict[str, List[int]] = defaultdict(list)
        for i, (rgb_path, _) in enumerate(self.pairs):
            cls = self._get_class(rgb_path)
            cls_map[cls].append(i)
        return {cls: idxs for cls, idxs in cls_map.items() if len(idxs) >= self.min_img_per_class}

    def _build_episodes(self) -> List[Dict]:
        episodes = []
        for cls, idxs in self.class_map.items():
            for qi in idxs:
                pool = [i for i in idxs if i != qi]
                if len(pool) >= self.shot:
                    episodes.append({'class': cls, 'query': qi, 'support': pool[:self.shot]})
        return episodes

    def _load_rgb(self, idx: int) -> torch.Tensor:
        path = self.pairs[idx][0]
        return self.rgb_tf(Image.open(path).convert('RGB'))

    def _load_depth(self, idx: int) -> torch.Tensor:
        path = self.pairs[idx][1]
        try:
            dep = Image.open(path)
            dep = dep.resize((self.img_size, self.img_size), Image.NEAREST)
            arr = np.array(dep, dtype=np.float32)
            if arr.max() > 100:
                arr = arr / 65535.0 * self.depth_max
            arr = arr / self.depth_max
            arr = np.clip(arr, 0.0, 1.0)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            return torch.from_numpy(arr).unsqueeze(0)
        except Exception:
            return torch.zeros(1, self.img_size, self.img_size)

    def _depth_to_mask(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        mask = (depth_tensor < self.depth_fg_thr).float()
        fg = mask.mean().item()
        if fg < 0.05 or fg > 0.95:
            arr = depth_tensor.squeeze().numpy()
            vals = arr[arr > 0]
            if len(vals) > 0:
                thr = float(np.percentile(vals, 35))
                mask = (depth_tensor < thr).float()
        return mask

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict:
        ep = self.episodes[idx]
        q_rgb = self._load_rgb(ep['query'])
        q_depth = self._load_depth(ep['query'])
        q_mask = self._depth_to_mask(q_depth)

        s_rgbs, s_depths, s_masks = [], [], []
        for si in ep['support']:
            sr = self._load_rgb(si)
            sd = self._load_depth(si)
            sm = self._depth_to_mask(sd)
            s_rgbs.append(sr)
            s_depths.append(sd)
            s_masks.append(sm)

        return {
            'query_rgb': q_rgb,
            'query_depth': q_depth,
            'query_mask': q_mask,
            'support_rgb': torch.stack(s_rgbs),
            'support_depth': torch.stack(s_depths),
            'support_masks': torch.stack(s_masks),
            'class': ep['class'],
        }
