import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .metrics import Metrics
from .model import RDCGPRNetPSO


class Trainer:
    def __init__(self, model: RDCGPRNetPSO, args):
        self.model = model
        self.args = args
        self.device = next(model.parameters()).device
        trainable = [p for n, p in model.named_parameters() if 'generic_branch' not in n and p.requires_grad]
        self.opt = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=args.epochs)
        os.makedirs(args.save_dir, exist_ok=True)

    def _unpack(self, batch):
        dev = self.device
        k = batch['support_rgb'].shape[1]
        return (
            batch['query_rgb'].to(dev),
            batch['query_depth'].to(dev),
            batch['query_mask'].to(dev),
            [batch['support_rgb'][:, i].to(dev) for i in range(k)],
            [batch['support_depth'][:, i].to(dev) for i in range(k)],
            [batch['support_masks'][:, i].to(dev) for i in range(k)],
        )

    def train_epoch(self, loader, epoch):
        self.model.train()
        losses, ious = [], []
        for step, batch in enumerate(loader):
            q_rgb, q_dep, q_msk, s_rgb, s_dep, s_msk = self._unpack(batch)
            self.opt.zero_grad()
            out = self.model(q_rgb, q_dep, s_rgb, s_dep, s_msk)
            loss = RDCGPRNetPSO.compute_loss(out['pred'], q_msk, out['R_rgb'], out['R_depth'])
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            losses.append(loss.item())
            ious.append(Metrics.iou(out['pred'].detach(), q_msk))
            if step % 50 == 0:
                print(f'Ep{epoch:3d} | step{step:4d} | loss {loss.item():.4f} | iou {ious[-1]:.4f}')
        self.sched.step()
        return {'loss': np.mean(losses), 'iou': np.mean(ious)}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        ious, dices, fbs, precs = [], [], [], []
        for batch in loader:
            q_rgb, q_dep, q_msk, s_rgb, s_dep, s_msk = self._unpack(batch)
            out = self.model(q_rgb, q_dep, s_rgb, s_dep, s_msk, query_mask=q_msk)
            pred = out['pred']
            ious.append(Metrics.iou(pred, q_msk))
            dices.append(Metrics.dice(pred, q_msk))
            fbs.append(Metrics.fb_iou(pred, q_msk))
            precs.append(Metrics.precision(pred, q_msk))
        return {'iou': np.mean(ious), 'dice': np.mean(dices), 'fb': np.mean(fbs), 'prec': np.mean(precs)}

    def run(self, train_loader, val_loader):
        best_iou = 0.0
        for epoch in range(1, self.args.epochs + 1):
            tr = self.train_epoch(train_loader, epoch)
            vl = self.evaluate(val_loader)
            print(f"Ep{epoch:3d}/{self.args.epochs} | loss={tr['loss']:.4f} iou={tr['iou']:.4f} | val iou={vl['iou']:.4f} dice={vl['dice']:.4f} fb={vl['fb']:.4f} prec={vl['prec']:.4f}")
            if vl['iou'] > best_iou:
                best_iou = vl['iou']
                ckpt = os.path.join(self.args.save_dir, f'best_{self.args.backbone}_{self.args.shot}shot.pth')
                torch.save({'epoch': epoch, 'state_dict': self.model.state_dict(), 'val': vl, 'args': vars(self.args)}, ckpt)
                print(f'Best checkpoint saved to {ckpt} (iou={best_iou:.4f})')
        print(f'Done. Best IoU = {best_iou:.4f}')


class AblationRunner:
    CONFIGS = {
        'Full RD-CGPRNet-PSO': dict(use_pso=True, num_iter=2),
        'w/o PSO (MRE weights only)': dict(use_pso=False, num_iter=2),
        'w/o Graph Refine': dict(use_pso=True, num_iter=0),
    }

    def __init__(self, args, val_loader, device):
        self.args = args
        self.loader = val_loader
        self.device = device

    @torch.no_grad()
    def _eval(self, model):
        model.eval()
        ious, dices, fbs = [], [], []
        for batch in self.loader:
            dev = self.device
            k = batch['support_rgb'].shape[1]
            q_rgb = batch['query_rgb'].to(dev)
            q_dep = batch['query_depth'].to(dev)
            q_msk = batch['query_mask'].to(dev)
            s_rgb = [batch['support_rgb'][:, i].to(dev) for i in range(k)]
            s_dep = [batch['support_depth'][:, i].to(dev) for i in range(k)]
            s_msk = [batch['support_masks'][:, i].to(dev) for i in range(k)]
            out = model(q_rgb, q_dep, s_rgb, s_dep, s_msk, query_mask=q_msk)
            ious.append(Metrics.iou(out['pred'], q_msk))
            dices.append(Metrics.dice(out['pred'], q_msk))
            fbs.append(Metrics.fb_iou(out['pred'], q_msk))
        return np.mean(ious), np.mean(dices), np.mean(fbs)

    def run(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        print(f"{'Config':<32} {'mIoU':>8} {'Dice':>8} {'FB-IoU':>8} {'Δ':>8}")
        base_iou = None
        for name, cfg in self.CONFIGS.items():
            model = RDCGPRNetPSO(
                backbone=self.args.backbone,
                embed_dim=self.args.embed_dim,
                num_iter=cfg['num_iter'],
                hidden_dim=self.args.hidden_dim,
                use_pso=cfg['use_pso'],
            ).to(self.device)
            model.load_state_dict(ckpt['state_dict'], strict=False)
            mi, md, mf = self._eval(model)
            if base_iou is None:
                base_iou = mi
            d = mi - base_iou
            ds = 'base' if d == 0 else f'{d:+.4f}'
            print(f'{name:<32} {mi:>8.4f} {md:>8.4f} {mf:>8.4f} {ds:>8}')
