import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import RGBEncoder, DepthEncoder
from .modules import (
    ModalityReliabilityEstimator,
    CrossModalGPG,
    ModalFusionModule,
    MultiScaleMatching,
    SegmentationDecoder,
)


class RDCGPRNetPSO(nn.Module):
    def __init__(self, backbone='resnet50', embed_dim=256, num_iter=2, hidden_dim=128, use_pso=True, pso_particles=30, pso_iters=40):
        super().__init__()
        c = embed_dim
        self.rgb_encoder = RGBEncoder(backbone, c)
        self.depth_encoder = DepthEncoder(c)
        self.mre = ModalityReliabilityEstimator(c)
        self.cm_gpg = CrossModalGPG(c, num_iter, hidden_dim)
        self.modal_fusion = ModalFusionModule(c, use_pso, pso_particles, pso_iters)
        self.ms_match = MultiScaleMatching(c)
        self.decoder = SegmentationDecoder(c)

    def forward(self, query_rgb, query_depth, support_rgb, support_depth, support_masks, query_mask=None):
        h, w = query_rgb.shape[-2:]
        f_q_rgb = self.rgb_encoder(query_rgb)
        f_q_dep = self.depth_encoder(query_depth)
        f_s_rgb = [self.rgb_encoder(s) for s in support_rgb]
        f_s_dep = [self.depth_encoder(s) for s in support_depth]

        r_rgb, r_depth = self.mre(f_q_rgb, f_q_dep)
        p_rgb_tilde, p_dep_tilde, f_q_att = self.cm_gpg(f_q_rgb, f_q_dep, f_s_rgb, f_s_dep, support_masks, r_rgb, r_depth)
        f_fused, pso_w = self.modal_fusion(
            f_q_rgb, f_q_dep, r_rgb, r_depth,
            p_rgb_tilde.mean(dim=1), p_dep_tilde.mean(dim=1), query_mask,
        )
        w_proto = float(pso_w['proto'].item()) if pso_w else 0.5
        s_q = self.ms_match(f_q_rgb, f_q_dep, p_rgb_tilde, p_dep_tilde, w_proto)
        pred = self.decoder(f_fused, f_q_att, s_q, (h, w))
        return {'pred': pred, 'R_rgb': r_rgb, 'R_depth': r_depth, 'pso_weights': pso_w}

    @staticmethod
    def compute_loss(pred, target, r_rgb, r_depth):
        l_seg = F.binary_cross_entropy(pred, target.float())
        target_ds = F.interpolate(target.float(), size=r_depth.shape[-2:], mode='nearest')
        l_rel = F.binary_cross_entropy(r_depth, target_ds)
        return l_seg + 0.1 * l_rel
