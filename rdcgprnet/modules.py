import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityReliabilityEstimator(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        c = in_channels
        self.net = nn.Sequential(
            nn.Conv2d(c * 2, c, 1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c // 2), nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, 2, 1),
        )

    def forward(self, f_rgb, f_depth):
        if f_depth.shape[-2:] != f_rgb.shape[-2:]:
            f_depth = F.interpolate(f_depth, size=f_rgb.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([f_rgb, f_depth], dim=1)
        x = self.net(x)
        x = F.softmax(x, dim=1)
        return x[:, 0:1], x[:, 1:2]


class PSOModalFusionOptimizer:
    def __init__(self, n_particles=30, n_iter=40, w=0.7, c1=1.5, c2=1.5, lambda_rel=0.3):
        self.n_p = n_particles
        self.n_i = n_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.lam = lambda_rel
        self.dim = 6

    @staticmethod
    def _softmax2(a, b):
        ea, eb = np.exp(a), np.exp(b)
        s = ea + eb + 1e-8
        return ea / s, eb / s

    def _decode(self, particle):
        w_rgb_fg, w_dep_fg = self._softmax2(particle[0], particle[1])
        w_rgb_bg, w_dep_bg = self._softmax2(particle[2], particle[3])
        return {
            'rgb_fg': w_rgb_fg,
            'dep_fg': w_dep_fg,
            'rgb_bg': w_rgb_bg,
            'dep_bg': w_dep_bg,
            'scale': float(np.clip(particle[4], 0, 1)),
            'proto': float(np.clip(particle[5], 0, 1)),
        }

    def _fitness(self, particles, rgb_sim, dep_sim, mask, r_rgb, r_depth):
        scores = np.zeros(self.n_p)
        mn = mask - mask.mean()
        mn_norm = np.linalg.norm(mn) + 1e-8
        for i in range(self.n_p):
            p = self._decode(particles[i])
            fg_fused = p['rgb_fg'] * rgb_sim + p['dep_fg'] * dep_sim
            bg_fused = p['rgb_bg'] * rgb_sim + p['dep_bg'] * dep_sim
            fused = mask * fg_fused + (1 - mask) * bg_fused
            fn = fused - fused.mean()
            ncc = (fn * mn).sum() / (np.linalg.norm(fn) * mn_norm + 1e-8)
            rel_pen = (p['dep_fg'] * (1 - r_depth) * mask).mean() + (p['dep_bg'] * (1 - r_depth) * (1 - mask)).mean()
            scores[i] = ncc - self.lam * rel_pen
        return scores

    def optimize(self, rgb_sim, dep_sim, query_mask, r_rgb, r_depth):
        device = rgb_sim.device
        h, w = rgb_sim.shape[-2:]

        def to_np(t):
            return t.detach().float().mean(0).squeeze().cpu().numpy()

        np_rgb = to_np(rgb_sim)
        np_dep = to_np(dep_sim)
        np_mask = to_np(F.interpolate(query_mask.float(), size=(h, w), mode='nearest'))
        np_r_rgb = to_np(r_rgb)
        np_r_dep = to_np(r_depth)

        pos = np.random.uniform(-2, 2, (self.n_p, self.dim))
        vel = np.random.uniform(-0.5, 0.5, (self.n_p, self.dim))
        pb_pos = pos.copy()
        pb_fit = self._fitness(pos, np_rgb, np_dep, np_mask, np_r_rgb, np_r_dep)
        gi = pb_fit.argmax()
        gb_pos = pb_pos[gi].copy()
        gb_fit = pb_fit[gi]

        for _ in range(self.n_i):
            r1 = np.random.rand(self.n_p, self.dim)
            r2 = np.random.rand(self.n_p, self.dim)
            vel = self.w * vel + self.c1 * r1 * (pb_pos - pos) + self.c2 * r2 * (gb_pos - pos)
            pos = np.clip(pos + vel, -3.0, 3.0)
            fit = self._fitness(pos, np_rgb, np_dep, np_mask, np_r_rgb, np_r_dep)
            better = fit > pb_fit
            pb_pos[better] = pos[better]
            pb_fit[better] = fit[better]
            bi = pb_fit.argmax()
            if pb_fit[bi] > gb_fit:
                gb_pos = pb_pos[bi].copy()
                gb_fit = pb_fit[bi]

        best = self._decode(gb_pos)
        return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in best.items()}


class ModalFusionModule(nn.Module):
    def __init__(self, channels=256, use_pso=True, pso_particles=30, pso_iters=40):
        super().__init__()
        self.use_pso = use_pso
        self.post_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.pso = PSOModalFusionOptimizer(n_particles=pso_particles, n_iter=pso_iters)

    def _compute_sim_maps(self, f_rgb, f_depth, proto_rgb, proto_dep):
        def cos_sim(feat, proto):
            p = F.normalize(proto, dim=1)[:, :, None, None]
            f = F.normalize(feat, dim=1)
            return (f * p).sum(dim=1)
        return cos_sim(f_rgb, proto_rgb), cos_sim(f_depth, proto_dep)

    def forward(self, f_rgb, f_depth, r_rgb, r_depth, proto_rgb, proto_dep, query_mask):
        if self.use_pso and not self.training and query_mask is not None and proto_rgb is not None and proto_dep is not None:
            rgb_sim, dep_sim = self._compute_sim_maps(f_rgb, f_depth, proto_rgb, proto_dep)
            pso_w = self.pso.optimize(rgb_sim, dep_sim, query_mask, r_rgb, r_depth)
            f_fg = pso_w['rgb_fg'] * f_rgb + pso_w['dep_fg'] * f_depth
            f_bg = pso_w['rgb_bg'] * f_rgb + pso_w['dep_bg'] * f_depth
            proxy = torch.sigmoid((rgb_sim + dep_sim).unsqueeze(1) * 3.0)
            f_fused = proxy * f_fg + (1 - proxy) * f_bg
        else:
            f_fused = r_rgb * f_rgb + r_depth * f_depth
            pso_w = None
        return self.post_proj(f_fused), pso_w


class CrossModalGPG(nn.Module):
    def __init__(self, embed_dim=256, num_iter=2, hidden_dim=128):
        super().__init__()
        self.T = num_iter
        self.scale = embed_dim ** -0.5
        self.Q_proj = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(inplace=True))
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mlp_rgb = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim * 2 + 1, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, embed_dim))
            for _ in range(num_iter)
        ])
        self.mlp_dep = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim * 2 + 1, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, embed_dim))
            for _ in range(num_iter)
        ])
        self.refine_rgb = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, embed_dim))
        self.refine_dep = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, embed_dim))

    @staticmethod
    def _map(feat, mask):
        m = F.interpolate(mask.float(), size=feat.shape[-2:], mode='nearest')
        num = (feat * m).sum(dim=(-2, -1))
        den = m.sum(dim=(-2, -1)).clamp(min=1e-5)
        return num / den

    @staticmethod
    def _map_scalar(smap, mask):
        m = F.interpolate(mask.float(), size=smap.shape[-2:], mode='nearest')
        num = (smap * m).sum(dim=(-2, -1))
        den = m.sum(dim=(-2, -1)).clamp(min=1e-5)
        return num / den

    def forward(self, f_q_rgb, f_q_dep, supp_rgb, supp_dep, supp_masks, r_rgb, r_depth):
        b, c, h, w = f_q_rgb.shape
        k = len(supp_rgb)
        p_rgb = torch.stack([self._map(supp_rgb[i], supp_masks[i]) for i in range(k)], dim=1)
        p_dep = torch.stack([self._map(supp_dep[i], supp_masks[i]) for i in range(k)], dim=1)
        rel_rgb_k = torch.stack([self._map_scalar(r_rgb, supp_masks[i]) for i in range(k)], dim=1)
        rel_dep_k = torch.stack([self._map_scalar(r_depth, supp_masks[i]) for i in range(k)], dim=1)

        qr = f_q_rgb.flatten(2).permute(0, 2, 1)
        qd = f_q_dep.flatten(2).permute(0, 2, 1)
        q = self.Q_proj(torch.cat([qr, qd], dim=-1))
        p_joint = (p_rgb + p_dep) / 2.0
        a_last = None

        for t in range(self.T):
            a = F.softmax(torch.bmm(self.W_Q(q), self.W_K(p_joint).transpose(1, 2)) * self.scale, dim=-1)
            a_last = a
            m = torch.bmm(a.transpose(1, 2), q)
            p_rgb = p_rgb + self.mlp_rgb[t](torch.cat([p_rgb, m, rel_rgb_k], dim=-1)) * rel_rgb_k
            p_dep = p_dep + self.mlp_dep[t](torch.cat([p_dep, m, rel_dep_k], dim=-1)) * rel_dep_k
            p_joint = (p_rgb + p_dep) / 2.0

        p_rgb_tilde = self.refine_rgb(p_rgb)
        p_dep_tilde = self.refine_dep(p_dep)
        f_q_att = torch.bmm(a_last, self.W_V(p_joint)).permute(0, 2, 1).view(b, c, h, w)
        return p_rgb_tilde, p_dep_tilde, f_q_att


class MultiScaleMatching(nn.Module):
    def __init__(self, channels=256):
        super().__init__()

        def sc(k):
            return nn.Sequential(
                nn.Conv2d(channels, channels, k, padding=k // 2, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            )

        self.rgb_conv3 = sc(3)
        self.rgb_conv5 = sc(5)
        self.rgb_conv7 = sc(7)
        self.dep_conv3 = sc(3)
        self.dep_conv5 = sc(5)
        self.dep_conv7 = sc(7)
        self.alpha = nn.Parameter(torch.zeros(3))

    @staticmethod
    def _cos(feat, proto):
        p = F.normalize(proto, dim=1)[:, :, None, None]
        return (F.normalize(feat, dim=1) * p).sum(dim=1)

    def forward(self, f_q_rgb, f_q_dep, p_rgb_tilde, p_dep_tilde, w_proto=0.5):
        b, k, c = p_rgb_tilde.shape
        scale_maps = []
        for rc, dc in [(self.rgb_conv3, self.dep_conv3), (self.rgb_conv5, self.dep_conv5), (self.rgb_conv7, self.dep_conv7)]:
            fr = rc(f_q_rgb)
            fd = dc(f_q_dep)
            s = torch.zeros(b, *fr.shape[-2:], device=f_q_rgb.device)
            for i in range(k):
                s += w_proto * self._cos(fr, p_rgb_tilde[:, i]) + (1 - w_proto) * self._cos(fd, p_dep_tilde[:, i])
            scale_maps.append(s / k)
        w = F.softmax(self.alpha, dim=0)
        return sum(w[i] * scale_maps[i] for i in range(3))


class SegmentationDecoder(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        c = channels
        self.net = nn.Sequential(
            nn.Conv2d(c * 2 + 1, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c // 2), nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, 1, 1),
        )

    def forward(self, f_fused, f_q_att, s_q, out_size):
        x = torch.cat([f_fused, f_q_att, s_q.unsqueeze(1)], dim=1)
        x = self.net(x)
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        return torch.sigmoid(x).clamp(1e-7, 1 - 1e-7)
