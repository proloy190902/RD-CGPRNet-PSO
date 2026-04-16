class Metrics:
    @staticmethod
    def iou(pred, target, thr=0.5):
        pb = (pred > thr).float()
        i = (pb * target).sum(dim=(-2, -1))
        u = pb.add(target).clamp(max=1).sum(dim=(-2, -1))
        return (i / (u + 1e-8)).mean().item()

    @staticmethod
    def dice(pred, target, thr=0.5):
        pb = (pred > thr).float()
        i = (pb * target).sum(dim=(-2, -1))
        return (2 * i / (pb.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + 1e-8)).mean().item()

    @staticmethod
    def fb_iou(pred, target, thr=0.5):
        return (Metrics.iou(pred, target, thr) + Metrics.iou(1 - pred, 1 - target, thr)) / 2.0

    @staticmethod
    def precision(pred, target, thr=0.5):
        pb = (pred > thr).float()
        tp = (pb * target).sum(dim=(-2, -1))
        fp = (pb * (1 - target)).sum(dim=(-2, -1))
        return (tp / (tp + fp + 1e-8)).mean().item()
