import torch
from torch.utils.data import DataLoader

from rdcgprnet.dataset import NYUDepthV2Dataset
from rdcgprnet.model import RDCGPRNetPSO
from rdcgprnet.trainer import Trainer, AblationRunner
from rdcgprnet.utils import build_parser, auto_detect_data_root, set_seed


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.data_root is None:
        args.data_root = auto_detect_data_root()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Device: {device}')
    print(f'Data root: {args.data_root}')

    train_ds = NYUDepthV2Dataset(
        root=args.data_root,
        shot=args.shot,
        img_size=args.img_size,
        split='train',
        depth_max=args.depth_max,
        depth_fg_threshold=args.depth_fg_thr,
    )
    val_ds = NYUDepthV2Dataset(
        root=args.data_root,
        shot=args.shot,
        img_size=args.img_size,
        split='test',
        depth_max=args.depth_max,
        depth_fg_threshold=args.depth_fg_thr,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = RDCGPRNetPSO(
        backbone=args.backbone,
        embed_dim=args.embed_dim,
        num_iter=args.num_iter,
        hidden_dim=args.hidden_dim,
        use_pso=args.use_pso,
        pso_particles=args.pso_particles,
        pso_iters=args.pso_iters,
    ).to(device)

    if args.mode == 'train':
        Trainer(model, args).run(train_loader, val_loader)
    elif args.mode == 'eval':
        assert args.checkpoint, '--checkpoint required for eval'
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        res = Trainer(model, args).evaluate(val_loader)
        print(res)
    elif args.mode == 'ablation':
        assert args.checkpoint, '--checkpoint required for ablation'
        AblationRunner(args, val_loader, device).run(args.checkpoint)


if __name__ == '__main__':
    main()
