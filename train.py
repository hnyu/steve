import math
import os.path
import argparse

import torch
import torchvision.utils as vutils

from torch.optim import Adam
from torch.nn import DataParallel as DP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from steve import STEVE
#from data import GlobVideoDataset
from movi_data_set import MoviDataset
from utils import cosine_anneal, linear_warmup


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0) #
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--ep_len', type=int, default=3)

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar') #
parser.add_argument('--data_path', default='data/*') #
parser.add_argument('--log_path', default='logs/') #

parser.add_argument('--lr_dvae', type=float, default=3e-4)
parser.add_argument('--lr_enc', type=float, default=1e-4)
parser.add_argument('--lr_dec', type=float, default=3e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=30000)
parser.add_argument('--lr_half_life', type=int, default=250000)
parser.add_argument('--clip', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--steps', type=int, default=200000)

parser.add_argument('--num_iterations', type=int, default=2)
parser.add_argument('--num_slots', type=int, default=10 + 1) #
parser.add_argument('--cnn_hidden_size', type=int, default=64)
parser.add_argument('--slot_size', type=int, default=64) #
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--num_predictor_blocks', type=int, default=1)
parser.add_argument('--num_predictor_heads', type=int, default=4)
parser.add_argument('--predictor_dropout', type=int, default=0.0)

parser.add_argument('--vocab_size', type=int, default=4096)
parser.add_argument('--num_decoder_blocks', type=int, default=8)
parser.add_argument('--num_decoder_heads', type=int, default=4)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--dropout', type=int, default=0.1)

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_steps', type=int, default=30000)

parser.add_argument('--hard', action='store_true')
parser.add_argument('--use_dp', default=True, action='store_true')

args = parser.parse_args()

if "movi_d" in args.data_path or "movi_e" in args.data_path:
    args.num_slots = 23 + 1

torch.manual_seed(args.seed)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(args.log_path, datetime.today().isoformat())
writer = SummaryWriter(log_dir)
writer.add_text('hparams', arg_str)

#train_dataset = GlobVideoDataset(root=args.data_path, phase='train', img_size=args.image_size, ep_len=args.ep_len, img_glob='????????_image.png')
#val_dataset = GlobVideoDataset(root=args.data_path, phase='val', img_size=args.image_size, ep_len=args.ep_len, img_glob='????????_image.png')

train_dataset = MoviDataset(data_dir=args.data_path, img_size=args.image_size, seq_len=args.ep_len, train=True)
val_dataset = MoviDataset(data_dir=args.data_path, img_size=args.image_size, train=False)

loader_kwargs = {
    'num_workers': args.num_workers,
    'pin_memory': True
}

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=None, shuffle=True, drop_last=True, **loader_kwargs)
val_loader = DataLoader(val_dataset, batch_size=10, sampler=None, shuffle=False, drop_last=False, **loader_kwargs)

train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

log_interval = train_epoch_size // 200
visualize_interval = log_interval * 5
eval_interval = log_interval * 5

model = STEVE(args)

if os.path.isfile(args.checkpoint_path):
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['model'])
else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0

model = model.cuda()
if args.use_dp:
    model = DP(model)

optimizer = Adam([
    {'params': (x[1] for x in model.named_parameters() if 'dvae' in x[0]), 'lr': args.lr_dvae},
    {'params': (x[1] for x in model.named_parameters() if 'steve_encoder' in x[0]), 'lr': 0.0},
    {'params': (x[1] for x in model.named_parameters() if 'steve_decoder' in x[0]), 'lr': 0.0},
])

if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])


def visualize(video, recon_dvae, recon_tf, attns, N=8):
    B, T, C, H, W = video.size()

    frames = []
    for t in range(T):
        video_t = video[:N, t, None, :, :, :]
        recon_dvae_t = recon_dvae[:N, t, None, :, :, :]
        recon_tf_t = recon_tf[:N, t, None, :, :, :]
        attns_t = attns[:N, t, :, :, :, :]

        # tile
        tiles = torch.cat((video_t, recon_dvae_t, recon_tf_t, attns_t), dim=1).flatten(end_dim=1)

        # grid
        frame = vutils.make_grid(tiles, nrow=(args.num_slots + 3), pad_value=0.8)
        frames += [frame]

    frames = torch.stack(frames, dim=0).unsqueeze(0)

    return frames


def compute_ari(seg,
                gt_seg,
                num_groups,
                ignore_background=True):
    """Converted from the JAX version:
    https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py#L111
    """
    # Following SAVI, it may be that num_groups <= max(segmentation). We prune
    # out extra objects here. For example, movi_e has an instance id of 64 in an
    # outlier video.
    # https://github.com/google-research/slot-attention-video/blob/main/savi/lib/preprocessing.py#L414
    gt_seg = torch.where(gt_seg >= num_groups, torch.zeros_like(gt_seg), gt_seg)
    seg = torch.where(seg >= num_groups, torch.zeros_like(seg), seg)

    seg = torch.nn.functional.one_hot(seg, num_groups).to(torch.float32)
    gt_seg = torch.nn.functional.one_hot(gt_seg, num_groups).to(torch.float32)

    if ignore_background:
        # remove background (id=0).
        gt_seg = gt_seg[..., 1:]

    N = torch.einsum('bthwc,bthwk->bck', gt_seg, seg)  # [B,c,k]
    A = N.sum(-1)  # row-sum  [B,c]
    B = N.sum(-2)  # col-sum  [B,k]
    num_points = A.sum(1)  # [B]

    rindex = (N * (N - 1)).sum((1, 2))  # [B]
    aindex = (A * (A - 1)).sum(1)  # [B]
    bindex = (B * (B - 1)).sum(1)  # [B]

    expected_rindex = aindex * bindex / torch.clamp(
        num_points * (num_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both label_pred and label_true assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == num_points * (num_points-1))
    # 2. If both label_pred and label_true assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    return torch.where(denominator > 0, ari, torch.ones_like(ari))


def evaluation(model, tau, eval_loader, writer, global_step):
    model.eval()
    with torch.no_grad():
        aris, fg_aris = [], []
        total = 0
        for sample, gt_seg  in eval_loader:
            total += sample.shape[0]
            video = sample.cuda()
            gt_seg = gt_seg.cuda().long()
            seg = model(video, tau, args.hard)[-1]
            gt_seg = gt_seg.squeeze(2)
            fg_ari = compute_ari(seg, gt_seg, num_groups=args.num_slots + 1)
            ari = compute_ari(seg, gt_seg, num_groups=args.num_slots + 1, ignore_background=False)
            fg_aris.append(fg_ari)
            aris.append(ari)
        mean_fg_ari = sum(fg_aris) / len(fg_aris)
        mean_ari = sum(aris) / len(aris)
        writer.add_scalar('eval/fg_ari', mean_fg_ari.mean(), global_step=global_step)
        writer.add_scalar('eval/ari', mean_ari.mean(), global_step=global_step)
        print("Total evaluated: ", total)
    model.train()


for epoch in range(start_epoch, args.epochs):
    model.train()

    for batch, (video, seg) in enumerate(train_loader):
        global_step = epoch * train_epoch_size + batch

        tau = cosine_anneal(
            global_step,
            args.tau_start,
            args.tau_final,
            0,
            args.tau_steps)

        lr_warmup_factor_enc = linear_warmup(
            global_step,
            0.,
            1.0,
            0.,
            args.lr_warmup_steps)

        lr_warmup_factor_dec = linear_warmup(
            global_step,
            0.,
            1.0,
            0,
            args.lr_warmup_steps)

        lr_decay_factor = math.exp(global_step / args.lr_half_life * math.log(0.5))

        optimizer.param_groups[0]['lr'] = args.lr_dvae
        optimizer.param_groups[1]['lr'] = lr_decay_factor * lr_warmup_factor_enc * args.lr_enc
        optimizer.param_groups[2]['lr'] = lr_decay_factor * lr_warmup_factor_dec * args.lr_dec

        video = video.cuda()

        optimizer.zero_grad()

        (recon, cross_entropy, mse, attns, seg) = model(video, tau, args.hard)

        if args.use_dp:
            mse = mse.mean()
            cross_entropy = cross_entropy.mean()

        loss = mse + cross_entropy

        loss.backward()
        clip_grad_norm_(model.parameters(), args.clip, 'inf')
        optimizer.step()

        with torch.no_grad():
            if batch % log_interval == 0:
                print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}'.format(
                      epoch+1, batch, train_epoch_size, loss.item(), mse.item()))

                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/cross_entropy', cross_entropy.item(), global_step)
                writer.add_scalar('TRAIN/mse', mse.item(), global_step)

                writer.add_scalar('TRAIN/tau', tau, global_step)
                writer.add_scalar('TRAIN/lr_dvae', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('TRAIN/lr_enc', optimizer.param_groups[1]['lr'], global_step)
                writer.add_scalar('TRAIN/lr_dec', optimizer.param_groups[2]['lr'], global_step)

            if batch % visualize_interval == 0:
                gen_video = (model.module if args.use_dp else model).reconstruct_autoregressive(video[:8])
                frames = visualize(video, recon, gen_video, attns, N=8)
                writer.add_video('TRAIN_recons/epoch={:03}'.format(epoch+1), frames, global_step, fps=1)

            if batch % eval_interval == 0:
                evaluation(model, tau, val_loader, writer, global_step)

    ##########
    checkpoint = {
        'epoch': epoch + 1,
        'model': model.module.state_dict() if args.use_dp else model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(checkpoint, args.checkpoint_path)

writer.close()
