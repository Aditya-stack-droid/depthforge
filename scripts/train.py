"""Train DepthForge-Net.

  # NYU v2 (recommended, on a Colab/Kaggle GPU ~15-25 min)
  python scripts/train.py --data nyu --mat nyu_depth_v2_labeled.mat --minutes 20

  # No download: procedural synthetic RGB-D scenes (CPU friendly smoke model)
  python scripts/train.py --data synth --minutes 10

Each training sample gets a random scale (2/4/8x) and a random degradation
(clean/noisy/holes/sparse/hard). Half the time the net sees the *classical*
pre-upsampled depth (hybrid mode) and half the time plain bicubic (DL-only mode),
so one checkpoint serves both pipelines.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from depthforge import classical as C, data, degrade  # noqa: E402
from depthforge.adaptive import choose_params  # noqa: E402
from depthforge.model import DepthForgeNet, count_params  # noqa: E402


class PatchSet(Dataset):
    def __init__(self, pairs, crop, length, train=True, seed=0, jitter=True):
        self.pairs, self.crop, self.length, self.train = pairs, crop, length, train
        self.jitter = jitter
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = random.Random(None if self.train else self.seed + i)
        rgb, gt = self.pairs[rng.randrange(len(self.pairs))]
        H, W = gt.shape
        c = self.crop
        y, x = rng.randrange(0, H - c + 1), rng.randrange(0, W - c + 1)
        rgb, gt = rgb[y:y + c, x:x + c], gt[y:y + c, x:x + c]
        if self.train and rng.random() < 0.5:
            rgb, gt = rgb[:, ::-1], gt[:, ::-1]
        rgb, gt = np.ascontiguousarray(rgb), np.ascontiguousarray(gt)
        if self.train and self.jitter:  # photometric jitter: depth is unchanged, so the net learns not to trust colour blindly
            g = rng.uniform(0.7, 1.4); b = rng.uniform(-25, 25)
            rgb = np.clip(255.0 * (rgb / 255.0) ** g + b, 0, 255).astype(np.uint8)
        gt = C.fill_holes(gt)  # NYU raw has a few zeros; train on dense targets
        scale = rng.choice([2, 4, 8])
        cond = rng.choice(list(degrade.CONDITIONS))
        lr = degrade.make_lr(gt, scale, *degrade.CONDITIONS[cond], seed=rng.randrange(1 << 30))
        if (lr > 0).sum() < 4:
            lr = gt[scale // 2::scale, scale // 2::scale].copy()
        if rng.random() < 0.5:
            p = choose_params(lr, rgb)
            coarse = C.classical_pipeline(lr, rgb, p, dict(snap=False))
        else:
            coarse = C.interpolate(C.fill_holes(lr), (c, c), "bicubic")
        s = max(float(np.percentile(coarse, 99)), 1e-3)
        to_t = lambda a: torch.from_numpy(a.astype(np.float32))
        return (to_t(coarse / s)[None], to_t(rgb / 255.0).permute(2, 0, 1), to_t(gt / s)[None])


def grad_loss(pred, gt):
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    return (dx(pred) - dx(gt)).abs().mean() + (dy(pred) - dy(gt)).abs().mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["nyu", "synth", "folder"], default="synth")
    ap.add_argument("--mat", default="nyu_depth_v2_labeled.mat")
    ap.add_argument("--folder", default=None)
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "weights" / "depthforge_net.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast on CUDA (about 2x faster on a Colab/Kaggle GPU)")
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay of the weights; the EMA copy is validated and saved (0 = off)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--init", default=None, help="warm-start from a checkpoint (e.g. the synthetic one)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if a.data == "nyu":
        pairs = list(data.load_nyu(a.mat, data.NYU_TRAIN))
        train_pairs, val_pairs = pairs[:900], pairs[900:]   # 900 train / 100 val; 449 test held out
    elif a.data == "folder":
        pairs = [(r, d) for _, r, d in data.load_folder(a.folder)]
        k = max(1, len(pairs) // 10)
        train_pairs, val_pairs = pairs[k:], pairs[:k]
    else:
        train_pairs = [data.synth_scene(240, 320, s) for s in range(1000, 1400)]
        val_pairs = [data.synth_scene(240, 320, s) for s in range(5000, 5040)]
    print(f"train {len(train_pairs)} / val {len(val_pairs)} images, device={dev}")

    tr = DataLoader(PatchSet(train_pairs, a.crop, 10 ** 7), a.batch, num_workers=a.workers,
                    persistent_workers=a.workers > 0)
    va = DataLoader(PatchSet(val_pairs, a.crop, 128, train=False), a.batch, num_workers=0)
    net = DepthForgeNet(a.width).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)["state_dict"])
        print("warm start from", a.init)
    print(f"params: {count_params(net):,}")
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
    use_amp = a.amp and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    import copy
    ema = copy.deepcopy(net).eval() if a.ema > 0 else None
    for p_ in (ema.parameters() if ema else []):
        p_.requires_grad_(False)
    budget = a.minutes * 60
    t0 = time.time()
    step, best = 0, 1e9
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    def evaluate():
        m = ema if ema is not None else net
        was = net.training; m.eval(); tot = n = 0
        with torch.no_grad():
            for d, r, g in va:
                d, r, g = d.to(dev), r.to(dev), g.to(dev)
                tot += F.l1_loss(m(d, r).float(), g, reduction="sum").item(); n += g.numel()
        net.train(was); return tot / n

    base = None
    for d, r, g in tr:
        frac = (time.time() - t0) / budget
        if frac >= 1:
            break
        for pg in opt.param_groups:  # cosine decay over the time budget
            pg["lr"] = a.lr * min(1.0, (step + 1) / 300) * 0.5 * (1 + np.cos(np.pi * min(frac, 1)))
        d, r, g = d.to(dev), r.to(dev), g.to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            pred = net(d, r)
        pred = pred.float()
        loss = F.l1_loss(pred, g) + 0.5 * grad_loss(pred, g)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); step += 1
        if ema is not None:
            dec = min(a.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(dec).add_(pn.detach(), alpha=1 - dec)
        if step % a.eval_every == 0 or step == 1:
            v = evaluate()
            if base is None:
                with torch.no_grad():
                    base = sum(F.l1_loss(dd, gg, reduction="sum").item() for dd, _, gg in va) / sum(gg.numel() for _, _, gg in va)
            print(f"step {step:5d}  {time.time()-t0:6.0f}s  loss {loss.item():.4f}  val L1 {v:.5f}  (input L1 {base:.5f})", flush=True)
            if v < best:
                best = v
                torch.save(dict(state_dict=(ema or net).state_dict(), width=a.width, data=a.data, crop=a.crop,
                                step=step, val_l1=v, input_val_l1=base), a.out)
    print(f"done: {step} steps, best val L1 {best:.5f} (input {base:.5f}) -> {a.out}")


if __name__ == "__main__":
    main()
