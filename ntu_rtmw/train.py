import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .constants import RUNS_DIR
from .dataset import SkeletonDataset


class GRUClassifier(nn.Module):
    def __init__(self, in_dim, classes, hidden=256):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, classes))

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.head(y[:, -1])


def run(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_set = SkeletonDataset(args.manifest, "train", args.frames)
    val_set = SkeletonDataset(args.manifest, "val", args.frames)
    if not train_set or not val_set:
        raise SystemExit("Manifest must contain both train and val samples.")

    classes = max(int(r["label"]) for r in train_set.rows + val_set.rows) + 1
    sample, _ = train_set[0]
    model = GRUClassifier(sample.shape[-1], classes, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * y.numel()
        acc = evaluate(model, val_loader, device)
        print("epoch {} loss {:.4f} val_acc {:.4f}".format(epoch, total_loss / len(train_set), acc))
        if acc >= best:
            best = acc
            torch.save({"model": model.state_dict(), "classes": classes, "frames": args.frames}, out_dir / "best.pt")
    print("best val_acc {:.4f}, saved {}".format(best, out_dir / "best.pt"))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    good = 0
    total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        good += int((pred == y).sum())
        total += y.numel()
    return good / max(1, total)


def parser():
    p = argparse.ArgumentParser(description="Train a baseline NTU action classifier.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", default=str(RUNS_DIR / "gru_baseline"))
    p.add_argument("--device")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    return p


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
