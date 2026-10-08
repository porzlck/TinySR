import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairedFolderDataset
from model import TinyAnimeSR


def psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    if mse.item() == 0:
        return 99.0
    return float(10.0 * torch.log10(torch.tensor(1.0, device=pred.device) / mse))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses, psnrs = [], []

    for batch in loader:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        pred = model(lr)
        loss = F.l1_loss(pred, hr)

        losses.append(loss.item())
        psnrs.append(psnr(pred, hr))

    return sum(losses) / len(losses), sum(psnrs) / len(psnrs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-dir", default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_set = PairedFolderDataset(args.data, "train")
    val_set = PairedFolderDataset(args.data, "val")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model = TinyAnimeSR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_psnr = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            pred = model(lr)

            # First version: only L1 loss.
            # This keeps the experiment easy to understand.
            loss = F.l1_loss(pred, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss, val_psnr = evaluate(model, val_loader, device)
        train_loss = running / len(train_loader)

        print(
            f"epoch={epoch:02d} "
            f"train_l1={train_loss:.4f} "
            f"val_l1={val_loss:.4f} "
            f"val_psnr={val_psnr:.2f} dB"
        )

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_psnr": val_psnr,
            },
            save_dir / "last.pt",
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_psnr": val_psnr,
                },
                save_dir / "best.pt",
            )

    print("best val PSNR:", best_psnr)


if __name__ == "__main__":
    main()
