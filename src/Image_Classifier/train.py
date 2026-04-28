"""
Training Script — Canonical entry point for training the 8-class product CNN.

Reads a manifest CSV produced by assemble_dataset.py and trains the same
5-conv-layer architecture used in classify_json_images.py.

Usage:
    python src/Image_Classifier/train.py --manifest data/training_manifest.csv
    python src/Image_Classifier/train.py --manifest data/training_manifest.csv --epochs 5 --resume
"""

import argparse
import csv
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

from classify_json_images import build_model, make_preprocess

log = logging.getLogger("train")


# ── Dataset ───────────────────────────────────────────────────────────────────

class ManifestDataset(Dataset):
    """PyTorch Dataset backed by a CSV manifest from assemble_dataset.py.

    Each row must have at least ``image_path`` and ``class_index`` columns.
    Optionally filters by the ``split`` column ("train" or "test").
    """

    def __init__(self, manifest_path: Path, transform=None, split: str | None = None):
        self.transform = transform
        self.rows: list[dict] = []

        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if split and row.get("split") != split:
                    continue
                self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row["class_index"])
        return img, label


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        str(path),
    )
    log.info(f"Checkpoint saved: {path} (epoch {epoch})")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    device: torch.device,
) -> int:
    """Load a checkpoint and return the next epoch number to resume from."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt.get("epoch", 0)
    log.info(f"Resumed from checkpoint: {path} (epoch {epoch})")
    return epoch + 1  # return the *next* epoch


# ── Training / validation loops ───────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        output = model(x)
        batch_loss = loss_fn(output, y)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item() * x.size(0)
        correct += output.argmax(dim=1).eq(y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        total_loss += loss_fn(output, y).item() * x.size(0)
        correct += output.argmax(dim=1).eq(y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy


# ── Main ──────────────────────────────────────────────────────────────────────

def _default_manifest() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data" / "training_manifest.csv"


def _default_model_out() -> Path:
    return Path(__file__).resolve().parent / "trained_model.pth"


def _default_checkpoint() -> Path:
    return Path(__file__).resolve().parent / "checkpoint.tar"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Train the 8-class product image CNN from a manifest CSV",
    )
    parser.add_argument(
        "--manifest", type=Path, default=_default_manifest(),
        help="Training manifest CSV from assemble_dataset.py",
    )
    parser.add_argument(
        "--model-out", type=Path, default=_default_model_out(),
        help="Output path for trained model weights",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=_default_checkpoint(),
        help="Checkpoint file path (saved each epoch)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from checkpoint",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        parser.error(f"Manifest not found: {args.manifest}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    preprocess = make_preprocess()

    # Load dataset — use split column if present
    train_dataset = ManifestDataset(args.manifest, transform=preprocess, split="train")
    test_dataset = ManifestDataset(args.manifest, transform=preprocess, split="test")

    # If manifest has no split column or all rows are one split, fall back to random split
    if len(train_dataset) == 0 and len(test_dataset) == 0:
        parser.error("Manifest has no rows")
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        log.info("No split column found or only one split — using random 80/20 split")
        full_dataset = ManifestDataset(args.manifest, transform=preprocess)
        train_size = int(len(full_dataset) * 0.8)
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    log.info(f"Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")

    # Model
    model = build_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume and args.checkpoint.exists():
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint, device)

    # Training loop
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = validate(model, test_loader, loss_fn, device)

        print(
            f"Epoch {epoch + 1:3d}  |  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}"
        )

        save_checkpoint(model, optimizer, epoch, test_loss, args.checkpoint)

    # Save final model weights
    torch.save(model.state_dict(), str(args.model_out))
    log.info(f"Saved model weights to {args.model_out}")
    print(f"\nTraining complete. Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
