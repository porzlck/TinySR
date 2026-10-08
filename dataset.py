from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class PairedFolderDataset(Dataset):
    """
    Expected layout:
      data/train/lr/000000.png
      data/train/hr/000000.png
    """
    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.lr_dir = self.root / split / "lr"
        self.hr_dir = self.root / split / "hr"
        self.lr_files = sorted(self.lr_dir.glob("*.png"))

        if not self.lr_files:
            raise RuntimeError(f"No PNG files found in {self.lr_dir}")

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_path = self.lr_files[idx]
        hr_path = self.hr_dir / lr_path.name

        lr = Image.open(lr_path).convert("RGB")
        hr = Image.open(hr_path).convert("RGB")

        return {
            "lr": to_tensor(lr),
            "hr": to_tensor(hr),
            "name": lr_path.name,
        }
