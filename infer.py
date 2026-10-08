import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import TinyAnimeSR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output", default="restored.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyAnimeSR().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img = Image.open(args.input).convert("RGB")
    x = to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(x)[0].cpu()

    out = to_pil_image(y)
    out.save(args.output)
    print("saved:", args.output)


if __name__ == "__main__":
    main()
