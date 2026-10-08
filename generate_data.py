from pathlib import Path
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def random_color(rng, low=20, high=235):
    return tuple(rng.randint(low, high) for _ in range(3))


def draw_anime_like(rng, size=64):
    """
    Create a small synthetic "anime-like" image:
    - flat colors
    - black clean outlines
    - circles/ellipses/polygons/arcs
    """
    bg = random_color(rng, 170, 245)
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)

    # Large flat polygon
    pts = []
    cx, cy = rng.randint(20, 44), rng.randint(20, 44)
    radius = rng.randint(12, 24)
    n = rng.randint(3, 6)
    for k in range(n):
        a = 2 * math.pi * k / n + rng.random() * 0.4
        rr = radius * (0.75 + 0.35 * rng.random())
        pts.append((int(cx + rr * math.cos(a)), int(cy + rr * math.sin(a))))
    d.polygon(pts, fill=random_color(rng, 60, 220), outline=(20, 20, 20))

    # Face-like ellipse sometimes
    if rng.random() < 0.8:
        x0 = rng.randint(8, 24)
        y0 = rng.randint(8, 24)
        w = rng.randint(22, 38)
        h = rng.randint(22, 38)
        d.ellipse(
            (x0, y0, min(size-2, x0+w), min(size-2, y0+h)),
            fill=random_color(rng, 120, 240),
            outline=(15, 15, 15),
            width=rng.randint(1, 2),
        )

        # Eyes
        ey = y0 + h // 2
        for ex in [x0 + w // 3, x0 + 2 * w // 3]:
            d.line((ex-3, ey, ex+3, ey), fill=(10,10,10), width=1)
            if rng.random() < 0.5:
                d.ellipse((ex-1, ey-1, ex+1, ey+1), fill=(10,10,10))

    # Random clean line strokes
    for _ in range(rng.randint(4, 10)):
        if rng.random() < 0.5:
            x1, y1 = rng.randint(2, size-3), rng.randint(2, size-3)
            x2, y2 = rng.randint(2, size-3), rng.randint(2, size-3)
            d.line((x1, y1, x2, y2), fill=(10, 10, 10), width=rng.randint(1, 2))
        else:
            box = (
                rng.randint(2, size//2),
                rng.randint(2, size//2),
                rng.randint(size//2, size-2),
                rng.randint(size//2, size-2),
            )
            d.arc(box, rng.randint(0, 180), rng.randint(181, 359), fill=(10,10,10), width=1)

    return img


def degrade(img, rng, lr_size=32):
    # Blur
    x = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.4)))

    # Downsample
    x = x.resize((lr_size, lr_size), Image.Resampling.BICUBIC)

    # Add light Gaussian noise
    arr = np.asarray(x).astype(np.float32)
    sigma = rng.uniform(1.0, 5.0)
    noise = np.random.default_rng(rng.randint(0, 10_000_000)).normal(
        0, sigma, arr.shape
    )
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    x = Image.fromarray(arr, mode="RGB")

    # Simulate mild JPEG compression in memory
    import io
    buffer = io.BytesIO()
    x.save(buffer, format="JPEG", quality=rng.randint(55, 90))
    buffer.seek(0)
    x = Image.open(buffer).convert("RGB")

    return x


def build_split(root, split, count, seed):
    rng = random.Random(seed)
    lr_dir = root / split / "lr"
    hr_dir = root / split / "hr"
    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        hr = draw_anime_like(rng, 64)
        lr = degrade(hr, rng, 32)

        name = f"{i:06d}.png"
        hr.save(hr_dir / name)
        lr.save(lr_dir / name)


if __name__ == "__main__":
    root = Path("data")
    build_split(root, "train", 300, seed=1234)
    build_split(root, "val", 60, seed=5678)
    print("done")
