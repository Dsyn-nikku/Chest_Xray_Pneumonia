from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("chest_xray/train")
EXTENSIONS = {".jpg", ".jpeg", ".png"}
random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for row, class_name in enumerate(("NORMAL", "PNEUMONIA")):
    folder = ROOT / class_name
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")
    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]
    if not images:
        raise SystemExit(f"No images found in {folder}")
    for col in range(3):
        path = random.choice(images)
        with Image.open(path) as image:
            axes[row, col].imshow(image.convert("L"), cmap="gray")
        axes[row, col].set_title(class_name)
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()
