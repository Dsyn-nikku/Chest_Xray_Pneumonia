from pathlib import Path

DATASET_PATH = Path("chest_xray")
SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")
EXTENSIONS = {".jpg", ".jpeg", ".png"}

print("CHEST X-RAY DATASET CHECK")
print("=" * 45)

if not DATASET_PATH.exists():
    raise SystemExit(
        "Dataset folder not found. Put the extracted 'chest_xray' folder "
        "beside this script."
    )

grand_total = 0
for split in SPLITS:
    print(f"\n{split.upper()}")
    split_total = 0
    for class_name in CLASSES:
        folder = DATASET_PATH / split / class_name
        if not folder.exists():
            print(f"  {class_name}: MISSING ({folder})")
            continue
        count = sum(
            1 for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        split_total += count
        print(f"  {class_name}: {count}")
    grand_total += split_total
    print(f"  Split total: {split_total}")

print("\n" + "=" * 45)
print(f"TOTAL IMAGES FOUND: {grand_total}")
