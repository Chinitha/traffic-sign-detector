"""
Remap 43 GTSDB classes into 2 superclasses.

  0  regulatory  → classes 0–15  (speed limits, overtaking, priority)
  1  warning      → classes 16–42 (warnings, danger, other)
"""

import pathlib, shutil
from collections import Counter

SRC = pathlib.Path("data/gtsdb_yolo")
OUT = pathlib.Path("data/gtsdb_yolo_grouped")

REMAP = {}
for i in range(0,  16): REMAP[i] = 0   # regulatory
for i in range(16, 43): REMAP[i] = 1   # warning

SUPERCLASS_NAMES = {
    0: "regulatory",
    1: "warning",
}

def remap_labels(src_lbl_dir, dst_lbl_dir):
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for f in src_lbl_dir.glob("*.txt"):
        lines = f.read_text().strip().splitlines()
        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            old_cls = int(parts[0])
            if old_cls > 42:           # skip corrupt labels
                continue
            new_cls = REMAP.get(old_cls, 1)
            counts[new_cls] += 1
            new_lines.append(f"{new_cls} {' '.join(parts[1:])}")
        (dst_lbl_dir / f.name).write_text("\n".join(new_lines))
    return counts

def copy_images(src_img_dir, dst_img_dir):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src_img_dir.glob("*.jpg"):
        shutil.copy2(f, dst_img_dir / f.name)
        count += 1
    return count

def write_data_yaml():
    yaml_lines = [
        f"path: {OUT.resolve()}",
        "train: images/train",
        "val:   images/val",
        "test:  images/test",
        f"nc: {len(SUPERCLASS_NAMES)}",
        "names:",
    ]
    for i, name in SUPERCLASS_NAMES.items():
        yaml_lines.append(f"  {i}: {name}")
    (OUT / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"{SRC} not found. Run download_data.py first.")

    print("=" * 55)
    print(" Remapping 43 classes → 2 superclasses")
    print("=" * 55)

    if OUT.exists():
        shutil.rmtree(OUT)

    total_counts = Counter()
    for split in ["train", "val", "test"]:
        n_imgs = copy_images(SRC / "images" / split, OUT / "images" / split)
        counts = remap_labels(SRC / "labels" / split, OUT / "labels" / split)
        total_counts += counts
        print(f"  {split:5s} → {n_imgs} images, {sum(counts.values())} boxes")

    write_data_yaml()
    print(f"\n✓ Wrote {OUT / 'data.yaml'}")

    print("\nSuperclass distribution in train split:")
    for cls_id, name in SUPERCLASS_NAMES.items():
        bar = "█" * (total_counts[cls_id] // 5)
        print(f"  {name:15s} {total_counts[cls_id]:4d}  {bar}")

    print(f"\n✓ Dataset ready at {OUT}/")
    print("  Next: python src/train.py --config grouped")

if __name__ == "__main__":
    main()