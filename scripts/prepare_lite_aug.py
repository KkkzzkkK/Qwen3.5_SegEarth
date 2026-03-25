import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_enabled(flag: str) -> bool:
    return str(flag).strip().lower() in {"1", "true", "yes", "y", "on"}


def lite_color_augment(img: Image.Image, rng: random.Random) -> Image.Image:
    out = img
    brightness_factor = rng.uniform(0.95, 1.05)
    contrast_factor = rng.uniform(0.95, 1.08)
    color_factor = rng.uniform(0.95, 1.08)

    out = ImageEnhance.Brightness(out).enhance(brightness_factor)
    out = ImageEnhance.Contrast(out).enhance(contrast_factor)
    out = ImageEnhance.Color(out).enhance(color_factor)
    return out


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def normalize_id(sample, fallback_idx: int):
    sid = sample.get("id", fallback_idx)
    return str(sid)


def main():
    parser = argparse.ArgumentParser(description="Prepare lite augmented train dataset (color-only, mask-safe).")
    parser.add_argument("--src", required=True, help="Original base_data_path")
    parser.add_argument("--dst", required=True, help="Augmented base_data_path output")
    parser.add_argument("--ratio", type=float, default=0.3, help="Augmented sample ratio in [0,1]")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", default="1", help="Whether to overwrite dst if exists")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        raise FileNotFoundError(f"src not found: {src_root}")

    if dst_root.exists() and not is_enabled(args.overwrite):
        raise FileExistsError(f"dst exists and overwrite disabled: {dst_root}")

    ratio = max(0.0, min(1.0, float(args.ratio)))
    rng = random.Random(args.seed)

    copy_tree(src_root, dst_root)

    ann_path = dst_root / "train" / "annotations" / "train_data.json"
    img_dir = dst_root / "train" / "images"

    if not ann_path.exists():
        raise FileNotFoundError(f"annotation not found: {ann_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"image dir not found: {img_dir}")

    samples = load_json(ann_path)
    if not isinstance(samples, list):
        raise ValueError("train_data.json must be a list of samples")

    total = len(samples)
    aug_count = int(total * ratio)
    if total > 0 and ratio > 0 and aug_count == 0:
        aug_count = 1

    indices = list(range(total))
    rng.shuffle(indices)
    selected = set(indices[:aug_count])

    new_samples = []
    used_names = {s.get("image_name", "") for s in samples}

    for idx, sample in enumerate(samples):
        if idx not in selected:
            continue

        image_name = sample.get("image_name")
        if not image_name:
            continue

        src_img = img_dir / image_name
        if not src_img.exists() or src_img.suffix.lower() not in VALID_EXTS:
            continue

        sid = normalize_id(sample, idx)
        aug_name = f"aug_lite_{sid}_{src_img.name}"
        if aug_name in used_names:
            aug_name = f"aug_lite_{sid}_{idx}_{src_img.name}"
        used_names.add(aug_name)

        with Image.open(src_img).convert("RGB") as im:
            aug_im = lite_color_augment(im, rng)
            aug_im.save(img_dir / aug_name)

        aug_sample = dict(sample)
        aug_sample["image_name"] = aug_name
        aug_sample["id"] = f"aug_lite_{sid}"
        new_samples.append(aug_sample)

    merged = samples + new_samples
    save_json(ann_path, merged)

    print(f"[AugLite] src={src_root}")
    print(f"[AugLite] dst={dst_root}")
    print(f"[AugLite] total={total}, selected={len(selected)}, generated={len(new_samples)}, final={len(merged)}")


if __name__ == "__main__":
    main()
