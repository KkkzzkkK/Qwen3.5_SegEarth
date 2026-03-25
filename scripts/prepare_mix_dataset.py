import argparse
import json
import os
import random
import shutil
from pathlib import Path


def is_enabled(flag: str) -> bool:
    return str(flag).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def symlink_force(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    os.symlink(src, dst, target_is_directory=src.is_dir())


def resolve_dataset_paths(root: Path):
    ann = root / "train" / "annotations" / "train_data.json"
    images = root / "train" / "images"
    if not ann.exists():
        raise FileNotFoundError(f"annotation not found: {ann}")
    if not images.exists():
        raise FileNotFoundError(f"image dir not found: {images}")
    return ann, images


def apply_keep_ratio(samples, keep_ratio: float, seed: int):
    keep_ratio = max(0.0, float(keep_ratio))
    if keep_ratio == 0.0:
        return []

    n = len(samples)
    if n == 0:
        return []

    if keep_ratio >= 1.0:
        rep = int(keep_ratio)
        frac = keep_ratio - rep
        out = samples * rep
        if frac > 0:
            k = int(round(n * frac))
            rng = random.Random(seed)
            idxs = list(range(n))
            rng.shuffle(idxs)
            out.extend([samples[i] for i in idxs[:k]])
        return out

    k = int(round(n * keep_ratio))
    if keep_ratio > 0 and k == 0:
        k = 1
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    return [samples[i] for i in idxs[:k]]


def main():
    parser = argparse.ArgumentParser(description="Mix two LaSeRS-format datasets into a new independent dataset root.")
    parser.add_argument("--src-a", required=True, help="Original dataset root")
    parser.add_argument("--src-b", required=True, help="New dataset root")
    parser.add_argument("--dst", required=True, help="Output mixed dataset root")
    parser.add_argument("--keep-ratio-a", type=float, default=1.0, help="Keep ratio for dataset A")
    parser.add_argument("--keep-ratio-b", type=float, default=1.0, help="Keep ratio for dataset B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", default="0", help="Overwrite dst if exists")
    args = parser.parse_args()

    src_a = Path(args.src_a)
    src_b = Path(args.src_b)
    dst = Path(args.dst)

    if src_a.resolve() == dst.resolve() or src_b.resolve() == dst.resolve():
        raise ValueError("--dst must be different from both --src-a and --src-b")

    ann_a_path, img_a_dir = resolve_dataset_paths(src_a)
    ann_b_path, img_b_dir = resolve_dataset_paths(src_b)

    if dst.exists() and any(dst.iterdir()) and not is_enabled(args.overwrite):
        raise FileExistsError(f"dst exists and is not empty: {dst}; set --overwrite 1 to replace")
    if dst.exists() and is_enabled(args.overwrite):
        shutil.rmtree(dst)

    out_ann_dir = dst / "train" / "annotations"
    out_img_dir = dst / "train" / "images"
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    data_a = load_json(ann_a_path)
    data_b = load_json(ann_b_path)
    if not isinstance(data_a, list) or not isinstance(data_b, list):
        raise ValueError("both train_data.json files must be arrays")

    picked_a = apply_keep_ratio(data_a, args.keep_ratio_a, args.seed)
    picked_b = apply_keep_ratio(data_b, args.keep_ratio_b, args.seed + 1001)

    merged = []

    def add_samples(samples, image_dir: Path, prefix: str):
        for i, sample in enumerate(samples):
            image_name = sample.get("image_name")
            if not image_name:
                continue

            src_img = image_dir / image_name
            if not src_img.exists():
                continue

            new_img_name = f"{prefix}_{i:07d}_{Path(image_name).name}"
            dst_img = out_img_dir / new_img_name
            symlink_force(src_img, dst_img)

            new_sample = dict(sample)
            old_id = str(sample.get("id", i))
            new_sample["id"] = f"{prefix}_{old_id}"
            new_sample["image_name"] = new_img_name
            new_sample["mix_source"] = prefix
            merged.append(new_sample)

    add_samples(picked_a, img_a_dir, "orig")
    add_samples(picked_b, img_b_dir, "new")

    random.Random(args.seed + 2024).shuffle(merged)

    if not merged:
        raise ValueError("No samples in mixed dataset. Check keep ratios and source paths.")

    save_json(out_ann_dir / "train_data.json", merged)
    save_json(
        out_ann_dir / "mix_meta.json",
        {
            "src_a": str(src_a),
            "src_b": str(src_b),
            "dst": str(dst),
            "keep_ratio_a": float(args.keep_ratio_a),
            "keep_ratio_b": float(args.keep_ratio_b),
            "picked_a": len(picked_a),
            "picked_b": len(picked_b),
            "final_count": len(merged),
            "seed": int(args.seed),
        },
    )

    src_a_test = src_a / "test"
    if src_a_test.exists():
        symlink_force(src_a_test, dst / "test")

    print(json.dumps({
        "picked_a": len(picked_a),
        "picked_b": len(picked_b),
        "final_count": len(merged),
        "dst": str(dst),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
