import argparse
import json
import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def is_enabled(flag: str) -> bool:
    return str(flag).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_seg_span(span: str) -> str:
    text = re.sub(r"\s+", " ", span).strip()
    text = re.sub(r"\s*</p>\s*", "</p> ", text)
    text = re.sub(r"\s*\[SEG\]\s*", " [SEG]", text)
    return text


def parse_object_names(description: str):
    return [name.strip() for name in re.findall(r"<p>\s*(.*?)\s*</p>", description or "", flags=re.IGNORECASE | re.DOTALL)]


def parse_seg_spans(description: str):
    spans = re.findall(r"<p>\s*.*?\s*</p>\s*\[SEG\]", description or "", flags=re.IGNORECASE | re.DOTALL)
    return [normalize_seg_span(span) for span in spans]


def polygon_group_to_binary_mask(poly_group, image_size=800):
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    for poly in poly_group:
        if not poly or len(poly) < 3:
            continue
        pts = np.asarray(poly, dtype=np.int32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, image_size - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, image_size - 1)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def binary_mask_to_rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def build_answer(description: str, polygons, answer_mode: str):
    mask_num = len(polygons)
    if answer_mode == "seg_spans":
        spans = parse_seg_spans(description)
        if len(spans) >= mask_num:
            return "\n".join(spans[:mask_num])
        names = parse_object_names(description)
        fallback = []
        for i in range(mask_num):
            if i < len(spans):
                fallback.append(spans[i])
            elif i < len(names):
                fallback.append(f"<p> {names[i]} </p> [SEG]")
            else:
                fallback.append(f"<p> object-{i} </p> [SEG]")
        return "\n".join(fallback)

    names = parse_object_names(description)
    spans = []
    for i in range(mask_num):
        name = names[i] if i < len(names) else f"object-{i}"
        spans.append(f"<p> {name} </p> [SEG]")
    return "\n".join(spans)


def map_rel_to_split_and_filename(rel_path: str):
    name = os.path.basename(rel_path)
    split = "train" if "/train/" in rel_path or "train" in rel_path else "test"
    return split, name


def main():
    parser = argparse.ArgumentParser(description="Convert GeoPixelD (DOTA patches) to LaSeRS train_data.json format.")
    parser.add_argument("--src", required=True, help="GeoPixelD root, e.g. /root/autodl-tmp/DOTA_patches")
    parser.add_argument("--dst", required=True, help="Output base_data_path for training")
    parser.add_argument("--metadata", default="GeoPixelD.json", help="Metadata file name under --src")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Which split to build")
    parser.add_argument("--description-mode", default="gcg", choices=["gcg", "human_query"], help="description source")
    parser.add_argument("--answer-mode", default="object_names", choices=["object_names", "seg_spans"], help="answer generation mode")
    parser.add_argument("--image-size", type=int, default=800)
    parser.add_argument("--overwrite", default="0", help="Overwrite dst if exists")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    metadata_path = src_root / args.metadata

    src_resolved = src_root.resolve()
    dst_resolved = dst_root.resolve()
    if src_resolved == dst_resolved:
        raise ValueError("--dst must be different from --src to avoid overwriting source dataset")

    if not src_root.exists():
        raise FileNotFoundError(f"src not found: {src_root}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata not found: {metadata_path}")

    if dst_root.exists() and any(dst_root.iterdir()) and not is_enabled(args.overwrite):
        raise FileExistsError(f"dst exists and is not empty: {dst_root}; set --overwrite 1 to replace")
    if dst_root.exists() and is_enabled(args.overwrite):
        shutil.rmtree(dst_root)

    data = load_json(metadata_path)
    if not isinstance(data, list):
        raise ValueError("GeoPixelD metadata must be a JSON array")

    (dst_root / "train" / "annotations").mkdir(parents=True, exist_ok=True)
    (dst_root / "train").mkdir(parents=True, exist_ok=True)

    src_split_dir = src_root / args.split
    if not src_split_dir.exists():
        raise FileNotFoundError(f"split directory not found: {src_split_dir}")

    dst_images = dst_root / "train" / "images"
    if dst_images.exists() or dst_images.is_symlink():
        if dst_images.is_dir() and not dst_images.is_symlink():
            shutil.rmtree(dst_images)
        else:
            dst_images.unlink()
    os.symlink(src_split_dir, dst_images, target_is_directory=True)

    converted = []
    skipped_no_json = 0
    skipped_no_seg = 0
    skipped_bad_poly = 0

    for idx, entry in enumerate(data):
        image_rel = entry.get("image", "")
        polygons_rel = entry.get("polygons", "")
        if not image_rel or not polygons_rel:
            skipped_no_json += 1
            continue

        split_img, img_name = map_rel_to_split_and_filename(image_rel)
        split_poly, poly_name = map_rel_to_split_and_filename(polygons_rel)

        if args.split != split_img or args.split != split_poly:
            continue

        poly_path = src_root / args.split / poly_name
        if not poly_path.exists():
            skipped_no_json += 1
            continue

        anno = load_json(poly_path)
        gcg_description = anno.get("gcg_description", "")
        polygons = anno.get("polygons", [])
        if not isinstance(polygons, list) or len(polygons) == 0:
            skipped_no_seg += 1
            continue

        if args.description_mode == "human_query":
            conversations = entry.get("conversations", [])
            human_query = ""
            if conversations and isinstance(conversations, list):
                for turn in conversations:
                    if turn.get("from") == "human":
                        human_query = turn.get("value", "")
                        break
            description = human_query or gcg_description
        else:
            description = gcg_description

        answer = build_answer(gcg_description, polygons, args.answer_mode)
        mask_num = answer.count("[SEG]")
        if mask_num <= 0:
            skipped_no_seg += 1
            continue

        masks_rle = []
        valid_mask_num = min(mask_num, len(polygons))
        for i in range(valid_mask_num):
            mask = polygon_group_to_binary_mask(polygons[i], image_size=args.image_size)
            if mask.sum() <= 0:
                continue
            masks_rle.append(binary_mask_to_rle(mask))

        if len(masks_rle) == 0:
            skipped_bad_poly += 1
            continue

        if len(masks_rle) < mask_num:
            answer_lines = [line for line in answer.splitlines() if "[SEG]" in line]
            answer = "\n".join(answer_lines[: len(masks_rle)])

        converted.append(
            {
                "id": str(entry.get("id", idx)),
                "image_name": img_name,
                "description": description,
                "answer": answer,
                "mask": masks_rle,
                "source": "GeoPixelD",
            }
        )

    ann_out = dst_root / "train" / "annotations" / "train_data.json"
    save_json(ann_out, converted)

    summary = {
        "src": str(src_root),
        "dst": str(dst_root),
        "split": args.split,
        "description_mode": args.description_mode,
        "answer_mode": args.answer_mode,
        "total_metadata": len(data),
        "converted": len(converted),
        "skipped_no_json": skipped_no_json,
        "skipped_no_seg": skipped_no_seg,
        "skipped_bad_poly": skipped_bad_poly,
    }
    save_json(dst_root / "train" / "annotations" / "geopixeld_convert_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
