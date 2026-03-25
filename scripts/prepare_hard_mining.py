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
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_bad_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid jsonl at line {line_no}: {path}") from exc
    return records


def normalize_key(sample):
    data_id = sample.get("data_id")
    if data_id is not None:
        return ("data_id", str(data_id))
    image_name = sample.get("image_name")
    if image_name:
        return ("image_name", image_name)
    return None


def build_bad_index(records, threshold: float):
    bad_index = {}

    def update_index(key, score: float, record):
        if key is None:
            return
        previous = bad_index.get(key)
        if previous is None or score > previous["loss_dice_weighted"]:
            bad_index[key] = {
                "loss_dice_weighted": score,
                "global_step": int(record.get("global_step", 0) or 0),
                "image_name": record.get("image_name"),
                "image_path": record.get("image_path"),
                "data_id": str(record.get("data_id")) if record.get("data_id") is not None else None,
            }

    for record in records:
        score = float(record.get("loss_dice_weighted", 0.0) or 0.0)
        if score < threshold:
            continue
        data_id = record.get("data_id")
        image_name = record.get("image_name")
        update_index(("data_id", str(data_id)) if data_id is not None else None, score, record)
        update_index(("image_name", image_name) if image_name else None, score, record)
    return bad_index


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    os.symlink(src, dst, target_is_directory=src.is_dir())


def main():
    parser = argparse.ArgumentParser(description="Prepare hard-mining dataset from bad_samples.jsonl")
    parser.add_argument("--src", required=True, help="Original base_data_path")
    parser.add_argument("--dst", required=True, help="Output base_data_path for hard mining")
    parser.add_argument("--bad-log", required=True, help="Path to bad_samples.jsonl")
    parser.add_argument("--threshold", type=float, default=40.0, help="Keep bad samples with weighted dice >= threshold")
    parser.add_argument("--top-k", type=int, default=0, help="Optional top-k hard samples, 0 means keep all above threshold")
    parser.add_argument("--replay-ratio", type=float, default=0.5, help="Random normal replay ratio relative to hard sample count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", default="1", help="Whether to overwrite dst if exists")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    bad_log_path = Path(args.bad_log)

    ann_path = src_root / "train" / "annotations" / "train_data.json"
    img_dir = src_root / "train" / "images"

    if not src_root.exists():
        raise FileNotFoundError(f"src not found: {src_root}")
    if not ann_path.exists():
        raise FileNotFoundError(f"annotation not found: {ann_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"image dir not found: {img_dir}")
    if not bad_log_path.exists():
        raise FileNotFoundError(f"bad sample log not found: {bad_log_path}")

    if dst_root.exists():
        if not is_enabled(args.overwrite):
            raise FileExistsError(f"dst exists and overwrite disabled: {dst_root}")
        shutil.rmtree(dst_root)

    rng = random.Random(args.seed)
    samples = load_json(ann_path)
    if not isinstance(samples, list):
        raise ValueError("train_data.json must be a list of samples")

    bad_index = build_bad_index(load_bad_records(bad_log_path), float(args.threshold))
    hard_samples = []
    hard_keys = set()
    for sample in samples:
        key = normalize_key(sample)
        if key is None or key not in bad_index:
            continue
        merged = dict(sample)
        merged["hard_mining"] = bad_index[key]
        hard_samples.append(merged)
        hard_keys.add(key)

    hard_samples.sort(
        key=lambda sample: float(sample.get("hard_mining", {}).get("loss_dice_weighted", 0.0)),
        reverse=True,
    )
    if args.top_k and args.top_k > 0:
        hard_samples = hard_samples[: args.top_k]
        hard_keys = {normalize_key(sample) for sample in hard_samples if normalize_key(sample) is not None}

    replay_candidates = [sample for sample in samples if normalize_key(sample) not in hard_keys]
    replay_count = int(round(len(hard_samples) * max(0.0, float(args.replay_ratio))))
    replay_count = min(replay_count, len(replay_candidates))
    replay_samples = rng.sample(replay_candidates, replay_count) if replay_count > 0 else []

    final_samples = hard_samples + replay_samples
    rng.shuffle(final_samples)

    if not final_samples:
        raise ValueError(
            "No samples selected for hard mining. Check threshold/top-k or whether bad_samples.jsonl is populated."
        )

    (dst_root / "train" / "annotations").mkdir(parents=True, exist_ok=True)
    (dst_root / "train").mkdir(parents=True, exist_ok=True)
    safe_symlink(img_dir, dst_root / "train" / "images")

    src_test = src_root / "test"
    if src_test.exists():
        safe_symlink(src_test, dst_root / "test")

    save_json(dst_root / "train" / "annotations" / "train_data.json", final_samples)
    save_json(
        dst_root / "train" / "annotations" / "hard_mining_meta.json",
        {
            "src": str(src_root),
            "bad_log": str(bad_log_path),
            "threshold": float(args.threshold),
            "top_k": int(args.top_k),
            "replay_ratio": float(args.replay_ratio),
            "hard_sample_count": len(hard_samples),
            "replay_sample_count": len(replay_samples),
            "final_sample_count": len(final_samples),
            "seed": int(args.seed),
            "top_examples": [
                {
                    "data_id": sample.get("id"),
                    "image_name": sample.get("image_name"),
                    "loss_dice_weighted": sample.get("hard_mining", {}).get("loss_dice_weighted"),
                }
                for sample in hard_samples[:20]
            ],
        },
    )

    print(f"[HardMining] src={src_root}")
    print(f"[HardMining] dst={dst_root}")
    print(f"[HardMining] bad_log={bad_log_path}")
    print(f"[HardMining] hard={len(hard_samples)} replay={len(replay_samples)} final={len(final_samples)}")


if __name__ == "__main__":
    main()