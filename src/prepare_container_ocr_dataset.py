from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


ALNUM_PATTERN = re.compile(r"^[A-Z0-9]+$")
DEFAULT_ALLOWED_LENGTHS = {4, 7, 10, 11}


def _sanitize_label(label: str) -> str:
    return label.strip().upper()


def _resolve_image_path(source_dir: Path, stem: str) -> Path | None:
    candidates = [
        source_dir / "images" / f"{stem}.jpg",
        source_dir / f"{stem}.jpg",
        source_dir / "images" / f"{stem}.png",
        source_dir / f"{stem}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _iter_annotations(source_dir: Path, label_dir: Path) -> Iterable[tuple[Path, int, tuple[int, int, int, int], str]]:
    for txt_path in sorted(label_dir.glob("*.txt")):
        image_path = _resolve_image_path(source_dir, txt_path.stem)
        if image_path is None:
            continue

        for idx, raw_line in enumerate(txt_path.read_text(encoding="utf-8").splitlines()):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < 5:
                continue

            try:
                x1, y1, x2, y2 = map(int, parts[:4])
            except ValueError:
                continue

            label = _sanitize_label(",".join(parts[4:]))
            yield image_path, idx, (x1, y1, x2, y2), label


def _expand_box(box: tuple[int, int, int, int], width: int, height: int, padding_ratio: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad_x = max(2, int((x2 - x1) * padding_ratio))
    pad_y = max(2, int((y2 - y1) * padding_ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def _split_name(image_path: Path, idx: int, label: str, val_ratio: float) -> tuple[str, str]:
    digest = hashlib.md5(f"{image_path.name}:{idx}:{label}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    subset = "val" if bucket < val_ratio else "train"
    filename = f"{image_path.stem}_{idx:02d}.png"
    return subset, filename


def build_dataset(
    source_dir: Path,
    output_dir: Path,
    allowed_lengths: set[int],
    padding_ratio: float,
    val_ratio: float,
) -> dict:
    label_dir = source_dir / "images_label"

    if not label_dir.exists():
        raise FileNotFoundError("Expected `images/` and `images_label/` inside source dataset.")

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[str] = []
    val_rows: list[str] = []
    stats = {
        "source_images": 0,
        "accepted_samples": 0,
        "skipped_bad_label": 0,
        "skipped_bad_box": 0,
        "train_samples": 0,
        "val_samples": 0,
    }

    seen_images: set[Path] = set()

    for image_path, idx, box, label in _iter_annotations(source_dir, label_dir):
        seen_images.add(image_path)

        if not ALNUM_PATTERN.fullmatch(label) or len(label) not in allowed_lengths:
            stats["skipped_bad_label"] += 1
            continue

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            x1, y1, x2, y2 = _expand_box(box, image.width, image.height, padding_ratio)
            if x2 <= x1 or y2 <= y1:
                stats["skipped_bad_box"] += 1
                continue

            crop = image.crop((x1, y1, x2, y2))

        subset, filename = _split_name(image_path, idx, label, val_ratio)
        subset_dir = train_dir if subset == "train" else val_dir
        output_path = subset_dir / filename
        crop.save(output_path)

        rel_path = f"{subset}/{filename}"
        row = f"{rel_path}\t{label}"
        if subset == "train":
            train_rows.append(row)
            stats["train_samples"] += 1
        else:
            val_rows.append(row)
            stats["val_samples"] += 1
        stats["accepted_samples"] += 1

    stats["source_images"] = len(seen_images)

    (output_dir / "train_label.txt").write_text(
        "\n".join(train_rows) + ("\n" if train_rows else ""),
        encoding="utf-8",
    )
    (output_dir / "val_label.txt").write_text(
        "\n".join(val_rows) + ("\n" if val_rows else ""),
        encoding="utf-8",
    )
    (output_dir / "alnum_dict.txt").write_text(
        "\n".join(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")) + "\n",
        encoding="utf-8",
    )
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ContainerNumber-OCR dataset to PaddleOCR recognition format."
    )
    parser.add_argument(
        "--source-dir",
        default="external_datasets/container_ocr_raw/unpacked",
        help="Directory that contains `images/` and `images_label/`.",
    )
    parser.add_argument(
        "--output-dir",
        default="train_data/rec_container_repo",
        help="Destination PaddleOCR recognition dataset directory.",
    )
    parser.add_argument(
        "--allowed-lengths",
        default="4,7,10,11",
        help="Comma-separated label lengths to keep. Default keeps container prefixes, serials, type codes, and full numbers.",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.04,
        help="Extra crop padding ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_lengths = {int(item) for item in args.allowed_lengths.split(",") if item.strip()}
    stats = build_dataset(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        allowed_lengths=allowed_lengths,
        padding_ratio=args.padding_ratio,
        val_ratio=args.val_ratio,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
