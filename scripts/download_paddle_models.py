"""Скачивает официальные inference-модели PaddleOCR в paddle_models/."""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "paddle_models"

BASE_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
    "official_inference_model/paddle3.0.0"
)

MODELS = {
    "det": {
        "url": f"{BASE_URL}/PP-OCRv6_medium_det_infer.tar",
        "archive": MODELS_DIR / "PP-OCRv6_medium_det_infer.tar",
        "extract_dir": MODELS_DIR / "PP-OCRv6_medium_det",
    },
    "rec": {
        "url": f"{BASE_URL}/PP-OCRv6_medium_rec_infer.tar",
        "archive": MODELS_DIR / "PP-OCRv6_medium_rec_infer.tar",
        "extract_dir": MODELS_DIR / "PP-OCRv6_medium_rec",
    },
    "textline": {
        "url": f"{BASE_URL}/PP-LCNet_x1_0_textline_ori_infer.tar",
        "archive": MODELS_DIR / "PP-LCNet_x1_0_textline_ori_infer.tar",
        "extract_dir": MODELS_DIR / "PP-LCNet_x1_0_textline_ori",
    },
}


def _progress(prefix: str):
    def hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size <= 0:
            print(f"\r{prefix}: {downloaded / 1_048_576:.1f} MB", end="", flush=True)
            return
        pct = min(100.0, downloaded * 100.0 / total_size)
        print(
            f"\r{prefix}: {pct:5.1f}%  {downloaded / 1_048_576:.1f}/{total_size / 1_048_576:.1f} MB",
            end="",
            flush=True,
        )

    return hook


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"Skip (exists): {dest.name}")
        return dest
    print(f"Downloading {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urlretrieve(url, tmp, _progress(dest.name))
    print()
    tmp.replace(dest)
    return dest


def extract_tar(archive: Path, dest_dir: Path) -> Path:
    marker = dest_dir / "inference.pdiparams"
    if marker.exists():
        print(f"Skip extract (exists): {dest_dir.name}")
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive.name} -> {dest_dir.name}")
    with tarfile.open(archive, "r") as tar:
        tar.extractall(dest_dir.parent)
    nested = dest_dir.parent / archive.name.replace(".tar", "")
    if nested.exists() and nested.resolve() != dest_dir.resolve():
        if dest_dir.exists() and not any(dest_dir.iterdir()):
            dest_dir.rmdir()
        if not dest_dir.exists():
            nested.replace(dest_dir)
        else:
            for item in nested.iterdir():
                target = dest_dir / item.name
                if target.exists():
                    continue
                item.replace(target)
            nested.rmdir()
    return dest_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PaddleOCR models to paddle_models/")
    parser.add_argument(
        "--only",
        choices=["all", "det", "rec", "textline"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = list(MODELS)
    if args.only != "all":
        selected = [args.only]

    for key in selected:
        item = MODELS[key]
        download_file(item["url"], item["archive"])
        extract_tar(item["archive"], item["extract_dir"])

    print("Done:")
    for key in selected:
        path = MODELS[key]["extract_dir"]
        ok = (path / "inference.pdiparams").exists()
        print(f"  {key}: {path} {'OK' if ok else 'MISSING'}")


if __name__ == "__main__":
    main()
