"""Скачивает Roboflow container-number и учит YOLO26s на RTX 5060."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
DATASET_DIR = DATASETS / "fmt-yolov8"
WEIGHTS_OUT = ROOT / "src" / "yolo_container_number_yolo26s.pt"


def download_dataset(api_key: str) -> Path:
    from roboflow import Roboflow

    DATASETS.mkdir(parents=True, exist_ok=True)
    yaml_path = DATASET_DIR / "data.yaml"
    if yaml_path.exists():
        print(f"Датасет уже есть: {DATASET_DIR}")
        return yaml_path

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("jq-zmko3").project("container-number-pmov4")
    version = project.version(4)
    dataset = version.download("yolov11", location=str(DATASET_DIR))
    print(f"Скачано: {dataset.location}")
    return Path(dataset.location) / "data.yaml"


def print_dataset_info(yaml_path: Path) -> None:
    text = yaml_path.read_text(encoding="utf-8")
    print("----- data.yaml -----")
    print(text)
    print("---------------------")
    for split in ("train", "valid", "val", "test"):
        images = list((yaml_path.parent / split / "images").glob("*")) if (yaml_path.parent / split / "images").exists() else []
        if images:
            print(f"{split}: {len(images)} картинок")


def train(yaml_path: Path, epochs: int, batch: int, imgsz: int) -> Path:
    model = YOLO("yolo26s.pt")
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=int(batch),
        device=0,
        workers=2,
        amp=True,
        patience=20,
        project=str(ROOT / "runs" / "container_det"),
        name="yolo26s",
        exist_ok=True,
        pretrained=True,
        close_mosaic=10,
        plots=True,
    )
    best = ROOT / "runs" / "container_det" / "yolo26s" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Не найден {best}")
    shutil.copy2(best, WEIGHTS_OUT)
    print(f"Веса скопированы в {WEIGHTS_OUT}")
    return WEIGHTS_OUT


def compare_on_test(new_weights: Path) -> None:
    test_dir = ROOT / "test"
    old = YOLO(str(ROOT / "src" / "yolo_container_number.pt"))
    new = YOLO(str(new_weights))
    files = sorted(
        p for p in test_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.stem.lower().startswith("car")
    )
    old_hits = 0
    new_hits = 0
    print(f"{'file':22} {'old':>6} {'new':>6}")
    for path in files:
        old_n = len(old(str(path), conf=0.25, verbose=False)[0].boxes or [])
        new_n = len(new(str(path), conf=0.25, verbose=False)[0].boxes or [])
        old_hits += int(old_n > 0)
        new_hits += int(new_n > 0)
        mark = " " if new_n >= old_n else " <"
        print(f"{path.name:22} {old_n:6} {new_n:6}{mark}")
    print(f"Детекций >0: old {old_hits}/{len(files)}  new {new_hits}/{len(files)}")
    print("Чтобы включить новую модель, в src/settings.py поставьте:")
    print('YOLO_CONTAINER_WEIGHTS = "src/yolo_container_number_yolo26s.pt"')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None, help="Roboflow API key, нужен только если датасета ещё нет")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    yaml_path = DATASET_DIR / "data.yaml"
    if yaml_path.exists():
        print(f"Датасет уже есть: {DATASET_DIR}")
    elif args.api_key:
        yaml_path = download_dataset(args.api_key)
    else:
        raise SystemExit("Нет datasets/fmt-yolov8/data.yaml. Передайте --api-key для скачивания.")

    print_dataset_info(yaml_path)
    weights = train(yaml_path, args.epochs, args.batch, args.imgsz)
    compare_on_test(weights)


if __name__ == "__main__":
    main()
