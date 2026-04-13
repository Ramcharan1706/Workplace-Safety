"""Utility helpers for dataset scanning, file IO, and confidence handling."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from core.config import FEATURE_VERSION, IMG_SIZE, SUPPORTED_EXTENSIONS


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_path(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def is_supported_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def iter_image_files(folder: str | Path, recursive: bool = True) -> Iterable[Path]:
    folder_path = Path(folder)
    iterator = folder_path.rglob("*") if recursive else folder_path.iterdir()
    for candidate in iterator:
        if candidate.is_file() and is_supported_image(candidate.name):
            yield candidate


def list_class_directories(dataset_path: str | Path) -> list[Path]:
    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        return []
    return [entry for entry in sorted(dataset_root.iterdir()) if entry.is_dir()]


def count_images_by_class(dataset_path: str | Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_dir in list_class_directories(dataset_path):
        counts[class_dir.name] = sum(1 for _ in iter_image_files(class_dir, recursive=True))
    return counts


def dataset_signature(dataset_path: str | Path) -> dict:
    dataset_root = Path(dataset_path)
    class_dirs = list_class_directories(dataset_root)
    class_labels = [class_dir.name for class_dir in class_dirs]
    class_counts = {label: 0 for label in class_labels}
    number_images = 0
    last_modified_time = 0.0

    for class_dir in class_dirs:
        for image_path in iter_image_files(class_dir, recursive=True):
            number_images += 1
            class_counts[class_dir.name] += 1
            try:
                stat_info = image_path.stat()
            except OSError:
                continue
            last_modified_time = max(last_modified_time, stat_info.st_mtime)

    return {
        "dataset_path": normalize_path(dataset_root),
        "number_images": number_images,
        "last_modified_time": last_modified_time,
        "class_labels": class_labels,
        "class_counts": class_counts,
        "img_size": list(IMG_SIZE),
        "feature_version": FEATURE_VERSION,
    }


def load_json(path: str | Path) -> dict | None:
    json_path = Path(path)
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_json(path: str | Path, payload: dict) -> None:
    json_path = Path(path)
    ensure_directory(json_path.parent)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_image_bgr(image_path: str | Path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    return image


def softmax_from_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-12, None)


def model_meta_path(model_path: str | Path) -> Path:
    model_file = Path(model_path)
    return model_file.with_suffix(model_file.suffix + ".meta.json")
