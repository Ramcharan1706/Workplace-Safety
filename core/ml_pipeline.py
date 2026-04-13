"""Classical image feature extraction, caching, training, and prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from core.config import (
    CACHE_META_FILE,
    CACHE_X_FILE,
    CACHE_Y_FILE,
    CLASSICAL_MODEL_FILES,
    DECISION_TREE_MAX_DEPTH,
    FEATURE_VERSION,
    IMG_SIZE,
    RANDOM_STATE,
    TEST_SIZE,
)
from utils import (
    dataset_signature,
    ensure_directory,
    iter_image_files,
    load_image_bgr,
    load_json,
    model_meta_path,
    normalize_path,
    save_json,
    softmax_from_scores,
)


@dataclass
class DatasetCacheResult:
    features: np.ndarray
    labels: np.ndarray
    class_labels: list[str]
    metadata: dict
    from_cache: bool
    unreadable_images: int


@dataclass
class ModelResult:
    name: str
    estimator: object
    metrics: dict[str, float]
    confusion_matrix: np.ndarray
    model_path: str


class FeatureExtractor:
    """Extract combined HOG, LBP, and HSV histogram features."""

    def __init__(self) -> None:
        self.image_size = IMG_SIZE
        self.lbp_points = 8
        self.lbp_radius = 1
        self.hsv_bins = (8, 8, 8)

    def extract_from_image(self, image_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image_bgr, self.image_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        hog_features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)

        lbp = local_binary_pattern(gray, P=self.lbp_points, R=self.lbp_radius, method="uniform")
        lbp_bins = self.lbp_points + 2
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=lbp_bins, range=(0, lbp_bins), density=True)
        lbp_hist = lbp_hist.astype(np.float32)

        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        color_parts: list[np.ndarray] = []
        for channel_index, bins in enumerate(self.hsv_bins):
            hist = cv2.calcHist([hsv], [channel_index], None, [bins], [0, 256 if channel_index else 180])
            hist = cv2.normalize(hist, None).flatten().astype(np.float32)
            color_parts.append(hist)

        return np.hstack([hog_features, lbp_hist, np.concatenate(color_parts)]).astype(np.float32)

    def extract_from_path(self, image_path: str | Path) -> np.ndarray | None:
        image = load_image_bgr(image_path)
        if image is None:
            return None
        return self.extract_from_image(image)


class ClassicalMLPipeline:
    """End-to-end cache, train, evaluate, and predict workflow for classical ML."""

    def __init__(self) -> None:
        self.extractor = FeatureExtractor()

    def dataset_summary(self, dataset_path: str | Path) -> dict:
        return dataset_signature(dataset_path)

    def _cache_metadata_matches(self, metadata: dict, signature: dict) -> bool:
        return (
            metadata.get("dataset_path") == signature.get("dataset_path")
            and metadata.get("number_images") == signature.get("number_images")
            and float(metadata.get("last_modified_time", -1)) == float(signature.get("last_modified_time", -2))
            and metadata.get("class_labels") == signature.get("class_labels")
            and metadata.get("feature_version") == FEATURE_VERSION
            and metadata.get("img_size") == signature.get("img_size")
        )

    def load_or_build_cache(self, dataset_path: str | Path) -> DatasetCacheResult:
        dataset_path = normalize_path(dataset_path)
        signature = dataset_signature(dataset_path)
        cached_meta = load_json(CACHE_META_FILE)

        if (
            CACHE_X_FILE.exists()
            and CACHE_Y_FILE.exists()
            and cached_meta is not None
            and self._cache_metadata_matches(cached_meta, signature)
        ):
            return DatasetCacheResult(
                features=np.load(CACHE_X_FILE),
                labels=np.load(CACHE_Y_FILE),
                class_labels=list(signature["class_labels"]),
                metadata=cached_meta,
                from_cache=True,
                unreadable_images=0,
            )

        class_labels = list(signature["class_labels"])
        class_to_index = {name: index for index, name in enumerate(class_labels)}
        features: list[np.ndarray] = []
        labels: list[int] = []
        unreadable_images = 0

        for class_name in class_labels:
            class_folder = Path(dataset_path) / class_name
            for image_path in iter_image_files(class_folder, recursive=True):
                feature_vector = self.extractor.extract_from_path(image_path)
                if feature_vector is None:
                    unreadable_images += 1
                    continue
                features.append(feature_vector)
                labels.append(class_to_index[class_name])

        features_array = np.asarray(features, dtype=np.float32)
        labels_array = np.asarray(labels, dtype=np.int32)
        save_json(
            CACHE_META_FILE,
            {
                **signature,
                "dataset_path": normalize_path(dataset_path),
                "class_labels": class_labels,
                "feature_version": FEATURE_VERSION,
                "img_size": list(IMG_SIZE),
            },
        )
        ensure_directory(CACHE_X_FILE.parent)
        np.save(CACHE_X_FILE, features_array)
        np.save(CACHE_Y_FILE, labels_array)

        return DatasetCacheResult(
            features=features_array,
            labels=labels_array,
            class_labels=class_labels,
            metadata=load_json(CACHE_META_FILE) or {},
            from_cache=False,
            unreadable_images=unreadable_images,
        )

    def _build_model(self, name: str):
        if name == "SVC":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
            ])
        if name == "DecisionTree":
            return DecisionTreeClassifier(max_depth=DECISION_TREE_MAX_DEPTH, random_state=RANDOM_STATE, class_weight="balanced")
        if name == "NearestCentroid":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", NearestCentroid()),
            ])
        if name == "GaussianProcess":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", GaussianProcessClassifier(random_state=RANDOM_STATE)),
            ])
        raise ValueError(f"Unsupported model: {name}")

    def train_test_split_data(self, features: np.ndarray, labels: np.ndarray):
        return train_test_split(
            features,
            labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels,
        )

    def _save_model_artifacts(self, name: str, estimator, metadata: dict) -> str:
        model_path = CLASSICAL_MODEL_FILES[name]
        ensure_directory(model_path.parent)
        joblib.dump(estimator, model_path)
        save_json(model_meta_path(model_path), metadata)
        return str(model_path)

    def train_model(
        self,
        name: str,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        class_labels: list[str],
        dataset_metadata: dict,
    ) -> ModelResult:
        estimator = self._build_model(name)
        estimator.fit(x_train, y_train)
        predictions = estimator.predict(x_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="macro", zero_division=0),
            "recall": recall_score(y_test, predictions, average="macro", zero_division=0),
            "f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        }
        matrix = confusion_matrix(y_test, predictions)
        model_path = self._save_model_artifacts(
            name,
            estimator,
            {
                "model_name": name,
                "dataset_path": dataset_metadata.get("dataset_path"),
                "number_images": dataset_metadata.get("number_images"),
                "last_modified_time": dataset_metadata.get("last_modified_time"),
                "class_labels": class_labels,
                "feature_version": FEATURE_VERSION,
                "img_size": list(IMG_SIZE),
                "train_size": int(len(x_train)),
                "test_size": int(len(x_test)),
                "model_type": type(estimator).__name__,
            },
        )

        return ModelResult(
            name=name,
            estimator=estimator,
            metrics=metrics,
            confusion_matrix=matrix,
            model_path=model_path,
        )

    def train_all_models(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        class_labels: list[str],
        dataset_metadata: dict,
        model_names: list[str] | None = None,
    ) -> tuple[dict[str, ModelResult], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        model_names = model_names or list(CLASSICAL_MODEL_FILES.keys())
        x_train, x_test, y_train, y_test = self.train_test_split_data(features, labels)
        results: dict[str, ModelResult] = {}
        for name in model_names:
            results[name] = self.train_model(name, x_train, x_test, y_train, y_test, class_labels, dataset_metadata)
        return results, (x_train, x_test, y_train, y_test)

    def load_model(self, name: str):
        model_path = CLASSICAL_MODEL_FILES[name]
        if not model_path.exists():
            raise FileNotFoundError(f"Train {name} before running predictions.")
        return joblib.load(model_path)

    def predict_features(self, estimator, features: np.ndarray) -> tuple[int, float | None]:
        features = np.asarray(features, dtype=np.float32).reshape(1, -1)
        prediction = int(estimator.predict(features)[0])

        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(features)[0]
            return prediction, float(np.max(probabilities))

        if hasattr(estimator, "decision_function"):
            scores = estimator.decision_function(features)
            probabilities = softmax_from_scores(scores)
            return prediction, float(np.max(probabilities[0]))

        return prediction, None

    def predict_image(self, image_path: str | Path, model_name: str, class_labels: list[str]) -> dict:
        image = load_image_bgr(image_path)
        if image is None:
            raise ValueError("The selected image could not be read.")

        estimator = self.load_model(model_name)
        feature_vector = self.extractor.extract_from_image(image)
        prediction, confidence = self.predict_features(estimator, feature_vector)
        return {
            "image_path": str(image_path),
            "predicted_index": prediction,
            "predicted_label": class_labels[prediction],
            "confidence_score": confidence,
        }

    def predict_folder(self, folder_path: str | Path, model_name: str, class_labels: list[str]) -> list[dict]:
        results: list[dict] = []
        for image_path in iter_image_files(folder_path, recursive=True):
            try:
                result = self.predict_image(image_path, model_name, class_labels)
                results.append(result)
            except Exception:
                continue
        return results
