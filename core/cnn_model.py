"""Optional TensorFlow / MobileNetV2 deep-learning workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from core.config import CNN_BATCH_SIZE, CNN_EPOCHS, CNN_IMAGE_SIZE, CNN_META_FILE, CNN_MODEL_FILE, FEATURE_VERSION, RANDOM_STATE
from utils import ensure_directory, load_json, save_json

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - optional dependency
    tf = None


@dataclass
class CNNTrainingResult:
    model_path: str
    metrics: dict[str, float]
    class_labels: list[str]


class CNNPipeline:
    """MobileNetV2 transfer-learning helper used when TensorFlow is installed."""

    def available(self) -> bool:
        return tf is not None

    def _require_tensorflow(self):
        if tf is None:
            raise RuntimeError(
                "TensorFlow is not installed. Install tensorflow to enable the CNN option."
            )

    def _build_model(self, num_classes: int):
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(*CNN_IMAGE_SIZE, 3),
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(*CNN_IMAGE_SIZE, 3))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _make_datasets(self, dataset_path: str | Path):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            validation_split=0.2,
            subset="training",
            seed=RANDOM_STATE,
            image_size=CNN_IMAGE_SIZE,
            batch_size=CNN_BATCH_SIZE,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            validation_split=0.2,
            subset="validation",
            seed=RANDOM_STATE,
            image_size=CNN_IMAGE_SIZE,
            batch_size=CNN_BATCH_SIZE,
        )
        class_labels = list(train_ds.class_names)
        autotune = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000, seed=RANDOM_STATE).prefetch(buffer_size=autotune)
        val_ds = val_ds.cache().prefetch(buffer_size=autotune)
        return train_ds, val_ds, class_labels

    def train(self, dataset_path: str | Path) -> CNNTrainingResult:
        self._require_tensorflow()
        train_ds, val_ds, class_labels = self._make_datasets(dataset_path)
        model = self._build_model(len(class_labels))

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ]
        model.fit(train_ds, validation_data=val_ds, epochs=CNN_EPOCHS, callbacks=callbacks, verbose=0)

        y_true: list[int] = []
        y_pred: list[int] = []
        for batch_images, batch_labels in val_ds:
            predictions = model.predict(batch_images, verbose=0)
            y_true.extend(batch_labels.numpy().tolist())
            y_pred.extend(np.argmax(predictions, axis=1).tolist())

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

        ensure_directory(CNN_MODEL_FILE.parent)
        model.save(CNN_MODEL_FILE)
        save_json(
            CNN_META_FILE,
            {
                "model_name": "CNN-MobileNetV2",
                "class_labels": class_labels,
                "image_size": list(CNN_IMAGE_SIZE),
                "feature_version": FEATURE_VERSION,
                "model_path": str(CNN_MODEL_FILE),
            },
        )
        return CNNTrainingResult(model_path=str(CNN_MODEL_FILE), metrics=metrics, class_labels=class_labels)

    def load(self):
        self._require_tensorflow()
        if not CNN_MODEL_FILE.exists():
            raise FileNotFoundError("Train the CNN model first.")
        return tf.keras.models.load_model(CNN_MODEL_FILE)

    def predict_image(self, image_path: str | Path) -> dict:
        self._require_tensorflow()
        metadata = load_json(CNN_META_FILE)
        if metadata is None:
            raise FileNotFoundError("Train the CNN model first.")

        model = self.load()
        image = tf.keras.utils.load_img(image_path, target_size=CNN_IMAGE_SIZE)
        array = tf.keras.utils.img_to_array(image)
        array = np.expand_dims(array, axis=0)
        probabilities = model.predict(array, verbose=0)[0]
        prediction = int(np.argmax(probabilities))
        class_labels = metadata.get("class_labels", [])
        return {
            "image_path": str(image_path),
            "predicted_index": prediction,
            "predicted_label": class_labels[prediction],
            "confidence_score": float(np.max(probabilities)),
        }
