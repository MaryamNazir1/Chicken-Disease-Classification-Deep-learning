import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf

tf.config.run_functions_eagerly(True)
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        """Load VGG16 base model and save it (without top layers)."""
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
        )

        # Save the base model (without optimizer state)
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """Attach classification head and compile."""
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False  # <-- fixed (previously `model.trainable`)
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Add classification head
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(
            flatten_in
        )

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

        # Always recompile with fresh optimizer
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        """Create full model with classification head and save it."""
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
        )

        # Save the updated model WITHOUT optimizer state
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # Ensure we don’t save optimizer state (so reload is clean)
        model.save(path, include_optimizer=False)
