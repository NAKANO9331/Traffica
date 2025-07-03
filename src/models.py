import os
import tensorflow as tf
import numpy as np
import logging
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    GRU,
    Conv1D,
    MaxPooling1D,
    Input,
    LayerNormalization,
    Add,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Flatten,
    Activation,
    Concatenate,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import tensorflow.keras.optimizers.legacy as legacy_optimizers
from . import config  # modify the import method
import warnings
from sklearn.metrics import r2_score
from tqdm import tqdm
from tensorflow.keras.callbacks import Callback
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import h5py
from .utils import DataPreprocessor

# Configure GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.enable_tensor_float_32_execution(True)
        tf.config.optimizer.set_jit(True)
        tf.config.threading.set_inter_op_parallelism_threads(
            8
        )
        tf.config.threading.set_intra_op_parallelism_threads(8)
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Set log level
tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Set environment variables for performance optimization
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = "8"
os.environ["TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT"] = "1"
os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"
os.environ["TF_SYNC_ON_FINISH"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Advanced TCN block
def AdvancedTCNBlock(x, filters, kernel_size, dilation_rate, dropout_rate):
    prev_x = x
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding="causal", dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding="causal", dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)
    if prev_x.shape[-1] != filters:
        prev_x = Conv1D(filters=filters, kernel_size=1, padding="same")(prev_x)
    x = Add()([x, prev_x])
    return x

class BaselineModels:
    """Baseline model class - use simple structures"""

    def __init__(self):
        self.models = {}

    def build_lstm(self, input_shape):
        """Build basic LSTM model"""
        # Get the latest configuration
        model_config = config.get_model_config()

        model = tf.keras.Sequential(
            [
                # Add input normalization layer
                tf.keras.layers.BatchNormalization(input_shape=input_shape),
                # First LSTM layer
                tf.keras.layers.LSTM(
                    units=model_config["LSTM"]["units"][0],
                    return_sequences=True,
                    dropout=model_config["LSTM"]["dropout"],
                    recurrent_dropout=0.1,  # Add recurrent dropout
                    kernel_regularizer=tf.keras.regularizers.l2(
                        model_config["LSTM"]["l2_regularization"]
                    ),
                ),
                tf.keras.layers.BatchNormalization(),
                # Second LSTM layer
                tf.keras.layers.LSTM(
                    units=model_config["LSTM"]["units"][1],
                    return_sequences=False,
                    dropout=model_config["LSTM"]["dropout"],
                    recurrent_dropout=0.1,
                ),
                # Fully connected layer
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1),
            ]
        )
        return model

    def build_gru(self, input_shape):
        """Build basic GRU model"""
        # Get the latest configuration
        model_config = config.get_model_config()

        model = tf.keras.Sequential(
            [
                tf.keras.layers.BatchNormalization(input_shape=input_shape),
                # Bidirectional GRU
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(
                        units=model_config["GRU"]["units"][0],
                        return_sequences=True,
                        dropout=model_config["GRU"]["dropout"],
                        recurrent_dropout=0.1,
                    )
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.GRU(
                    units=model_config["GRU"]["units"][1],
                    return_sequences=False,
                    dropout=model_config["GRU"]["dropout"],
                ),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1),
            ]
        )
        return model

    def build_tcn(self, input_shape, filters=64, kernel_sizes=[2,3,4], dropout_rate=0.2, n_blocks=4, n_heads=4):
        inputs = Input(shape=input_shape)
        x = inputs
        skips = []
        for i in range(n_blocks):
            k = kernel_sizes[i % len(kernel_sizes)]
            x = AdvancedTCNBlock(x, filters=filters, kernel_size=k, dilation_rate=2**i, dropout_rate=dropout_rate)
            skips.append(x)
        x = Add()(skips)
        x = MultiHeadAttention(num_heads=n_heads, key_dim=filters)(x, x)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """Train the model"""
        try:
            training_config = config.get_training_config()
            model_config = config.get_model_config()
            logging.info(
                f"Current training configuration: epochs={training_config['epochs']}, batch_size={training_config['batch_size']}"
            )

            # Preprocess data
            X_train, y_train = DataPreprocessor.prepare_data(X_train, y_train)
            X_val, y_val = DataPreprocessor.prepare_data(X_val, y_val)

            # Use silent mode
            with tf.keras.utils.CustomObjectScope({}):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # Use mixed precision training
                    policy = tf.keras.mixed_precision.Policy("mixed_float16")
                    tf.keras.mixed_precision.set_global_policy(policy)

                    if model_name not in self.models:
                        if model_name == "LSTM":
                            model = self.build_lstm(X_train.shape[1:])
                        elif model_name == "GRU":
                            model = self.build_gru(X_train.shape[1:])
                        elif model_name == "TCN":
                            model = self.build_tcn(X_train.shape[1:])
                        else:
                            raise ValueError(f"Unknown model name: {model_name}")

                        # Compile the model
                        model.compile(
                            optimizer=model_config[model_name]["optimizer"](
                                learning_rate=model_config[model_name]["learning_rate"]
                            ),
                            loss=model_config[model_name]["loss"],
                            metrics=["mae", "mse", "mape"],
                        )

                        self.models[model_name] = model

                    logging.info(
                        f"Training {model_name} for {training_config['epochs']} epochs"
                    )

                    # Add custom metrics callback
                    custom_metrics_callback = CustomMetricsCallback(
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        batch_size=training_config["batch_size"],
                    )
                    callbacks = training_config["callbacks"] + [custom_metrics_callback, RichProgressBar()]

                    # Use tf.data.Dataset for data loading
                    train_dataset = (
                        tf.data.Dataset.from_tensor_slices((X_train, y_train))
                        .batch(training_config["batch_size"])
                        .prefetch(tf.data.AUTOTUNE)
                    )

                    val_dataset = (
                        tf.data.Dataset.from_tensor_slices((X_val, y_val))
                        .batch(training_config["batch_size"])
                        .prefetch(tf.data.AUTOTUNE)
                    )

                    # Train the model, make sure to use the epochs value in the configuration
                    history = self.models[model_name].fit(
                        train_dataset,
                        validation_data=val_dataset,
                        epochs=int(
                            training_config["epochs"]
                        ),  # Make sure epochs is an integer
                        batch_size=int(
                            training_config["batch_size"]
                        ),  # Make sure batch_size is an integer
                        verbose=0,  # 關閉原生進度條
                        callbacks=callbacks,
                        use_multiprocessing=False,  # Disable multiprocessing
                        workers=1,  # Reduce the number of workers
                    )

                    if 'X_test' in locals() and 'y_test' in locals():
                        y_pred = self.models[model_name].predict(X_test)
                        with h5py.File(f'filled_result_{model_name}.h5', 'w') as hf:
                            hf.create_dataset('y_pred', data=y_pred)
                            hf.create_dataset('y_true', data=y_test)

                    return self.models[model_name], history

        except Exception as e:
            logging.error(f"Error occurred during training: {str(e)}")
            raise e


class EnhancedModels:
    """Enhanced model class - use more complex structures to handle weather features"""

    def __init__(self):
        self.models = {}

    def build_lstm(self, input_shape):
        """Build enhanced LSTM model"""
        inputs = tf.keras.Input(shape=input_shape)

        # 1. Separate traffic and weather features
        traffic_features = inputs[:, :, :207]  # Traffic features
        weather_features = inputs[:, :, 207:]  # Weather features

        # 2. Traffic feature processing branch
        traffic = tf.keras.layers.BatchNormalization()(traffic_features)
        traffic = tf.keras.layers.LSTM(
            units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2
        )(traffic)

        # 3. Weather feature processing branch
        weather = tf.keras.layers.BatchNormalization()(weather_features)
        weather = tf.keras.layers.LSTM(units=64, return_sequences=True, dropout=0.2)(
            weather
        )

        # 4. Feature fusion
        concat = tf.keras.layers.Concatenate()([traffic, weather])

        # 5. Main network
        x = tf.keras.layers.LSTM(units=128, return_sequences=False, dropout=0.2)(concat)

        # 6. Output layer
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1)(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def build_gru(self, input_shape):
        """Build enhanced GRU model"""
        inputs = tf.keras.Input(shape=input_shape)

        # 1. Separate features
        traffic_features = inputs[:, :, :207]
        weather_features = inputs[:, :, 207:]

        # 2. Traffic feature processing
        traffic = tf.keras.layers.BatchNormalization()(traffic_features)
        traffic = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2
            )
        )(traffic)

        # 3. Weather feature processing
        weather = tf.keras.layers.BatchNormalization()(weather_features)
        weather_attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(
            weather, weather
        )
        weather = tf.keras.layers.Add()([weather, weather_attention])
        weather = tf.keras.layers.LayerNormalization()(weather)

        weather = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units=64, return_sequences=True, dropout=0.2)
        )(weather)

        # 4. Feature fusion
        concat = tf.keras.layers.Concatenate()([traffic, weather])

        # 5. Temporal attention
        temporal_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=32
        )(concat, concat)
        concat = tf.keras.layers.Add()([concat, temporal_attention])
        concat = tf.keras.layers.LayerNormalization()(concat)

        # 6. Main network
        x = tf.keras.layers.GRU(units=128, return_sequences=False, dropout=0.2)(concat)

        # 7. Residual connection
        traffic_pooled = tf.keras.layers.GlobalAveragePooling1D()(traffic)
        weather_pooled = tf.keras.layers.GlobalAveragePooling1D()(weather)
        residual = tf.keras.layers.Concatenate()([traffic_pooled, weather_pooled])

        # 8. Feature fusion
        x = tf.keras.layers.Concatenate()([x, residual])

        # 9. Output layer
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1)(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def build_tcn(self, input_shape, filters=64, kernel_sizes=[2,3,4], dropout_rate=0.2, n_blocks=4, n_heads=4):
        """Advanced dual-branch TCN: traffic/weather each use advanced TCN, then attention fusion."""
        inputs = Input(shape=input_shape)
        traffic = inputs[:, :, :207]
        weather = inputs[:, :, 207:]
        # Traffic branch
        x1 = traffic
        skips1 = []
        for i in range(n_blocks):
            k = kernel_sizes[i % len(kernel_sizes)]
            x1 = AdvancedTCNBlock(x1, filters=filters, kernel_size=k, dilation_rate=2**i, dropout_rate=dropout_rate)
            skips1.append(x1)
        x1 = Add()(skips1)
        # Weather branch
        x2 = weather
        skips2 = []
        for i in range(n_blocks):
            k = kernel_sizes[i % len(kernel_sizes)]
            x2 = AdvancedTCNBlock(x2, filters=filters//2, kernel_size=k, dilation_rate=2**i, dropout_rate=dropout_rate)
            skips2.append(x2)
        x2 = Add()(skips2)
        x = Concatenate()([x1, x2])
        x = MultiHeadAttention(num_heads=n_heads, key_dim=filters)(x, x)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """Train the model"""
        try:
            # Use silent mode
            with tf.keras.utils.CustomObjectScope({}):
                # Use context manager to suppress warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # Get the latest configuration every time you train
                    training_config = config.get_training_config()
                    model_config = config.get_model_config()

                    # Validate configuration values
                    logging.info(
                        f"Current training configuration: epochs={training_config['epochs']}, batch_size={training_config['batch_size']}"
                    )

                    # Use GPU device
                    with tf.device("/GPU:0"):
                        if model_name not in self.models:
                            if model_name == "LSTM":
                                model = self.build_lstm(X_train.shape[1:])
                            elif model_name == "GRU":
                                model = self.build_gru(X_train.shape[1:])
                            elif model_name == "TCN":
                                model = self.build_tcn(X_train.shape[1:])
                            else:
                                raise ValueError(f"Unknown model name: {model_name}")

                            # Compile the model
                            model.compile(
                                optimizer=model_config[model_name]["optimizer"](
                                    learning_rate=model_config[model_name][
                                        "learning_rate"
                                    ]
                                ),
                                loss=model_config[model_name]["loss"],
                                metrics=[
                                    "mae",
                                    "mse",
                                    "mape",
                                ],  # Add all required metrics
                            )

                            self.models[model_name] = model

                        logging.info(
                            f"Before training - epochs value: {training_config['epochs']}"
                        )
                        logging.info(
                            f"Training {model_name} for {training_config['epochs']} epochs"
                        )

                        # Add custom metrics callback
                        custom_metrics_callback = CustomMetricsCallback(
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            batch_size=training_config["batch_size"],
                        )
                        callbacks = training_config["callbacks"] + [custom_metrics_callback, RichProgressBar()]

                        # Use tf.data.Dataset for data loading
                        train_dataset = (
                            tf.data.Dataset.from_tensor_slices((X_train, y_train))
                            .batch(training_config["batch_size"])
                            .prefetch(tf.data.AUTOTUNE)
                        )

                        val_dataset = (
                            tf.data.Dataset.from_tensor_slices((X_val, y_val))
                            .batch(training_config["batch_size"])
                            .prefetch(tf.data.AUTOTUNE)
                        )

                        # Train the model, make sure to use the epochs value in the configuration
                        history = self.models[model_name].fit(
                            train_dataset,
                            validation_data=val_dataset,
                            epochs=int(
                                training_config["epochs"]
                            ),  # Make sure epochs is an integer
                            batch_size=int(
                                training_config["batch_size"]
                            ),  # Make sure batch_size is an integer
                            verbose=0,
                            callbacks=callbacks,
                            use_multiprocessing=False,
                            workers=1,
                        )

                        if 'X_test' in locals() and 'y_test' in locals():
                            y_pred = self.models[model_name].predict(X_test)
                            with h5py.File(f'filled_result_{model_name}.h5', 'w') as hf:
                                hf.create_dataset('y_pred', data=y_pred)
                                hf.create_dataset('y_true', data=y_test)

                        logging.info(
                            f"After training - epochs value: {training_config['epochs']}"
                        )
                        return self.models[model_name], history

        except Exception as e:
            logging.error(f"Error occurred during training: {str(e)}")
            raise e


class CustomMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val, batch_size=128):
        super(CustomMetricsCallback, self).__init__()
        # Preprocess the data
        self.X_train, self.y_train = DataPreprocessor.prepare_data(X_train, y_train)
        self.X_val, self.y_val = DataPreprocessor.prepare_data(X_val, y_val)
        self.batch_size = batch_size

    @tf.function
    def predict_batch(self, x):
        """Use @tf.function to accelerate prediction"""
        return self.model(x, training=False)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        try:
            # Calculate predictions in batches
            train_pred = []
            val_pred = []

            # Predict on training set
            for i in range(0, len(self.X_train), self.batch_size):
                batch_x = self.X_train[i : i + self.batch_size]
                batch_pred = self.predict_batch(batch_x)
                train_pred.append(batch_pred)

            # Predict on validation set
            for i in range(0, len(self.X_val), self.batch_size):
                batch_x = self.X_val[i : i + self.batch_size]
                batch_pred = self.predict_batch(batch_x)
                val_pred.append(batch_pred)

            # Concatenate the predictions
            train_pred = tf.concat(train_pred, axis=0)
            val_pred = tf.concat(val_pred, axis=0)

            # Convert to numpy arrays for metric calculation
            train_pred_np = train_pred.numpy().flatten()
            val_pred_np = val_pred.numpy().flatten()
            y_train_np = self.y_train.numpy().flatten()
            y_val_np = self.y_val.numpy().flatten()

            # Calculate metrics on training set
            logs["rmse"] = np.sqrt(np.mean((y_train_np - train_pred_np) ** 2))
            logs["mae"] = np.mean(np.abs(y_train_np - train_pred_np))
            logs["mape"] = (
                np.mean(np.abs((y_train_np - train_pred_np) / y_train_np)) * 100
            )
            logs["r2"] = r2_score(y_train_np, train_pred_np)

            # Calculate metrics on validation set
            logs["val_rmse"] = np.sqrt(np.mean((y_val_np - val_pred_np) ** 2))
            logs["val_mae"] = np.mean(np.abs(y_val_np - val_pred_np))
            logs["val_mape"] = (
                np.mean(np.abs((y_val_np - val_pred_np) / y_val_np)) * 100
            )
            logs["val_r2"] = r2_score(y_val_np, val_pred_np)

            # Record the current learning rate
            logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.lr)

        except Exception as e:
            logging.error(f"Error in CustomMetricsCallback: {str(e)}")
            raise


class RichProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.progress = Progress(
            TextColumn("[bold blue]Training Progress"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self.task = self.progress.add_task("epoch", total=self.epochs)
        self.progress.start()

    def on_epoch_end(self, epoch, logs=None):
        self.progress.update(self.task, advance=1, description=f"[bold blue]Training Progress Epoch {epoch+1}/{self.epochs}")
        if logs:
            postfix = "  ".join([f"{k}={v:.4f}" for k, v in logs.items() if isinstance(v, float)])
            self.progress.console.print(postfix, highlight=False)

    def on_train_end(self, logs=None):
        # If stopped early, manually complete the progress bar and show EarlyStopping message
        if not self.progress.finished:
            self.progress.update(self.task, completed=self.epochs)
            self.progress.console.print("[yellow]Training stopped early (EarlyStopping)[/yellow]")
        self.progress.stop()
