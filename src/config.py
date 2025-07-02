# Configuration file paths and model parameters
import os
import tensorflow as tf
import tensorflow.keras.optimizers.legacy as legacy_optimizers
import importlib
import numpy as np
import random

# Data paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
RESULTS_DIR = "results"

# File names
TRAFFIC_FILE = "traffic/metr-la.h5"
WEATHER_FILE = "weather/noaa_weather_5min.csv"

# Data processing parameters
TRAIN_RATIO = 0.7  # Adjust training set ratio
VAL_RATIO = 0.15  # Increase validation set ratio
TEST_RATIO = 0.15  # Increase test set ratio

# Global random seed
RANDOM_SEED = 42


def set_global_random_seed():
    """Set global random seed to ensure experiment reproducibility"""
    # Python built-in random module
    random.seed(RANDOM_SEED)

    # Numpy
    np.random.seed(RANDOM_SEED)

    # TensorFlow
    tf.random.set_seed(RANDOM_SEED)

    # Set TensorFlow deterministic operations
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # Enable TensorFlow deterministic operations
    tf.config.experimental.enable_op_determinism()

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

    # Set TensorFlow thread count to reduce uncertainty
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


# Set random seed when importing configuration
set_global_random_seed()

# Create a configuration class to store training parameters
class TrainingConfig:
    def __init__(self):
        self.config = {
            "batch_size": 64,  # Increase batch size for better training stability
            "epochs": 100,  # Increase training epochs
            "verbose": 1,
            "callbacks": [
                # More gentle learning rate warmup and decay strategy
                tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch: 1e-4 * tf.math.exp(0.05 * epoch)
                    if epoch < 20
                    else 1e-4 * tf.math.exp(-0.05 * (epoch - 20))
                ),
                # More gentle ReduceLROnPlateau
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.2,  # More gentle learning rate decay
                    patience=15,  # Increase patience value
                    min_lr=1e-7,
                    verbose=1,
                ),
                # Adjust EarlyStopping
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=30,  # Increase patience value
                    restore_best_weights=True,
                    min_delta=1e-5,  # Lower minimum improvement threshold
                    verbose=1,
                ),
            ],
        }


# Create global configuration instance
training_config = TrainingConfig()


def get_training_config():
    """Get training configuration"""
    # Force module reload
    importlib.reload(tf.keras.callbacks)
    return training_config.config


def custom_combined_loss(y_true, y_pred):
    """Custom combined loss function, integrating RMSE, MAE, MAPE and R²"""
    # Calculate metrics
    # MSE (for RMSE)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    rmse = tf.sqrt(mse)

    # MAE
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))

    # MAPE
    epsilon = 1e-7  # Prevent division by zero
    mape = tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    # R² (1 - ratio of residual sum of squares to total sum of squares)
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - (ss_res / (ss_tot + epsilon))

    # Combined loss (adjustable weights)
    alpha_rmse = 0.3  # RMSE weight
    alpha_mae = 0.3  # MAE weight
    alpha_mape = 0.2  # MAPE weight
    alpha_r2 = 0.2  # R² weight

    # Note: Since R² is better when larger, we use 1-R²
    combined_loss = (
        alpha_rmse * rmse
        + alpha_mae * mae
        + alpha_mape * mape * 0.01
        + alpha_r2 * (1 - r2)  # Scale MAPE to similar range
    )

    return combined_loss


def get_model_config():
    """Get model configuration"""
    return {
        "LSTM": {
            "units": [128, 64, 32],
            "dropout": 0.2,
            "l2_regularization": 1e-6,
            "optimizer": legacy_optimizers.Adam,
            "learning_rate": 1e-4,
            "loss": custom_combined_loss,
            "metrics": ["mae", "mse", "mape"],
        },
        "GRU": {
            "units": [128, 64, 32],
            "dropout": 0.2,
            "l2_regularization": 1e-6,
            "optimizer": legacy_optimizers.Adam,
            "learning_rate": 1e-4,
            "loss": custom_combined_loss,
            "metrics": ["mae", "mse", "mape"],
        },
        "TCN": {
            "optimizer": legacy_optimizers.Adam,
            "learning_rate": 1e-4,
            "loss": custom_combined_loss,
            "metrics": ["mae", "mse", "mape"],
        },
    }


# Add custom metrics
def custom_r2_score(y_true, y_pred):
    """Custom R² score calculation"""
    epsilon = 1e-7
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - (ss_res / (ss_tot + epsilon))


def custom_rmse(y_true, y_pred):
    """Custom RMSE calculation"""
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


# Add custom metrics to TensorFlow's custom objects scope
tf.keras.utils.get_custom_objects().update(
    {
        "custom_combined_loss": custom_combined_loss,
        "r2_score": custom_r2_score,
        "rmse": custom_rmse,
    }
)

# Experiment settings
EXPERIMENT_TYPES = ["baseline", "enhanced"]

# Visualization configuration
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "style": "seaborn",
    "color_palette": "Set2",
    "dpi": 300,
}

# Data processing parameter validation
assert (
    abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-10
), "Dataset split ratios must sum to 1"

# Add data processing configuration
DATA_CONFIG = {
    "sequence_length": 24,  # Increase sequence length for better CNN temporal pattern capture
    "prediction_horizon": 3,  # Keep unchanged
    "weather_feature_selection": {
        "correlation_threshold": 0.1,
        "importance_threshold": 0.05,
    },
    "features": {
        "traffic": ["avg_speed", "volume", "occupancy"],
        "weather": [
            "TMAX",
            "TMIN",
            "PRCP",
            "AWND",  # Keep key weather features
            "temp_range",
            "feels_like",
            "severe_weather",
            "rush_hour_rain",
            "wind_direction",  # Add wind direction feature
        ],
    },
}

# Add weather analysis configuration
WEATHER_ANALYSIS_CONFIG = {
    "extreme_weather_threshold": 0.85,  # Adjust extreme weather threshold
    "rush_hour_periods": [(6, 10), (16, 20)],  # Expand rush hour periods
    "weather_features": [
        "Temperature",
        "Precipitation",
        "Wind Speed",
        "Humidity",
        "Wind Chill",
        "Heat Index",
        "Visibility",
        "Pressure",
    ],
}

def get_config():
    """Get all configurations"""
    return {"training": get_training_config(), "model": get_model_config()}
