import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import importlib
import sys
from rich.console import Console
from rich.logging import RichHandler

import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('tableau-colorblind10')
sns.set_palette('Set2')               
# Ensure to reload the configuration every time
if "src.config" in sys.modules:
    importlib.reload(sys.modules["src.config"])
    # Reload related modules
    if "src.models" in sys.modules:
        importlib.reload(sys.modules["src.models"])

from . import config

from .data_utils import (
    load_traffic_data,
    load_weather_data,
    unified_filled_traffic_data,
    load_filled_traffic_data,
    DataProcessor,
    calculate_missing_rate_and_plot
)
from .models import BaselineModels, EnhancedModels
from .analysis import evaluate_model, evaluate_models, calculate_improvements, DataVisualizer

# Set environment variables to disable all TensorFlow and CUDA warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=info, 2=warning, 3=error
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimization
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"
os.environ["TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE"] = "0"
os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "0"
os.environ["TF_ENABLE_RESOURCE_VARIABLES"] = "true"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_DISABLE_CONTROL_FLOW_V2"] = "1"
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "0"
os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM"] = "1"

# Disable all Python warnings
import warnings

warnings.filterwarnings("ignore")

# Configure TensorFlow log level
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

# Disable TensorFlow executor warnings
tf.debugging.disable_traceback_filtering()

class CustomRichHandler(RichHandler):
    def emit(self, record):
        if record.levelno == logging.INFO:
            record.msg = f'[bold green][INFO][/bold green] {record.msg}'
        super().emit(record)

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[CustomRichHandler(console=console, rich_tracebacks=True, markup=True)]
)

IMPUTATION_METHODS = ["btmf"]
MODEL_NAMES = ["LSTM", "GRU", "TCN"]

def setup_logging():
    """Configure logging settings"""
    # Create logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Generate log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")

    # Configure log format
    log_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Configure file handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)

    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)

    # Create custom log filter
    class TensorFlowFilter(logging.Filter):
        def filter(self, record):
            return not any(
                msg in str(record.getMessage()).lower()
                for msg in [
                    "executing op",
                    "gradient",
                    "executor",
                    "custom operations",
                    "numa node",
                    "tf-trt",
                    "tensorflow",
                    "cuda",
                    "gpu",
                    "warning",
                    "warn",
                ]
            )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Add filters to both handlers
    tf_filter = TensorFlowFilter()
    file_handler.addFilter(tf_filter)
    console_handler.addFilter(tf_filter)

    # Configure TensorFlow log
    tf_logger = logging.getLogger("tensorflow")
    tf_logger.addFilter(tf_filter)
    tf_logger.setLevel(logging.ERROR)

    # Disable NumPy warnings
    np.seterr(all="ignore")

    # Disable Pandas warnings
    pd.options.mode.chained_assignment = None

    # Log GPU information
    if tf.config.list_physical_devices("GPU"):
        for gpu in tf.config.list_physical_devices("GPU"):
            logging.info(f"Found GPU device: {gpu}")
    else:
        logging.info("No GPU device found, using CPU for training")

    return log_file  # Return log file path for later use



def setup_directories():
    directories = [
        os.path.join(config.RESULTS_DIR, "figures"),
        os.path.join(config.RESULTS_DIR, "figures", "traffic"),
        os.path.join(config.RESULTS_DIR, "figures", "weather"),
        os.path.join(config.RESULTS_DIR, "figures", "models"),
        os.path.join(config.RESULTS_DIR, "figures", "comparison"),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_or_load_model(model, model_path):
    if os.path.exists(model_path):
        logging.info(f"Model exists, loading: {model_path}")
        return tf.keras.models.load_model(model_path, compile=False)
    else:
        model.save(model_path)
        logging.info(f"Model saved: {model_path}")
        return model


def run_full_experiment():
    all_results = {}
    data_processor = DataProcessor()
    visualizer = DataVisualizer()
    for impute_method in IMPUTATION_METHODS:
        logging.info(f"\n=== Imputation method: {impute_method} ===")
        traffic_data = load_filled_traffic_data(method=impute_method)
        weather_data = load_weather_data()
        visualizer.plot_weather_analysis(weather_data, visualizer.subdirs["weather"])
        visualizer.plot_traffic_weather_relationship(traffic_data, weather_data, visualizer.subdirs["traffic"])
        visualizer.plot_weather_box_by_condition(weather_data, visualizer.subdirs["weather"])
        _, scaler, target_scaler = data_processor.prepare_data(
            traffic_data=traffic_data,
            weather_data=weather_data
        )
        X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx = data_processor.prepare_sequences(
            traffic_data=traffic_data,
            sequence_length=config.DATA_CONFIG["sequence_length"],
        )
        X_train_enh, y_train_enh, X_val_enh, y_val_enh, X_test_enh, y_test_enh, train_idx_enh, val_idx_enh, test_idx_enh = data_processor.prepare_sequences(
            traffic_data=traffic_data,
            weather_data=weather_data,
            sequence_length=config.DATA_CONFIG["sequence_length"],
        )
        all_results[impute_method] = {}
        model_predictions = {} 
        for model_name in MODEL_NAMES:
            logging.info(f"\n--- Model: {model_name} ---")
            model_b_path = os.path.join(config.RESULTS_DIR, f"{impute_method}_{model_name}_baseline.h5")
            if os.path.exists(model_b_path):
                logging.info(f"Model exists, loading: {model_b_path}")
                model_b = tf.keras.models.load_model(model_b_path, compile=False)
                history_b = None  
            else:
                baseline_models = BaselineModels()
                model_b, history_b = baseline_models.train_model(model_name, X_train, y_train, X_val, y_val)
                model_b.save(model_b_path)
                logging.info(f"Model saved: {model_b_path}")
            pred_b = model_b.predict(X_test, batch_size=32, verbose=0)
            metrics_b = evaluate_model(y_test, pred_b.flatten(), model_name, model=None, feature_names=None, history=history_b)
            model_e_path = os.path.join(config.RESULTS_DIR, f"{impute_method}_{model_name}_enhanced.h5")
            if os.path.exists(model_e_path):
                logging.info(f"Model exists, loading: {model_e_path}")
                model_e = tf.keras.models.load_model(model_e_path, compile=False)
                history_e = None
            else:
                enhanced_models = EnhancedModels()
                model_e, history_e = enhanced_models.train_model(model_name, X_train_enh, y_train_enh, X_val_enh, y_val_enh)
                model_e.save(model_e_path)
                logging.info(f"Model saved: {model_e_path}")
            feature_names = [f"traffic_t-{i}" for i in range(config.DATA_CONFIG["sequence_length"])]
            feature_names.extend([f"weather_{i}" for i in range(X_train_enh.shape[-1] - config.DATA_CONFIG["sequence_length"])])
            pred_e = model_e.predict(X_test_enh, batch_size=32, verbose=0)
            metrics_e = evaluate_model(y_test_enh, pred_e.flatten(), model_name, model=None, feature_names=feature_names, history=history_e)
            all_results[impute_method][model_name] = {
                "baseline": {"metrics": metrics_b, "pred": pred_b.flatten(), "history": history_b},
                "enhanced": {"metrics": metrics_e, "pred": pred_e.flatten(), "history": history_e},
            }
            visualizer.plot_prediction_vs_actual(y_true=y_test_enh, y_pred=pred_b.flatten(), timestamps=traffic_data.index[-len(y_test_enh):], model_name=f"{impute_method}_{model_name}_baseline", save_path=visualizer.subdirs["models"])
            visualizer.plot_prediction_vs_actual(y_true=y_test_enh, y_pred=pred_e.flatten(), timestamps=traffic_data.index[-len(y_test_enh):], model_name=f"{impute_method}_{model_name}_enhanced", save_path=visualizer.subdirs["models"])
            last_pred_y = pred_e.flatten()
            last_pred_y_real = target_scaler.inverse_transform(last_pred_y.reshape(-1, 1)).flatten()
            model_predictions[model_name] = last_pred_y_real
        visualizer.plot_model_improvements(
            baseline_metrics={m: all_results[impute_method][m]["baseline"]["metrics"] for m in MODEL_NAMES if m in all_results[impute_method]},
            enhanced_metrics={m: all_results[impute_method][m]["enhanced"]["metrics"] for m in MODEL_NAMES if m in all_results[impute_method]},
            save_path=visualizer.subdirs["models"]
        )
        for model_name in MODEL_NAMES:
            model_e_path = os.path.join(config.RESULTS_DIR, f"{impute_method}_{model_name}_enhanced.h5")
            model_e = tf.keras.models.load_model(model_e_path, compile=False)
            feature_names = [f"traffic_t-{i}" for i in range(config.DATA_CONFIG["sequence_length"])]
            feature_names.extend([f"weather_{i}" for i in range(X_train_enh.shape[-1] - config.DATA_CONFIG["sequence_length"])])
            visualizer.plot_feature_importance_analysis(
                model=model_e,
                feature_names=feature_names,
                save_path=visualizer.subdirs["models"]
            )
        predictions_dict = {f"{model_name}_baseline": all_results[impute_method][model_name]["baseline"]["pred"] for model_name in MODEL_NAMES}
        predictions_dict.update({f"{model_name}_enhanced": all_results[impute_method][model_name]["enhanced"]["pred"] for model_name in MODEL_NAMES})
        visualizer.plot_prediction_comparison(
            y_true=y_test_enh,
            predictions_dict=predictions_dict,
            save_path=visualizer.subdirs["comparison"]
        )
        visualizer.create_performance_table(
            baseline_metrics={m: all_results[impute_method][m]["baseline"]["metrics"] for m in MODEL_NAMES if m in all_results[impute_method]},
            enhanced_metrics={m: all_results[impute_method][m]["enhanced"]["metrics"] for m in MODEL_NAMES if m in all_results[impute_method]},
            improvements=None,
            save_path=visualizer.subdirs["comparison"]
        )
        # DEBUG: Output current all_results[impute_method] keys
        logging.info(f"all_results[{impute_method}] keys: {list(all_results[impute_method].keys())}")
    logging.info("All experiments completed. Results saved.")
    # === Save the complete dataset with predictions ===
    # Use TCN enhanced as example
    try:
        best_model = "TCN"
        impute_method = IMPUTATION_METHODS[0]
        # Reload the filled dataset and predictions
        traffic_data = load_filled_traffic_data(method=impute_method)
        data_processor = DataProcessor()
        weather_data = load_weather_data()
        _, _, target_scaler = data_processor.prepare_data(traffic_data=traffic_data, weather_data=weather_data)
        _, _, _, _, _, _, _, _, test_idx = data_processor.prepare_sequences(traffic_data=traffic_data, weather_data=weather_data, sequence_length=config.DATA_CONFIG["sequence_length"])
        # Get predictions (inverse transformed)
        pred = all_results[impute_method][best_model]["enhanced"]["pred"]
        pred_real = target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        # Add a new column for predictions corresponding to test_idx
        traffic_data_result = traffic_data.copy()
        traffic_data_result.loc[test_idx, "prediction_TCN_enhanced"] = pred_real
        # Save to h5 (relative path)
        save_path = "data/completed/traffic_result.h5"
        traffic_data_result.to_hdf(save_path, key='df', mode='w')
        logging.info(f"Traffic result with predictions saved to: {save_path}")
    except Exception as e:
        logging.error(f"Failed to save traffic_result.h5: {e}")
    # === End of save ===
    # Consistency check (log only, no duplicate print)
    for impute_method in all_results:
        model_keys = set(all_results[impute_method].keys())
        model_names_set = set(MODEL_NAMES)
        missing_in_results = model_names_set - model_keys
        extra_in_results = model_keys - model_names_set
        if missing_in_results:
            logging.warning(f"The following models are missing in all_results['{impute_method}']: {missing_in_results}")
        if extra_in_results:
            logging.warning(f"all_results['{impute_method}'] contains models not in MODEL_NAMES: {extra_in_results}")
    return all_results


def main():
    """Main function"""
    try:
        # Ensure random seed is correctly set
        config.set_global_random_seed()

        # Set up logging
        log_file = setup_logging()

        # Log experiment configuration
        logging.info("Experiment configuration:")
        logging.info(f"Random seed: {config.RANDOM_SEED}")
        logging.info(
            f"Data set split ratio: Training set={config.TRAIN_RATIO}, Validation set={config.VAL_RATIO}, Test set={config.TEST_RATIO}"
        )
        logging.info(f"Sequence length: {config.DATA_CONFIG['sequence_length']}")
        logging.info(f"Prediction horizon: {config.DATA_CONFIG['prediction_horizon']}")

        # Create necessary directories
        setup_directories()
        run_full_experiment()

        logging.info("\n" + "-" * 30)
        logging.info("Experiment Completed")
        logging.info("-" * 30)

    except Exception as e:
        logging.error(f"\nError occurred during the experiment: {str(e)}")
        logging.error("Detailed error information:", exc_info=True)
        raise


if __name__ == "__main__":
    main()
