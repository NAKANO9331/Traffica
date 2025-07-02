import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

COLORS = {
    "orange": "#FFB347",
    "blue": "#2E86C1",
    "red": "#FF4C4C",
    "gray": "#888888",
    "baseline": "#FF9900",
    "enhanced": "#2E86C1",
    "bar": "#FFB347",
    "line": "#2E86C1",
    "fill": "#B0C4DE",
    "bbox_white": {"facecolor": "white", "alpha": 0.8},
}

def zscore(x):
    """Z-score normalization"""
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

class DataPreprocessor:
    @staticmethod
    def prepare_data(X, y):
        """Prepare data, ensure correct data types and formats"""
        # Use float32 instead of float16, as some operations require higher precision
        import tensorflow as tf
        X = tf.cast(X, tf.float32)
        y = tf.cast(y, tf.float32)
        return X, y 

def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def save_figure(path, dpi=300, bbox_inches="tight"):
    """Save current matplotlib figure to file."""
    plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches)

def close_plt():
    """Close current matplotlib figure."""
    plt.close()

def tight_layout():
    """Apply tight layout to current matplotlib figure."""
    plt.tight_layout()

def create_figure(figsize=(8, 6)):
    """Create a new matplotlib figure."""
    return plt.figure(figsize=figsize)

def add_subplot(rows, cols, idx):
    """Add a subplot to the current figure."""
    return plt.subplot(rows, cols, idx)

def add_title(title, fontsize=14):
    """Set the title for the current axes."""
    plt.title(title, fontsize=fontsize)

def add_xlabel(label, fontsize=12):
    """Set the x-axis label for the current axes."""
    plt.xlabel(label, fontsize=fontsize)

def add_ylabel(label, fontsize=12):
    """Set the y-axis label for the current axes."""
    plt.ylabel(label, fontsize=fontsize)

def add_legend(ax=None, **kwargs):
    """Add a legend to the current axes or given axes."""
    if ax is None:
        plt.legend(**kwargs)
    else:
        ax.legend(**kwargs)

def add_grid(ax=None, **kwargs):
    """Add a grid to the current axes or given axes."""
    if ax is None:
        plt.grid(**kwargs)
    else:
        ax.grid(**kwargs)

def add_xticks(ticks, labels=None, rotation=0):
    """Set x-ticks and optionally labels and rotation."""
    plt.xticks(ticks, labels, rotation=rotation)

def add_yticks(ticks, labels=None, rotation=0):
    """Set y-ticks and optionally labels and rotation."""
    plt.yticks(ticks, labels, rotation=rotation)

def add_suptitle(title, fontsize=16, y=1.05):
    """Set the super title for the current figure."""
    plt.suptitle(title, fontsize=fontsize, y=y)

def plot_line(x, y, **kwargs):
    """Plot a line on the current axes."""
    plt.plot(x, y, **kwargs)

def plot_bar(x, height, **kwargs):
    """Plot a bar chart on the current axes."""
    plt.bar(x, height, **kwargs)

def plot_scatter(x, y, **kwargs):
    """Plot a scatter plot on the current axes."""
    plt.scatter(x, y, **kwargs)

def plot_hist(data, bins=50, **kwargs):
    """Plot a histogram on the current axes."""
    plt.hist(data, bins=bins, **kwargs)

def plot_heatmap(data, **kwargs):
    """Plot a heatmap using seaborn."""
    sns.heatmap(data, **kwargs)

def plot_box(data, **kwargs):
    """Plot a boxplot using seaborn."""
    sns.boxplot(data=data, **kwargs)

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def r2(y_true, y_pred):
    """R2 Score."""
    return r2_score(y_true, y_pred)

def evaluate_metrics(y_true, y_pred):
    """Return a dict of common regression metrics."""
    return {
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'R2': r2(y_true, y_pred),
        'MAPE': mape(y_true, y_pred)
    }

def save_csv(df, path):
    """Save a DataFrame to CSV."""
    df.to_csv(path, index=False)

def load_csv(path):
    """Load a DataFrame from CSV."""
    return pd.read_csv(path)

def setup_logger(log_path=None, level=logging.INFO):
    """Setup logging to file and console."""
    handlers = [logging.StreamHandler()]
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(level=level, handlers=handlers, format='%(asctime)s | %(levelname)s | %(message)s')

def add_noise(arr, std=0.01):
    """Add Gaussian noise to an array."""
    return arr + np.random.normal(0, std, arr.shape)

def random_scale(arr, low=0.95, high=1.05):
    """Randomly scale an array."""
    return arr * np.random.uniform(low, high, arr.shape)

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    import torch
    import tensorflow as tf
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

def flatten_list(nested_list):
    """Flatten a nested list."""
    return [item for sublist in nested_list for item in sublist] 