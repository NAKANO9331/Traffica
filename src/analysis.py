import os
import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.inspection import permutation_importance
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import shap
import tensorflow as tf

from .config import RESULTS_DIR, RANDOM_SEED, VISUALIZATION_CONFIG
from .utils import ensure_dir, save_figure, close_plt, tight_layout, COLORS

plt.style.use("tableau-colorblind10")
sns.set_palette("Set2")
plt.style.use(VISUALIZATION_CONFIG["style"])
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]


def evaluate_model(
    y_true, y_pred, model_name, model=None, feature_names=None, history=None
):
    """Evaluate model performance"""
    results = {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "errors": y_pred - y_true,  # Save prediction errors
    }

    # Save training history
    if history is not None:
        results["history"] = history

    # Calculate feature importance
    if model is not None and feature_names is not None:
        try:
            # Calculate feature importance using model weights
            feature_importance = {}
            for layer in model.layers:
                if "dense" in layer.name.lower():
                    weights = layer.get_weights()[0]
                    importance = np.abs(weights).mean(axis=1)
                    # Ensure feature names and importance scores have the same length
                    min_len = min(len(feature_names), len(importance))
                    for i in range(min_len):
                        feature_importance[feature_names[i]] = float(importance[i])
                    break  # Only use the weights of the first dense layer

            if feature_importance:
                results["feature_importance"] = feature_importance
        except Exception as e:
            logging.warning(
                f"Error occurred when calculating feature importance: {str(e)}"
            )

    return results


def plot_predictions(y_true, y_pred, model_name, results_dir):
    """Plot prediction comparison"""
    plt.figure(figsize=(15, 6))
    plt.plot(y_true.reset_index(drop=True), label="Actual", linewidth=2)
    plt.plot(y_pred, label="Predicted", linewidth=2, linestyle="--")
    plt.title(f"{model_name} Prediction Results", fontsize=14)
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    if results_dir:
        figures_dir = os.path.join(results_dir, "figures")
        if y_pred is not None and y_true is not None:
            ensure_dir(figures_dir)
            save_path = os.path.join(figures_dir, f"{model_name}_predictions.png")
            save_figure(save_path, dpi=300, bbox_inches="tight")
    close_plt()


def plot_residuals(y_true, y_pred, model_name):
    """Plot residual analysis"""
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Residual scatter plot
    ax1.scatter(y_pred, residuals, alpha=0.5, color=COLORS["blue"])
    ax1.axhline(y=0, color=COLORS["r"], linestyle="--")
    ax1.set_xlabel("Predicted Values", fontsize=12)
    ax1.set_ylabel("Residuals", fontsize=12)
    ax1.set_title("Residuals vs Predicted", fontsize=14)
    ax1.grid(True)

    # Residual distribution plot
    sns.histplot(residuals, kde=True, ax=ax2, color=COLORS["blue"])
    ax2.axvline(x=0, color=COLORS["r"], linestyle="--")
    ax2.set_xlabel("Residuals", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Residual Distribution", fontsize=14)
    ax2.grid(True)

    plt.suptitle(f"{model_name} Residual Analysis", fontsize=16, y=1.05)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "figures", f"{model_name}_residuals.png")
    save_figure(save_path, dpi=300, bbox_inches="tight")
    close_plt()

    # Record residual statistics
    residuals_stats = {
        "mean": np.mean(residuals),
        "std": np.std(residuals),
        "skewness": pd.Series(residuals).skew(),
        "kurtosis": pd.Series(residuals).kurtosis(),
    }

    logging.info(f"\n{model_name} Residual Statistics:")
    for stat, value in residuals_stats.items():
        logging.info(f"{stat}: {value:.4f}")


def plot_feature_importance(
    model, X_test, y_test, feature_names, model_name, results_dir
):
    """Use Permutation Importance for feature importance analysis"""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1
    )
    importance = result.importances_mean

    # Plot feature importance
    indices = np.argsort(importance)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title(f"{model_name} Feature Importance (Permutation Importance)", fontsize=16)
    plt.bar(range(len(feature_names)), importance[indices], align="center")
    plt.xticks(
        range(len(feature_names)), [feature_names[i] for i in indices], rotation=90
    )
    plt.xlabel("Feature")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    save_path = os.path.join(results_dir, f"{model_name}_feature_importance.png")
    save_figure(save_path, dpi=300)
    close_plt()


def find_best_model(improvements):
    """Find the best model based on performance improvement"""
    model_scores = {}

    # Calculate the overall score of each model
    for model, metrics in improvements.items():
        # Calculate the score based on the decrease in RMSE and MAE and the increase in R2
        rmse_score = metrics["RMSE_improvement"]
        mae_score = metrics["MAE_improvement"]
        r2_score = metrics["R2_improvement"]

        # Overall score (weights can be adjusted as needed)
        model_scores[model] = (rmse_score + mae_score + r2_score) / 3

    # Find the model with the highest score
    best_model = max(model_scores.items(), key=lambda x: x[1])

    logging.info("\nBest Model Analysis Results:")
    logging.info(f"Best Model: {best_model[0]}")
    logging.info(f"Average Performance Improvement: {best_model[1]:.2f}%")

    return best_model[0]


def plot_prediction_distribution(y_true, y_pred, model_name):
    """Plot the prediction distribution"""
    plt.figure(figsize=(15, 5))

    # Scatter plot of predictions vs actual values
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Prediction vs Actual")
    plt.grid(True)

    # Prediction error distribution plot
    plt.subplot(1, 2, 2)
    residuals = y_pred - y_true
    sns.histplot(residuals, kde=True)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title("Error Distribution")
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(
        RESULTS_DIR, "figures", f"{model_name}_prediction_analysis.png"
    )
    save_figure(save_path, dpi=300, bbox_inches="tight")
    close_plt()


def plot_time_series_decomposition(data, model_name):
    """Plot the time series decomposition"""
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Perform time series decomposition
    decomposition = seasonal_decompose(
        data, period=24 * 7
    )  # Assuming the data is one point per hour, with a period of

    plt.figure(figsize=(15, 12))

    # Original data
    plt.subplot(411)
    plt.plot(data)
    plt.title("Original Time Series")
    plt.grid(True)

    # Trend
    plt.subplot(412)
    plt.plot(decomposition.trend)
    plt.title("Trend")
    plt.grid(True)

    # Seasonality
    plt.subplot(413)
    plt.plot(decomposition.seasonal)
    plt.title("Seasonal")
    plt.grid(True)

    # Residuals
    plt.subplot(414)
    plt.plot(decomposition.resid)
    plt.title("Residual")
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(
        RESULTS_DIR, "figures", f"{model_name}_time_series_decomposition.png"
    )
    save_figure(save_path, dpi=300, bbox_inches="tight")
    close_plt()


def analyze_weather_impact(baseline_results, enhanced_results, weather_data):
    """Analyze the impact of weather factors"""
    impact_analysis = {
        "overall_improvement": {},
        "weather_condition_impact": {},
        "feature_importance": {},
    }

    # 1. Calculate the overall performance improvement
    for metric in ["rmse", "mae", "mape"]:
        improvement = (
            (baseline_results[metric] - enhanced_results[metric])
            / baseline_results[metric]
            * 100
        )
        impact_analysis["overall_improvement"][metric] = improvement

    # 2. Analyze the impact under different weather conditions
    weather_conditions = weather_data["condition"].unique()
    for condition in weather_conditions:
        condition_mask = weather_data["condition"] == condition
        condition_impact = calculate_condition_impact(
            baseline_results, enhanced_results, condition_mask
        )
        impact_analysis["weather_condition_impact"][condition] = condition_impact

    # 3. Feature importance analysis
    weather_features = ["temperature", "precipitation", "wind_speed", "humidity"]
    impact_analysis["feature_importance"] = analyze_feature_importance(
        enhanced_results, weather_data[weather_features]
    )

    return impact_analysis


def detailed_model_analysis(y_true, y_pred, model_name):
    """Detailed model performance analysis"""
    residuals = y_pred - y_true
    stats_results = {
        "mean_error": np.mean(residuals),
        "std_error": np.std(residuals),
        "skewness": stats.skew(residuals),
        "kurtosis": stats.kurtosis(residuals),
    }

    return stats_results


def shap_analysis(model, X_sample, model_name, results_dir):
    """Analyze feature importance using perturbation method"""
    try:
        base_predictions = model.predict(X_sample)
        n_features = X_sample.shape[2]
        importances = np.zeros(n_features)

        for i in range(n_features):
            X_perturbed = X_sample.copy()
            X_perturbed[:, :, i] = np.random.permutation(X_perturbed[:, :, i])
            perturbed_predictions = model.predict(X_perturbed)
            importances[i] = np.mean((base_predictions - perturbed_predictions) ** 2)

        plt.figure(figsize=(10, 6))
        feature_indices = np.argsort(importances)[::-1]
        plt.bar(range(len(importances)), importances[feature_indices])
        plt.title(f"{model_name} Feature Importance Analysis")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance Score")
        plt.xticks(range(len(importances)), feature_indices)
        plt.tight_layout()

        figures_dir = os.path.join(results_dir, "figures")
        ensure_dir(figures_dir)
        save_path = os.path.join(figures_dir, f"{model_name}_feature_importance.png")
        save_figure(save_path, dpi=300, bbox_inches="tight")
        close_plt()

    except Exception as e:
        logging.error(f"Feature importance analysis error: {str(e)}")
        raise


def plot_training_history(history, model_name, results_dir):
    """Plot the model training history"""
    # Ensure the directory exists
    figures_dir = os.path.join(results_dir, "figures")
    ensure_dir(figures_dir)

    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} Training History - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # MAE curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"], label="Training MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.title(f"{model_name} Training History - MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(figures_dir, f"{model_name}_training_history.png")
    save_figure(save_path, dpi=300, bbox_inches="tight")
    close_plt()


def analyze_weather_features(df, weather_features, target="avg_speed"):
    """Analyze the relationship between weather features and the target variable"""
    # Ensure the directory exists
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    ensure_dir(figures_dir)

    # Correlation analysis
    plt.figure(figsize=(12, 8))
    corr = df[weather_features + [target]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Weather Features Correlation with Traffic Speed")
    save_path = os.path.join(figures_dir, "weather_correlation.png")
    save_figure(save_path, dpi=300, bbox_inches="tight")
    close_plt()


def plot_prediction_comparison(
    y_true, y_pred_baseline, y_pred_enhanced, model_name, results_dir
):
    """Plot the comparison of baseline and enhanced model prediction results"""
    plt.figure(figsize=(15, 6))

    # Select the first 200 data points for clear display
    n_points = 200
    x = np.arange(n_points)

    plt.plot(x, y_true[:n_points], label="Actual", linewidth=2)
    plt.plot(x, y_pred_baseline[:n_points], "--", label="Baseline", linewidth=2)
    plt.plot(x, y_pred_enhanced[:n_points], "--", label="Enhanced", linewidth=2)

    plt.title(f"{model_name} Prediction Comparison", fontsize=14)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Traffic Flow", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

    save_path = os.path.join(
        results_dir, "figures", f"{model_name}_prediction_comparison.png"
    )
    save_figure(save_path, dpi=300, bbox_inches="tight")
    close_plt()


def plot_feature_importance(model, feature_names, results_dir):
    """Plot the feature importance analysis chart"""
    plt.figure(figsize=(12, 6))

    # Get feature importance scores
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(
        range(len(importances)), [feature_names[i] for i in indices], rotation=45
    )
    plt.title("Feature Importance Analysis", fontsize=14)
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Importance Score", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(results_dir, "figures", "feature_importance.png")
    save_figure(save_path, dpi=300, bbox_inches="tight")
    close_plt()


def analyze_feature_importance(model, X_train, y_train, feature_names, results_dir):
    """Analyze feature importance"""
    plt.figure(figsize=(15, 8))

    # Using permutation importance
    perm_importance = permutation_importance(
        model, X_train, y_train, n_repeats=10, random_state=42
    )

    # Get feature importance scores
    importances = pd.DataFrame(
        {"feature": feature_names, "importance": perm_importance.importances_mean}
    )
    importances = importances.sort_values("importance", ascending=False)

    # Plot feature importance bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importances.head(20), x="importance", y="feature")
    plt.title("Top 20 Most Important Features", fontsize=14)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()

    # Save the image
    save_path = os.path.join(results_dir, "figures", "feature_importance.png")
    save_figure(save_path, dpi=300, bbox_inches="tight")
    close_plt()

    return importances


def evaluate_models(predictions, y_true):
    """Evaluate model performance"""
    results = {}

    for model_name, y_pred in predictions.items():
        # Calculate evaluation metrics
        results[model_name] = {
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }

        # Print evaluation results
        logging.info(f"\n{model_name} Model Evaluation Results:")
        for metric, value in results[model_name].items():
            logging.info(f"{metric}: {value:.4f}")

    return results


def calculate_improvements(baseline_results, enhanced_results):
    """Calculate performance improvements"""
    improvements = {}

    for model_name in baseline_results.keys():
        improvements[model_name] = {}
        metrics = ["RMSE", "MAE", "R2"]  # Use uppercase metric names

        for metric in metrics:
            baseline = baseline_results[model_name][metric]
            enhanced = enhanced_results[model_name][metric]

            # For R2, the improvement calculation is different
            if metric == "R2":
                improvement = (enhanced - baseline) * 100
            else:
                improvement = (baseline - enhanced) / baseline * 100

            improvements[model_name][
                f"{metric}_improvement"
            ] = improvement  # Keep uppercase

    return improvements


class DataVisualizer:
    def __init__(self):
        plt.style.use("tableau-colorblind10")
        sns.set_palette("Set2")
        plt.style.use(VISUALIZATION_CONFIG["style"])
        self.figure_size = VISUALIZATION_CONFIG["figure_size"]
        self.dpi = VISUALIZATION_CONFIG["dpi"]
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
        self.results_dir = RESULTS_DIR
        self.figures_dir = os.path.join(self.results_dir, "figures")
        self.subdirs = {
            "comparison": os.path.join(self.figures_dir, "comparison"),
            "models": os.path.join(self.figures_dir, "models"),
            "traffic": os.path.join(self.figures_dir, "traffic"),
            "weather": os.path.join(self.figures_dir, "weather"),
        }
        for dir_path in self.subdirs.values():
            ensure_dir(dir_path)

    def get_figure_save_folder(self, filename):
        base_dir = self.figures_dir
        if "traffic" in filename:
            folder = "traffic"
        elif "weather" in filename:
            folder = "weather"
        elif (
            "model" in filename
            or "performance" in filename
            or "feature_importance" in filename
        ):
            folder = "models"
        elif "comparison" in filename or "impact" in filename or "table" in filename:
            folder = "comparison"
        else:
            folder = "others"
        return os.path.join(base_dir, folder)

    def plot_traffic_patterns(self, traffic_data, save_path):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        hourly_pattern = (
            traffic_data.groupby(traffic_data.index.hour).mean().mean(axis=1)
        )
        plt.plot(hourly_pattern.index, hourly_pattern.values, marker="o")
        plt.title("Average Daily Traffic Pattern", fontsize=14)
        plt.xlabel("Hour of Day")
        plt.ylabel("Average Traffic Flow")
        plt.grid(True)
        plt.subplot(2, 1, 2)
        weekly_pattern = (
            traffic_data.groupby(traffic_data.index.dayofweek).mean().mean(axis=1)
        )
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        plt.plot(weekly_pattern.index, weekly_pattern.values, marker="o")
        plt.xticks(range(7), days, rotation=45)
        plt.title("Weekly Traffic Pattern", fontsize=14)
        plt.xlabel("Day of Week")
        plt.ylabel("Average Traffic Flow")
        plt.grid(True)
        plt.tight_layout()
        filename = "traffic_patterns.png"
        folder = self.subdirs["traffic"]
        save_figure(os.path.join(folder, filename), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_weather_analysis(self, weather_data, save_path):
        plt.figure(figsize=(12, 6))
        plt.plot(weather_data.index, weather_data["TMAX"], label="Max Temperature")
        plt.plot(weather_data.index, weather_data["TMIN"], label="Min Temperature")
        plt.title("Temperature Variation Over Time", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename_temp = "weather_temperature.png"
        folder = self.subdirs["weather"]
        save_figure(os.path.join(folder, filename_temp), dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(12, 6))
        humidity_bins = pd.qcut(weather_data["RHAV"], q=10)
        weather_data["humidity_bin"] = humidity_bins
        # Boxplot showing precipitation distribution for each humidity range
        sns.boxplot(x="humidity_bin", y="PRCP", data=weather_data)
        plt.title("Precipitation Distribution by Humidity Range", fontsize=14)
        plt.xlabel("Relative Humidity Range (%)")
        plt.ylabel("Precipitation (mm)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        corr = weather_data["PRCP"].corr(weather_data["RHAV"])
        plt.text(
            0.02,
            0.98,
            f"Correlation: {corr:.2f}",
            transform=plt.gca().transAxes,
            bbox=COLORS["bbox_white"],
            verticalalignment="top",
        )
        plt.tight_layout()
        filename_precip = "weather_precipitation_by_humidity.png"
        folder = self.subdirs["weather"]
        save_figure(os.path.join(folder, filename_precip), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_prediction_comparison(self, y_true, predictions_dict, save_path=None):
        """Visualize prediction comparison between different models"""
        # Ensure save directory exists
        if save_path:
            ensure_dir(save_path)

        # 1. Prediction vs actual values comparison
        plt.figure(figsize=(12, 8))
        plt.plot(y_true[:100], "k-", label="Actual", alpha=0.7)
        for model_name, pred in predictions_dict.items():
            plt.plot(pred[:100], "--", label=f"{model_name}_pred", alpha=0.7)
        plt.title("Prediction vs Actual")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = "prediction_vs_actual.png"
        folder = self.subdirs["comparison"]
        save_figure(os.path.join(folder, filename), dpi=300)
        plt.close()

    def plot_model_improvements(self, baseline_metrics, enhanced_metrics, save_path):
        pass

    def plot_prediction_vs_actual(
        self, y_true, y_pred, timestamps, model_name, save_path=None
    ):
        """Visualize prediction vs actual values"""
        plt.figure(figsize=(15, 6))
        plt.plot(timestamps, y_true, label="Actual", color="blue", alpha=0.6)
        plt.plot(timestamps, y_pred, label="Predicted", color="red", alpha=0.6)

        plt.title(f"{model_name} Model Prediction Results", fontsize=12)
        plt.xlabel("Time")
        plt.ylabel("Traffic Flow")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add shadow area for prediction errors
        plt.fill_between(
            timestamps,
            y_true,
            y_pred,
            color="gray",
            alpha=0.2,
            label="Prediction Error",
        )

        if save_path:
            filename = f"{model_name}_prediction.png"
            folder = self.subdirs["models"]
            save_figure(os.path.join(folder, filename), dpi=300, bbox_inches="tight")
            plt.close()

    def plot_feature_importance_analysis(self, model, feature_names, save_path):
        """Visualize feature importance analysis"""
        plt.figure(figsize=(12, 6))

        # Use model weights to analyze feature importance
        weights = []
        for layer in model.layers:
            if "dense" in layer.name.lower():
                w = layer.get_weights()[0]
                weights.append(np.abs(w).mean(axis=1))

        if weights:
            # Calculate average feature importance
            importance_scores = np.abs(weights[0])

            # Ensure feature names and importance scores match in length
            min_len = min(len(feature_names), len(importance_scores))
            feature_names = feature_names[:min_len]
            importance_scores = importance_scores[:min_len]

            # Create feature importance dataframe
            importance_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": importance_scores}
            )

            # Sort by importance
            importance_df = importance_df.sort_values("Importance", ascending=False)

            # Plot top 15 most important features (since we reduced feature count)
            plt.figure(figsize=(12, 6))
            top_n = min(15, len(importance_df))
            sns.barplot(
                data=importance_df.head(top_n),
                x="Importance",
                y="Feature",
                palette="viridis",
            )

            plt.title("Top Most Important Features", fontsize=14)
            plt.xlabel("Importance Score", fontsize=12)
            plt.ylabel("Feature", fontsize=12)

            plt.tight_layout()
            filename = "feature_importance.png"
            folder = self.subdirs["models"]
            save_figure(os.path.join(folder, filename), dpi=300)
            plt.close()

            return importance_df
        else:
            logging.warning(
                "No dense layers found in the model for feature importance analysis"
            )
            return None

    def plot_weather_impact_analysis(
        self, baseline_results, enhanced_results, weather_data
    ):
        """Visualize weather impact on prediction performance"""
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)

        # 1. Prediction error comparison under different weather conditions
        ax1 = plt.subplot(gs[0, 0])
        self._plot_weather_condition_comparison(
            baseline_results, enhanced_results, weather_data, ax1
        )

        # 2. Extreme weather event analysis
        ax2 = plt.subplot(gs[0, 1])
        self._plot_extreme_weather_analysis(
            baseline_results, enhanced_results, weather_data, ax2
        )

        # 3. Weather feature importance analysis
        ax3 = plt.subplot(gs[1, 0])
        self._plot_weather_feature_importance(enhanced_results, weather_data, ax3)

        # 4. Performance improvement statistics
        ax4 = plt.subplot(gs[1, 1])
        self._plot_performance_improvement(baseline_results, enhanced_results, ax4)

        plt.tight_layout()
        filename = "weather_impact_analysis.png"
        folder = self.subdirs["comparison"]
        save_figure(os.path.join(folder, filename), dpi=300)
        plt.close()
        return fig

    def create_performance_table(
        self, baseline_metrics, enhanced_metrics, improvements, save_path
    ):
        metrics = ["RMSE", "MAE", "R2"]
        models = list(baseline_metrics.keys())
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        bar_width = 0.35
        index = np.arange(len(models))
        baseline_color = COLORS["baseline"]
        enhanced_color = COLORS["enhanced"]
        for i, metric in enumerate(metrics):
            ax = axes[i]
            baseline_values = [baseline_metrics[m][metric] for m in models]
            enhanced_values = [enhanced_metrics[m][metric] for m in models]
            bars1 = ax.bar(
                index - bar_width / 2,
                baseline_values,
                width=bar_width,
                label="Baseline",
                color=baseline_color,
                alpha=0.8,
            )
            bars2 = ax.bar(
                index + bar_width / 2,
                enhanced_values,
                width=bar_width,
                label="Enhanced",
                color=enhanced_color,
                alpha=0.8,
            )
            ax.set_title(metric, fontsize=14)
            ax.set_xticks(index)
            ax.set_xticklabels(models, rotation=45)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            for bar in bars1:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                )
            for bar in bars2:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                )
            if metric != "R2":
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(0, ymax * 1.1)
        plt.tight_layout()
        filename = "performance_metrics.png"
        folder = self.subdirs["comparison"]
        save_figure(os.path.join(folder, filename), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_traffic_weather_relationship(self, traffic_data, weather_data, save_path):
        """Analyze the relationship between traffic flow and weather"""
        ensure_dir(save_path)
        avg_traffic = traffic_data.mean(axis=1)
        # 1. Violin plot of traffic flow in different temperature ranges
        plt.figure(figsize=(12, 8))
        temp_bins = pd.cut(
            weather_data["TMAX"], bins=5, labels=[f"{i+1}" for i in range(5)]
        )
        sns.violinplot(x=temp_bins, y=avg_traffic)
        plt.title("Traffic Flow Distribution by Temperature Range")
        plt.xlabel("Temperature Level (1: Coldest, 5: Hottest)")
        plt.ylabel("Traffic Flow")
        plt.tight_layout()
        filename = "traffic_temp_distribution.png"
        folder = self.subdirs["traffic"]
        save_figure(os.path.join(folder, filename), dpi=300)
        plt.close()
        # 2. Violin plot of traffic flow changes in different precipitation levels
        plt.figure(figsize=(12, 8))
        weather_data["rain_category"] = pd.cut(
            weather_data["PRCP"],
            bins=[-np.inf, 0, 0.1, 1, np.inf],
            labels=["No Rain", "Light", "Moderate", "Heavy"],
        )
        sns.violinplot(
            x="rain_category",
            y=avg_traffic,
            data=pd.DataFrame(
                {"rain_category": weather_data["rain_category"], "traffic": avg_traffic}
            ),
        )
        plt.title("Traffic Flow by Precipitation Level")
        plt.xlabel("Precipitation Level")
        plt.ylabel("Traffic Flow")
        plt.tight_layout()
        filename = "traffic_precip_distribution.png"
        folder = self.subdirs["traffic"]
        save_figure(os.path.join(folder, filename), dpi=300)
        plt.close()
        # 4. Weather event markers on time series
        plt.figure(figsize=(12, 8))
        plt.plot(avg_traffic.index, avg_traffic, alpha=0.5, label="Traffic Flow")
        extreme_weather = weather_data["PRCP"] > weather_data["PRCP"].quantile(0.95)
        plt.scatter(
            avg_traffic.index[extreme_weather],
            avg_traffic[extreme_weather],
            color=COLORS["red"],
            alpha=0.5,
            label="Heavy Rain",
        )
        plt.title("Traffic Flow with Extreme Weather Events")
        plt.xlabel("Time")
        plt.ylabel("Traffic Flow")
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.xticks(rotation=45)
        plt.tight_layout()
        filename = "traffic_extreme_events.png"
        folder = self.subdirs["traffic"]
        save_figure(os.path.join(folder, filename), dpi=300)
        plt.close()

    def create_comprehensive_report(
        self, baseline_metrics, enhanced_metrics, weather_data, save_path
    ):
        """Create a comprehensive performance report"""
        # Define colors and line styles
        colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
        linestyles = ["-", "--", ":", "-."]

        # 1. Overall performance comparison - use 2x2 subgraph layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Performance Metrics Comparison Across Models", fontsize=16, y=1.02
        )

        metrics = {
            "RMSE": {
                "ax": ax1,
                "color": COLORS["skyblue"],
                "title": "Root Mean Square Error (RMSE)",
            },
            "MAE": {
                "ax": ax2,
                "color": COLORS["lightgreen"],
                "title": "Mean Absolute Error (MAE)",
            },
            "R2": {
                "ax": ax3,
                "color": COLORS["lightcoral"],
                "title": "R-squared Score",
            },
        }

        models = list(baseline_metrics.keys())  # ['LSTM', 'GRU', 'TCN']
        bar_width = 0.35
        index = np.arange(len(models))

        def add_value_labels(ax, bars):
            """Add value labels to the bar chart"""
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                )

        # Use the same colors as weather_impact_conditions
        baseline_color = COLORS["baseline"]
        enhanced_color = COLORS["enhanced"]

        # Draw a subgraph for each metric
        for metric_name, metric_info in metrics.items():
            ax = metric_info["ax"]

            # Get the baseline and enhanced values
            baseline_values = [baseline_metrics[model][metric_name] for model in models]
            enhanced_values = [enhanced_metrics[model][metric_name] for model in models]

            # Draw the bar chart
            bars1 = ax.bar(
                index - bar_width / 2,
                baseline_values,
                width=bar_width,
                label="Baseline",
                color=baseline_color,
                alpha=0.8,
            )
            bars2 = ax.bar(
                index + bar_width / 2,
                enhanced_values,
                width=bar_width,
                label="Enhanced",
                color=enhanced_color,
                alpha=0.8,
            )

            add_value_labels(ax, bars1)
            add_value_labels(ax, bars2)

            ax.set_title(metric_info["title"], fontsize=12, pad=10)
            ax.set_xticks(index)
            ax.set_xticklabels(models, rotation=45)
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

            if metric_name != "R2":  # R2 score can be negative, so don't start from 0
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(0, ymax * 1.1)  # Leave 10% space to display value labels

            # Add y-axis label
            if metric_name == "R2":
                ax.set_ylabel("R-squared Score")
            else:
                ax.set_ylabel(metric_name)

        plt.tight_layout()

        filename = "performance_metrics.png"
        folder = self.subdirs["comparison"]
        save_figure(os.path.join(folder, filename), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_weather_box_by_condition(self, weather_data, save_dir):
        if "condition" in weather_data.columns and "TMAX" in weather_data.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x="condition", y="TMAX", data=weather_data)
            plt.title("Max Temperature by Weather Condition")
            plt.xticks(rotation=45)
            plt.tight_layout()
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "weather_condition_temp_box.png"), dpi=300)
        plt.close()
