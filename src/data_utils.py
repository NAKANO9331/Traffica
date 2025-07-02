import os
import logging
import pandas as pd
import numpy as np
import torch
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .config import RAW_DATA_DIR, TRAFFIC_FILE, WEATHER_FILE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED, PROCESSED_DATA_DIR, DATA_CONFIG
from .btmf_fill import btmf_fill, run_btmf_fill_and_save
from tqdm import tqdm
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
except ImportError:
    IterativeImputer = None
from .utils import COLORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

FILLED_DATA_DIR = "data/filled"
FILLED_TRAFFIC_FILE = "traffic_filled_btmf.csv"

def load_traffic_data():
    """Load raw traffic data from HDF5."""
    try:
        file_path = os.path.join(RAW_DATA_DIR, TRAFFIC_FILE)
        traffic_data = pd.read_hdf(file_path, key='df')
        logging.info(f"Loaded traffic data: {traffic_data.shape}")
        return traffic_data
    except FileNotFoundError:
        logging.error(f"Traffic data file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Unknown error occurred while loading traffic data: {e}")
        raise

def load_weather_data():
    """Load weather data."""
    try:
        file_path = os.path.join(RAW_DATA_DIR, WEATHER_FILE)
        weather_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logging.info(f"Loaded weather data: {weather_data.shape}")
        return weather_data
    except FileNotFoundError:
        logging.error(f"Weather data file not found: {file_path}")
        raise
    except pd.errors.ParserError:
        logging.error(f"Weather data file format error: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Unknown error occurred while loading weather data: {e}")
        raise

def load_filled_traffic_data(method='btmf'):
    """Load BTMF-imputed traffic data (always save and load from h5)."""
    traffic_data = load_traffic_data()
    weather_data = load_weather_data()
    output_dir = FILLED_DATA_DIR
    filled_h5 = os.path.join(output_dir, 'btmf_fill.h5')
    if os.path.exists(filled_h5):
        logging.info(f"BTMF imputation result already exists, loading directly: {filled_h5}")
        filled_df = pd.read_hdf(filled_h5, key='df')
    else:
        filled_df = run_btmf_fill_and_save(traffic_data, weather_data, output_dir)
        # filled_df.to_hdf(filled_h5, key='df')
        logging.info(f"BTMF imputation completed. Data saved to: {filled_h5}")
    return filled_df

def load_filled_traffic_data_from_h5():
    """Load filled traffic data from HDF5."""
    file_path = os.path.join(FILLED_DATA_DIR, "btmf_fill.h5")
    try:
        traffic_data = pd.read_hdf(file_path, key='df')
        logging.info(f"Loaded filled traffic data from HDF5: {traffic_data.shape}")
        return traffic_data
    except FileNotFoundError:
        logging.error(f"Filled traffic data file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Unknown error occurred while loading filled traffic data: {e}")
        raise

def unified_filled_traffic_data():
    """Run BTMF imputation and return result."""
    traffic_data = load_traffic_data()
    weather_data = load_weather_data()
    output_dir = FILLED_DATA_DIR
    filled_df = run_btmf_fill_and_save(traffic_data, weather_data, output_dir)
    logging.info(f"Filled traffic data (BTMF) and all analysis charts have been saved to: {output_dir}")
    return filled_df

def calculate_missing_rate_and_plot():
    """Calculate missing rate of the original traffic dataset (0 as missing) and save a pie chart to the specified path (all in English, using COLORS from utils.py)."""
    import matplotlib.pyplot as plt
    import os
    import logging
    # Read original traffic data
    file_path = os.path.join(RAW_DATA_DIR, TRAFFIC_FILE)
    df = pd.read_csv(file_path, index_col=0)
    total = df.size
    missing = (df.values == 0).sum()
    filled = total - missing
    missing_rate = missing / total
    # Unified output to log and terminal
    msg = (
        f"[Original Dataset Missing Statistics]\n"
        f"Total data points: {total}\n"
        f"Missing count: {missing}\n"
        f"Missing rate: {missing_rate:.4%}\n"
        f"Non-missing count: {filled}"
    )
    print(msg)
    logging.info(msg)
    # Draw pie chart
    labels = ['Not Missing', 'Missing']
    sizes = [filled, missing]
    colors = [COLORS['blue'], COLORS['orange']]
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90, counterclock=False)
    plt.title('Missing Rate of Original Traffic Dataset')
    plt.axis('equal')
    out_path = '/home/ldf/Traffica/data/filled/missing_pattern.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Pie chart saved to: {out_path}")
    logging.info(f"Pie chart saved to: {out_path}")

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
    def process_traffic_data(self, df):
        """Process traffic data and remove outliers."""
        df.index = pd.to_datetime(df.index)
        df["avg_speed"] = df.mean(axis=1)
        df = df.dropna()
        Q1 = df["avg_speed"].quantile(0.25)
        Q3 = df["avg_speed"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df["avg_speed"] >= lower_bound) & (df["avg_speed"] <= upper_bound)]
        return df
    def process_weather_data(self, df):
        """Process weather data, fill missing values, and remove outliers."""
        missing_summary = df.isnull().sum()
        total_missing = missing_summary.sum()
        logging.info(f"Total missing values in weather data: {total_missing}")
        df_filled = df.copy()
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype == "float":
                    df_filled[column] = df[column].interpolate(method="time")
                else:
                    df_filled[column] = df[column].fillna(method="ffill")
        df_filled = df_filled.fillna(method="bfill")
        remaining_missing = df_filled.isnull().sum().sum()
        if remaining_missing > 0:
            logging.warning(f"Number of missing values after filling: {remaining_missing}")
        df = df_filled
        selected_features = ["TMAX", "TMIN", "PRCP", "AWND", "RHAV", "ASLP"]
        df = df[selected_features]
        df["temp_diff"] = df["TMAX"] - df["TMIN"]
        df["is_raining"] = (df["PRCP"] > 0).astype(int)
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
        logging.info("Weather data processing completed")
        return df
    def align_and_merge_data(self, traffic_df, weather_df):
        """Align and merge traffic and weather data."""
        traffic_df = traffic_df.resample("5T").mean()
        weather_df = weather_df.resample("5T").mean()
        common_idx = traffic_df.index.intersection(weather_df.index)
        traffic_df = traffic_df.loc[common_idx]
        weather_df = weather_df.loc[common_idx]
        merged_df = pd.concat([traffic_df, weather_df], axis=1)
        merged_df = merged_df.dropna()
        logging.info(f"Data merge completed, final data shape: {merged_df.shape}")
        return merged_df
    def create_features(self, df, include_weather=True):
        """Create features for modeling."""
        df = df.sort_index()
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        import holidays
        us_holidays = holidays.US()
        df["is_holiday"] = df.index.to_series().apply(lambda x: 1 if x in us_holidays else 0)
        df["speed_lag1"] = df["avg_speed"].shift(1)
        df["speed_lag2"] = df["avg_speed"].shift(2)
        df["speed_lag3"] = df["avg_speed"].shift(3)
        df["speed_ma5"] = df["avg_speed"].rolling(window=5).mean()
        df["speed_ma10"] = df["avg_speed"].rolling(window=10).mean()
        df["speed_ma15"] = df["avg_speed"].rolling(window=15).mean()
        df["speed_ma30"] = df["avg_speed"].rolling(window=30).mean()
        if include_weather and "weather_description" in df.columns:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df["weather_encoded"] = le.fit_transform(df["weather_description"])
        df = df.dropna()
        return df
    def split_data(self, data):
        """Split dataset into train, val, and test sets."""
        np.random.seed(RANDOM_SEED)
        y = data["target"]
        X = data.drop("target", axis=1)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_SEED)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(TEST_RATIO / (TEST_RATIO + VAL_RATIO)), random_state=RANDOM_SEED)
        return X_train, X_val, X_test, y_train, y_val, y_test
    def prepare_data(self, traffic_data, weather_data=None):
        """Standardize and clean data."""
        processed_data = traffic_data.copy()
        processed_data["target"] = processed_data.iloc[:, -1]
        if weather_data is not None:
            processed_data = pd.concat([processed_data, weather_data], axis=1)
        processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
        processed_data = processed_data.fillna(method="ffill")
        processed_data = processed_data.fillna(method="bfill")
        processed_data = processed_data.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        target_scaler = StandardScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(processed_data), columns=processed_data.columns, index=processed_data.index)
        scaled_data["target"] = target_scaler.fit_transform(processed_data[["target"]])
        return scaled_data, scaler, target_scaler
    def preprocess_weather_features(self, weather_data):
        """Generate derived weather features."""
        weather_data["temp_range"] = weather_data["TMAX"] - weather_data["TMIN"]
        weather_data["feels_like"] = weather_data["TMAX"] - 0.55 * (1 - weather_data["RHAV"] / 100) * (weather_data["TMAX"] - 14.5)
        weather_data["wind_chill"] = 13.12 + 0.6215 * weather_data["TMAX"] - 11.37 * (weather_data["AWND"] ** 0.16) + 0.3965 * weather_data["TMAX"] * (weather_data["AWND"] ** 0.16)
        weather_data["severe_weather"] = ((weather_data["PRCP"] > weather_data["PRCP"].quantile(0.95)) | (weather_data["AWND"] > weather_data["AWND"].quantile(0.95))).astype(int)
        weather_data["hour"] = weather_data.index.hour
        weather_data["is_rush_hour"] = ((weather_data["hour"] >= 7) & (weather_data["hour"] <= 9) | (weather_data["hour"] >= 16) & (weather_data["hour"] <= 18)).astype(int)
        weather_data["rush_hour_rain"] = weather_data["is_rush_hour"] * (weather_data["PRCP"] > 0).astype(int)
        weather_data["temp_change"] = weather_data["TMAX"].diff()
        weather_data["precip_change"] = weather_data["PRCP"].diff()
        weather_data["wind_change"] = weather_data["AWND"].diff()
        weather_data["temp_trend"] = weather_data["TMAX"].rolling(window=12).mean()
        weather_data["precip_trend"] = weather_data["PRCP"].rolling(window=12).mean()
        weather_data["wind_trend"] = weather_data["AWND"].rolling(window=12).mean()
        weather_data = weather_data.fillna(method="ffill").fillna(method="bfill")
        return weather_data
    def prepare_sequences(self, traffic_data, weather_data=None, sequence_length=12):
        """Prepare sequence data for modeling, and return indices for each split."""
        np.random.seed(RANDOM_SEED)
        if weather_data is not None:
            weather_data = self.preprocess_weather_features(weather_data)
            weather_features = [
                "TMAX", "TMIN", "PRCP", "AWND", "RHAV", "temp_range", "feels_like", "wind_chill", "severe_weather", "rush_hour_rain"
            ]
            weather_data = weather_data[weather_features]
            weather_data = (weather_data - weather_data.mean()) / weather_data.std()
        traffic_processed = self.process_traffic_data(traffic_data)
        if weather_data is not None:
            data = self.align_and_merge_data(traffic_processed, weather_data)
        else:
            data = traffic_processed
        data = self.create_features(data, include_weather=(weather_data is not None))
        data_scaled, _, _ = self.prepare_data(data)
        X = data_scaled.drop("target", axis=1)
        y = data_scaled["target"]
        X_sequences = []
        y_sequences = []
        for i in range(len(X) - sequence_length):
            X_sequences.append(X.iloc[i : (i + sequence_length)].values)
            y_sequences.append(y.iloc[i + sequence_length])
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        total_samples = len(X_sequences)
        train_size = int(total_samples * TRAIN_RATIO)
        val_size = int(total_samples * VAL_RATIO)
        X_train = X_sequences[:train_size]
        y_train = y_sequences[:train_size]
        X_val = X_sequences[train_size : train_size + val_size]
        y_val = y_sequences[train_size : train_size + val_size]
        X_test = X_sequences[train_size + val_size :]
        y_test = y_sequences[train_size + val_size :]
        idx = data.index[sequence_length:]
        train_idx = idx[:train_size]
        val_idx = idx[train_size : train_size + val_size]
        test_idx = idx[train_size + val_size :]
        logging.info("Data sequence preparation completed")
        logging.info(f"Sequence shape: {X_sequences.shape}")
        logging.info(f"Training set: {X_train.shape}")
        logging.info(f"Validation set: {X_val.shape}")
        logging.info(f"Test set: {X_test.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx
    def augment_weather_data(self, X_weather, y):
        """Augment weather data for training."""
        augmented_X = [X_weather]
        augmented_y = [y]
        noise = np.random.normal(0, 0.01, X_weather.shape)
        augmented_X.append(X_weather + noise)
        augmented_y.append(y)
        scale = np.random.uniform(0.95, 1.05, X_weather.shape)
        augmented_X.append(X_weather * scale)
        augmented_y.append(y)
        return np.concatenate(augmented_X, axis=0), np.concatenate(augmented_y, axis=0) 