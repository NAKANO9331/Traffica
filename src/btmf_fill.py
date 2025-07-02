import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from .utils import zscore, COLORS

plt.style.use('bmh')
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# Unified image save directory
RESULTS_DATA_DIR = 'results/figures/data'
os.makedirs(RESULTS_DATA_DIR, exist_ok=True)

def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / (var + 1e-8)) / np.sum(var != 0)

def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / np.sum(var != 0))

def plot_metrics(mape_history, rmse_history, name, output_dir=None):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(
        np.arange(50, len(mape_history) * 50 + 1, 50), mape_history, "o-", color="blue"
    )
    plt.xlabel("Iteration")
    plt.ylabel("MAPE")
    plt.title("Training MAPE Change")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(
        np.arange(50, len(rmse_history) * 50 + 1, 50), rmse_history, "o-", color="red"
    )
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("Training RMSE Change")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DATA_DIR, f"training_metrics_{name}.png"), dpi=300)
    plt.close()

def plot_prediction(dense_mat, mat_hat, name, output_dir=None):
    var_idx = 0
    plt.figure(figsize=(15, 6))
    plt.plot(dense_mat[var_idx, :], "b-", label="True Value", alpha=0.7)
    plt.plot(mat_hat[var_idx, :], "r--", label="Filled/Predicted Value", alpha=0.7)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(f"Prediction Result Comparison for Variable {var_idx+1}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DATA_DIR, f"prediction_comparison_{name}.png"), dpi=300)
    plt.close()

def plot_error_distribution(dense_mat, mat_hat, name, output_dir=None):
    pos = dense_mat != 0
    errors = dense_mat[pos] - mat_hat[pos]
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=30, density=True, alpha=0.7, color="green")
    plt.axvline(x=0, color="r", linestyle="--", label="Zero Error")
    plt.xlabel("Prediction Error")
    plt.ylabel("Density")
    plt.title(f"Prediction Error Distribution ({name})")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(RESULTS_DATA_DIR, f"prediction_error_distribution_{name}.png"), dpi=300
    )
    plt.close()

def plot_alpha(alpha, name, output_dir=None):
    plt.figure(figsize=(10, 4))
    plt.plot(torch.sigmoid(alpha).detach().cpu().numpy(), label="Fusion Weight alpha")
    plt.xlabel("Time Step")
    plt.ylabel("alpha (sigmoid)")
    plt.title(f"Dynamic Fusion Weight alpha Change Over Time ({name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DATA_DIR, f"alpha_dynamic_{name}.png"), dpi=300)
    plt.close()

class BTMF_ARLSTM(nn.Module):
    def __init__(self, n_entity, n_time, rank, time_lags, weather_feat, device):
        super().__init__()
        self.n_entity = n_entity
        self.n_time = n_time
        self.rank = rank
        self.time_lags = torch.tensor(time_lags, dtype=torch.long, device=device)
        self.d = len(time_lags)
        self.device = device
        self.W = nn.Parameter(torch.randn(n_entity, rank, device=device) * 0.1)
        self.A = nn.Parameter(torch.randn(rank * self.d, rank, device=device) * 0.1)
        self.X_ar = nn.Parameter(torch.randn(n_time, rank, device=device) * 0.1)
        self.weather_feat = weather_feat
        self.lstm_input_dim = rank + (
            weather_feat.shape[1] if weather_feat is not None else 0
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=2 * rank,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.lstm_norm = nn.LayerNorm(2 * rank)
        self.X_init = nn.Parameter(torch.randn(1, rank, device=device) * 0.1)
        self.alpha = nn.Parameter(torch.ones(n_time, 1, device=device) * 0.8)

    def ar_part(self):
        tmax = torch.max(self.time_lags).item()
        X = self.X_ar
        Z = X[tmax:, :]
        Q = []
        for k in range(self.d):
            Q.append(X[tmax - self.time_lags[k] : self.n_time - self.time_lags[k], :])
        Q = torch.cat(Q, dim=1)
        pred = torch.matmul(Q, self.A)
        return Z, pred, X

    def lstm_part(self):
        X_seq = self.X_init.expand(self.n_time, -1)
        if self.weather_feat is not None:
            lstm_input = torch.cat([X_seq, self.weather_feat], dim=1).unsqueeze(0)
        else:
            lstm_input = X_seq.unsqueeze(0)
        X_lstm, _ = self.lstm(lstm_input)
        X_lstm = X_lstm.squeeze(0)
        X_lstm = self.lstm_norm(X_lstm)
        X_lstm = X_lstm[:, : self.rank]
        return X_lstm

    def forward(self):
        Z, pred_ar, X_ar = self.ar_part()
        X_lstm = self.lstm_part()
        alpha = torch.sigmoid(self.alpha)
        X = alpha * X_ar + (1 - alpha) * X_lstm
        mat_hat = torch.matmul(self.W, X.t())
        return mat_hat, Z, pred_ar, X_ar, X_lstm, X, alpha

    def predict_multi(self, multi_step):
        tmax = torch.max(self.time_lags).item()
        X_ar = self.X_ar.detach().clone()
        for _ in range(multi_step):
            Q = []
            for k in range(self.d):
                Q.append(X_ar[-self.time_lags[k] :][0:1, :])
            Q = torch.cat(Q, dim=1)
            next_x = torch.matmul(Q, self.A)
            X_ar = torch.cat([X_ar, next_x], dim=0)
        X_seq = self.X_init.expand(self.n_time, -1)
        if self.weather_feat is not None:
            last_weather = self.weather_feat[-1:, :].repeat(multi_step, 1)
            lstm_input = torch.cat([X_seq, self.weather_feat], dim=1)
            lstm_input_pred = torch.cat(
                [X_seq[-1:, :].repeat(multi_step, 1), last_weather], dim=1
            )
            lstm_input_full = torch.cat([lstm_input, lstm_input_pred], dim=0).unsqueeze(
                0
            )
        else:
            lstm_input_full = X_seq.unsqueeze(0)
        X_lstm, (h, c) = self.lstm(lstm_input_full)
        X_lstm_pred = X_lstm.squeeze(0)[-multi_step:, :]
        X_lstm_pred = X_lstm_pred[:, : self.rank]
        alpha_pred = torch.sigmoid(self.alpha[-multi_step:])
        X_pred_fusion = alpha_pred * X_ar[-multi_step:] + (1 - alpha_pred) * X_lstm_pred
        mat_pred = torch.matmul(self.W, X_pred_fusion.t())
        return mat_pred.detach().cpu().numpy()

def btmf_fill(
    traffic_mat,
    weather_feat,
    device="cpu",
    rank=30,
    burn_iter=2000,
    lr=0.02,
    time_lags=None,
):
    if time_lags is None:
        time_lags = [1, 2, 3, 24, 25, 26, 168, 169]
    n_entity, n_time = traffic_mat.shape
    weather_feat = zscore(weather_feat)
    weather_feat_tensor = torch.tensor(weather_feat, dtype=torch.float32, device=device)
    model = BTMF_ARLSTM(
        n_entity, n_time, rank, time_lags, weather_feat_tensor, torch.device(device)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mask = torch.tensor(traffic_mat != 0, dtype=torch.float32, device=device)
    Y = torch.tensor(traffic_mat, dtype=torch.float32, device=device)
    best_state = None
    best_mape = float("inf")
    patience = 10
    patience_counter = 0
    for epoch in range(burn_iter):
        model.train()
        optimizer.zero_grad()
        mat_hat, Z, pred_ar, X_ar, X_lstm, X, alpha = model()
        loss_rec = ((mat_hat - Y) ** 2 * mask).sum() / mask.sum()
        loss_ar = nn.functional.mse_loss(Z, pred_ar)
        loss_smooth = ((X_lstm[1:] - X_lstm[:-1]) ** 2).mean()
        loss_fusion = ((X_ar - X_lstm) ** 2).mean()
        mape_loss = torch.mean(
            torch.abs((mat_hat - Y)[mask.bool()] / (Y[mask.bool()] + 1e-8))
        )
        loss = (
            loss_rec
            + 0.1 * loss_ar
            + 0.01 * loss_smooth
            + 0.05 * loss_fusion
            + 0.01 * mape_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        # early stopping
        with torch.no_grad():
            mat_hat_eval, _, _, _, _, _, _ = model()
            pos = traffic_mat != 0
            mape = np.sum(
                np.abs(traffic_mat[pos] - mat_hat_eval.cpu().numpy()[pos])
                / (traffic_mat[pos] + 1e-8)
            ) / np.sum(traffic_mat[pos] != 0)
            if mape < best_mape:
                best_mape = mape
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter > patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        if (epoch + 1) % 200 == 0:
            logging.info(f"Epoch {epoch+1}/{burn_iter}")
    if best_state is not None:
        model.load_state_dict(best_state)
    mat_hat, *_ = model()
    filled_mat = mat_hat.detach().cpu().numpy()
    return filled_mat

def run_btmf_fill_and_save(traffic_df, weather_df, output_dir, name="AR_LSTM_Weather"):
    os.makedirs(output_dir, exist_ok=True)
    traffic_mat = traffic_df.values.astype(np.float32).T
    n_entity, n_time = traffic_mat.shape
    # Missing Pattern Analysis + original missing rate pie chart
    traffic_mat_raw = traffic_df.values
    total = traffic_mat_raw.size
    missing = (traffic_mat_raw == 0).sum()
    filled = total - missing
    missing_rate = missing / total
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # Left: missing pattern
    missing_mask = traffic_mat == 0.0
    im = axes[0].imshow(missing_mask, aspect="auto", cmap="gray_r")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Node")
    axes[0].set_title("Missing Pattern (White=Missing, Black=Data)")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    # Right: missing rate pie chart
    labels = ['Not Missing', 'Missing']
    sizes = [filled, missing]
    colors = [COLORS['blue'], COLORS['orange']]
    axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90, counterclock=False)
    axes[1].set_title('Missing Rate of Original Traffic Dataset')
    axes[1].axis('equal')
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DATA_DIR, "missing_pattern_and_pie.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()
    logging.info(f"Missing pattern and pie chart saved to: {out_path}")
    # Weather Feature
    weather_cols = ["AWND", "PRCP", "RHAV", "TMAX", "TMIN", "WSF2"]
    weather_feat = weather_df[weather_cols].values.astype(np.float32)
    if weather_feat.shape[0] > n_time:
        weather_feat = weather_feat[:n_time, :]
    elif weather_feat.shape[0] < n_time:
        pad = np.tile(weather_feat[-1], (n_time - weather_feat.shape[0], 1))
        weather_feat = np.vstack([weather_feat, pad])
    weather_feat = zscore(weather_feat)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weather_feat_tensor = torch.tensor(weather_feat, dtype=torch.float32, device=device)
    sparse_mat = np.copy(traffic_mat)
    dense_mat = np.copy(traffic_mat)
    rank = 30
    burn_iter = 5000
    lr = 0.02
    multi_step = 54
    pred_step = 7 * 144
    time_lags = [1, 2, 3, 24, 25, 26, 168, 169]
    model = BTMF_ARLSTM(
        n_entity, n_time, rank, time_lags, weather_feat_tensor, device
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    mask = torch.tensor(sparse_mat != 0, dtype=torch.float32, device=device)
    Y = torch.tensor(sparse_mat, dtype=torch.float32, device=device)
    mape_history = []
    rmse_history = []
    best_mape = float("inf")
    best_state = None
    patience = 20
    patience_counter = 0
    for epoch in range(burn_iter):
        model.train()
        optimizer.zero_grad()
        mat_hat, Z, pred_ar, X_ar, X_lstm, X, alpha = model()
        loss_rec = ((mat_hat - Y) ** 2 * mask).sum() / mask.sum()
        loss_ar = nn.functional.mse_loss(Z, pred_ar)
        loss_smooth = ((X_lstm[1:] - X_lstm[:-1]) ** 2).mean()
        loss_fusion = ((X_ar - X_lstm) ** 2).mean()
        mape_loss = torch.mean(
            torch.abs((mat_hat - Y)[mask.bool()] / (Y[mask.bool()] + 1e-8))
        )
        loss = (
            loss_rec
            + 0.1 * loss_ar
            + 0.01 * loss_smooth
            + 0.05 * loss_fusion
            + 0.01 * mape_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        scheduler.step(loss)
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                mat_hat_eval, _, _, _, _, _, _ = model()
                pos = dense_mat != 0
                mape = compute_mape(dense_mat[pos], mat_hat_eval.cpu().numpy()[pos])
                rmse = compute_rmse(dense_mat[pos], mat_hat_eval.cpu().numpy()[pos])
                mape_history.append(mape)
                rmse_history.append(rmse)
                if mape < best_mape:
                    best_mape = mape
                    best_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter > patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break
        if (epoch + 1) % 200 == 0:
            logging.info(f"Epoch {epoch+1}/{burn_iter}")
    if best_state is not None:
        model.load_state_dict(best_state)
    mat_hat, _, _, _, _, _, alpha = model()
    mat_hat = mat_hat.detach().cpu().numpy()
    mat_pred = model.predict_multi(multi_step)
    pos = dense_mat != 0
    mape = compute_mape(dense_mat[pos], mat_hat[pos])
    rmse = compute_rmse(dense_mat[pos], mat_hat[pos])
    # Linear regression correction
    y_true = dense_mat[pos].reshape(-1, 1)
    y_pred = mat_hat[pos].reshape(-1, 1)
    reg = LinearRegression().fit(y_pred, y_true)
    mat_hat_corrected = reg.predict(mat_hat.reshape(-1, 1)).reshape(mat_hat.shape)
    mat_pred_corrected = reg.predict(mat_pred.reshape(-1, 1)).reshape(mat_pred.shape)
    # Corrected metrics
    mape_corr = compute_mape(dense_mat[pos], mat_hat_corrected[pos])
    rmse_corr = compute_rmse(dense_mat[pos], mat_hat_corrected[pos])
    logging.info(f"MAPE: {mape:.6f}, RMSE: {rmse:.6f} | Corrected MAPE: {mape_corr:.6f}, RMSE: {rmse_corr:.6f}")
    plot_metrics(mape_history, rmse_history, name)
    plot_alpha(alpha, name)
    # Save complete filled data
    filled_dir = 'data/filled'
    os.makedirs(filled_dir, exist_ok=True)
    filled_df = pd.DataFrame(
        mat_hat_corrected.T, index=traffic_df.index, columns=traffic_df.columns
    )
    filled_df.to_hdf(os.path.join(filled_dir, 'btmf_fill.h5'), key='df', mode='w')

    return filled_df 