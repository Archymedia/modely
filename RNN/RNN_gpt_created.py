#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN_full.py
Vanilla SimpleRNN for next-day Z-score simple return prediction and trading strategy backtest.

- Same feature logic as in your earlier RNN (technical indicators preserved).
- Target: next-day SimpleReturn Z-score (computed without data leakage).
- Train/test split consistent with MLP (train up to 2020-12-31, test from 2021-01-01).
- Strategy: each day pick top 10 longs and bottom 10 shorts; PT/SL = ±2% from entry close; hold until a barrier hits.
- Metrics: MSE, MAE, RMSE, R²; strategy metrics (CAGR, annualized vol, Sharpe, max drawdown, win rate, profit factor,
  realized alpha vs equal-weight market, significance tests).
- Visualization: single PNG "RNN_results.png" with loss curve, equity curve (vertical line at 2020-12-31), and two tables.

Author: Assistant
Date: 2025-08-10
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

import warnings
warnings.filterwarnings("ignore")

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
@dataclass
class Config:
    DATA_PATH: str = "/Users/lindawaisova/Desktop/DP/data/SP_100/READY_DATA/9DATA_FINAL.csv"
    VIX_PATH: str = "/Users/lindawaisova/Desktop/DP/data/SP_100/VIX/VIX_2005_2023.csv"
    TRAIN_END_DATE: str = "2020-12-31"
    TEST_START_DATE: str = "2021-01-01"
    TIMESTEPS: int = 5
    RANDOM_SEED: int = 42

    # Model / training
    RNN_LAYERS: int = 2
    UNITS: int = 64
    LR: float = 0.001
    DROPOUT: float = 0.2
    L2_REG: float = 0.001
    BATCH_SIZE: int = 64
    EPOCHS: int = 150
    ES_PATIENCE: int = 25

    # Strategy
    N_LONG: int = 10
    N_SHORT: int = 10
    PT_PCT: float = 0.02  # 2%
    SL_PCT: float = 0.02  # 2%

    # Risk-free rate (annual, for Sharpe/alpha)
    RF_ANNUAL: float = 0.02  # 2% p.a.
CFG = Config()

np.random.seed(CFG.RANDOM_SEED)
tf.random.set_seed(CFG.RANDOM_SEED)

print("="*80)
print("RNN (SimpleRNN) pipeline starting...")
print("Hyperparameters / Settings:")
for k, v in asdict(CFG).items():
    print(f"  {k}: {v}")
print("="*80)

# =========================
# DATA LOADING
# =========================
def load_data():
    if not os.path.exists(CFG.DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {CFG.DATA_PATH}")
    if not os.path.exists(CFG.VIX_PATH):
        raise FileNotFoundError(f"VIX file not found: {CFG.VIX_PATH}")
    df = pd.read_csv(CFG.DATA_PATH)
    vix = pd.read_csv(CFG.VIX_PATH)

    # Ensure datetime
    for dcol in ["Date"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol])
        if dcol in vix.columns:
            vix[dcol] = pd.to_datetime(vix[dcol])

    # Merge VIX on Date if present
    if "Date" in df.columns and "Date" in vix.columns and "VIX" in vix.columns:
        df = df.merge(vix[["Date","VIX"]], on="Date", how="left")

    return df

# =========================
# FEATURE ENGINEERING (technical indicators; look-ahead safe)
# =========================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[INFO] Computing technical indicators (look-ahead safe)...")
    df = df.sort_values(["ID","Date"]).reset_index(drop=True)
    out = []

    required_cols = ["CloseAdj","HighAdj","LowAdj","OpenAdj","Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data.")

    for sid, g in df.groupby("ID", sort=False):
        s = g.copy().sort_values("Date").reset_index(drop=True)

        # Basic lags
        if "SimpleReturn" in s.columns:
            for lag in [1,2,3,5,10]:
                s[f"SimpleReturn_lag_{lag}"] = s["SimpleReturn"].shift(lag)

        # SMA / EMA, ratios
        close_hist = s["CloseAdj"].shift(1)
        for p in [5,10,20]:
            sma = close_hist.rolling(p).mean()
            ema = close_hist.ewm(span=p, adjust=False).mean()
            s[f"SMA_{p}"] = sma
            s[f"EMA_{p}"] = ema
            s[f"Price_SMA_{p}_ratio"] = close_hist / sma
            s[f"Price_EMA_{p}_ratio"] = close_hist / ema

        # MACD
        ema12 = close_hist.ewm(span=12, adjust=False).mean()
        ema26 = close_hist.ewm(span=26, adjust=False).mean()
        s["MACD"] = ema12 - ema26
        s["MACD_signal"] = s["MACD"].ewm(span=9, adjust=False).mean()
        s["MACD_hist"] = s["MACD"] - s["MACD_signal"]

        # RSI
        for p in [7,14]:
            delta = close_hist.diff()
            gain = delta.clip(lower=0).rolling(p).mean()
            loss = (-delta.clip(upper=0)).rolling(p).mean()
            rs = gain / loss
            s[f"RSI_{p}"] = 100 - (100/(1+rs))

        # Bollinger
        sma20 = close_hist.rolling(20).mean()
        std20 = close_hist.rolling(20).std()
        s["BB_upper"] = sma20 + 2*std20
        s["BB_lower"] = sma20 - 2*std20
        s["BB_pos"] = (close_hist - s["BB_lower"]) / (s["BB_upper"] - s["BB_lower"])

        # ATR (14)
        high_hist = s["HighAdj"].shift(1)
        low_hist  = s["LowAdj"].shift(1)
        prev_close = s["CloseAdj"].shift(1)
        tr = np.maximum(high_hist-low_hist, np.maximum((high_hist-prev_close).abs(), (low_hist-prev_close).abs()))
        s["ATR_14"] = tr.rolling(14).mean()

        # HV
        if "SimpleReturn" in s.columns:
            for p in [10,20]:
                s[f"HV_{p}"] = s["SimpleReturn"].shift(1).rolling(p).std() * np.sqrt(252)

        # OBV
        if "SimpleReturn" in s.columns:
            s["OBV"] = (s["Volume"].shift(1) * np.sign(s["SimpleReturn"].shift(1))).cumsum()

        # VIX derived
        if "VIX" in s.columns:
            s["VIX_SMA_5"] = s["VIX"].shift(1).rolling(5).mean()
            s["VIX_change"] = s["VIX"].shift(1).pct_change()

        out.append(s)

    return pd.concat(out, axis=0).reset_index(drop=True)

# =========================
# TARGET: Z-score of next-day SimpleReturn (no leakage)
# =========================
def add_target_zscore(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Preparing target: next-day Z-score of SimpleReturn (no leakage)...")
    df = df.sort_values(["ID","Date"]).reset_index(drop=True)
    # Next-day return as base target
    df["target_raw"] = df.groupby("ID")["SimpleReturn"].shift(-1)

    # Train/test split by date
    train_mask = df["Date"] <= pd.to_datetime(CFG.TRAIN_END_DATE)
    test_mask  = df["Date"] >= pd.to_datetime(CFG.TEST_START_DATE)

    # Compute per-ID mean/std on TRAIN ONLY
    mu = df.loc[train_mask].groupby("ID")["SimpleReturn"].mean()
    sd = df.loc[train_mask].groupby("ID")["SimpleReturn"].std().replace(0, np.nan)

    # Map to all rows (per ID), but using train stats
    df["SR_mu_train"] = df["ID"].map(mu)
    df["SR_sd_train"] = df["ID"].map(sd).fillna(1.0)  # avoid div by zero

    # Z-score for next-day return (target_raw)
    df["target"] = (df["target_raw"] - df["SR_mu_train"]) / df["SR_sd_train"]

    return df

# =========================
# TRAIN/TEST SPLIT + FEATURE SELECTION
# =========================
def build_feature_set(df: pd.DataFrame):
    print("[INFO] Selecting features (matching MLP feature philosophy)...")
    exclude = {
        "ID","RIC","Name","Date","target","target_raw","SR_mu_train","SR_sd_train",
        "TotRet","SimpleReturn","Close","Volume","Open","High","Low",
        "CloseAdj","OpenAdj","HighAdj","LowAdj","VIX"
    }
    feature_cols = []
    for c in df.columns:
        if c in exclude: 
            continue
        if df[c].dtype.kind not in "fi": 
            continue
        # Keep engineered features
        if any(k in c for k in ["_lag","SMA_","EMA_","MACD","RSI_","BB_","ATR_","HV_","OBV","VIX_","Price_"]):
            feature_cols.append(c)
    print(f"[INFO] Selected {len(feature_cols)} features.")
    return feature_cols

def split_train_test(df: pd.DataFrame, feature_cols):
    train_df = df[df["Date"] <= pd.to_datetime(CFG.TRAIN_END_DATE)].copy()
    test_df  = df[df["Date"] >= pd.to_datetime(CFG.TEST_START_DATE)].copy()

    # Drop NaNs
    train_df = train_df.dropna(subset=feature_cols + ["target"])
    test_df  = test_df.dropna(subset=feature_cols + ["target"])

    return train_df, test_df

# =========================
# SEQUENCE BUILDER
# =========================
def build_sequences(df: pd.DataFrame, feature_cols, timesteps: int):
    X_list, y_list = [], []
    ids = df["ID"].unique()
    for sid in ids:
        s = df[df["ID"]==sid].sort_values("Date")
        arr = s[feature_cols].values
        targ = s["target"].values
        for i in range(len(s) - timesteps):
            X_list.append(arr[i:i+timesteps])
            y_list.append(targ[i+timesteps-1])
    if not X_list:
        return np.empty((0,timesteps,len(feature_cols))), np.empty((0,))
    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y

# =========================
# MODEL
# =========================
def make_model(input_timesteps: int, input_dim: int) -> tf.keras.Model:
    m = Sequential()
    for li in range(CFG.RNN_LAYERS):
        return_seq = (li < CFG.RNN_LAYERS - 1)
        if li == 0:
            m.add(SimpleRNN(
                units=CFG.UNITS, activation="tanh", return_sequences=return_seq,
                input_shape=(input_timesteps, input_dim),
                kernel_regularizer=l2(CFG.L2_REG)
            ))
        else:
            m.add(SimpleRNN(
                units=CFG.UNITS, activation="tanh", return_sequences=return_seq,
                kernel_regularizer=l2(CFG.L2_REG)
            ))
        m.add(Dropout(CFG.DROPOUT))
    m.add(Dense(1, activation="linear"))
    m.compile(optimizer=Adam(CFG.LR), loss="mse", metrics=["mae"])
    return m

# =========================
# STRATEGY SIMULATOR (PT/SL ±2% from entry close, hold until barrier)
# =========================
class Position:
    __slots__ = ("sid","direction","entry_date","entry_price","open_idx")

    def __init__(self, sid, direction, entry_date, entry_price, open_idx):
        self.sid = sid
        self.direction = direction  # +1 long, -1 short
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.open_idx = open_idx  # index in the daily series to resume checks

def simulate_strategy(test_df: pd.DataFrame, pred_df: pd.DataFrame):
    """
    test_df columns required per (ID, Date): CloseAdj, HighAdj, LowAdj
    pred_df: per (ID, Date) prediction for next-day z-score target; we use it for ranking at each Date.

    Logic:
    - Each date D: select top N_LONG longs and bottom N_SHORT shorts based on prediction at D.
      Entry at CloseAdj[D].
    - Then for each open position, from D+1 onward, check High/Low relative to entry price:
        * LONG: TP=+PT_PCT, SL=-SL_PCT; SHORT vice versa.
        * If both breached the same day, assume SL first (conservative).
      Daily MTM uses close-to-close for open positions; on exit day, cap return to barrier level.
    - Daily portfolio return = average of per-position daily returns across all open positions that day.
    """
    if not {"ID","Date","CloseAdj","HighAdj","LowAdj"}.issubset(set(test_df.columns)):
        raise ValueError("test_df must contain ID, Date, CloseAdj, HighAdj, LowAdj")

    df = test_df.sort_values(["Date","ID"]).copy()
    pred = pred_df.sort_values(["Date","ID"]).copy()

    # Align predictions to create daily rankings (use prediction at date D to open at close D)
    by_date = pred.groupby("Date")

    # Build quick access maps
    df = df.set_index(["ID","Date"]).sort_index()
    all_dates = sorted(test_df["Date"].unique())

    rf_daily = (1 + CFG.RF_ANNUAL)**(1/252) - 1

    positions = {}  # sid -> Position (allow only one per stock)
    daily_returns = []  # (Date, ret, mkt_ret)

    # Precompute daily close returns for MTM
    test_sorted = test_df.sort_values(["ID","Date"]).copy()
    test_sorted["close_ret"] = test_sorted.groupby("ID")["CloseAdj"].pct_change().fillna(0.0)
    close_ret_map = test_sorted.set_index(["ID","Date"])["close_ret"]

    # Market equal-weight daily return:
    mkt_daily = test_sorted.groupby("Date")["close_ret"].mean().reindex(all_dates).fillna(0.0)

    # Loop by date
    for di, D in enumerate(all_dates):
        # 1) Update open positions, check barriers, compute daily contributions
        todays_pos_returns = []

        ids_today = [sid for sid in list(positions.keys())]
        for sid in ids_today:
            pos = positions[sid]
            if (sid, D) not in df.index:
                continue
            row = df.loc[(sid, D)]
            close_today = row["CloseAdj"]
            high_today = row["HighAdj"]
            low_today  = row["LowAdj"]
            daily_mtm = close_ret_map.get((sid, D), 0.0)

            if pos.direction == +1:
                tp = pos.entry_price * (1 + CFG.PT_PCT)
                sl = pos.entry_price * (1 - CFG.SL_PCT)
                hit_tp = (high_today >= tp)
                hit_sl = (low_today  <= sl)
                if hit_tp or hit_sl:
                    ret_total = (CFG.PT_PCT if (hit_tp and not hit_sl) else
                                 (-CFG.SL_PCT if (hit_sl and not hit_tp) else -CFG.SL_PCT))
                    # Assign today's contribution as ret_total minus cumulative since entry until yesterday
                    try:
                        sid_series = test_sorted[test_sorted["ID"]==sid].set_index("Date")["CloseAdj"].sort_index()
                        if pos.entry_date in sid_series.index and D in sid_series.index:
                            entry_close = sid_series.loc[pos.entry_date]
                            exit_price = tp if (hit_tp and not hit_sl) else (sl if (hit_sl and not hit_tp) else sl)
                            ret_since_entry_today = (exit_price / entry_close) - 1.0
                            prev_day_idx = max([dt for dt in sid_series.index if dt < D], default=None)
                            if prev_day_idx is not None and prev_day_idx >= pos.entry_date:
                                prev_close = sid_series.loc[prev_day_idx]
                                ret_until_prev = (prev_close / entry_close) - 1.0
                            else:
                                ret_until_prev = 0.0
                            today_contrib = ret_since_entry_today - ret_until_prev
                        else:
                            today_contrib = ret_total
                    except Exception:
                        today_contrib = ret_total
                    todays_pos_returns.append(today_contrib)
                    positions.pop(sid, None)
                else:
                    todays_pos_returns.append(daily_mtm)
            else:
                tp = pos.entry_price * (1 - CFG.PT_PCT)
                sl = pos.entry_price * (1 + CFG.SL_PCT)
                hit_tp = (low_today  <= tp)
                hit_sl = (high_today >= sl)
                if hit_tp or hit_sl:
                    ret_total = (CFG.PT_PCT if (hit_tp and not hit_sl) else
                                 (-CFG.SL_PCT if (hit_sl and not hit_tp) else -CFG.SL_PCT))
                    try:
                        sid_series = test_sorted[test_sorted["ID"]==sid].set_index("Date")["CloseAdj"].sort_index()
                        if pos.entry_date in sid_series.index and D in sid_series.index:
                            entry_close = sid_series.loc[pos.entry_date]
                            exit_price = tp if (hit_tp and not hit_sl) else (sl if (hit_sl and not hit_tp) else sl)
                            ret_since_entry_today = (entry_close - exit_price) / entry_close
                            prev_day_idx = max([dt for dt in sid_series.index if dt < D], default=None)
                            if prev_day_idx is not None and prev_day_idx >= pos.entry_date:
                                prev_close = sid_series.loc[prev_day_idx]
                                ret_until_prev = (entry_close - prev_close)/entry_close
                            else:
                                ret_until_prev = 0.0
                            today_contrib = ret_since_entry_today - ret_until_prev
                        else:
                            today_contrib = ret_total
                    except Exception:
                        today_contrib = ret_total
                    todays_pos_returns.append(today_contrib)
                    positions.pop(sid, None)
                else:
                    todays_pos_returns.append(-daily_mtm)

        # 2) Open new positions based on predictions at D
        if D in pred["Date"].values:
            g = pred[pred["Date"]==D].copy()
            g = g[~g["ID"].isin(positions.keys())]
            g = g.sort_values("pred", ascending=False)
            longs  = g.head(CFG.N_LONG)
            shorts = g.tail(CFG.N_SHORT)

            for _, row in longs.iterrows():
                sid = row["ID"]
                if (sid, D) in df.index:
                    entry_price = df.loc[(sid, D)]["CloseAdj"]
                    positions[sid] = Position(sid, +1, D, entry_price, None)
            for _, row in shorts.iterrows():
                sid = row["ID"]
                if (sid, D) in df.index:
                    entry_price = df.loc[(sid, D)]["CloseAdj"]
                    positions[sid] = Position(sid, -1, D, entry_price, None)

        # 3) Daily portfolio return
        port_ret = np.mean(todays_pos_returns) if todays_pos_returns else 0.0
        daily_returns.append((D, port_ret, mkt_daily.loc[D] if D in mkt_daily.index else 0.0))

    df_ret = pd.DataFrame(daily_returns, columns=["Date","strategy_ret","market_ret"]).sort_values("Date")
    df_ret["cum_strategy"] = (1 + df_ret["strategy_ret"]).cumprod()
    df_ret["cum_market"] = (1 + df_ret["market_ret"]).cumprod()
    df_ret["excess_ret"] = df_ret["strategy_ret"] - rf_daily
    return df_ret

# =========================
# METRICS
# =========================
def compute_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}

def compute_strategy_metrics(df_ret: pd.DataFrame):
    rf_daily = (1 + CFG.RF_ANNUAL)**(1/252) - 1
    rets = df_ret["strategy_ret"].values
    n = len(rets)
    if n == 0:
        return {}

    # CAGR
    cum = (1 + rets).prod()
    years = n / 252.0
    cagr = cum**(1/years) - 1 if years > 0 else 0.0

    # Vol / Sharpe
    vol_ann = np.std(rets, ddof=1) * np.sqrt(252)
    mean_daily = np.mean(rets)
    sharpe = 0.0 if vol_ann == 0 else ((mean_daily - rf_daily) / np.std(rets, ddof=1)) * np.sqrt(252)

    # Max drawdown
    equity = (1 + df_ret["strategy_ret"]).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak) - 1
    max_dd = drawdown.min()

    # Win rate & Profit factor based on daily returns sign
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    win_rate = len(wins) / len(rets) if len(rets) > 0 else 0.0
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan

    # Realized alpha via CAPM: (R_p - R_f) = alpha + beta*(R_m - R_f) + eps
    ex_p = df_ret["strategy_ret"].values - rf_daily
    ex_m = df_ret["market_ret"].values - rf_daily
    X = np.vstack([np.ones_like(ex_m), ex_m]).T
    beta_hat = np.linalg.lstsq(X, ex_p, rcond=None)[0]
    alpha_daily = beta_hat[0]
    beta = beta_hat[1]
    alpha_ann = alpha_daily * 252

    # t-stat and p-value for alpha (simple OLS with homoskedastic SE)
    resid = ex_p - X @ beta_hat
    s2 = (resid**2).sum() / (len(ex_p) - 2)
    var_alpha = s2 * np.linalg.inv(X.T @ X)[0,0]
    se_alpha = np.sqrt(var_alpha)
    t_alpha = alpha_daily / se_alpha if se_alpha > 0 else np.nan

    # t-stat for mean excess return
    se_mean = np.std(ex_p, ddof=1) / np.sqrt(len(ex_p)) if len(ex_p) > 1 else np.nan
    t_mean = (np.mean(ex_p) / se_mean) if se_mean and se_mean>0 else np.nan

    return {
        "CAGR": cagr,
        "Ann_Vol": vol_ann,
        "Sharpe": sharpe,
        "Max_Drawdown": max_dd,
        "Win_Rate": win_rate,
        "Profit_Factor": profit_factor,
        "Alpha_ann": alpha_ann,
        "Beta": beta,
        "t_alpha": t_alpha,
        "t_mean_excess": t_mean,
    }

# =========================
# VISUALIZATION
# =========================
def save_summary_png(history, df_ret, reg_metrics: dict, model_info: dict, out_path="RNN_results.png"):
    print("\n[INFO] Saving summary PNG ->", out_path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Loss curve
    ax = axes[0,0]
    ax.plot(history.history.get("loss", []), label="Train Loss")
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="Val Loss")
    ax.set_title("RNN Loss / Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2) Equity curve with vertical line at 2020-12-31
    ax = axes[0,1]
    ax.plot(df_ret["Date"], df_ret["cum_strategy"], label="Strategy (cum)")
    ax.plot(df_ret["Date"], df_ret["cum_market"], label="Market (cum)")
    vline_date = pd.to_datetime("2020-12-31")
    ax.axvline(vline_date, color="k", linestyle="--", linewidth=1, label="Train/Test split")
    ax.set_title("Cumulative Returns (Strategy vs Market)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth (x)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3) Table: best hyperparams & model characteristics
    ax = axes[1,0]
    ax.axis("off")
    table_data = [[k, f"{v}"] for k, v in model_info.items()]
    table = ax.table(cellText=table_data, colLabels=["Model / Setting", "Value"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)
    ax.set_title("RNN Hyperparameters & Settings", pad=12)

    # 4) Table: regression & strategy metrics
    ax = axes[1,1]
    ax.axis("off")
    strat_metrics = compute_strategy_metrics(df_ret)
    def fmt(v, pct=False):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "NaN"
        if pct:
            return f"{v*100:.2f}%"
        return f"{v:.4f}"
    table_rows = [
        ["MSE", fmt(reg_metrics.get("MSE"))],
        ["MAE", fmt(reg_metrics.get("MAE"))],
        ["RMSE", fmt(reg_metrics.get("RMSE"))],
        ["R²", fmt(reg_metrics.get("R2"))],
        ["CAGR", fmt(strat_metrics.get("CAGR"), pct=True)],
        ["Ann. Vol", fmt(strat_metrics.get("Ann_Vol"))],
        ["Sharpe", fmt(strat_metrics.get("Sharpe"))],
        ["Max DD", fmt(strat_metrics.get("Max_Drawdown"), pct=True)],
        ["Win Rate", fmt(strat_metrics.get("Win_Rate"), pct=True)],
        ["Profit Factor", fmt(strat_metrics.get("Profit_Factor"))],
        ["Alpha (ann.)", fmt(strat_metrics.get("Alpha_ann"))],
        ["Beta", fmt(strat_metrics.get("Beta"))],
        ["t(alpha)", fmt(strat_metrics.get("t_alpha"))],
        ["t(mean excess)", fmt(strat_metrics.get("t_mean_excess"))],
    ]
    table2 = ax.table(cellText=table_rows, colLabels=["Metric", "Value"], loc="center")
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.2, 1.2)
    ax.set_title("Regression & Strategy Metrics", pad=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

# =========================
# MAIN
# =========================
def main():
    t0 = time.time()
    print("\n[STEP] Loading data...")
    df = load_data()
    print(f"[INFO] Raw rows: {len(df):,}, columns: {len(df.columns)}")

    print("[STEP] Computing indicators...")
    df = compute_indicators(df)

    print("[STEP] Preparing target (Z-score next-day SimpleReturn)...")
    df = add_target_zscore(df)

    print("[STEP] Building feature set...")
    feature_cols = build_feature_set(df)

    print("[STEP] Train/Test split...")
    train_df, test_df = split_train_test(df, feature_cols)

    # Scale features using train only
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    def transform_seq(X_seq):
        n_s, n_t, n_f = X_seq.shape
        X2 = X_seq.reshape(-1, n_f)
        X2s = scaler.transform(X2)
        return X2s.reshape(n_s, n_t, n_f)

    print("[STEP] Building sequences (train/test)...")
    X_train_seq, y_train = build_sequences(train_df, feature_cols, CFG.TIMESTEPS)
    X_test_seq,  y_test  = build_sequences(test_df, feature_cols, CFG.TIMESTEPS)
    print(f"[INFO] Train sequences: {X_train_seq.shape}, Test sequences: {X_test_seq.shape}")

    X_train_seq = transform_seq(X_train_seq)
    X_test_seq  = transform_seq(X_test_seq)

    # Model
    print("[STEP] Compiling RNN model...")
    model = make_model(CFG.TIMESTEPS, X_train_seq.shape[2])
    print(model.summary())

    # Train
    print("[STEP] Training...")
    es = EarlyStopping(monitor="val_loss", patience=CFG.ES_PATIENCE, restore_best_weights=True, verbose=1)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_seq, y_train, test_size=0.1, random_state=CFG.RANDOM_SEED)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=CFG.EPOCHS,
        batch_size=CFG.BATCH_SIZE,
        verbose=2,
        callbacks=[es]
    )

    # Predict on test
    print("[STEP] Predicting on test sequences...")
    y_pred = model.predict(X_test_seq).reshape(-1)

    # Regression metrics
    reg_metrics = compute_regression_metrics(y_test, y_pred)
    print("\n[RESULT] Regression metrics (test):")
    for k, v in reg_metrics.items():
        print(f"  {k}: {v:.6f}")

    # Build per-day predictions dataframe for ranking
    # We need to reconstruct (Date, ID) pairs corresponding to each sequence end in test_df
    print("[STEP] Building prediction DataFrame for ranking...")
    meta = []
    X_list = []
    ids = test_df["ID"].unique()
    for sid in ids:
        s = test_df[test_df["ID"]==sid].sort_values("Date")
        arr = s[feature_cols].values
        n = len(s)
        for i in range(n - CFG.TIMESTEPS):
            meta.append((sid, s["Date"].iloc[i+CFG.TIMESTEPS-1]))
            X_list.append(arr[i:i+CFG.TIMESTEPS])
    X_meta = np.stack(X_list) if len(X_list)>0 else np.empty((0, CFG.TIMESTEPS, len(feature_cols)))
    # scale
    n_s, n_t, n_f = X_meta.shape
    if n_s > 0:
        X2 = X_meta.reshape(-1, n_f)
        X2s = StandardScaler().fit(train_df[feature_cols].values).transform(X2)  # ensure same scaler
        X_meta_scaled = X2s.reshape(n_s, n_t, n_f)
        preds = model.predict(X_meta_scaled).reshape(-1)
        pred_df = pd.DataFrame(meta, columns=["ID","Date"])
        pred_df["Date"] = pd.to_datetime(pred_df["Date"])
        pred_df["pred"] = preds
    else:
        pred_df = pd.DataFrame(columns=["ID","Date","pred"])

    # Strategy simulation
    print("[STEP] Simulating trading strategy (PT/SL, hold-until-barrier)...")
    df_ret = simulate_strategy(test_df[["ID","Date","CloseAdj","HighAdj","LowAdj","SimpleReturn"]].copy(), pred_df)

    # Model info for table
    model_info = {
        "Model": "Vanilla SimpleRNN",
        "RNN_LAYERS": CFG.RNN_LAYERS,
        "UNITS": CFG.UNITS,
        "LR": CFG.LR,
        "DROPOUT": CFG.DROPOUT,
        "L2_REG": CFG.L2_REG,
        "BATCH_SIZE": CFG.BATCH_SIZE,
        "EPOCHS": CFG.EPOCHS,
        "TIMESTEPS": CFG.TIMESTEPS,
        "TRAIN_END_DATE": CFG.TRAIN_END_DATE,
        "TEST_START_DATE": CFG.TEST_START_DATE,
        "PT/SL": f"{CFG.PT_PCT*100:.1f}% / {CFG.SL_PCT*100:.1f}%",
        "N_LONG/N_SHORT": f"{CFG.N_LONG}/{CFG.N_SHORT}",
        "RF_ANNUAL": f"{CFG.RF_ANNUAL*100:.2f}%"
    }

    # Visualization
    save_summary_png(history, df_ret, reg_metrics, model_info, out_path="RNN_results.png")

    t1 = time.time()
    print("\n" + "="*80)
    print("RNN run completed successfully.")
    print(f"Total runtime: {(t1 - t0)/60:.2f} minutes")
    print("Summary image saved: RNN_results.png")
    print("="*80)

if __name__ == "__main__":
    main()
