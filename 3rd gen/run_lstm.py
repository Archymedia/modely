#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM pipeline for predicting next-day Z-scored simple returns (SP100) + trading backtest.
- Strict time-aware splits (global date boundaries)
- Per-ID, train-only standardization to avoid leakage
- Feature engineering with 1-day lag for all predictors
- Manual randomized search over compact hyperparameter space
- EarlyStopping on validation (time-split from training period)
- Realistic trading: signal D-1 -> enter at open D+1; PT/SL ±2% from entry open
- PNG recap: losses, cumulative P&L, and metric tables
- Verbose terminal prints and total elapsed time

Author: ChatGPT (LSTM version)
"""

import os
import sys
import time
import json
import math
import random
import traceback
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks, optimizers

np.set_printoptions(suppress=True)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 200)

# ===== HYPERPARAMETER CONFIGURATION =====
HYPERPARAMS = {
    # DATA & SPLIT
    'data': {
        'data_path': ["/Users/lindawaisova/Desktop/DP/data/SP_100/READY_DATA/9DATA_FINAL.csv", r"C:\Users\david\Desktop\SP100\9DATA_FINAL.csv"],
        'vix_path': ["/Users/lindawaisova/Desktop/DP/data/SP_100/READY_DATA/VIX_2005_2023.csv", r"C:\Users\david\Desktop\SP100\VIX_2005_2023.csv"],
        'train_end_date': '2020-12-31',
        'test_start_date': '2021-01-01',
    },

    # Training configuration
'final_model_epochs': 100,
'cv_epochs': 20,
'patience': 10,
'cv_folds': 3,
'n_iter': 20,

    # HYPERPARAMETER SEARCH SPACE (LSTM)
    'search_space': {
        'layers': [1, 2],                   # number of LSTM layers
        'units': [32, 64, 128],             # hidden units per layer
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'window': [20, 40, 60]              # shared with RNN/CNN
    },

    # FIXED PARAMS (non-tuned)
    'fixed_params': {
        'dropout_rate': 0.15,
        'batch_size': 128,
        'dropout_rate': 0.15,
        'l2_reg': 3e-4,
        'random_seed': 42,
    },

    # STRATEGY
    'strategy': {
        'n_long': 10,
        'n_short': 10,
        'pt_pct': 0.02,    # 2%
        'sl_pct': 0.02,    # 2%
        'rf_annual': 0.02  # 2% p.a.
    }
}

# ---------------------- 0. Utilities & Repro ----------------------
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def verbose_header(title: str):
    print("\n" + "="*12 + f" {title} " + "="*12)

def parse_dates(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
    return df

def to_daily_rf(rf_annual: float) -> float:
    return (1 + rf_annual)**(1/252) - 1

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

# ---------------------- 1. Data Loading and Preprocessing ----------------------
def resolve_first_existing(p):
    import os
    if isinstance(p, (list, tuple)):
        for cand in p:
            if isinstance(cand, str) and os.path.exists(cand):
                return cand
        return p[0]
    return p

def load_and_merge(data_path: str, vix_path: str) -> pd.DataFrame:
    verbose_header("1. Data Loading and Preprocessing")
    data_path = resolve_first_existing(data_path)
    print("# Loading main data:", data_path)
    df = pd.read_csv(data_path)
    df = parse_dates(df, "Date")

    vix_path = resolve_first_existing(vix_path)
    print("# Loading VIX data:", vix_path)
    vix = pd.read_csv(vix_path)
    vix_date_col = "Date" if "Date" in vix.columns else vix.columns[0]
    vix = vix.rename(columns={vix_date_col: "Date"})
    vix = parse_dates(vix, "Date")
    vix_val_col = "VIX" if "VIX" in vix.columns else [c for c in vix.columns if c.lower().startswith("vix") or c.lower() in ("close", "value")]
    if isinstance(vix_val_col, list):
        vix_val_col = vix_val_col[0]
    vix = vix[["Date", vix_val_col]].rename(columns={vix_val_col: "VIX"})

    print("# Merging VIX on Date ...")
    df = df.merge(vix, on="Date", how="left")
    df.sort_values(["Date", "ID"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_features(df: pd.DataFrame):
    # ===== REQUIRED COLUMNS =====
    required_cols = [
        "ID", "Date", "SimpleReturn",
        "OpenAdj", "HighAdj", "LowAdj", "CloseAdj", "Volume"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Chybějící sloupce v datech: {missing}")

    verbose_header("2. Feature Engineering (lagged by 1 day)")

    def by_id_apply(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date").copy()

        # Basic lagged return
        g["ret_lag1"] = g["SimpleReturn"].shift(1)

        # SMA/EMA
        for w in (5,10,20):
            g[f"sma_{w}"] = g["CloseAdj"].rolling(w).mean().shift(1)
            g[f"ema_{w}"] = g["CloseAdj"].ewm(span=w, adjust=False).mean().shift(1)

        # MACD (12,26,9)
        ema12 = g["CloseAdj"].ewm(span=12, adjust=False).mean()
        ema26 = g["CloseAdj"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        g["macd"] = macd.shift(1)
        g["macd_signal"] = signal.shift(1)

        # RSI
        def rsi(series, period=14):
            delta = series.diff()
            up = np.where(delta > 0, delta, 0.0)
            down = np.where(delta < 0, -delta, 0.0)
            roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
            roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
            rs = roll_up / (roll_down + 1e-12)
            rsi_val = 100 - (100 / (1 + rs))
            return rsi_val

        g["rsi_14"] = rsi(g["CloseAdj"], 14).shift(1)
        g["rsi_7"]  = rsi(g["CloseAdj"], 7).shift(1)

        # Bollinger (20,2)
        sma20 = g["CloseAdj"].rolling(20).mean()
        std20 = g["CloseAdj"].rolling(20).std(ddof=0)
        g["bb_mid_20"] = sma20.shift(1)
        g["bb_up_20"]  = (sma20 + 2*std20).shift(1)
        g["bb_low_20"] = (sma20 - 2*std20).shift(1)

        # ATR(14)
        tr1 = g["HighAdj"] - g["LowAdj"]
        tr2 = (g["HighAdj"] - g["CloseAdj"].shift(1)).abs()
        tr3 = (g["LowAdj"]  - g["CloseAdj"].shift(1)).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        g["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean().shift(1)

        # Historical volatility
        for w in (10,20):
            g[f"hv_{w}"] = g["SimpleReturn"].rolling(w).std(ddof=0).shift(1)

        # OBV
        direction = np.sign(g["CloseAdj"].diff().fillna(0.0))
        g["obv"] = (direction * g["Volume"]).fillna(0.0).cumsum().shift(1)

        # VROC
        for w in (5,10):
            g[f"vroc_{w}"] = (g["Volume"] - g["Volume"].shift(w)) / (g["Volume"].shift(w) + 1e-12)
            g[f"vroc_{w}"] = g[f"vroc_{w}"].shift(1)

        # ROC (price)
        for w in (5,10):
            g[f"roc_{w}"] = (g["CloseAdj"] - g["CloseAdj"].shift(w)) / (g["CloseAdj"].shift(w) + 1e-12)
            g[f"roc_{w}"] = g[f"roc_{w}"].shift(1)

        # VIX & VIX_SMA5
        if "VIX" in g.columns:
            g["VIX"] = g["VIX"].ffill()
            g["VIX_sma5"] = g["VIX"].rolling(5).mean().shift(1)
        else:
            g["VIX"] = np.nan
            g["VIX_sma5"] = np.nan

        return g

    print("# Computing technical indicators per ID (shifted by 1 day to avoid leakage) ...")
    df = df.groupby("ID", group_keys=False).apply(by_id_apply)

    # Target: next-day SimpleReturn z-scored per ID (train-only stats later)
    df["target_next_ret"] = df.groupby("ID")["SimpleReturn"].shift(-1)

    feature_cols = [
        "ret_lag1",
        "sma_5","sma_10","sma_20",
        "ema_5","ema_10","ema_20",
        "macd","macd_signal",
        "rsi_14","rsi_7",
        "bb_mid_20","bb_up_20","bb_low_20",
        "atr_14",
        "hv_10","hv_20",
        "obv",
        "vroc_5","vroc_10",
        "roc_5","roc_10",
        "VIX","VIX_sma5"
    ]

    df = df.sort_values(["Date","ID"]).reset_index(drop=True)
    cols_out = ["ID","Date","OpenAdj","HighAdj","LowAdj","CloseAdj","Volume","target_next_ret"] + feature_cols
    df_features = df[cols_out].copy()
    return df_features, feature_cols

def make_time_splits(df: pd.DataFrame, train_end: str, test_start: str, n_folds: int = 5) -> Dict:
    verbose_header("3. Time-based splits")
    train_end = pd.to_datetime(train_end)
    test_start = pd.to_datetime(test_start)

    train_df = df[df["Date"] <= train_end].copy()
    test_df  = df[df["Date"] >= test_start].copy()

    dates = np.sort(train_df["Date"].unique())
    folds = []
    chunk_edges = np.linspace(0, len(dates), n_folds+1, dtype=int)
    for i in range(1, len(chunk_edges)):
        val_start_idx = max(chunk_edges[i-1], 1)
        val_end_idx   = chunk_edges[i]
        val_start = dates[val_start_idx-1]
        val_end   = dates[val_end_idx-1]
        tr_idx = train_df["Date"] < val_start
        vl_idx = (train_df["Date"] >= val_start) & (train_df["Date"] <= val_end)
        if tr_idx.sum() == 0 or vl_idx.sum() == 0:
            continue
        folds.append((train_df.index[tr_idx].values, train_df.index[vl_idx].values))

    print(f"# Train size: {len(train_df):,} rows | Test size: {len(test_df):,} rows | Folds: {len(folds)}")
    return {"train_df": train_df, "test_df": test_df, "folds": folds}

def per_id_train_scalers(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, StandardScaler]:
    print("# Fitting per-ID StandardScaler on TRAIN only ...")
    scalers = {}
    for id_, g in train_df.groupby("ID"):
        X = g[feature_cols].values
        mask = np.isfinite(X).all(axis=1)
        sc = StandardScaler()
        if mask.sum() >= 5:
            sc.fit(X[mask])
        else:
            sc.fit(np.zeros((5, len(feature_cols))) + 1e-6)
        scalers[id_] = sc
    return scalers

def apply_scalers(df: pd.DataFrame, feature_cols: List[str], scalers: Dict[str, StandardScaler]) -> pd.DataFrame:
    print("# Applying per-ID scalers to a dataframe ...")
    df = df.copy()
    outs = []
    for id_, g in df.groupby("ID"):
        X = g[feature_cols].values
        sc = scalers.get(id_)
        if sc is None:
            sc = StandardScaler().fit(np.zeros((5, len(feature_cols))) + 1e-6)
        Xs = sc.transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
        g[feature_cols] = Xs
        outs.append(g)
    return pd.concat(outs, axis=0).sort_values(["Date","ID"]).reset_index(drop=True)

def make_target_zscore(train_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    print("# Computing per-ID z-score of next-day return using TRAIN statistics only ...")
    full_df = full_df.copy()
    z_list = []
    for id_, g in full_df.groupby("ID"):
        g = g.sort_values("Date").copy()
        g_train = g[g.index.isin(train_df.index)]
        mu = g_train["target_next_ret"].mean()
        sd = g_train["target_next_ret"].std(ddof=0)
        sd = sd if (sd is not None and np.isfinite(sd) and sd > 1e-12) else 1.0
        g["target_z"] = (g["target_next_ret"] - mu) / sd
        z_list.append(g)
    return pd.concat(z_list, axis=0).sort_values(["Date","ID"]).reset_index(drop=True)

# ---------------------- 3. Sequence Building ----------------------
def build_sequences(df: pd.DataFrame, feature_cols: List[str], window: int):
    X_list, y_list, meta = [], [], []
    for id_, g in df.groupby("ID"):
        g = g.sort_values("Date").copy()
        mat = g[feature_cols].values
        y = g["target_z"].values
        idx = g.index.values
        if len(g) < window:
            continue
        for t in range(window-1, len(g)):
            x_win = mat[t-window+1:t+1, :]
            y_t = y[t]
            if not (np.isfinite(x_win).all() and np.isfinite(y_t)):
                continue
            X_list.append(x_win)
            y_list.append(y_t)
            meta.append((id_, idx[t]))
    if not X_list:
        return np.zeros((0, window, len(feature_cols))), np.zeros((0,)), []
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y, meta

# ---------------------- 4. LSTM Model ----------------------
def build_lstm(input_dim: int, timesteps: int, layers_n: int, units: int, lr: float, l2_reg: float, dropout_rate: float) -> keras.Model:
    reg = regularizers.l2(l2_reg)
    model = keras.Sequential(name="LSTM_regressor")
    model.add(layers.Input(shape=(timesteps, input_dim)))
    for i in range(layers_n):
        return_seq = (i < layers_n - 1)
        model.add(layers.LSTM(units, return_sequences=return_seq,
                              kernel_regularizer=reg, recurrent_regularizer=None,
                              activation="tanh", recurrent_activation="sigmoid"))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(64, activation="relu", kernel_regularizer=reg))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation="linear"))
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse")
    return model

# ---------------------- 5. Random Search with Time CV ----------------------
def sample_space(space: Dict[str, List], n_iter: int, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    keys = list(space.keys())
    configs = []
    for _ in range(n_iter):
        conf = {k: rng.choice(space[k]) for k in keys}
        configs.append(conf)
    uniq, seen = [], set()
    for c in configs:
        t = tuple(sorted(c.items()))
        if t not in seen:
            seen.add(t); uniq.append(c)
    return uniq

def time_cv_evaluate(train_df: pd.DataFrame, folds: List[Tuple[np.ndarray, np.ndarray]], feature_cols: List[str],
                     cfg: Dict, fixed: Dict, epochs: int, patience: int, batch_size: int, l2_reg: float, dropout: float):
    val_losses = []
    last_history = None

    for fi, (tr_idx, vl_idx) in enumerate(folds, 1):
        tr = train_df.loc[tr_idx]
        vl = train_df.loc[vl_idx]

        X_tr, y_tr, _ = build_sequences(tr, feature_cols, cfg["window"])
        X_vl, y_vl, _ = build_sequences(vl, feature_cols, cfg["window"])

        if len(y_tr) == 0 or len(y_vl) == 0:
            print(f"    [Fold {fi}] not enough sequences; skipping fold.")
            continue

        model = build_lstm(
            input_dim=X_tr.shape[2],
            timesteps=X_tr.shape[1],
            layers_n=cfg["layers"],
            units=cfg["units"],
            lr=cfg["learning_rate"],
            l2_reg=l2_reg,
            dropout_rate=dropout
        )

        es = callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0)
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_vl, y_vl),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[es]
        )
        val_loss = min(hist.history.get("val_loss", [np.inf]))
        val_losses.append(val_loss)
        last_history = hist.history
        print(f"    [Fold {fi}/{len(folds)}] best val_loss = {val_loss:.6f}")

    if not val_losses:
        raise RuntimeError("No valid folds produced sequences; aborting search.")

    avg_val = float(np.mean(val_losses))
    return avg_val, {"history": last_history, "config": cfg}

def random_search(train_df: pd.DataFrame, folds: List[Tuple[np.ndarray, np.ndarray]], feature_cols: List[str],
                  search_space: Dict, fixed: Dict, cfg: Dict) -> Dict:
    verbose_header("4. Randomized Search (time-aware CV)")
    n_iter = int(cfg["n_iter"])
    cv_epochs = int(cfg["cv_epochs"])
    patience = int(cfg["patience"])
    batch_size = int(fixed["batch_size"])
    l2_reg = float(fixed["l2_reg"])
    dropout = float(fixed["dropout_rate"])

    candidates = sample_space(search_space, n_iter, seed=fixed["random_seed"])
    if not candidates:
        raise RuntimeError("RandomizedSearch produced zero candidates. Aborting.")

    best = None
    best_val = np.inf
    for i, cand in enumerate(candidates, 1):
        print(f"# Trial {i}/{len(candidates)}: {cand}")
        try:
            avg_val, info = time_cv_evaluate(train_df, folds, feature_cols, cand, fixed,
                                             epochs=cv_epochs, patience=patience, batch_size=batch_size,
                                             l2_reg=l2_reg, dropout=dropout)
        except Exception as e:
            print("!! Error during CV trial, aborting search.")
            traceback.print_exc()
            sys.exit(1)

        if avg_val < best_val:
            best_val = avg_val
            best = {"params": cand, "avg_val_loss": avg_val, "last_history": info["history"]}

    print(f"# Best average CV val_loss: {best_val:.6f} with params: {best['params']}")
    return best

# ---------------------- 6. Train Final Model on Full Train ----------------------
def train_final(train_df: pd.DataFrame, feature_cols: List[str], best_params: Dict, fixed: Dict, cfg: Dict):
    verbose_header("5. Final Training on full TRAIN period")
    X_tr, y_tr, _ = build_sequences(train_df, feature_cols, best_params["window"])

    model = build_lstm(
        input_dim=X_tr.shape[2],
        timesteps=X_tr.shape[1],
        layers_n=best_params["layers"],
        units=best_params["units"],
        lr=best_params["learning_rate"],
        l2_reg=fixed["l2_reg"],
        dropout_rate=fixed["dropout_rate"]
    )

    es = callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"], restore_best_weights=True)
    hist = model.fit(
        X_tr, y_tr,
        validation_split=0.1,
        epochs=cfg["final_model_epochs"],
        batch_size=fixed["batch_size"],
        verbose=1,
        callbacks=[es]
    )
    final_info = {
        "train_loss_min": min(hist.history.get("loss", [np.inf])),
        "val_loss_min": min(hist.history.get("val_loss", [np.inf])),
        "history": hist.history
    }
    return model, final_info

# ---------------------- 7. Prediction Helpers ----------------------
def predict_over_all_dates(model: keras.Model, df_scaled: pd.DataFrame, feature_cols: List[str], window: int) -> pd.Series:
    X_all, y_all, meta = build_sequences(df_scaled, feature_cols, window)
    preds = np.full((len(df_scaled),), np.nan, dtype=float)
    if len(meta) == 0:
        return pd.Series(preds, index=df_scaled.index, name="y_pred")
    yhat = model.predict(X_all, batch_size=1024, verbose=0).reshape(-1)
    for (id_, row_idx), val in zip(meta, yhat):
        preds[row_idx] = val
    return pd.Series(preds, index=df_scaled.index, name="y_pred")

# ---------------------- 8. Trading Simulation ----------------------
def simulate_trading(df: pd.DataFrame, preds: pd.Series, strategy: Dict) -> Dict:
    verbose_header("6. Trading Backtest (open D+1, PT/SL from entry open)")
    df = df.copy()
    df["y_pred"] = preds
    df.sort_values(["Date","ID"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    nL = int(strategy["n_long"])
    nS = int(strategy["n_short"])
    pt = float(strategy["pt_pct"])
    sl = float(strategy["sl_pct"])

    unique_dates = np.sort(df["Date"].unique())
    daily_ranks = {}
    for d in unique_dates:
        if len(g) == 0:
            continue
        g = g[np.isfinite(g["y_pred"])]; srt = g.sort_values("y_pred", ascending=False)
        longs = srt["ID"].head(nL).tolist()
        shorts = srt["ID"].tail(nS).tolist()
        daily_ranks[d] = {"longs": longs, "shorts": shorts}

    def process_position(pos, day_slice):
        entry_open = pos["entry_open"]
        pt_price = entry_open * (1 + pt) if pos["side"] == "long" else entry_open * (1 - pt)
        sl_price = entry_open * (1 - sl) if pos["side"] == "long" else entry_open * (1 + sl)
        for _, r in day_slice.iterrows():
            hi = r["HighAdj"]; lo = r["LowAdj"]; close = r["CloseAdj"]
            hit_pt = (hi >= pt_price) if pos["side"] == "long" else (lo <= pt_price)
            hit_sl = (lo <= sl_price) if pos["side"] == "long" else (hi >= sl_price)
            if hit_pt and hit_sl:
                hit_pt = False  # conservative: SL first
            if hit_pt:
                exit_price = pt_price; exit_date = r["Date"]
                ret = (exit_price - entry_open) / entry_open if pos["side"] == "long" else (entry_open - exit_price) / entry_open
                return exit_date, ret
            if hit_sl:
                exit_price = sl_price; exit_date = r["Date"]
                ret = (exit_price - entry_open) / entry_open if pos["side"] == "long" else (entry_open - exit_price) / entry_open
                return exit_date, ret
        r = day_slice.iloc[-1]
        exit_price = r["CloseAdj"]; exit_date = r["Date"]
        ret = (exit_price - entry_open) / entry_open if pos["side"] == "long" else (entry_open - exit_price) / entry_open
        return exit_date, ret

    date_to_row = {d: df[df["Date"] == d] for d in unique_dates}
    pnl_by_day = {d: 0.0 for d in unique_dates}

    for i, d in enumerate(unique_dates[:-1]):
        ranks = daily_ranks.get(d)
        next_day = unique_dates[i+1]
        if ranks:
            day_next_rows = date_to_row[next_day]
            for sid in ranks["longs"]:
                row = day_next_rows[day_next_rows["ID"] == sid]
                if len(row) == 0: continue
                entry_open = float(row["OpenAdj"].values[0])
                pos = {"id": sid, "entry_date": next_day, "entry_open": entry_open, "side": "long"}
                series_future = df[(df["ID"] == sid) & (df["Date"] >= next_day)]
                exit_date, ret = process_position(pos, series_future)
                pnl_by_day[exit_date] += ret
            for sid in ranks["shorts"]:
                row = day_next_rows[day_next_rows["ID"] == sid]
                if len(row) == 0: continue
                entry_open = float(row["OpenAdj"].values[0])
                pos = {"id": sid, "entry_date": next_day, "entry_open": entry_open, "side": "short"}
                series_future = df[(df["ID"] == sid) & (df["Date"] >= next_day)]
                exit_date, ret = process_position(pos, series_future)
                pnl_by_day[exit_date] += ret

    positions_per_day = (nL + nS)
    daily_pnl = []
    for d in unique_dates:
        daily_ret = pnl_by_day.get(d, 0.0) / max(positions_per_day, 1)
        daily_pnl.append(daily_ret)

    equity = np.cumprod(1 + np.array(daily_pnl))
    equity_curve = pd.DataFrame({"Date": unique_dates, "Strategy": equity}).set_index("Date")

    # Benchmark EW close-to-close
    bm_daily = []
    for d in unique_dates[:-1]:
        common = set(g_today["ID"]).intersection(set(g_next["ID"]))
        if not common:
            bm_daily.append(0.0); continue
        g_t = g_today[g_today["ID"].isin(common)].set_index("ID")
        g_n = g_next[g_next["ID"].isin(common)].set_index("ID")
        r = (g_n["CloseAdj"] / g_t["CloseAdj"] - 1.0).mean()
        bm_daily.append(float(r))
    bm_daily.append(0.0)
    bm_equity = np.cumprod(1 + np.array(bm_daily))
    equity_curve["Benchmark_SP100_EW"] = bm_equity

    return {
        "equity_curve": equity_curve,
        "daily_returns": np.array(daily_pnl),
        "bm_daily": np.array(bm_daily),
    }

# ---------------------- 9. Metrics ----------------------
def portfolio_metrics(equity_curve: pd.DataFrame, daily_ret: np.ndarray, bm_daily: np.ndarray, rf_annual: float) -> Dict:
    n_days = len(daily_ret)
    if n_days == 0:
        return {}
    total_return = float(equity_curve["Strategy"].iloc[-1] - 1.0)
    cagr = (equity_curve["Strategy"].iloc[-1])**(252/max(n_days,1)) - 1.0
    vol_ann = np.std(daily_ret, ddof=1) * np.sqrt(252)
    rf_daily = to_daily_rf(rf_annual)
    excess = daily_ret - rf_daily
    sharpe = float(np.mean(excess) / (np.std(excess, ddof=1) + 1e-12) * np.sqrt(252))
    curve = equity_curve["Strategy"].values
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    maxdd = float(np.min(dd))
    X = bm_daily.reshape(-1,1)
    y = daily_ret
    try:
        reg = LinearRegression().fit(X, y)
        beta = float(reg.coef_[0])
        alpha_daily = float(reg.intercept_)
        alpha_ann = (1 + alpha_daily)**252 - 1
    except Exception:
        beta = np.nan; alpha_ann = np.nan
    t_stat = float(np.mean(daily_ret) / (np.std(daily_ret, ddof=1) / np.sqrt(max(n_days,1) )))
    return {
        "TotalReturn": total_return,
        "CAGR": float(cagr),
        "VolAnn": float(vol_ann),
        "Sharpe": float(sharpe),
        "MaxDrawdown": float(maxdd),
        "AlphaAnn": float(alpha_ann),
        "Beta": float(beta),
        "T_stat": float(t_stat)
    }

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": float(mse), "MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}

# ---------------------- 10. Plot PNG Recap ----------------------
def make_report_png(out_dir: str, train_hist: Dict, equity_curve: pd.DataFrame, best_params: Dict,
                    reg_train: Dict, reg_test: Dict, port_metrics: Dict, train_end_date: str):
    verbose_header("7. Creating PNG report")
    safe_makedirs(out_dir)
    png_path = os.path.join(out_dir, "lstm_recap.png")

    plt.figure(figsize=(12, 9))

    plt.subplot(2,2,1)
    if train_hist:
        plt.plot(train_hist.get("loss", []), label="train")
        plt.plot(train_hist.get("val_loss", []), label="val")
        plt.title("Loss vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No training history", ha="center")

    plt.subplot(2,2,2)
    equity_curve[["Strategy","Benchmark_SP100_EW"]].plot(ax=plt.gca())
    plt.title("Cumulative Performance")
    plt.xlabel("Date"); plt.ylabel("Equity (normalized)")
    xline = pd.to_datetime(train_end_date)
    plt.axvline(x=xline, linestyle="--")
    plt.legend()

    plt.subplot(2,2,3)
    cell_text = [[k, str(v)] for k, v in best_params.items()]
    table = plt.table(cellText=cell_text, colLabels=["Hyperparam", "Value"], loc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.5)
    plt.axis("off")
    plt.title("Best Hyperparameters")

    plt.subplot(2,2,4)
    metrics_pairs = [
        ("Train MSE", reg_train["MSE"]),
        ("Train MAE", reg_train["MAE"]),
        ("Train RMSE", reg_train["RMSE"]),
        ("Train R2", reg_train["R2"]),
        ("Test MSE", reg_test["MSE"]),
        ("Test MAE", reg_test["MAE"]),
        ("Test RMSE", reg_test["RMSE"]),
        ("Test R2", reg_test["R2"]),
        ("CAGR", port_metrics.get("CAGR", np.nan)),
        ("VolAnn", port_metrics.get("VolAnn", np.nan)),
        ("Sharpe", port_metrics.get("Sharpe", np.nan)),
        ("MaxDD", port_metrics.get("MaxDrawdown", np.nan)),
        ("AlphaAnn", port_metrics.get("AlphaAnn", np.nan)),
        ("AlphaPValue", port_metrics.get("AlphaPValue", np.nan)),
        ("Beta", port_metrics.get("Beta", np.nan)),
        ("T-stat", port_metrics.get("T_stat", np.nan)),
    ]
    cell_text = [[k, f"{v:.4f}"] for k, v in metrics_pairs]
    table = plt.table(cellText=cell_text, colLabels=["Metric", "Value"], loc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.5)
    plt.axis("off"); plt.title("Metrics")

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(f"# Saved report to: {png_path}")
    return png_path

# ---------------------- 11. Grading with Emoji ----------------------
def grade_strategy(port_metrics: Dict) -> Dict:
    cagr = port_metrics.get("CAGR", 0.0)
    sharpe = port_metrics.get("Sharpe", 0.0)
    maxdd = abs(port_metrics.get("MaxDrawdown", 0.0))
    score = 0.0
    score += max(0.0, min(10.0, cagr*100)) * 0.2
    score += max(0.0, min(10.0, sharpe*2)) * 0.5
    score += max(0.0, min(10.0, (1.0-maxdd)*10)) * 0.3
    if score >= 7:
        grade = "Výborný"; emoji = "🌟"
    elif score >= 5:
        grade = "Dobrý"; emoji = "✅"
    elif score >= 3:
        grade = "Průměrný"; emoji = "⚠️"
    else:
        grade = "Slabý"; emoji = "❌"
    return {"score": float(score), "grade": grade, "emoji": emoji}

# ---------------------- 12. Main ----------------------
def main():
    t0 = time.time()
    set_global_seed(HYPERPARAMS["fixed_params"]["random_seed"])

    out_dir = os.path.abspath("./lstm_outputs")
    safe_makedirs(out_dir)

    verbose_header("CONFIGURATION PARAMETERS")
    print(json.dumps(HYPERPARAMS, indent=2, ensure_ascii=False))

    df_raw = load_and_merge(HYPERPARAMS["data"]["data_path"], HYPERPARAMS["data"]["vix_path"])
    df_feat, feature_cols = compute_features(df_raw)

    df_feat = df_feat.dropna(subset=["target_next_ret"] + feature_cols).reset_index(drop=True)

    splits = make_time_splits(df_feat, HYPERPARAMS["data"]["train_end_date"], HYPERPARAMS["data"]["test_start_date"], n_folds=HYPERPARAMS["cv_folds"])
    train_df, test_df, folds = splits["train_df"], splits["test_df"], splits["folds"]

    scalers = per_id_train_scalers(train_df, feature_cols)
    train_scaled = apply_scalers(train_df, feature_cols, scalers)
    test_scaled  = apply_scalers(test_df, feature_cols, scalers)

    df_all_scaled = pd.concat([train_scaled, test_scaled], axis=0).sort_values(["Date","ID"]).reset_index(drop=True)
    df_all_scaled = make_target_zscore(train_scaled, df_all_scaled)

    train_scaled = df_all_scaled[df_all_scaled.index.isin(train_scaled.index)].copy()
    test_scaled  = df_all_scaled[df_all_scaled.index.isin(test_scaled.index)].copy()

    # Random search
    best = random_search(train_scaled, folds, feature_cols, HYPERPARAMS["search_space"],
                         HYPERPARAMS["fixed_params"], HYPERPARAMS)
    best_params = best["params"]

    # Final train
    model, final_info = train_final(train_scaled, feature_cols, best_params, HYPERPARAMS["fixed_params"], HYPERPARAMS)

    # Regression diagnostics
    X_tr, y_tr, _ = build_sequences(train_scaled, feature_cols, best_params["window"])
    X_te, y_te, _ = build_sequences(test_scaled, feature_cols, best_params["window"])
    yhat_tr = model.predict(X_tr, batch_size=1024, verbose=0).reshape(-1) if len(y_tr)>0 else np.array([])
    yhat_te = model.predict(X_te, batch_size=1024, verbose=0).reshape(-1) if len(y_te)>0 else np.array([])

    reg_train = regression_metrics(y_tr, yhat_tr) if len(y_tr)>0 else {"MSE":np.nan,"MAE":np.nan,"RMSE":np.nan,"R2":np.nan}
    reg_test  = regression_metrics(y_te, yhat_te) if len(y_te)>0 else {"MSE":np.nan,"MAE":np.nan,"RMSE":np.nan,"R2":np.nan}

    # Trading simulation over full timeline
    preds_all = predict_over_all_dates(model, df_all_scaled, feature_cols, best_params["window"])
    trade_res = simulate_trading(df_all_scaled, preds_all, HYPERPARAMS["strategy"])

    port_m = portfolio_metrics(trade_res["equity_curve"], trade_res["daily_returns"], trade_res["bm_daily"], HYPERPARAMS["strategy"]["rf_annual"])

    png_path = make_report_png(out_dir, final_info["history"], trade_res["equity_curve"], best_params, reg_train, reg_test, port_m, HYPERPARAMS["data"]["train_end_date"])

    grade = grade_strategy(port_m)

    summary = {
        "best_params": best_params,
        "regression": {"train": reg_train, "test": reg_test},
        "portfolio": port_m,
        "grade": grade,
        "report_png": png_path
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    m, s = divmod(elapsed, 60)
    print("\n" + "-"*60)
    print(f"TOTAL ELAPSED TIME: {int(m)} min {int(s)} sec")
    print(f"Outputs saved in: {out_dir}")
    print("-"*60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFATAL ERROR:")
        traceback.print_exc()
        sys.exit(1)