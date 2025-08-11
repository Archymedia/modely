
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        'data_path': ["/Users/lindawaisova/Desktop/DP/data/SP_100/READY_DATA/9DATA_FINAL.csv",
                      r"C:\Users\david\Desktop\SP100\9DATA_FINAL.csv"],
        'vix_path':  ["/Users/lindawaisova/Desktop/DP/data/SP_100/READY_DATA/VIX_2005_2023.csv",
                      r"C:\Users\david\Desktop\SP100\VIX_2005_2023.csv"],
        'train_end_date': '2020-12-31',
        'test_start_date': '2021-01-01',
    },

    # Training configuration
    'final_model_epochs': 2,        # Počet epoch pro finální trénink
    'cv_epochs': 2,                  # Počet epoch pro cross-validation
    'patience': 1,
    'cv_folds': 2,
    'n_iter': 1,

    # HYPERPARAMETER SEARCH SPACE (MLP)
    'search_space': {
        'layers': [1, 2],
        'units': [64, 128, 256],
        'learning_rate': [1e-4, 3e-4, 1e-3]
    },

    # FIXNÍ PARAMETRY (netunované)
    'fixed_params': {
        'batch_size': 128,
        'dropout_rate': 0.15,
        'l2_reg': 3e-4,
        'random_seed': 42,
    },

    # STRATEGY
    'strategy': {
        'n_long': 10,
        'n_short': 10,
        'pt_pct': 0.02,   # 2%
        'sl_pct': 0.02,   # 2%
        'rf_annual': 0.02, # 2% p.a.
        'fast_backtest': True  # zapnout rychlou vektorizovanou simulaci
    }
}

# ---------------------- Utilities & reproducibility ----------------------
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

def resolve_first_existing(p):
    if isinstance(p, (list, tuple)):
        for cand in p:
            if isinstance(cand, str) and os.path.exists(cand):
                return cand
        return p[0]
    return p

# ---------------------- 1. Data Loading and Preprocessing ----------------------
def load_and_merge(data_path, vix_path) -> pd.DataFrame:
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

def data_self_check(df: pd.DataFrame):
    verbose_header("DATA SELF-CHECK")
    cols = df.columns.tolist()
    print(f"# Columns ({len(cols)}): {cols}")
    n_ids = df["ID"].nunique() if "ID" in df.columns else 0
    date_min = df["Date"].min() if "Date" in df.columns else None
    date_max = df["Date"].max() if "Date" in df.columns else None
    print(f"# Rows: {len(df):,} | Unique IDs: {n_ids:,} | Date range: {date_min} .. {date_max}")
    na = df.isna().mean().sort_values(ascending=False)
    print("# Top-10 NA columns (share):")
    print(na.head(10))

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

# ---------------------- 2. Time Splits & Scaling ----------------------
def make_time_splits(df: pd.DataFrame, train_end: str, test_start: str, n_folds: int = 3) -> Dict:
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
    from sklearn.preprocessing import StandardScaler
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

def apply_scalers(df: pd.DataFrame, feature_cols: List[str], scalers: Dict[str, 'StandardScaler']) -> pd.DataFrame:
    print("# Applying per-ID scalers to a dataframe ...")
    df = df.copy()
    outs = []
    for id_, g in df.groupby("ID"):
        X = g[feature_cols].values
        sc = scalers.get(id_)
        if sc is None:
            from sklearn.preprocessing import StandardScaler
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

# ---------------------- Self-checks after features ----------------------
def features_self_check(df_feat: pd.DataFrame, feature_cols: List[str]):
    verbose_header("FEATURES SELF-CHECK")
    na_share = df_feat[feature_cols + ["target_next_ret"]].isna().mean().sort_values(ascending=False)
    print("# Top-15 NA share (features + target_next_ret):")
    print(na_share.head(15))
    ids = df_feat["ID"].nunique()
    dmin, dmax = df_feat["Date"].min(), df_feat["Date"].max()
    print(f"# Feature rows: {len(df_feat):,} | Feature cols: {len(feature_cols)} | IDs: {ids} | Dates: {dmin}..{dmax}")

# ---------------------- 3. Model (MLP) ----------------------
def build_mlp(input_dim: int, layers_n: int, units: int, lr: float, l2_reg: float, dropout_rate: float) -> keras.Model:
    reg = regularizers.l2(l2_reg)
    model = keras.Sequential(name="MLP_regressor")
    model.add(layers.Input(shape=(input_dim,)))
    for _ in range(layers_n):
        model.add(layers.Dense(units, activation="relu", kernel_regularizer=reg))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation="linear"))
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse")
    return model

# ---------------------- 4. Random Search with Time CV ----------------------
def sample_space(space: Dict[str, List], n_iter: int, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    keys = list(space.keys())
    configs = []
    for _ in range(n_iter):
        conf = {k: rng.choice(space[k]) for k in keys}
        configs.append(conf)
    return configs

def time_cv_evaluate_tabular(train_df: pd.DataFrame, folds, feature_cols, cfg, fixed, epochs, patience, batch_size, l2_reg, dropout):
    val_losses = []
    last_history = None

    for fi, (tr_idx, vl_idx) in enumerate(folds, 1):
        tr = train_df.loc[tr_idx]
        vl = train_df.loc[vl_idx]

        X_tr = tr[feature_cols].values
        y_tr = tr["target_z"].values
        X_vl = vl[feature_cols].values
        y_vl = vl["target_z"].values

        model = build_mlp(
            input_dim=X_tr.shape[1],
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
        raise RuntimeError("No valid folds; aborting search.")

    avg_val = float(np.mean(val_losses))
    return avg_val, {"history": last_history, "config": cfg}

def random_search_tabular(train_df, folds, feature_cols, search_space, fixed, cfg):
    verbose_header("4. Randomized Search (time-aware CV)")
    candidates = sample_space(search_space, int(cfg["n_iter"]), fixed["random_seed"])
    best = None; best_val = np.inf
    for i, cand in enumerate(candidates, 1):
        print(f"# Trial {i}/{len(candidates)}: {cand}")
        try:
            avg_val, info = time_cv_evaluate_tabular(
                train_df, folds, feature_cols, cand, fixed,
                epochs=cfg["cv_epochs"], patience=cfg["patience"],
                batch_size=fixed["batch_size"], l2_reg=fixed["l2_reg"], dropout=fixed["dropout_rate"]
            )
        except Exception as e:
            print("!! Error during CV trial, aborting search.")
            traceback.print_exc()
            sys.exit(1)
        if avg_val < best_val:
            best_val = avg_val
            best = {"params": cand, "avg_val_loss": avg_val, "last_history": info["history"]}
    print(f"# Best average CV val_loss: {best_val:.6f} with params: {best['params']}")
    return best

# ---------------------- 6. Trading Simulation (Fast NumPy) ----------------------
def simulate_trading_fast(df: pd.DataFrame, preds: pd.Series, strategy: Dict) -> Dict:
    verbose_header("6. Trading Backtest (FAST NumPy — open D+1, PT/SL from entry open)")
    t0 = time.time()

    df = df.copy()
    df["y_pred"] = preds
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["Date","ID"], inplace=True, ignore_index=True)

    nL = int(strategy["n_long"]); nS = int(strategy["n_short"])
    pt = float(strategy["pt_pct"]); sl = float(strategy["sl_pct"])

    dates = np.array(sorted(df["Date"].unique()))
    ids = np.array(sorted(df["ID"].unique()))
    T = len(dates); N = len(ids)
    print(f"# Universe: {N} tickers × {T} trading days")

    # mapy na indexy
    date_to_idx = {d:i for i,d in enumerate(dates)}
    id_to_idx = {sid:i for i,sid in enumerate(ids)}

    # 2D matice (T, N) plněné NaN
    OPEN = np.full((T,N), np.nan, dtype=float)
    HIGH = np.full((T,N), np.nan, dtype=float)
    LOW  = np.full((T,N), np.nan, dtype=float)
    CLOSE= np.full((T,N), np.nan, dtype=float)
    PRED = np.full((T,N), np.nan, dtype=float)

    # Jednorázové plnění matic
    for _, r in df.iterrows():
        di = date_to_idx[r["Date"]]
        ii = id_to_idx[r["ID"]]
        OPEN[di, ii]  = r["OpenAdj"]
        HIGH[di, ii]  = r["HighAdj"]
        LOW[di, ii]   = r["LowAdj"]
        CLOSE[di, ii] = r["CloseAdj"]
        PRED[di, ii]  = r["y_pred"]

    pnl_by_day = np.zeros(T, dtype=float)

    # Pomocné funkce pro výběr top/bottom bez plného sortu
    def pick_top_k(values, k):
        # values: 1D s NaN -> vybírá největší k na finite mask
        mask = np.isfinite(values)
        idx = np.where(mask)[0]
        if idx.size == 0:
            return np.array([], dtype=int)
        vals = values[idx]
        k_eff = min(k, idx.size)
        part = np.argpartition(vals, -k_eff)[-k_eff:]
        top_local = idx[part]
        # seřadit top lokální dle hodnot sestupně
        order = np.argsort(values[top_local])[::-1]
        return top_local[order]

    def pick_bottom_k(values, k):
        mask = np.isfinite(values)
        idx = np.where(mask)[0]
        if idx.size == 0:
            return np.array([], dtype=int)
        vals = values[idx]
        k_eff = min(k, idx.size)
        part = np.argpartition(vals, k_eff-1)[:k_eff]
        bottom_local = idx[part]
        order = np.argsort(values[bottom_local])  # vzestupně
        return bottom_local[order]

    for d in range(T-1):
        if (d+1) % 100 == 0:
            elapsed = time.time() - t0
            mm, ss = divmod(int(elapsed), 60)
            print(f"[Trading Progress] Day {d+1}/{T} ({pd.Timestamp(dates[d]).date()}) — elapsed {mm:02d}:{ss:02d}")

        preds_today = PRED[d, :]
        longs_idx  = pick_top_k(preds_today, nL)
        shorts_idx = pick_bottom_k(preds_today, nS)

        next_d = d + 1
        # LONGS
        for ii in longs_idx:
            entry_open = OPEN[next_d, ii]
            if not np.isfinite(entry_open):
                continue
            pt_price = entry_open * (1 + pt)
            sl_price = entry_open * (1 - sl)

            highs = HIGH[next_d:, ii]
            lows  = LOW[next_d:, ii]
            closes= CLOSE[next_d:, ii]

            # první index zásahu bariér
            hit_pt_idx = np.where(highs >= pt_price)[0]
            hit_sl_idx = np.where(lows  <= sl_price)[0]

            # konzervativně: když obě v ten samý den -> bereme SL
            if hit_sl_idx.size == 0 and hit_pt_idx.size == 0:
                exit_rel = len(highs) - 1
                exit_price = closes[exit_rel]
                exit_day = next_d + exit_rel
            else:
                first_pt = hit_pt_idx[0] if hit_pt_idx.size else np.inf
                first_sl = hit_sl_idx[0] if hit_sl_idx.size else np.inf
                if first_sl < first_pt:
                    exit_rel = int(first_sl)
                    exit_price = sl_price
                elif first_pt < first_sl:
                    exit_rel = int(first_pt)
                    exit_price = pt_price
                else:
                    # stejné (vzácně) -> SL priorita
                    exit_rel = int(first_sl) if np.isfinite(first_sl) else int(first_pt)
                    exit_price = sl_price if np.isfinite(first_sl) else pt_price
                exit_day = next_d + exit_rel

            if np.isfinite(exit_price):
                ret = (exit_price - entry_open) / entry_open
                pnl_by_day[exit_day] += ret

        # SHORTS
        for ii in shorts_idx:
            entry_open = OPEN[next_d, ii]
            if not np.isfinite(entry_open):
                continue
            pt_price = entry_open * (1 - pt)  # pro short: zisk, když cena klesne
            sl_price = entry_open * (1 + sl)

            highs = HIGH[next_d:, ii]
            lows  = LOW[next_d:, ii]
            closes= CLOSE[next_d:, ii]

            hit_pt_idx = np.where(lows <= pt_price)[0]
            hit_sl_idx = np.where(highs >= sl_price)[0]

            if hit_sl_idx.size == 0 and hit_pt_idx.size == 0:
                exit_rel = len(highs) - 1
                exit_price = closes[exit_rel]
                exit_day = next_d + exit_rel
            else:
                first_pt = hit_pt_idx[0] if hit_pt_idx.size else np.inf
                first_sl = hit_sl_idx[0] if hit_sl_idx.size else np.inf
                if first_sl < first_pt:
                    exit_rel = int(first_sl)
                    exit_price = sl_price
                elif first_pt < first_sl:
                    exit_rel = int(first_pt)
                    exit_price = pt_price
                else:
                    exit_rel = int(first_sl) if np.isfinite(first_sl) else int(first_pt)
                    exit_price = sl_price if np.isfinite(first_sl) else pt_price
                exit_day = next_d + exit_rel

            if np.isfinite(exit_price):
                ret = (entry_open - exit_price) / entry_open  # short PnL
                pnl_by_day[exit_day] += ret

    positions_per_day = (nL + nS)
    daily_returns = pnl_by_day / max(positions_per_day, 1)

    # equity curve
    equity = np.cumprod(1 + daily_returns)
    equity_curve = pd.DataFrame({"Date": dates, "Strategy": equity}).set_index("Date")

    # Benchmark EW close-to-close (SP100 universe from data) — vektorizovaně
    bm_daily = np.zeros(T, dtype=float)
    for d in range(T-1):
        c0 = CLOSE[d, :]
        c1 = CLOSE[d+1, :]
        mask = np.isfinite(c0) & np.isfinite(c1)
        if not np.any(mask):
            bm_daily[d] = 0.0
        else:
            bm_daily[d] = np.mean(c1[mask] / c0[mask] - 1.0)
    bm_equity = np.cumprod(1 + bm_daily)
    equity_curve["Benchmark_SP100_EW"] = bm_equity

    total_elapsed = time.time() - t0
    mm, ss = divmod(int(total_elapsed), 60)
    print(f"[Trading Progress] FAST backtest done in {mm:02d}:{ss:02d}")

    return {
        "equity_curve": equity_curve,
        "daily_returns": daily_returns,
        "bm_daily": bm_daily,
    }

# ---------------------- 7. Metrics & Report ----------------------
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

    # OLS alpha/beta vs benchmark + p-value for alpha (normal approx)
    X = bm_daily.reshape(-1,1)
    y = daily_ret
    alpha_ann = np.nan
    beta = np.nan
    alpha_p = np.nan
    try:
        X1 = np.column_stack([np.ones(len(X)), X])
        XtX = X1.T @ X1
        XtX_inv = np.linalg.pinv(XtX)
        b = XtX_inv @ (X1.T @ y)
        alpha = float(b[0]); beta = float(b[1])
        y_hat = X1 @ b
        resid = y - y_hat
        dof = max(len(y) - X1.shape[1], 1)
        sigma2 = float((resid @ resid) / dof)
        se_alpha = math.sqrt(max((XtX_inv * sigma2)[0,0], 1e-12))
        from math import erf, sqrt
        t_alpha = alpha / se_alpha
        p_two = 2.0 * (1.0 - 0.5*(1.0 + erf(abs(t_alpha)/sqrt(2.0))))
        alpha_p = float(p_two)
        alpha_ann = (1 + alpha)**252 - 1
    except Exception:
        pass

    t_stat = float(np.mean(daily_ret) / (np.std(daily_ret, ddof=1) / np.sqrt(max(n_days,1) )))
    return {
        "TotalReturn": total_return,
        "CAGR": float(cagr),
        "VolAnn": float(vol_ann),
        "Sharpe": float(sharpe),
        "MaxDrawdown": float(maxdd),
        "AlphaAnn": float(alpha_ann),
        "AlphaPValue": float(alpha_p),
        "Beta": float(beta),
        "T_stat": float(t_stat)
    }

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": float(mse), "MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}

def make_report_png(out_dir: str, train_hist: Dict, equity_curve: pd.DataFrame, best_params: Dict,
                    reg_train: Dict, reg_test: Dict, port_metrics: Dict, train_end_date: str, model_tag: str):
    verbose_header("7. Creating PNG report")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{model_tag}_recap.png")

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

# ---------------------- 5. Final Training (same as before) ----------------------
def train_final_tabular(train_df, feature_cols, best_params, fixed, cfg):
    verbose_header("5. Final Training on full TRAIN period")
    X_tr = train_df[feature_cols].values
    y_tr = train_df["target_z"].values

    model = build_mlp(
        input_dim=X_tr.shape[1],
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
    return model, {"history": hist.history}

# ---------------------- 8. Main ----------------------
def main():
    t0 = time.time()
    set_global_seed(HYPERPARAMS["fixed_params"]["random_seed"])

    out_dir = os.path.abspath("./mlp_outputs")
    os.makedirs(out_dir, exist_ok=True)

    verbose_header("CONFIGURATION PARAMETERS")
    print(json.dumps(HYPERPARAMS, indent=2, ensure_ascii=False))

    # Load + self-check
    df_raw = load_and_merge(HYPERPARAMS["data"]["data_path"], HYPERPARAMS["data"]["vix_path"])

    # Features + self-check
    df_feat, feature_cols = compute_features(df_raw)

    # Drop NaN rows for modeling
    before = len(df_feat)
    df_feat = df_feat.dropna(subset=["target_next_ret"] + feature_cols).reset_index(drop=True)
    after = len(df_feat)
    print(f"# Dropped rows due to NaN: {before-after:,} ({(before-after)/max(before,1):.2%})")
    print(f"# Remaining rows: {after:,} | Features: {len(feature_cols)}")

    # Splits
    train_end = HYPERPARAMS["data"]["train_end_date"]
    test_start = HYPERPARAMS["data"]["test_start_date"]
    train_end = pd.to_datetime(train_end); test_start = pd.to_datetime(test_start)

    train_df = df_feat[df_feat["Date"] <= train_end].copy()
    test_df  = df_feat[df_feat["Date"] >= test_start].copy()

    # Scalers
    scalers = per_id_train_scalers(train_df, feature_cols)
    train_scaled = apply_scalers(train_df, feature_cols, scalers)
    test_scaled  = apply_scalers(test_df, feature_cols, scalers)

    df_all_scaled = pd.concat([train_scaled, test_scaled], axis=0).sort_values(["Date","ID"]).reset_index(drop=True)
    df_all_scaled = make_target_zscore(train_scaled, df_all_scaled)

    train_scaled = df_all_scaled[df_all_scaled.index.isin(train_scaled.index)].copy()
    test_scaled  = df_all_scaled[df_all_scaled.index.isin(test_scaled.index)].copy()

    # Simple CV (time-aware) for hyperparam search
    dates = np.sort(train_scaled["Date"].unique())
    n_folds = HYPERPARAMS["cv_folds"]
    folds = []
    chunk_edges = np.linspace(0, len(dates), n_folds+1, dtype=int)
    for i in range(1, len(chunk_edges)):
        val_start_idx = max(chunk_edges[i-1], 1)
        val_end_idx   = chunk_edges[i]
        val_start = dates[val_start_idx-1]
        val_end   = dates[val_end_idx-1]
        tr_idx = train_scaled["Date"] < val_start
        vl_idx = (train_scaled["Date"] >= val_start) & (train_scaled["Date"] <= val_end)
        if tr_idx.sum() == 0 or vl_idx.sum() == 0:
            continue
        folds.append((train_scaled.index[tr_idx].values, train_scaled.index[vl_idx].values))

    # Search & Train
    best = random_search_tabular(train_scaled, folds, feature_cols, HYPERPARAMS["search_space"], HYPERPARAMS["fixed_params"], HYPERPARAMS)
    best_params = best["params"]
    model, final_info = train_final_tabular(train_scaled, feature_cols, best_params, HYPERPARAMS["fixed_params"], HYPERPARAMS)

    # Regression diagnostics (tabular)
    X_tr = train_scaled[feature_cols].values; y_tr = train_scaled["target_z"].values
    X_te = test_scaled[feature_cols].values;  y_te = test_scaled["target_z"].values
    yhat_tr = model.predict(X_tr, batch_size=1024, verbose=0).reshape(-1) if len(y_tr)>0 else np.array([])
    yhat_te = model.predict(X_te, batch_size=1024, verbose=0).reshape(-1) if len(y_te)>0 else np.array([])

    reg_train = regression_metrics(y_tr, yhat_tr) if len(y_tr)>0 else {"MSE":np.nan,"MAE":np.nan,"RMSE":np.nan,"R2":np.nan}
    reg_test  = regression_metrics(y_te, yhat_te) if len(y_te)>0 else {"MSE":np.nan,"MAE":np.nan,"RMSE":np.nan,"R2":np.nan}

    # Trading over full timeline (FAST or SLOW)
    preds_all = pd.Series(np.nan, index=df_all_scaled.index)
    preds_all.loc[train_scaled.index] = model.predict(train_scaled[feature_cols].values, batch_size=1024, verbose=0).reshape(-1)
    preds_all.loc[test_scaled.index]  = model.predict(test_scaled[feature_cols].values,  batch_size=1024, verbose=0).reshape(-1)

    if HYPERPARAMS["strategy"].get("fast_backtest", True):
        trade_res = simulate_trading_fast(df_all_scaled, preds_all, HYPERPARAMS["strategy"])
    else:
        raise NotImplementedError("Slow backtest path not included in this fast build. Set fast_backtest=True.")

    # Portfolio metrics & report
    port_m = portfolio_metrics(trade_res["equity_curve"], trade_res["daily_returns"], trade_res["bm_daily"], HYPERPARAMS["strategy"]["rf_annual"])
    png_path = make_report_png(out_dir, final_info["history"], trade_res["equity_curve"], best_params, reg_train, reg_test, port_m, HYPERPARAMS["data"]["train_end_date"], model_tag="mlp_fast")

    # Save summary
    summary = {
        "best_params": best_params,
        "regression": {"train": reg_train, "test": reg_test},
        "portfolio": port_m,
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
