#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN_full6.py
Vanilla SimpleRNN for next-day Z-score simple return prediction and trading strategy backtest.

Updates in this version:
- Cross-validation uses TimeSeriesSplit (respects temporal order).
- Strategy simulator records trade-level data (entry/exit, return, holding days, reason TP/SL).
- Added trade-level metrics and included them in the PNG summary table and terminal output.
- SciKeras param grid uses 'model__' prefix (prevents "invalid parameter units" error).
- CV training now shows epoch progress bars (fit__verbose=2 / verbose=2).
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

try:
    # Prefer scikeras if available
    from scikeras.wrappers import KerasRegressor
    USE_SCIKERAS = True
except Exception:
    from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
    USE_SCIKERAS = False


# ===== HYPERPARAMETER CONFIGURATION =====
HYPERPARAMS = {
    # DATA & SPLIT
    'data': {
        'data_path': 'C:\\Users\\david\\Desktop\\SP100\\9DATA_FINAL.csv',
        'vix_path': 'C:\\Users\\david\\Desktop\\SP100\\VIX_2005_2023.csv',
        'train_end_date': '2020-12-31',
        'test_start_date': '2021-01-01',
    },

    # Training configuration
    'final_model_epochs': 100,        # Počet epoch pro finální trénink
    'cv_epochs': 20,                  # Počet epoch pro cross-validation (nižší kvůli rychlosti)
    'patience': 10,
    'cv_folds': 3,
    'n_iter': 20,

    # HYPERPARAMETER SEARCH SPACE
    'search_space': {
        'rnn_layers': [1, 2, 3],
        'neurons_per_layer': [32, 64, 128],
        'learning_rate': [0.0001, 0.001, 0.01]
        # optionally: 'l2_reg': [0.0, 1e-4, 1e-3], 'dropout_rate': [0.0, 0.1, 0.2, 0.3]
    },

    # FIXNÍ PARAMETRY (netunované)
    'fixed_params': {
        'batch_size': 128,
        'dropout_rate': 0.15,
        'l2_reg': 0.0003,
        'timesteps': 10, # Počet časových kroků pro RNN
        'random_seed': 42,
    },

    # STRATEGY
    'strategy': {
        'n_long': 10,
        'n_short': 10,
        'pt_pct': 0.02,   # 2%
        'sl_pct': 0.02,   # 2%
        'rf_annual': 0.02 # 2% p.a.
    }
}

# --- Convenience getters (keep script readable) ---
DSET = HYPERPARAMS['data']
FIX  = HYPERPARAMS['fixed_params']
STR  = HYPERPARAMS['strategy']

np.random.seed(FIX['random_seed'])
tf.random.set_seed(FIX['random_seed'])

print("="*80)
print("RNN (SimpleRNN) pipeline starting...")
print("Hyperparameters / Settings:")
for section, content in HYPERPARAMS.items():
    if isinstance(content, dict):
        print(f"  [{section}]")
        for k, v in content.items():
            print(f"    {k}: {v}")
    else:
        print(f"  {section}: {content}")
print("="*80)

# =========================
# DATA LOADING
# =========================
def load_data():
    if not os.path.exists(DSET['data_path']):
        raise FileNotFoundError(f"Data file not found: {DSET['data_path']}")
    if not os.path.exists(DSET['vix_path']):
        raise FileNotFoundError(f"VIX file not found: {DSET['vix_path']}")
    df = pd.read_csv(DSET['data_path'])
    vix = pd.read_csv(DSET['vix_path'])

    # Ensure datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    if 'Date' in vix.columns:
        vix['Date'] = pd.to_datetime(vix['Date'])

    # Merge VIX on Date if present
    if 'Date' in df.columns and 'Date' in vix.columns and 'VIX' in vix.columns:
        df = df.merge(vix[['Date','VIX']], on='Date', how='left')

    return df

# =========================
# FEATURE ENGINEERING (look-ahead safe)
# =========================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[INFO] Computing technical indicators (look-ahead safe)...")
    df = df.sort_values(['ID','Date']).reset_index(drop=True)
    out = []

    required_cols = ['CloseAdj','HighAdj','LowAdj','OpenAdj','Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data.")

    for sid, g in df.groupby('ID', sort=False):
        s = g.copy().sort_values('Date').reset_index(drop=True)

        # Basic lags
        if 'SimpleReturn' in s.columns:
            for lag in [1,2,3,5,10]:
                s[f'SimpleReturn_lag_{lag}'] = s['SimpleReturn'].shift(lag)

        # SMA / EMA, ratios
        close_hist = s['CloseAdj'].shift(1)
        for p in [5,10,20]:
            sma = close_hist.rolling(p).mean()
            ema = close_hist.ewm(span=p, adjust=False).mean()
            s[f'SMA_{p}'] = sma
            s[f'EMA_{p}'] = ema
            s[f'Price_SMA_{p}_ratio'] = close_hist / sma
            s[f'Price_EMA_{p}_ratio'] = close_hist / ema

        # MACD
        ema12 = close_hist.ewm(span=12, adjust=False).mean()
        ema26 = close_hist.ewm(span=26, adjust=False).mean()
        s['MACD'] = ema12 - ema26
        s['MACD_signal'] = s['MACD'].ewm(span=9, adjust=False).mean()
        s['MACD_hist'] = s['MACD'] - s['MACD_signal']

        # RSI
        for p in [7,14]:
            delta = close_hist.diff()
            gain = delta.clip(lower=0).rolling(p).mean()
            loss = (-delta.clip(upper=0)).rolling(p).mean()
            rs = gain / loss
            s[f'RSI_{p}'] = 100 - (100/(1+rs))

        # Bollinger
        sma20 = close_hist.rolling(20).mean()
        std20 = close_hist.rolling(20).std()
        s['BB_upper'] = sma20 + 2*std20
        s['BB_lower'] = sma20 - 2*std20
        s['BB_pos'] = (close_hist - s['BB_lower']) / (s['BB_upper'] - s['BB_lower'])

        # ATR (14)
        high_hist = s['HighAdj'].shift(1)
        low_hist  = s['LowAdj'].shift(1)
        prev_close = s['CloseAdj'].shift(1)
        tr = np.maximum(high_hist-low_hist, np.maximum((high_hist-prev_close).abs(), (low_hist-prev_close).abs()))
        s['ATR_14'] = tr.rolling(14).mean()

        # HV
        if 'SimpleReturn' in s.columns:
            for p in [10,20]:
                s[f'HV_{p}'] = s['SimpleReturn'].shift(1).rolling(p).std() * np.sqrt(252)

        # OBV
        if 'SimpleReturn' in s.columns:
            s['OBV'] = (s['Volume'].shift(1) * np.sign(s['SimpleReturn'].shift(1))).cumsum()

        # VIX derived
        if 'VIX' in s.columns:
            s['VIX_SMA_5'] = s['VIX'].shift(1).rolling(5).mean()
            s['VIX_change'] = s['VIX'].shift(1).pct_change()

        out.append(s)

    return pd.concat(out, axis=0).reset_index(drop=True)

# =========================
# TARGET: Z-score of next-day SimpleReturn (no leakage)
# =========================
def add_target_zscore(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Preparing target: next-day Z-score of SimpleReturn (no leakage)...")
    df = df.sort_values(['ID','Date']).reset_index(drop=True)
    df['target_raw'] = df.groupby('ID')['SimpleReturn'].shift(-1)

    # Train/test split by date
    train_mask = df['Date'] <= pd.to_datetime(DSET['train_end_date'])

    # Compute per-ID mean/std on TRAIN ONLY
    mu = df.loc[train_mask].groupby('ID')['SimpleReturn'].mean()
    sd = df.loc[train_mask].groupby('ID')['SimpleReturn'].std().replace(0, np.nan)

    # Map to all rows using train stats
    df['SR_mu_train'] = df['ID'].map(mu)
    df['SR_sd_train'] = df['ID'].map(sd).fillna(1.0)

    # Z-score for next-day return (target_raw)
    df['target'] = (df['target_raw'] - df['SR_mu_train']) / df['SR_sd_train']
    return df

# =========================
# TRAIN/TEST SPLIT + FEATURE SELECTION
# =========================
def build_feature_set(df: pd.DataFrame):
    print("[INFO] Selecting features (matching MLP feature philosophy)...")
    exclude = {
        'ID','RIC','Name','Date','target','target_raw','SR_mu_train','SR_sd_train',
        'TotRet','SimpleReturn','Close','Volume','Open','High','Low',
        'CloseAdj','OpenAdj','HighAdj','LowAdj','VIX'
    }
    feature_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype.kind not in 'fi':
            continue
        if any(k in c for k in ['_lag','SMA_','EMA_','MACD','RSI_','BB_','ATR_','HV_','OBV','VIX_','Price_']):
            feature_cols.append(c)
    print(f"[INFO] Selected {len(feature_cols)} features.")
    return feature_cols

def split_train_test(df: pd.DataFrame, feature_cols):
    train_df = df[df['Date'] <= pd.to_datetime(DSET['train_end_date'])].copy()
    test_df  = df[df['Date'] >= pd.to_datetime(DSET['test_start_date'])].copy()
    train_df = train_df.dropna(subset=feature_cols + ['target'])
    test_df  = test_df.dropna(subset=feature_cols + ['target'])
    return train_df, test_df

# =========================
# SEQUENCE BUILDER
# =========================
def build_sequences(df: pd.DataFrame, feature_cols, timesteps: int):
    X_list, y_list = [], []
    for sid, s in df.groupby('ID', sort=False):
        s = s.sort_values('Date')
        arr = s[feature_cols].values
        targ = s['target'].values
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
def build_rnn_model_for_search(rnn_layers=2, units=64, learning_rate=0.001, l2_reg=0.001, dropout_rate=0.2, input_timesteps=None, input_dim=None):
    # Use provided input shapes; fall back to FIX if not given
    tsteps = input_timesteps if input_timesteps is not None else FIX['timesteps']
    dim = input_dim if input_dim is not None else None
    m = Sequential()
    for li in range(int(rnn_layers)):
        return_seq = (li < int(rnn_layers) - 1)
        if li == 0:
            m.add(SimpleRNN(
                units=int(units), activation='tanh', return_sequences=return_seq,
                input_shape=(tsteps, dim),
                kernel_regularizer=l2(float(l2_reg))
            ))
        else:
            m.add(SimpleRNN(
                units=int(units), activation='tanh', return_sequences=return_seq,
                kernel_regularizer=l2(float(l2_reg))
            ))
        m.add(Dropout(float(dropout_rate)))
    m.add(Dense(1, activation='linear'))
    m.compile(optimizer=Adam(float(learning_rate)), loss='mse', metrics=['mae'])
    return m

def make_model(input_timesteps: int, input_dim: int, rnn_layers: int, units: int, learning_rate: float, l2_reg: float, dropout_rate: float) -> tf.keras.Model:
    m = Sequential()
    for li in range(rnn_layers):
        return_seq = (li < rnn_layers - 1)
        if li == 0:
            m.add(SimpleRNN(
                units=units, activation='tanh', return_sequences=return_seq,
                input_shape=(input_timesteps, input_dim),
                kernel_regularizer=l2(l2_reg)
            ))
        else:
            m.add(SimpleRNN(
                units=units, activation='tanh', return_sequences=return_seq,
                kernel_regularizer=l2(l2_reg)
            ))
        m.add(Dropout(dropout_rate))
    m.add(Dense(1, activation='linear'))
    m.compile(optimizer=Adam(learning_rate), loss='mse', metrics=['mae'])
    return m


# =========================
# STRATEGY SIMULATOR (PT/SL ±2% from entry close, hold until barrier)
# =========================
class Position:
    __slots__ = ('sid','direction','entry_date','entry_price')
    def __init__(self, sid, direction, entry_date, entry_price):
        self.sid = sid
        self.direction = direction  # +1 long, -1 short
        self.entry_date = entry_date
        self.entry_price = entry_price

def simulate_strategy(test_df: pd.DataFrame, pred_df: pd.DataFrame):
    if not {'ID','Date','CloseAdj','HighAdj','LowAdj'}.issubset(set(test_df.columns)):
        raise ValueError("test_df must contain ID, Date, CloseAdj, HighAdj, LowAdj")

    df = test_df.sort_values(['Date','ID']).copy()
    pred = pred_df.sort_values(['Date','ID']).copy()
    by_date = pred.groupby('Date')

    df = df.set_index(['ID','Date']).sort_index()
    all_dates = sorted(test_df['Date'].unique())

    # Precompute MTM daily returns & equal-weight market
    ts = test_df.sort_values(['ID','Date']).copy()
    ts['close_ret'] = ts.groupby('ID')['CloseAdj'].pct_change().fillna(0.0)
    close_ret_map = ts.set_index(['ID','Date'])['close_ret']
    mkt_daily = ts.groupby('Date')['close_ret'].mean().reindex(all_dates).fillna(0.0)

    positions = {}
    daily_returns = []
    trades = []  # record finished trades

    # Precompute per-ID close series for barrier math
    close_series = {sid: ts[ts['ID']==sid].set_index('Date')['CloseAdj'].sort_index() for sid in ts['ID'].unique()}

    for D in all_dates:
        todays_pos_returns = []

        # Update open positions
        for sid in list(positions.keys()):
            if (sid, D) not in df.index:
                continue
            row = df.loc[(sid, D)]
            high_today  = row['HighAdj']
            low_today   = row['LowAdj']
            daily_mtm   = close_ret_map.get((sid, D), 0.0)

            pos = positions[sid]
            if pos.direction == +1:  # LONG
                tp = pos.entry_price * (1 + STR['pt_pct'])
                sl = pos.entry_price * (1 - STR['sl_pct'])
                hit_tp = (high_today >= tp)
                hit_sl = (low_today  <= sl)
                if hit_tp or hit_sl:
                    entry_close = pos.entry_price
                    exit_price = tp if (hit_tp and not hit_sl) else (sl if (hit_sl and not hit_tp) else sl)
                    total_ret = (exit_price / entry_close) - 1.0
                    cs = close_series.get(sid)
                    if cs is not None:
                        prev_dates = cs.index[cs.index < D]
                        if len(prev_dates) and prev_dates.min() <= pos.entry_date:
                            prev_day = prev_dates.max()
                            ret_until_prev = (cs.loc[prev_day] / entry_close) - 1.0
                        else:
                            ret_until_prev = 0.0
                    else:
                        ret_until_prev = 0.0
                    today_contrib = total_ret - ret_until_prev
                    todays_pos_returns.append(today_contrib)

                    trades.append({
                        "ID": sid,
                        "direction": "long",
                        "entry_date": pos.entry_date,
                        "exit_date": D,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "return_trade": total_ret,
                        "holding_days": int((pd.to_datetime(D) - pd.to_datetime(pos.entry_date)).days),
                        "reason": "TP" if (hit_tp and not hit_sl) else ("SL" if (hit_sl and not hit_tp) else "SL")
                    })
                    positions.pop(sid, None)
                else:
                    todays_pos_returns.append(daily_mtm)

            else:  # SHORT
                tp = pos.entry_price * (1 - STR['pt_pct'])
                sl = pos.entry_price * (1 + STR['sl_pct'])
                hit_tp = (low_today  <= tp)
                hit_sl = (high_today >= sl)
                if hit_tp or hit_sl:
                    entry_close = pos.entry_price
                    exit_price = tp if (hit_tp and not hit_sl) else (sl if (hit_sl and not hit_tp) else sl)
                    total_ret = (entry_close - exit_price) / entry_close
                    cs = close_series.get(sid)
                    if cs is not None:
                        prev_dates = cs.index[cs.index < D]
                        if len(prev_dates) and prev_dates.min() <= pos.entry_date:
                            prev_day = prev_dates.max()
                            ret_until_prev = (entry_close - cs.loc[prev_day]) / entry_close
                        else:
                            ret_until_prev = 0.0
                    else:
                        ret_until_prev = 0.0
                    today_contrib = total_ret - ret_until_prev
                    todays_pos_returns.append(today_contrib)

                    trades.append({
                        "ID": sid,
                        "direction": "short",
                        "entry_date": pos.entry_date,
                        "exit_date": D,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "return_trade": total_ret,
                        "holding_days": int((pd.to_datetime(D) - pd.to_datetime(pos.entry_date)).days),
                        "reason": "TP" if (hit_tp and not hit_sl) else ("SL" if (hit_sl and not hit_tp) else "SL")
                    })
                    positions.pop(sid, None)
                else:
                    todays_pos_returns.append(-daily_mtm)

        # Open new positions based on predictions at D
        if D in by_date.groups:
            g = by_date.get_group(D).copy()
            g = g[~g['ID'].isin(positions.keys())]
            g = g.sort_values('pred', ascending=False)
            longs  = g.head(STR['n_long'])
            shorts = g.tail(STR['n_short'])

            for _, row in longs.iterrows():
                sid = row['ID']
                if (sid, D) in df.index:
                    entry_price = df.loc[(sid, D)]['CloseAdj']
                    positions[sid] = Position(sid, +1, D, entry_price)
            for _, row in shorts.iterrows():
                sid = row['ID']
                if (sid, D) in df.index:
                    entry_price = df.loc[(sid, D)]['CloseAdj']
                    positions[sid] = Position(sid, -1, D, entry_price)

        port_ret = np.mean(todays_pos_returns) if todays_pos_returns else 0.0
        daily_returns.append((D, port_ret, mkt_daily.loc[D] if D in mkt_daily.index else 0.0))

    df_ret = pd.DataFrame(daily_returns, columns=['Date','strategy_ret','market_ret']).sort_values('Date')
    df_ret['cum_strategy'] = (1 + df_ret['strategy_ret']).cumprod()
    df_ret['cum_market'] = (1 + df_ret['market_ret']).cumprod()
    df_trades = pd.DataFrame(trades)
    return df_ret, df_trades

# =========================
# METRICS
# =========================
def compute_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

def compute_strategy_metrics(df_ret: pd.DataFrame):
    rf_daily = (1 + STR['rf_annual'])**(1/252) - 1
    rets = df_ret['strategy_ret'].values
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
    equity = (1 + df_ret['strategy_ret']).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak) - 1
    max_dd = drawdown.min()

    # Win rate & Profit factor
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    win_rate = len(wins) / len(rets) if len(rets) > 0 else 0.0
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan

    # Realized alpha via CAPM vs equal-weight market
    ex_p = df_ret['strategy_ret'].values - rf_daily
    ex_m = df_ret['market_ret'].values - rf_daily
    X = np.vstack([np.ones_like(ex_m), ex_m]).T
    beta_hat = np.linalg.lstsq(X, ex_p, rcond=None)[0]
    alpha_daily = beta_hat[0]
    beta = beta_hat[1]
    alpha_ann = alpha_daily * 252

    # t-stats (homoskedastic; pro striktnost lze nahradit HAC/Newey-West)
    resid = ex_p - X @ beta_hat
    s2 = (resid**2).sum() / (len(ex_p) - 2) if len(ex_p) > 2 else np.nan
    var_alpha = s2 * np.linalg.inv(X.T @ X)[0,0] if isinstance(s2, float) else np.nan
    se_alpha = np.sqrt(var_alpha) if isinstance(var_alpha, float) and var_alpha >= 0 else np.nan
    t_alpha = alpha_daily / se_alpha if (se_alpha and se_alpha>0) else np.nan

    se_mean = np.std(ex_p, ddof=1) / np.sqrt(len(ex_p)) if len(ex_p) > 1 else np.nan
    t_mean = (np.mean(ex_p) / se_mean) if (se_mean and se_mean>0) else np.nan

    return {
        'CAGR': cagr,
        'Ann_Vol': vol_ann,
        'Sharpe': sharpe,
        'Max_Drawdown': max_dd,
        'Win_Rate': win_rate,
        'Profit_Factor': profit_factor,
        'Alpha_ann': alpha_ann,
        'Beta': beta,
        't_alpha': t_alpha,
        't_mean_excess': t_mean,
    }

def compute_trade_metrics(df_tr: pd.DataFrame):
    if df_tr is None or df_tr.empty:
        return {
            "n_trades": 0,
            "win_rate_trades": np.nan,
            "profit_factor_trades": np.nan,
            "avg_trade_return": np.nan,
            "median_trade_return": np.nan,
            "avg_holding_days": np.nan,
            "median_holding_days": np.nan,
            "share_TP": np.nan,
            "share_SL": np.nan,
        }
    wins = df_tr.loc[df_tr['return_trade'] > 0, 'return_trade']
    losses = df_tr.loc[df_tr['return_trade'] < 0, 'return_trade']
    return {
        "n_trades": int(len(df_tr)),
        "win_rate_trades": float((df_tr['return_trade'] > 0).mean()) if len(df_tr)>0 else np.nan,
        "profit_factor_trades": float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.nan,
        "avg_trade_return": float(df_tr['return_trade'].mean()),
        "median_trade_return": float(df_tr['return_trade'].median()),
        "avg_holding_days": float(df_tr['holding_days'].mean()),
        "median_holding_days": float(df_tr['holding_days'].median()),
        "share_TP": float((df_tr['reason'] == 'TP').mean()),
        "share_SL": float((df_tr['reason'] == 'SL').mean()),
    }


# =========================
# HYPERPARAMETER TUNING
# =========================
def random_search_tuning(X_train_seq, y_train):
    print("[STEP] RandomizedSearchCV tuning (TimeSeriesSplit)...")
    timesteps = X_train_seq.shape[1]
    input_dim = X_train_seq.shape[2]

    # Wrap builder
    if USE_SCIKERAS:
        reg = KerasRegressor(
            model=build_rnn_model_for_search,
            model__input_timesteps=timesteps,
            model__input_dim=input_dim,
            epochs=HYPERPARAMS['cv_epochs'],
            batch_size=FIX['batch_size'],
            fit__verbose=2   # progress bary z Kerasu při ladění
        )
    else:
        reg = KerasRegressor(
            build_fn=lambda rnn_layers, units, learning_rate, l2_reg, dropout_rate: build_rnn_model_for_search(
                rnn_layers=rnn_layers, units=units, learning_rate=learning_rate, l2_reg=l2_reg,
                dropout_rate=dropout_rate, input_timesteps=timesteps, input_dim=input_dim
            ),
            epochs=HYPERPARAMS['cv_epochs'],
            batch_size=FIX['batch_size'],
            verbose=2        # progress bary u fallback wrapperu
        )

    # Map search space keys (SciKeras needs 'model__' prefix)
    space = HYPERPARAMS['search_space']
    param_dist = {}

    def key(k):
        return f"model__{k}" if USE_SCIKERAS else k

    if 'rnn_layers' in space:        param_dist[key('rnn_layers')]    = space['rnn_layers']
    if 'neurons_per_layer' in space: param_dist[key('units')]         = space['neurons_per_layer']
    if 'learning_rate' in space:     param_dist[key('learning_rate')] = space['learning_rate']
    if 'l2_reg' in space:            param_dist[key('l2_reg')]        = space['l2_reg']
    if 'dropout_rate' in space:      param_dist[key('dropout_rate')]  = space['dropout_rate']

    cv = TimeSeriesSplit(n_splits=HYPERPARAMS['cv_folds'])

    rnd = RandomizedSearchCV(
        estimator=reg,
        param_distributions=param_dist,
        n_iter=HYPERPARAMS['n_iter'],
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=1,
        verbose=2,
        random_state=FIX['random_seed']
    )

    rnd.fit(X_train_seq, y_train)
    best_params = rnd.best_params_
    best_score = rnd.best_score_
    print(f"[RESULT] Best params from RandomizedSearchCV: {best_params}")
    print(f"[RESULT] Best CV score (neg MSE): {best_score:.6f}")

    # Normalize keys across wrappers (scikeras uses model__ prefix)
    normalized = {}
    for k, v in best_params.items():
        if k.startswith("model__"):
            normalized[k.split("__",1)[1]] = v
        else:
            normalized[k] = v

    # Safe defaults for any missing keys
    normalized.setdefault('rnn_layers', 2)
    normalized.setdefault('units', 64)
    normalized.setdefault('learning_rate', 0.001)
    normalized.setdefault('l2_reg', FIX['l2_reg'])
    normalized.setdefault('dropout_rate', FIX['dropout_rate'])

    return normalized


# =========================
# VISUALIZATION
# =========================
def save_summary_png(history, df_ret, df_trades, reg_metrics: dict, model_info: dict, out_path="RNN_results.png"):
    print("\n[INFO] Saving summary PNG ->", out_path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Loss curve
    ax = axes[0,0]
    ax.plot(history.history.get('loss', []), label='Train Loss')
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Val Loss')
    ax.set_title('RNN Loss / Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2) Equity curve with vertical line at 2020-12-31
    ax = axes[0,1]
    ax.plot(df_ret['Date'], df_ret['cum_strategy'], label='Strategy (cum)')
    ax.plot(df_ret['Date'], df_ret['cum_market'], label='Market (cum)')
    vline_date = pd.to_datetime('2020-12-31')
    ax.axvline(vline_date, color='k', linestyle='--', linewidth=1, label='Train/Test split')
    ax.set_title('Cumulative Returns (Strategy vs Market)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Growth (x)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3) Table: hyperparams & model info
    ax = axes[1,0]
    ax.axis('off')
    table_data = [[k, f"{v}"] for k, v in model_info.items()]
    table = ax.table(cellText=table_data, colLabels=['Model / Setting', 'Value'], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)
    ax.set_title('RNN Hyperparameters & Settings', pad=12)

    # 4) Table: regression, strategy & trade metrics
    ax = axes[1,1]
    ax.axis('off')
    strat_metrics = compute_strategy_metrics(df_ret)
    trade_metrics = compute_trade_metrics(df_trades)

    def fmt(v, pct=False):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return 'NaN'
        if pct:
            return f"{v*100:.2f}%"
        return f"{v:.4f}"

    rows_top = [
        ['MSE', fmt(reg_metrics.get('MSE'))],
        ['MAE', fmt(reg_metrics.get('MAE'))],
        ['RMSE', fmt(reg_metrics.get('RMSE'))],
        ['R²', fmt(reg_metrics.get('R2'))],
        ['CAGR', fmt(strat_metrics.get('CAGR'), pct=True)],
        ['Ann. Vol', fmt(strat_metrics.get('Ann_Vol'))],
        ['Sharpe', fmt(strat_metrics.get('Sharpe'))],
        ['Max DD', fmt(strat_metrics.get('Max_Drawdown'), pct=True)],
        ['Win Rate (daily)', fmt(strat_metrics.get('Win_Rate'), pct=True)],
        ['Profit Factor (daily)', fmt(strat_metrics.get('Profit_Factor'))],
        ['Alpha (ann.)', fmt(strat_metrics.get('Alpha_ann'))],
        ['Beta', fmt(strat_metrics.get('Beta'))],
        ['t(alpha)', fmt(strat_metrics.get('t_alpha'))],
        ['t(mean excess)', fmt(strat_metrics.get('t_mean_excess'))],
    ]

    rows_trade = [
        ['Trades (count)', f"{int(trade_metrics.get('n_trades', 0))}"],
        ['Win Rate (trades)', fmt(trade_metrics.get('win_rate_trades'), pct=True)],
        ['Profit Factor (trades)', fmt(trade_metrics.get('profit_factor_trades'))],
        ['Avg Trade Return', fmt(trade_metrics.get('avg_trade_return'))],
        ['Median Trade Return', fmt(trade_metrics.get('median_trade_return'))],
        ['Avg Holding Days', f"{trade_metrics.get('avg_holding_days'):.1f}" if not np.isnan(trade_metrics.get('avg_holding_days', np.nan)) else 'NaN'],
        ['Median Holding Days', f"{trade_metrics.get('median_holding_days'):.0f}" if not np.isnan(trade_metrics.get('median_holding_days', np.nan)) else 'NaN'],
        ['Share TP', fmt(trade_metrics.get('share_TP'), pct=True)],
        ['Share SL', fmt(trade_metrics.get('share_SL'), pct=True)],
    ]

    table_rows = rows_top + rows_trade
    table2 = ax.table(cellText=table_rows, colLabels=['Metric', 'Value'], loc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.1, 1.0)
    ax.set_title('Regression, Strategy & Trade Metrics', pad=12)

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
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train_df[feature_cols].values)
    def transform_seq(X_seq):
        n_s, n_t, n_f = X_seq.shape
        X2 = X_seq.reshape(-1, n_f)
        X2s = scaler.transform(X2)
        return X2s.reshape(n_s, n_t, n_f)

    print("[STEP] Building sequences (train/test)...")
    X_train_seq, y_train = build_sequences(train_df, feature_cols, FIX['timesteps'])
    X_test_seq,  y_test  = build_sequences(test_df, feature_cols, FIX['timesteps'])
    print(f"[INFO] Train sequences: {X_train_seq.shape}, Test sequences: {X_test_seq.shape}")

    X_train_seq = transform_seq(X_train_seq)
    X_test_seq  = transform_seq(X_test_seq)

    # Tuning (TimeSeriesSplit)
    best = random_search_tuning(X_train_seq, y_train)

    # Model with best params
    print("[STEP] Compiling FINAL RNN model with best hyperparameters...]")
    final_units = int(best.get('units', 64))
    final_layers = int(best.get('rnn_layers', 2))
    final_lr = float(best.get('learning_rate', 0.001))
    final_l2 = float(best.get('l2_reg', FIX['l2_reg']))
    final_dropout = float(best.get('dropout_rate', FIX['dropout_rate']))

    def make_final_model():
        m = Sequential()
        for li in range(final_layers):
            return_seq = (li < final_layers - 1)
            if li == 0:
                m.add(SimpleRNN(units=final_units, activation='tanh', return_sequences=return_seq,
                                input_shape=(FIX['timesteps'], X_train_seq.shape[2]), kernel_regularizer=l2(final_l2)))
            else:
                m.add(SimpleRNN(units=final_units, activation='tanh', return_sequences=return_seq,
                                kernel_regularizer=l2(final_l2)))
            m.add(Dropout(final_dropout))
        m.add(Dense(1, activation='linear'))
        m.compile(optimizer=Adam(final_lr), loss='mse', metrics=['mae'])
        return m

    model = make_final_model()
    print(model.summary())

    # Train final (keep time order in validation split by taking last 10% as val)
    print("[STEP] Training FINAL model...")
    es = EarlyStopping(monitor='val_loss', patience=HYPERPARAMS['patience'], restore_best_weights=True, verbose=1)

    # Time-based validation: last 10% of sequences as validation
    n_train = X_train_seq.shape[0]
    val_size = max(1, int(0.1 * n_train))
    X_tr, X_val = X_train_seq[:-val_size], X_train_seq[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=HYPERPARAMS['final_model_epochs'],
        batch_size=FIX['batch_size'],
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
    print("[STEP] Building prediction DataFrame for ranking...]")
    meta = []
    X_list = []
    for sid, s in test_df.groupby('ID', sort=False):
        s = s.sort_values('Date')
        arr = s[feature_cols].values
        n = len(s)
        for i in range(n - FIX['timesteps']):
            meta.append((sid, s['Date'].iloc[i+FIX['timesteps']-1]))
            X_list.append(arr[i:i+FIX['timesteps']])
    X_meta = np.stack(X_list) if len(X_list)>0 else np.empty((0, FIX['timesteps'], len(feature_cols)))
    if len(X_list) > 0:
        n_s, n_t, n_f = X_meta.shape
        X2 = X_meta.reshape(-1, n_f)
        X2s = scaler.transform(X2)            # use the SAME scaler (train-only stats)
        X_meta_scaled = X2s.reshape(n_s, n_t, n_f)
        preds = model.predict(X_meta_scaled).reshape(-1)
        pred_df = pd.DataFrame(meta, columns=['ID','Date'])
        pred_df['Date'] = pd.to_datetime(pred_df['Date'])
        pred_df['pred'] = preds
    else:
        pred_df = pd.DataFrame(columns=['ID','Date','pred'])

    # Strategy simulation (returns df_ret + df_trades)
    print("[STEP] Simulating trading strategy (PT/SL, hold-until-barrier)...")
    df_ret, df_trades = simulate_strategy(test_df[['ID','Date','CloseAdj','HighAdj','LowAdj','SimpleReturn']].copy(), pred_df)

    # Trade-level metrics
    trade_metrics = compute_trade_metrics(df_trades)
    print("\n[RESULT] Trade-level metrics:")
    for k, v in trade_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    # Model info table
    model_info = {
        'tuned_rnn_layers': final_layers,
        'tuned_units': final_units,
        'tuned_learning_rate': final_lr,
        'tuned_l2_reg': final_l2,
        'tuned_dropout': final_dropout,
        'Model': 'Vanilla SimpleRNN',
        'rnn_layers': final_layers,
        'units': final_units,
        'learning_rate': final_lr,
        'dropout': final_dropout,
        'l2_reg': FIX['l2_reg'],
        'batch_size': FIX['batch_size'],
        'epochs': HYPERPARAMS['final_model_epochs'],
        'timesteps': FIX['timesteps'],
        'train_end_date': DSET['train_end_date'],
        'test_start_date': DSET['test_start_date'],
        'PT/SL': f"{STR['pt_pct']*100:.1f}% / {STR['sl_pct']*100:.1f}%",
        'N_LONG/N_SHORT': f"{STR['n_long']}/{STR['n_short']}",
        'RF_ANNUAL': f"{STR['rf_annual']*100:.2f}%"
    }

    # Visualization
    save_summary_png(history, df_ret, df_trades, reg_metrics, model_info, out_path="RNN_results.png")

    t1 = time.time()
    print("\n" + "="*80)
    print("RNN run completed successfully.")
    print(f"Total runtime: {(t1 - t0)/60:.2f} minutes")
    print("Summary image saved: RNN_results.png")
    print("="*80)

if __name__ == "__main__":
    main()
