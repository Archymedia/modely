
# ------------------------------------------------------------
# Fičura call 2 – LSTM pipeline (SP100 + VIX), bariéry ±2 %
# ------------------------------------------------------------
# Závislosti: pandas, numpy, scikit-learn, tensorflow (keras), scipy, statsmodels (volitelně)
# ------------------------------------------------------------

import os
import math
import json
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, deque

# TF / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error

# Stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Simple timestamped logger
def log(msg: str):
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# -----------------------------
# 1) HYPERPARAMS & REPRODUCIBILITA
# -----------------------------

K_FOLDS = 2  # number of folds for cross-validation

HYPERPARAMS = {
    'data': {
        # 👉 tvoje požadované cesty:
        'data_path': "/Users/lindawaisova/Desktop/DP/data/SP_100/READY_DATA/9DATA_FINAL.csv",
        'vix_path': "/Users/lindawaisova/Desktop/DP/data/SP_100/VIX/VIX_2005_2023.csv",
        'train_end_date': '2020-12-31',
        'test_start_date': '2021-01-01',
    },
    'seq': {
        'window': 20,        # h
        'rsi_period': 14,
        'cci_period': 20,
        'stoch_period': 14,
        'min_obs_per_id': 20 # pro jistotu
    },
    'model': {
        'lstm_units': 64,
        'dense_units': 64,
        'batch_size': 64,
        'epochs': 3,
        'patience': 1,
        'learning_rate': 1e-3,
        'loss': 'mse'
    },
    'strategy': {
        'top_k': 10,
        'bottom_k': 10,
        'tp': 0.02,   # +2 %
        'sl': 0.02,   # -2 %
        'sl_first': True, # priorita zásahu: nejdřív SL, pak TP
        'entry_on_open': True,  # vstup na t+1 OpenAdj
    },
    'random_seed': 42
}

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(HYPERPARAMS['random_seed'])

# -----------------------------
# 2) UTIL – TA indikátory (bez leaků – jen z minulosti)
# -----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    out = 100 - 100 / (1 + rs)
    return out

def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period).mean()
    md = (tp - sma).abs().rolling(period).mean()
    cci = (tp - sma) / (0.015 * md.replace(0, np.nan))
    return cci

def stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    return k

# -----------------------------
# 3) DATA LOAD & IDContIndex
# -----------------------------
def load_data(data_path: str, vix_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log("Loading SP100 and VIX data…")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(['ID', 'Date'], inplace=True)
    vix = pd.read_csv(vix_path)
    vix['Date'] = pd.to_datetime(vix['Date'])
    vix.sort_values('Date', inplace=True)
    log(f"Data loaded: SP100 rows={len(df):,}, VIX rows={len(vix):,}")
    return df, vix

def add_id_cont_index(df: pd.DataFrame) -> pd.DataFrame:
    # předpoklad: df je seřazené dle ['ID','Date']
    # Excel vzorec: =P2 + OR((A3<>A2),(D3>(D2+20)))  -> A=ID, D=Date
    # implementace v pandas:
    log("Computing IDContIndex (continuous listing periods)…")
    df = df.copy()
    df['Date_int'] = (df['Date'].view('int64') // 10**9)  # sekundy
    # posuny po skupinách ID
    df['ID_shift'] = df['ID'].shift(1)
    df['Date_shift'] = df['Date'].shift(1)
    change_id = (df['ID'] != df['ID_shift']).astype(int)
    gap = (df['Date'] > (df['Date_shift'] + pd.Timedelta(days=20))).astype(int)
    trigger = ((change_id == 1) | (gap == 1)).astype(int)
    # kumulativní součet přes celé df – ale musíme resetovat, když ID se změní:
    # trik: při change_id==1 dáme trigger=1 (už je), a kumulujeme v rámci celého df
    df['IDContIndex'] = trigger.cumsum()
    # Aby šlo lépe číst: spojíme původní ID s IDContIndex (nepovinné)
    log("IDContIndex computed.")
    return df.drop(columns=['Date_int', 'ID_shift', 'Date_shift'])

# -----------------------------
# 4) FEATURE ENGINEERING – subsekvence per IDContIndex
# -----------------------------
@dataclass
class SequenceData:
    X: np.ndarray        # (N, h, F)
    y: np.ndarray        # (N,)
    dates_t: np.ndarray  # (N,) datum t (konec okna)
    ids: np.ndarray      # (N,) IDContIndex (nebo RIC) pro mapování
    ric: np.ndarray      # (N,) RIC (kvůli strategii)

def build_sequences(df: pd.DataFrame, vix: pd.DataFrame, hparams: dict) -> SequenceData:
    log(f"Building sequences (h={hparams['seq']['window']}) and technical indicators…")
    h = hparams['seq']['window']
    rsi_p = hparams['seq']['rsi_period']
    cci_p = hparams['seq']['cci_period']
    stoch_p = hparams['seq']['stoch_period']
    min_obs = hparams['seq']['min_obs_per_id']

    # Merge VIX_Change (globální feature) podle Date
    vix_small = vix[['Date', 'VIX_Change']].copy()
    df = df.merge(vix_small, on='Date', how='left')

    # Technické indikátory per IDContIndex (bez leaků – rolling)
    def compute_ta(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values('Date').copy()
        g['RSI'] = rsi(g['CloseAdj'], rsi_p)
        g['CCI'] = cci(g['HighAdj'], g['LowAdj'], g['CloseAdj'], cci_p)
        g['STOCHK'] = stochastic_k(g['HighAdj'], g['LowAdj'], g['CloseAdj'], stoch_p)
        return g

    df = df.groupby('IDContIndex', group_keys=False).apply(compute_ta)

    # Příprava konstruovaných relativních kanálů (viz zadání)
    def make_relatives(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values('Date').copy()
        # subsequence_return
        g['RET'] = g['SimpleReturn']

        # close normalized to first in window -> budeme stavět až při skládání okna

        # high/low vs. own close
        g['HIGH_vs_CLOSE'] = g['HighAdj'] / g['CloseAdj'] - 1.0
        g['LOW_vs_CLOSE']  = g['LowAdj']  / g['CloseAdj'] - 1.0

        # close vs open
        g['CLOSE_vs_OPEN'] = g['CloseAdj'] / g['OpenAdj'] - 1.0

        # volume normalized to first in window -> až při skládání

        return g

    df = df.groupby('IDContIndex', group_keys=False).apply(make_relatives)

    # Budování oken (h) a targetu y = SimpleReturn(t+1)
    X_list, y_list, dates_list, id_list, ric_list = [], [], [], [], []

    # Pomocné funkce pro "within-window" normalizace Close/Volume sekvencí podle 1. prvku okna
    def rel_to_first(arr: np.ndarray) -> np.ndarray:
        base = arr[0]
        if base == 0 or np.isnan(base):
            return np.full_like(arr, np.nan)
        return arr / base - 1.0

    # pro každý IDContIndex
    for idc, g in df.groupby('IDContIndex'):
        g = g.sort_values('Date')
        if len(g) < max(min_obs, h + 2):
            continue

        # Vektory, ze kterých skládáme okna
        ret = g['RET'].values
        close = g['CloseAdj'].values
        high = g['HighAdj'].values
        low = g['LowAdj'].values
        openp = g['OpenAdj'].values
        vol = g['VolumeAdj'].values
        rsi_v = g['RSI'].values
        cci_v = g['CCI'].values
        stoch_v = g['STOCHK'].values
        vixc = g['VIX_Change'].values
        dates = g['Date'].values
        rics = g['RIC'].values

        # skládání oken
        for t in range(h-1, len(g)-1):  # -1 kvůli targetu t+1
            # okno končí v t, target je SimpleReturn(t+1)
            y = ret[t+1]
            # subsekvence
            win_slice = slice(t-h+1, t+1)
            ret_seq   = ret[win_slice]
            close_seq = rel_to_first(close[win_slice])
            high_seq  = (high[win_slice] / close[win_slice]) - 1.0
            low_seq   = (low[win_slice]  / close[win_slice]) - 1.0
            open_seq  = (close[win_slice] / openp[win_slice]) - 1.0
            vol_seq   = rel_to_first(vol[win_slice])

            rsi_seq   = rsi_v[win_slice]
            cci_seq   = cci_v[win_slice]
            stoch_seq = stoch_v[win_slice]
            vix_seq   = vixc[win_slice]

            seq_mat = np.vstack([
                ret_seq,
                close_seq,
                high_seq,
                low_seq,
                open_seq,
                vol_seq,
                rsi_seq,
                cci_seq,
                stoch_seq,
                vix_seq
            ]).T  # (h, F=10)

            if np.isnan(seq_mat).any() or np.isinf(seq_mat).any():
                continue

            X_list.append(seq_mat)
            y_list.append(y)
            dates_list.append(dates[t])     # datum t (konec okna)
            id_list.append(idc)
            ric_list.append(rics[t])

    X = np.array(X_list, dtype=np.float32)      # (N,h,F)
    y = np.array(y_list, dtype=np.float32)      # (N,)
    dates_t = np.array(dates_list)              # (N,)
    ids = np.array(id_list)
    ric = np.array(ric_list)

    log(f"Sequences ready: X={len(X_list):,} windows, features per step={X_list[0].shape[1] if X_list else 'N/A'}")
    return SequenceData(X, y, dates_t, ids, ric)

# -----------------------------
# 5) DEV/TEST SPLIT + 3-fold CV s anti‑leakage normalizací
# -----------------------------
@dataclass
class SplitIdx:
    dev_idx: np.ndarray
    test_idx: np.ndarray

def dev_test_split(dates_t: np.ndarray, train_end_date: str, test_start_date: str) -> SplitIdx:
    dev_mask  = dates_t <= np.datetime64(train_end_date)
    test_mask = dates_t >= np.datetime64(test_start_date)
    dev_idx = np.where(dev_mask)[0]
    test_idx = np.where(test_mask)[0]
    return SplitIdx(dev_idx, test_idx)

class PerFeatureScaler:
    """μ,σ per feature (agregace přes všechny timesteps), fit pouze na train."""
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, X: np.ndarray):
        # X: (N,h,F)
        # zploštíme timesteps
        N, h, F = X.shape
        Xf = X.reshape(-1, F)
        self.mu = np.nanmean(Xf, axis=0)
        self.sd = np.nanstd(Xf, axis=0)
        self.sd[self.sd == 0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

def make_lstm(input_shape: Tuple[int, int], hparams: dict) -> keras.Model:
    h, F = input_shape
    inp = keras.Input(shape=(h, F))
    x = layers.Masking(mask_value=np.nan)(inp)  # pro jistotu, neměly by tam být NaNs
    x = layers.LSTM(hparams['model']['lstm_units'])(x)
    x = layers.Dense(hparams['model']['dense_units'], activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)
    model = keras.Model(inp, out)
    opt = keras.optimizers.Adam(learning_rate=hparams['model']['learning_rate'])
    model.compile(optimizer=opt, loss=hparams['model']['loss'])
    return model

@dataclass
class CVResult:
    val_losses: List[float]
    models: List[keras.Model]
    scalers: List[PerFeatureScaler]

def cross_val_train(X_dev: np.ndarray, y_dev: np.ndarray, dates_dev: np.ndarray, hparams: dict) -> CVResult:
    # 3 časové foldy podle dat (bez míšení budoucnosti do minulosti)
    # rozdělíme podle kvantilů datumů
    uniq_dates = np.sort(np.unique(dates_dev))
    folds = K_FOLDS
    log("Starting 3-fold time-series cross-validation…")
    tss = TimeSeriesSplit(n_splits=folds)
    val_losses, models, scalers = [], [], []

    for fold_id, (train_idx_pos, val_idx_pos) in enumerate(tss.split(uniq_dates)):
        # map zpět na indexy vzorků podle data
        train_dates = set(uniq_dates[train_idx_pos])
        val_dates   = set(uniq_dates[val_idx_pos])
        train_idx = np.where(np.isin(dates_dev, list(train_dates)))[0]
        val_idx   = np.where(np.isin(dates_dev, list(val_dates)))[0]
        log(f"CV fold {fold_id+1}/{folds}: train={len(train_idx):,}, val={len(val_idx):,}")

        X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
        X_va, y_va = X_dev[val_idx], y_dev[val_idx]

        # anti‑leakage scaler fit jen na train
        scaler = PerFeatureScaler()
        scaler.fit(X_tr)
        X_trn = scaler.transform(X_tr)
        X_val = scaler.transform(X_va)

        cb = [
            callbacks.EarlyStopping(monitor='val_loss', patience=hparams['model']['patience'],
                                    restore_best_weights=True, verbose=0)
        ]

        model = make_lstm((X_trn.shape[1], X_trn.shape[2]), hparams)
        hist = model.fit(
            X_trn, y_tr,
            validation_data=(X_val, y_va),
            epochs=hparams['model']['epochs'],
            batch_size=hparams['model']['batch_size'],
            verbose=1,
            callbacks=cb
        )
        best_val = min(hist.history['val_loss'])
        val_losses.append(best_val)
        models.append(model)
        scalers.append(scaler)
        print(f"[CV fold {fold_id+1}/{folds}] best val_loss={best_val:.6f} (n_train={len(train_idx)}, n_val={len(val_idx)})")

    return CVResult(val_losses, models, scalers)

def retrain_on_full_dev(X_dev: np.ndarray, y_dev: np.ndarray, hparams: dict) -> Tuple[keras.Model, PerFeatureScaler]:
    log("Retraining final model on full DEV set…")
    scaler = PerFeatureScaler()
    scaler.fit(X_dev)
    X_devn = scaler.transform(X_dev)
    model = make_lstm((X_devn.shape[1], X_devn.shape[2]), hparams)
    cb = [
        callbacks.EarlyStopping(monitor='loss', patience=hparams['model']['patience'],
                                restore_best_weights=True, verbose=0)
    ]
    model.fit(
        X_devn, y_dev,
        validation_split=0.0,
        epochs=hparams['model']['epochs'],
        batch_size=hparams['model']['batch_size'],
        verbose=1,
        callbacks=cb
    )
    log("Final DEV model trained.")
    return model, scaler

# -----------------------------
# 6) PREDIKCE + STRATEGIE (long/short, bariéry ±2 %)
# -----------------------------
@dataclass
class Predictions:
    dates_t: np.ndarray
    ric: np.ndarray
    idc: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray

def predict_with_model(model: keras.Model, scaler: PerFeatureScaler, X: np.ndarray, y: np.ndarray,
                       dates_t: np.ndarray, ric: np.ndarray, idc: np.ndarray) -> Predictions:
    log(f"Predicting {len(y)} samples…")
    Xn = scaler.transform(X)
    yhat = model.predict(Xn, verbose=0).reshape(-1)
    log("Predictions done.")
    return Predictions(dates_t, ric, idc, y, yhat)

# --- strategie s bariérami ---
@dataclass
class Trade:
    ric: str
    side: int        # +1 long, -1 short
    entry_date: np.datetime64
    entry_price: float
    size: float      # notional (1.0 == 100%)
    open: bool = True

@dataclass
class StrategyResult:
    daily_pnl: pd.Series              # realizované PnL (vrací se v den exitů)
    equity_curve: pd.Series
    sharpe_pd: float
    sharpe_pa: float
    trades: List[Dict]
    alpha: float
    alpha_tstat: float

def simulate_barrier_strategy(pred: Predictions, raw_df: pd.DataFrame, hparams: dict) -> StrategyResult:
    log("Simulating barrier strategy (±2%)…")
    """
    - Signál z okna končícího v t => vstup t+1 OpenAdj
    - Každý den vyber top_k/bottom_k podle predikce
    - Bariéry ± tp/sl (priorita SL first pokud sl_first=True)
    - Výnos realizujeme v den zasažení bariéry (připsán do daily PnL)
    - OHLC z raw_df (per RIC & Date); používáme *_Adj
    """
    top_k = hparams['strategy']['top_k']
    bottom_k = hparams['strategy']['bottom_k']
    tp = hparams['strategy']['tp']
    sl = hparams['strategy']['sl']
    sl_first = hparams['strategy']['sl_first']

    # Připravíme OHLC pivoty: pro rychlý přístup k cenám
    px = raw_df[['RIC','Date','OpenAdj','HighAdj','LowAdj','CloseAdj']].copy()
    px['Date'] = pd.to_datetime(px['Date'])
    px = px.sort_values(['RIC','Date'])

    # groupby pro vyhledávání podle RIC
    ric_groups = {ric: g.set_index('Date') for ric, g in px.groupby('RIC')}
    all_dates = np.sort(np.unique(px['Date'].values))

    # Predikce jako DataFrame podle dne t a RIC
    pred_df = pd.DataFrame({
        'Date_t': pd.to_datetime(pred.dates_t),
        'RIC': pred.ric,
        'y_pred': pred.y_pred,
        'y_true': pred.y_true
    })
    pred_df = pred_df.sort_values(['Date_t','y_pred'], ascending=[True, False])

    # Z denních predikcí (Date_t) vyber top/bottom
    daily_signals = {}
    for d, g in pred_df.groupby('Date_t'):
        longs = g.nlargest(top_k, 'y_pred')['RIC'].tolist()
        shorts = g.nsmallest(bottom_k, 'y_pred')['RIC'].tolist()
        daily_signals[d] = {'longs': longs, 'shorts': shorts}

    # Simulace portfolia
    open_trades: Dict[str, List[Trade]] = defaultdict(list)
    pnl = defaultdict(float)  # pnl per date
    executed_trades = []

    # projdeme kalendářně
    for i, d in enumerate(all_dates[:-1]):  # poslední den už nemá t+1 open k vstupu
        # 1) Uzavírání starých podle bariér – kontrolujeme pro všechny otevřené
        for ric, trades in list(open_trades.items()):
            g = ric_groups.get(ric)
            if g is None:
                continue
            if d not in g.index:
                continue
            o = g.at[d, 'OpenAdj']
            h = g.at[d, 'HighAdj']
            l = g.at[d, 'LowAdj']
            c = g.at[d, 'CloseAdj']

            new_list = []
            for tr in trades:
                if not tr.open:
                    continue
                # od vstupního dne dál sledujeme OHLC – dnes kontrolujeme
                # výstupní cena při hitu bariéry = přesně bariéra (konzervativní předpoklad)
                hit = None
                if tr.side == +1:
                    tp_px = tr.entry_price * (1 + tp)
                    sl_px = tr.entry_price * (1 - sl)
                    first_hit = None
                    if sl_first:
                        if l <= sl_px:
                            first_hit = ('SL', sl_px)
                        elif h >= tp_px:
                            first_hit = ('TP', tp_px)
                    else:
                        if h >= tp_px:
                            first_hit = ('TP', tp_px)
                        elif l <= sl_px:
                            first_hit = ('SL', sl_px)
                    if first_hit:
                        kind, exit_px = first_hit
                        ret = (exit_px / tr.entry_price - 1.0) * tr.size
                        pnl[d] += ret
                        tr.open = False
                        executed_trades.append({
                            'Date_exit': d, 'RIC': ric, 'side': 'LONG',
                            'entry_date': tr.entry_date, 'entry_px': tr.entry_price,
                            'exit_px': exit_px, 'ret': ret, 'event': kind
                        })
                    else:
                        new_list.append(tr)
                else:  # short
                    tp_px = tr.entry_price * (1 - tp)  # TP pro short => pokles
                    sl_px = tr.entry_price * (1 + sl)
                    first_hit = None
                    if sl_first:
                        if h >= sl_px:
                            first_hit = ('SL', sl_px)
                        elif l <= tp_px:
                            first_hit = ('TP', tp_px)
                    else:
                        if l <= tp_px:
                            first_hit = ('TP', tp_px)
                        elif h >= sl_px:
                            first_hit = ('SL', sl_px)
                    if first_hit:
                        kind, exit_px = first_hit
                        ret = (1.0 - exit_px / tr.entry_price) * tr.size  # short zisk
                        pnl[d] += ret
                        tr.open = False
                        executed_trades.append({
                            'Date_exit': d, 'RIC': ric, 'side': 'SHORT',
                            'entry_date': tr.entry_date, 'entry_px': tr.entry_price,
                            'exit_px': exit_px, 'ret': ret, 'event': kind
                        })
                    else:
                        new_list.append(tr)
            open_trades[ric] = new_list

        # 2) Vstupy (signály z t=d => vstup na t+1 open)
        d_next = all_dates[i+1]
        sig = daily_signals.get(d, None)
        if sig:
            # Longy
            for ric in sig['longs']:
                g = ric_groups.get(ric)
                if g is None: 
                    continue
                if d_next not in g.index:
                    continue
                entry_px = g.at[d_next, 'OpenAdj']
                open_trades[ric].append(Trade(ric=ric, side=+1, entry_date=d_next,
                                              entry_price=entry_px, size=1.0))
            # Shorty
            for ric in sig['shorts']:
                g = ric_groups.get(ric)
                if g is None:
                    continue
                if d_next not in g.index:
                    continue
                entry_px = g.at[d_next, 'OpenAdj']
                open_trades[ric].append(Trade(ric=ric, side=-1, entry_date=d_next,
                                              entry_price=entry_px, size=1.0))

    # Daily PnL série
    pnl_series = pd.Series(pnl).sort_index()
    equity = pnl_series.cumsum()

    # Sharpe (denní a annualizovaný; pokud je std=0, dá NaN)
    sharpe_pd = pnl_series.mean() / (pnl_series.std(ddof=1) + 1e-12) if len(pnl_series) > 1 else np.nan
    sharpe_pa = sharpe_pd * math.sqrt(252) if not np.isnan(sharpe_pd) else np.nan

    # Alfa vůči EW benchmarku (equal‑weighted SimpleReturn napříč dostupnými tickery)
    bench = raw_df.groupby('Date')['SimpleReturn'].mean().sort_index()
    # zarovnat na pnl_series index (realizace v dny výstupů)
    df_alpha = pd.DataFrame({
        'pnl': pnl_series
    })
    df_alpha['bench'] = bench.reindex(df_alpha.index).fillna(0.0)
    Xreg = sm.add_constant(df_alpha['bench'].values)
    model_ols = sm.OLS(df_alpha['pnl'].values, Xreg, missing='drop').fit()
    alpha = model_ols.params[0]
    alpha_t = model_ols.tvalues[0]

    log(f"Simulation done: trades executed={len(executed_trades):,}, pnl days={len(pnl):,}")
    return StrategyResult(
        daily_pnl=pnl_series,
        equity_curve=equity,
        sharpe_pd=sharpe_pd,
        sharpe_pa=sharpe_pa,
        trades=executed_trades,
        alpha=alpha,
        alpha_tstat=alpha_t
    )

# -----------------------------
# 7) MAIN PIPELINE
# -----------------------------
def main(hparams=HYPERPARAMS):
    log("==== LSTM pipeline started ====")
    # Load
    df, vix = load_data(hparams['data']['data_path'], hparams['data']['vix_path'])

    # IDContIndex
    df = add_id_cont_index(df)

    # Build sequences
    seq = build_sequences(df, vix, hparams)
    log(f"Sequences built: X={seq.X.shape}, y={seq.y.shape}, unique dates={len(np.unique(seq.dates_t))}")

    # Dev/Test split
    sp = dev_test_split(seq.dates_t, hparams['data']['train_end_date'], hparams['data']['test_start_date'])
    X_dev, y_dev, d_dev, ric_dev, idc_dev = seq.X[sp.dev_idx], seq.y[sp.dev_idx], seq.dates_t[sp.dev_idx], seq.ric[sp.dev_idx], seq.ids[sp.dev_idx]
    X_test, y_test, d_test, ric_test, idc_test = seq.X[sp.test_idx], seq.y[sp.test_idx], seq.dates_t[sp.test_idx], seq.ric[sp.test_idx], seq.ids[sp.test_idx]

    log(f"DEV: {X_dev.shape[0]} samples (<= {hparams['data']['train_end_date']})")
    log(f"TEST: {X_test.shape[0]} samples (>= {hparams['data']['test_start_date']})")

    # 3-fold CV (anti‑leakage scaling per fold)
    cvres = cross_val_train(X_dev, y_dev, d_dev, hparams)
    log(f"CV val losses: {cvres.val_losses} | mean={np.mean(cvres.val_losses):.6f}")

    # Finální trén na celém DEV (nový scaler z celého DEV)
    final_model, final_scaler = retrain_on_full_dev(X_dev, y_dev, hparams)

    # Predikce DEV & TEST
    pred_dev = predict_with_model(final_model, final_scaler, X_dev, y_dev, d_dev, ric_dev, idc_dev)
    pred_tst = predict_with_model(final_model, final_scaler, X_test, y_test, d_test, ric_test, idc_test)

    # R^2 / RMSE pro info
    rmse_dev = math.sqrt(mean_squared_error(pred_dev.y_true, pred_dev.y_pred))
    rmse_tst = math.sqrt(mean_squared_error(pred_tst.y_true, pred_tst.y_pred))
    r2_dev = r2_score(pred_dev.y_true, pred_dev.y_pred)
    r2_tst = r2_score(pred_tst.y_true, pred_tst.y_pred)
    log(f"DEV: RMSE={rmse_dev:.6f}, R2={r2_dev:.4f}")
    log(f"TEST: RMSE={rmse_tst:.6f}, R2={r2_tst:.4f}")

    # Strategie s bariérami (na DEV i TEST zvlášť)
    # DEV subset df
    df_dev = df[df['Date'] <= pd.to_datetime(hparams['data']['train_end_date'])].copy()
    df_tst = df[df['Date'] >= pd.to_datetime(hparams['data']['test_start_date'])].copy()

    strat_dev = simulate_barrier_strategy(pred_dev, df_dev, hparams)
    strat_tst = simulate_barrier_strategy(pred_tst, df_tst, hparams)

    log("\n--- Strategy DEV ---")
    log(f"Sharpe_pd={strat_dev.sharpe_pd:.3f}, Sharpe_pa={strat_dev.sharpe_pa:.3f}, alpha={strat_dev.alpha:.6f} (t={strat_dev.alpha_tstat:.2f})")
    log(f"Trades executed: {len(strat_dev.trades)} | last equity={strat_dev.equity_curve.iloc[-1]:.4f}")

    log("\n--- Strategy TEST ---")
    log(f"Sharpe_pd={strat_tst.sharpe_pd:.3f}, Sharpe_pa={strat_tst.sharpe_pa:.3f}, alpha={strat_tst.alpha:.6f} (t={strat_tst.alpha_tstat:.2f})")
    if len(strat_tst.equity_curve) > 0:
        log(f"Trades executed: {len(strat_tst.trades)} | last equity={strat_tst.equity_curve.iloc[-1]:.4f}")

    # --- Plot cumulative returns (PnL) over 2005–2023 and save as PNG ---
    # Build a daily index across the whole data range
    date_min = df['Date'].min().normalize()
    date_max = df['Date'].max().normalize()
    daily_index = pd.date_range(date_min, date_max, freq='D')

    # Forward-fill equity curves on daily grid (DEV starts at 0)
    dev_eq = strat_dev.equity_curve.reindex(daily_index).ffill().fillna(0.0)
    tst_eq_raw = strat_tst.equity_curve.reindex(daily_index).ffill()

    # Offset TEST equity so that it continues from the last DEV value
    if not tst_eq_raw.dropna().empty:
        first_valid = tst_eq_raw.dropna().index[0]
        baseline = tst_eq_raw.loc[first_valid]
        last_dev_val = dev_eq.loc[first_valid] if first_valid in dev_eq.index else dev_eq.iloc[-1]
        tst_eq = last_dev_val + (tst_eq_raw - baseline)
    else:
        tst_eq = pd.Series(index=daily_index, dtype=float)

    # Combine: DEV everywhere, then override with TEST (offset) once it starts
    combined_eq = dev_eq.copy()
    if not tst_eq.dropna().empty:
        mask = tst_eq.notna()
        combined_eq.loc[mask.index] = np.where(mask, tst_eq, combined_eq.loc[mask.index])

    # Save PNG
    plt.figure()
    plt.plot(combined_eq.index, combined_eq.values, label='Cumulative PnL (DEV+TEST)')
    plt.axvline(pd.to_datetime(hparams['data']['test_start_date']), linestyle='--', label='Test start')
    plt.title('Cumulative returns (PnL) – 2005–2023')
    plt.xlabel('Date')
    plt.ylabel('Cumulative return')
    plt.legend()
    out_path = os.path.join(os.path.dirname(__file__), 'equity_2005_2023.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Saved equity curve to: {out_path}")

    # Vrátíme artefakty pro případné další použití
    artifacts = {
        'final_model': final_model,
        'final_scaler_mu': final_scaler.mu.tolist(),
        'final_scaler_sd': final_scaler.sd.tolist(),
        'metrics': {
            'rmse_dev': rmse_dev, 'r2_dev': r2_dev,
            'rmse_test': rmse_tst, 'r2_test': r2_tst,
            'sharpe_pd_dev': strat_dev.sharpe_pd, 'sharpe_pa_dev': strat_dev.sharpe_pa,
            'alpha_dev': strat_dev.alpha, 'alpha_t_dev': strat_dev.alpha_tstat,
            'sharpe_pd_test': strat_tst.sharpe_pd, 'sharpe_pa_test': strat_tst.sharpe_pa,
            'alpha_test': strat_tst.alpha, 'alpha_t_test': strat_tst.alpha_tstat
        }
    }
    log("==== Pipeline finished ====")
    return artifacts, (pred_dev, pred_tst), (strat_dev, strat_tst)

if __name__ == "__main__":
    artifacts, preds, strategies = main(HYPERPARAMS)
