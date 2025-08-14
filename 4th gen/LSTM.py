# -*- coding: utf-8 -*-
"""
LSTM pipeline pro SP100 (2005-2023) s VIX, M2M PnL a bariérami ±2 %.

- Příprava IDContIndex podle pravidla: P2 + OR((A3<>A2), (D3>(D2+20)))
- Tvorba subsekvencí (okno h) pro LSTM (returns, close norm, high/low rel, open ratio, volume norm)
- TA indikátory (RSI, CCI, Stochastic %K)
- VIX subsekvence (VIX_Change)
- Split: Development (<= 2020-12-31), Test (>= 2021-01-01)
- k-fold time-based CV na Developmentu (anti-leak na normalizaci: mu/sigma jen z train části; `k_folds` nastavíš v HYPERPARAMS)
- LSTM: 64 LSTM -> 64 Dense, Adam, batch 64, EarlyStopping patience 10
- Signály v t → vstup na t+1 OpenAdj, držení do bariéry ±2 % (SL prioritní, pak TP)
- M2M PnL: denní přeceňování všech otevřených pozic
- Benchmark: equal-weighted průměr SimpleReturn napříč dostupnými tickery daného dne
- Realizovaná alfa na test sample (OLS: strategy ~ alpha + beta*benchmark)
- Graf kumulativních výnosů (strategie vs. benchmark) a uložení PNG

Poznámky:
- TensorFlow 2.17 (tensorflow-macos) + tensorflow-metal 1.3 funguje s tímto kódem.
- Kód je paměťově úsporný, ale i tak může trvat déle (hodně sekvencí).
"""

import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle as sk_shuffle

import statsmodels.api as sm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# =========================
# ====== HYPERPARAMS ======
# =========================

HYPERPARAMS = {
    'data': {
        # Použij nejprve Mac cestu, pokud neexistuje, použij Windows; fallback /mnt/data
        'data_paths': [
            "/Users/lindawaisova/Desktop/DP/data/SP_100/READY_DATA/9DATA_FINAL.csv",
            r"C:\Users\david\Desktop\SP100\9DATA_FINAL.csv",
            "/mnt/data/9DATA_FINAL.csv",
        ],
        'vix_paths': [
            "/Users/lindawaisova/Desktop/DP/data/SP_100/READY_DATA/VIX_2005_2023.csv",
            r"C:\Users\david\Desktop\SP100\VIX_2005_2023.csv",
            "/mnt/data/VIX_2005_2023.csv",
        ],
        'train_end_date': '2020-12-31',
        'test_start_date': '2021-01-01',
        'date_col': 'Date',
        'id_col': 'ID',
        'id_cont_col': 'IDContIndex',
        'ric_col': 'RIC',
    },
    'features': {
        'window': 15,             # h (délka subsekvence)
        'use_RSI': True,
        'use_CCI': True,
        'use_STOCH': True,
        'rsi_period': 14,
        'cci_period': 20,
        'stoch_period': 14
    },
    'model': {
        'lstm_units': 64,
        'dense_units': 64,
        'optimizer': 'adam',
        'loss': 'mse',
        'batch_size': 64,
        'CV_epochs': 100,          # kratší trénink pro CV
        'final_epochs': 100,      # finální trénink na celém developmentu
        'early_stopping_patience': 1,
        'seed': 42,
        'k_folds': 3,
        'keras_verbose': 1
    },
    'strategy': {
        'top_n': 10,
        'bottom_n': 10,
        'tp': 0.02,      # +2 %
        'sl': -0.02,     # -2 %
        'priority': 'SL_first',  # pokud High i Low zasáhnou v jednom dni: nejdřív SL, pak TP
        'engine': 'exact_cached',  # rychlejší přístup k datům (keše indexů/NumPy)
    },
    'output': {
        'png_paths': [
            "/Users/lindawaisova/Desktop/DP/4th generation/LSTM_dashboard.png",
            r"C:\Users\david\Desktop\DP\LSTM_dashboard.png",
            os.path.join(os.getcwd(), "LSTM_dashboard.png"),
        ]
    }
}

np.random.seed(HYPERPARAMS['model']['seed'])
tf.random.set_seed(HYPERPARAMS['model']['seed'])

# =========================
# ====== UTIL FUNCS =======
# =========================

def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")
    sys.stdout.flush()

def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[-1]  # poslední jako fallback

def load_data():
    cfg = HYPERPARAMS['data']
    data_path = find_first_existing(cfg['data_paths'])
    vix_path = find_first_existing(cfg['vix_paths'])
    log(f"Načítám SP100 data: {data_path}")
    df = pd.read_csv(data_path)
    log(f"Načítám VIX data: {vix_path}")
    vix = pd.read_csv(vix_path)

    # Parsování dat
    df[cfg['date_col']] = pd.to_datetime(df[cfg['date_col']])
    vix['Date'] = pd.to_datetime(vix['Date'])

    # Seřaď podle ID a data
    df.sort_values([cfg['id_col'], cfg['date_col']], inplace=True)
    vix.sort_values(['Date'], inplace=True)
    vix = vix[['Date', 'VIX_Change']].copy()
    return df, vix

def compute_IDContIndex(df: pd.DataFrame):
    """ Implementace: P3 = P2 + OR((A3<>A2), (D3 > D2+20)), kde:
        A = ID, D = Date, P = IDContIndex (1-based; zde vytvoříme 0-based a pak posuneme na 1-based)
    """
    cfg = HYPERPARAMS['data']
    id_col, date_col = cfg['id_col'], cfg['date_col']

    log("Vypočítávám IDContIndex (kontinuální periody členství v indexu)...")
    df = df.copy()
    df['__id_shift'] = df[id_col].shift(1)
    df['__date_shift'] = df[date_col].shift(1)
    # Nový segment startuje, když se změní ID nebo mezera > 20 dní
    new_segment = (df[id_col] != df['__id_shift']) | ((df[date_col] - df['__date_shift']).dt.days > 20)
    # kumulativní součet nových segmentů per ID "resetuje" až když se ID změní;
    # proto to uděláme globálně a pak přemapujeme v rámci každého ID zvlášť.
    # Jednodušeji: uděláme groupby dle ID a v něm kumulativní sumu new_segment.
    df['segment'] = new_segment.groupby(df[id_col]).cumsum()
    # Aby byla IDContIndex unikátní napříč ID, zkombinujeme (ID, segment) do běžícího čítače:
    # ale pro jednodušší práci zachováme sloupec jako "lokální" index per ID a segment:
    # Požadavek říká "Proměnná IDContIndex nahrazuje ID akcie." — stačí (ID, segment) → jedinečná kombinace.
    # Uděláme číselný index přes kategorizaci.
    df['IDContIndex'] = pd.Categorical(df[id_col].astype(str) + '_' + df['segment'].astype(str)).codes
    # Úklid
    df.drop(columns=['__id_shift', '__date_shift', 'segment'], inplace=True)
    return df

# ----- TA indikátory -----

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    r = 100 - (100 / (1 + rs))
    return r.fillna(0.0)

def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20):
    tp = (high + low + close) / 3.0
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    c = (tp - ma) / (0.015 * md)
    return c.fillna(0.0)

def stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    return k.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ----- Subsekvence builder -----

def build_sequences_for_group(g: pd.DataFrame, h: int, feat_cfg):
    """
    Vytvoří subsekvence pro jednu kontinuální periodu (IDContIndex).
    Vrací: list of dicts: {'date_t': t_date, 'X': (h, C), 'y': scalar, 'keys': ...}
    """
    # Očekávané sloupce:
    # SimpleReturn, OpenAdj, HighAdj, LowAdj, CloseAdj, VolumeAdj
    # TA: RSI, CCI, STOCH již spočteno v g
    needed = ['SimpleReturn','OpenAdj','HighAdj','LowAdj','CloseAdj','VolumeAdj','RSI','CCI','STOCH']
    for col in needed:
        if col not in g.columns:
            g[col] = np.nan

    arr = []
    g = g.reset_index(drop=True)
    T = len(g)

    # Pomocné funkce pro subsekvence dle definic
    def subseq_return(t0, t1):  # inclusive t0..t1
        return g['SimpleReturn'].iloc[t0:t1+1].values

    def subseq_close_norm(t0, t1):
        base = g['CloseAdj'].iloc[t0]
        seq = g['CloseAdj'].iloc[t0:t1+1].values
        return (seq / base) - 1.0

    def subseq_rel(series_name):
        # e.g. HighAdj/CloseAdj - 1
        s = (g[series_name] / g['CloseAdj']) - 1.0
        return s

    def subseq_open_ratio():
        # CloseAdj/OpenAdj - 1
        s = (g['CloseAdj'] / g['OpenAdj']) - 1.0
        return s

    def subseq_volume_norm(t0, t1):
        base = g['VolumeAdj'].iloc[t0]
        seq = g['VolumeAdj'].iloc[t0:t1+1].values
        return (seq / base) - 1.0

    # Předpočítáme rel a ratio řady pro rychlost
    rel_high = subseq_rel('HighAdj').values
    rel_low  = subseq_rel('LowAdj').values
    open_ratio = subseq_open_ratio().values

    # Pro indexování budeme procházet konce oken t = h-1 .. T-2 (protože y=SimpleReturn(t+1))
    for t in range(h-1, T-1):
        t0 = t - (h - 1)
        t1 = t

        # target(t) = SimpleReturn(t+1)
        y = g['SimpleReturn'].iloc[t+1]
        if pd.isna(y):
            continue  # nelze použít

        # Skládání kanálů: returns, close_norm, high_rel, low_rel, open_ratio, volume_norm, RSI, CCI, STOCH
        r_seq = subseq_return(t0, t1)
        c_seq = subseq_close_norm(t0, t1)
        h_seq = rel_high[t0:t1+1]
        l_seq = rel_low[t0:t1+1]
        o_seq = open_ratio[t0:t1+1]
        v_seq = subseq_volume_norm(t0, t1)
        rsi_seq = g['RSI'].iloc[t0:t1+1].values if feat_cfg['use_RSI'] else None
        cci_seq = g['CCI'].iloc[t0:t1+1].values if feat_cfg['use_CCI'] else None
        sto_seq = g['STOCH'].iloc[t0:t1+1].values if feat_cfg['use_STOCH'] else None

        # Stack kanálů do (h, C)
        channels = [r_seq, c_seq, h_seq, l_seq, o_seq, v_seq]
        names = ["RET","CLOSE_N","HIGH_REL","LOW_REL","OPEN_RATIO","VOL_N"]
        if rsi_seq is not None:
            channels.append(rsi_seq); names.append("RSI")
        if cci_seq is not None:
            channels.append(cci_seq); names.append("CCI")
        if sto_seq is not None:
            channels.append(sto_seq); names.append("STOCH")

        X = np.vstack(channels).T  # shape (h, C)
        # Meta informace pro strategii a pro VIX doplníme později
        arr.append({
            'date_t': g['Date'].iloc[t],          # konec okna (signál v t)
            'idcont': g['IDContIndex'].iloc[t],
            'id': g['ID'].iloc[t],
            'ric': g.get('RIC', pd.Series([None])).iloc[t] if 'RIC' in g.columns else None,
            'X': X,
            'y': y,
            'OpenAdj_t1': g['OpenAdj'].iloc[t+1],  # vstupní cena na t+1
            'HighAdj_t1': g['HighAdj'].iloc[t+1],
            'LowAdj_t1': g['LowAdj'].iloc[t+1],
            'CloseAdj_t1': g['CloseAdj'].iloc[t+1],
            'row_idx_t': t
        })
    return arr, names

def attach_vix_sequences(seq_list, vix_df, h):
    """
    Ke každé sekvenci doplní kanál VIX_Change subseq [t-h+1..t]
    Standardizace VIX se bude dělat až při normalizaci (globální mu/sigma z dev-train).
    """
    # Precompute dict: date -> rolling window of VIX_Change ending at date
    vix = vix_df.set_index('Date')['VIX_Change'].sort_index()
    # Pro rychlost naplníme do Series a budeme si tahat okno per date.
    for item in seq_list:
        t_date = item['date_t']
        t0_date = t_date - timedelta(days=365*10)  # nepotřebujeme, jen placeholder
        # Vezmeme posledních h obchodních vix hodnot <= t_date
        # protože VIX má trading dny, použijeme index slice
        if t_date not in vix.index:
            # pokud není přesně tento obchodní den ve VIX (svátky), vezmi nejbližší menší
            prior_dates = vix.index[vix.index <= t_date]
            if len(prior_dates) == 0:
                vix_seq = np.zeros(h)
                item['X'] = np.hstack([item['X'], vix_seq.reshape(-1,1)])
                item['feat_names'] = item.get('feat_names', []) + ['VIX_CHG']
                continue
            t_date_eff = prior_dates[-1]
        else:
            t_date_eff = t_date
        # okno h hodnot končící v t_date_eff
        idx_pos = vix.index.get_loc(t_date_eff)
        start = max(0, idx_pos - (h - 1))
        window = vix.iloc[start:idx_pos+1].values
        if len(window) < h:
            # dopadujeme nulami na začátku (konzervativně)
            window = np.pad(window, (h - len(window), 0))
        vix_seq = window[-h:]
        item['X'] = np.hstack([item['X'], vix_seq.reshape(-1,1)])
        item['feat_names'] = item.get('feat_names', []) + ['VIX_CHG']
    return seq_list

def compute_ta_per_idcont(df):
    feat = HYPERPARAMS['features']
    log("Počítám TA indikátory (RSI, CCI, Stochastic %K) per IDContIndex...")
    out = []
    for _, g in df.groupby('IDContIndex'):
        g = g.sort_values('Date').copy()
        if feat['use_RSI']:
            g['RSI'] = rsi(g['CloseAdj'], feat['rsi_period'])
        else:
            g['RSI'] = 0.0
        if feat['use_CCI']:
            g['CCI'] = cci(g['HighAdj'], g['LowAdj'], g['CloseAdj'], feat['cci_period'])
        else:
            g['CCI'] = 0.0
        if feat['use_STOCH']:
            g['STOCH'] = stochastic_k(g['HighAdj'], g['LowAdj'], g['CloseAdj'], feat['stoch_period'])
        else:
            g['STOCH'] = 0.0
        out.append(g)
    return pd.concat(out, axis=0).sort_values(['IDContIndex','Date']).reset_index(drop=True)

def build_all_sequences(df, vix):
    h = HYPERPARAMS['features']['window']
    feat_cfg = HYPERPARAMS['features']
    log(f"Tvořím subsekvence (okno h={h}) pro všechny kontinuální periody...")
    all_seqs = []
    feat_names_ref = None
    for _, g in df.groupby('IDContIndex'):
        seqs, feat_names = build_sequences_for_group(g, h, feat_cfg)
        if feat_names_ref is None:
            feat_names_ref = feat_names[:]  # základ bez VIX
        all_seqs.extend(seqs)
    log(f"Počet subsekvencí (vzorků): {len(all_seqs):,}")
    # Připojíme VIX subsekvence
    all_seqs = attach_vix_sequences(all_seqs, vix, h)
    # finální seznam jmen feature kanálů:
    feat_names_final = feat_names_ref + ['VIX_CHG']
    return all_seqs, feat_names_final

def split_dev_test(all_seqs):
    train_end = pd.to_datetime(HYPERPARAMS['data']['train_end_date'])
    test_start = pd.to_datetime(HYPERPARAMS['data']['test_start_date'])
    dev = [s for s in all_seqs if s['date_t'] <= train_end]
    test = [s for s in all_seqs if s['date_t'] >= test_start]
    log(f"Split dev/test: dev={len(dev):,} vzorků (<= {train_end.date()}), test={len(test):,} vzorků (>= {test_start.date()})")
    return dev, test

def stack_Xy(seq_list):
    X = np.stack([s['X'] for s in seq_list], axis=0)  # (N, h, C)
    y = np.array([s['y'] for s in seq_list], dtype=float)  # (N,)
    dates = np.array([s['date_t'] for s in seq_list])
    ids = np.array([s['id'] for s in seq_list])
    idconts = np.array([s['idcont'] for s in seq_list])
    meta = {
        'OpenAdj_t1': np.array([s['OpenAdj_t1'] for s in seq_list]),
        'HighAdj_t1': np.array([s['HighAdj_t1'] for s in seq_list]),
        'LowAdj_t1':  np.array([s['LowAdj_t1'] for s in seq_list]),
        'CloseAdj_t1':np.array([s['CloseAdj_t1'] for s in seq_list]),
        'RIC': np.array([s['ric'] for s in seq_list], dtype=object)
    }
    return X, y, dates, ids, idconts, meta

def compute_mu_sigma(X):
    # mu,sigma přes všechny akcie a všechny subsekvence – per feature kanál, přes čas i vzorky
    # X shape: (N, h, C) -> agregujeme přes N a h → (C,)
    mu = X.reshape(-1, X.shape[-1]).mean(axis=0)
    sigma = X.reshape(-1, X.shape[-1]).std(axis=0, ddof=1)
    sigma[sigma == 0] = 1.0
    return mu, sigma

def standardize_with(X, mu, sigma):
    return (X - mu) / sigma

def build_lstm_model(input_shape, hp):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(hp['lstm_units']),
        layers.Dense(hp['dense_units'], activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=hp['optimizer'], loss=hp['loss'])
    return model

def cv_train_lstm(X_dev, y_dev, dates_dev):
    """ k-fold time-based split přes datumy v dev. """
    hp = HYPERPARAMS['model']
    k_folds = int(hp.get('k_folds', 3))
    log(f"Spouštím {k_folds}-fold time-based cross-validaci (bez leaků v normalizaci)...")
    N = len(y_dev)
    order = np.argsort(dates_dev)
    X_dev = X_dev[order]
    y_dev = y_dev[order]
    dates_dev = dates_dev[order]

    # rozdělíme development na k_folds chronologických bloků
    folds = np.array_split(np.arange(N), k_folds)
    val_losses = []
    models_trained = []

    for fold_idx in range(k_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(k_folds) if i != fold_idx])

        X_train, y_train = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx], y_dev[val_idx]

        # mu/sigma jen z train části fold-u
        mu, sigma = compute_mu_sigma(X_train)
        X_train_n = standardize_with(X_train, mu, sigma)
        X_val_n   = standardize_with(X_val,   mu, sigma)

        model = build_lstm_model(input_shape=X_train_n.shape[1:], hp=hp)
        es = callbacks.EarlyStopping(monitor='val_loss', patience=hp['early_stopping_patience'], restore_best_weights=True)
        hist = model.fit(
            X_train_n, y_train,
            validation_data=(X_val_n, y_val),
            epochs=hp['CV_epochs'],
            batch_size=hp['batch_size'],
            verbose=HYPERPARAMS['model'].get('keras_verbose', 1),
            callbacks=[es]
        )
        best_val = min(hist.history['val_loss'])
        val_losses.append(best_val)
        models_trained.append((model, mu, sigma))
        log(f"Fold {fold_idx+1}/{k_folds} hotov. Nejlepší val_loss={best_val:.6f}")

    best_fold = int(np.argmin(val_losses))
    log(f"CV hotovo. Val loss per fold: {[round(v,6) for v in val_losses]} → vybírám fold {best_fold+1}")
    best_model, best_mu, best_sigma = models_trained[best_fold]
    return best_model, best_mu, best_sigma

def final_train_lstm(X_dev, y_dev):
    """ Finální trénink na celém developmentu; mu/sigma z celého developmentu. """
    log("Finální trénink LSTM na celém development setu...")
    hp = HYPERPARAMS['model']
    mu, sigma = compute_mu_sigma(X_dev)
    X_dev_n = standardize_with(X_dev, mu, sigma)

    model = build_lstm_model(input_shape=X_dev_n.shape[1:], hp=hp)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=hp['early_stopping_patience'], restore_best_weights=True)
    hist = model.fit(
        X_dev_n, y_dev,
        validation_split=0.1,   # malý holdout pro kontrolu ES
        epochs=hp['final_epochs'],
        batch_size=hp['batch_size'],
        verbose=HYPERPARAMS['model'].get('keras_verbose', 1),
        callbacks=[es]
    )
    log(f"Finální trénink hotov. Nejlepší val_loss={min(hist.history['val_loss']):.6f}")
    return model, mu, sigma

def predict_with(model, X, mu, sigma):
    Xn = standardize_with(X, mu, sigma)
    preds = model.predict(Xn, verbose=HYPERPARAMS['model'].get('keras_verbose', 1)).ravel()
    return preds

# ----- Benchmark a PnL -----

def compute_benchmark(df):
    """ Equal-weighted průměrný SimpleReturn napříč všemi akciemi, pro každý den. """
    log("Počítám benchmark (equal-weighted průměr SimpleReturn napříč tickery)...")
    bench = df.groupby('Date')['SimpleReturn'].mean().sort_index()
    return bench

class Position:
    def __init__(self, ric, direction, entry_date, entry_price, tp, sl):
        self.ric = ric
        self.direction = direction  # +1 long, -1 short
        self.entry_date = entry_date
        self.entry_price = float(entry_price)
        self.tp = tp
        self.sl = sl
        self.is_open = True
        self.last_price = entry_price  # pro M2M referenci (Close předchozího dne)
        self.exit_date = None
        self.exit_price = None

    def check_barriers(self, date, high, low, close, priority='SL_first'):
        """ Kontrola zásahu bariér v 'date'.
            Pokud zasáhne, nastaví exit a uzavře pozici s realizovaným výnosem od entry.
            Vrať tuple (closed_today: bool, realized_pnl: float or 0 for today portion).
        """
        if not self.is_open:
            return False, 0.0

        # Bariérové ceny
        if self.direction == +1:  # long
            tp_price = self.entry_price * (1 + self.tp)
            sl_price = self.entry_price * (1 + self.sl)
            hit_tp = high >= tp_price
            hit_sl = low <= sl_price
        else:  # short
            tp_price = self.entry_price * (1 + self.sl)  # pro short je TP při poklesu o 2 % (zrcadlově)
            sl_price = self.entry_price * (1 + self.tp)  # SL při růstu o 2 %
            hit_tp = low <= tp_price
            hit_sl = high >= sl_price

        # Pokud zasáhne obě: použij prioritu
        hit_today = False
        realized = 0.0
        if hit_sl and hit_tp:
            if priority == 'SL_first':
                # konzervativně nejdřív SL
                exit_price = sl_price
            else:
                exit_price = tp_price
            self.exit_date = date
            self.exit_price = exit_price
            self.is_open = False
            hit_today = True
            # Realizovaný celkový výnos od entry do exit:
            # pro long: (exit/entry - 1), pro short: (entry/exit - 1)
            if self.direction == +1:
                realized = (self.exit_price / self.entry_price) - 1.0
            else:
                realized = (self.entry_price / self.exit_price) - 1.0

        elif hit_sl:
            exit_price = sl_price
            self.exit_date = date
            self.exit_price = exit_price
            self.is_open = False
            hit_today = True
            if self.direction == +1:
                realized = (self.exit_price / self.entry_price) - 1.0
            else:
                realized = (self.entry_price / self.exit_price) - 1.0

        elif hit_tp:
            exit_price = tp_price
            self.exit_date = date
            self.exit_price = exit_price
            self.is_open = False
            hit_today = True
            if self.direction == +1:
                realized = (self.exit_price / self.entry_price) - 1.0
            else:
                realized = (self.entry_price / self.exit_price) - 1.0
        else:
            # Bez zásahu: pro M2M denní zisk použijeme Close dne t vůči včerejší referenci:
            # pro long: (Close/last - 1), pro short: (last/Close - 1)
            if self.direction == +1:
                realized = (close / self.last_price) - 1.0
            else:
                realized = (self.last_price / close) - 1.0
            # Posuň referenci pro další den
            self.last_price = close

        return hit_today, realized

def run_strategy(pred_df, ohlc_map, dates_ordered, top_n=10, bottom_n=10, tp=0.02, sl=-0.02, priority='SL_first', phase=None):
    """
    pred_df: DataFrame s predikcemi na úrovni (date_t, RIC, pred, entry OpenAdj_{t+1})
    ohlc_map: dict[RIC] -> DataFrame se sloupci [Date, OpenAdj, HighAdj, LowAdj, CloseAdj] (Sorted)
    dates_ordered: seřazený list unikátních date_t (signálové dny)
    Výstup: series denních PnL strategie (equal-weighted across open positions, M2M) a pomocné info
    """
    if phase:
        log(f"Simuluji obchodní strategii ({phase}) s M2M přeceňováním a bariérami ±2 %...")
    else:
        log("Simuluji obchodní strategii s M2M přeceňováním a bariérami ±2 %...")

    # --- Rychlejší přístup k datům: keše per RIC ---
    ric_dates = {}
    ric_pos_of_date = {}
    ric_ohlc_np = {}
    for ric, df_ric in ohlc_map.items():
        dates_arr = pd.DatetimeIndex(df_ric['Date'].values)
        ric_dates[ric] = dates_arr
        # Mapování date -> pozice řádku (místo searchsorted v hot‑loopu)
        ric_pos_of_date[ric] = {d: i for i, d in enumerate(dates_arr)}
        # OHLC jako NumPy (rychlé čtení)
        ric_ohlc_np[ric] = df_ric[['OpenAdj','HighAdj','LowAdj','CloseAdj']].to_numpy()

    # Omez kalendář jen na potřebné dny: od (min signálu + 1 den) dál
    min_signal_date = pd.to_datetime(pred_df['date_t'].min())
    start_exec_date = (min_signal_date + pd.Timedelta(days=1))
    all_dates = pd.DatetimeIndex(
        np.unique(np.concatenate([ric_dates[ric].values for ric in ric_dates]))
    )
    all_dates = all_dates[all_dates >= start_exec_date].sort_values()
    heartbeat_every = 250  # malé heartbeat logování

    positions = []  # otevřené pozice
    daily_pnl = {}
    # Přes den t (signál v t → vstup na t+1 OpenAdj)
    for t in dates_ordered:
        day_preds = pred_df[pred_df['date_t'] == t]
        day_preds = day_preds.sort_values('pred', ascending=False)

        # 1) Vstupy do nových pozic podle predikcí (na t+1 Open)
        longs = day_preds.head(top_n)
        shorts = day_preds.tail(bottom_n)
        entries = pd.concat([longs.assign(direction=+1), shorts.assign(direction=-1)], axis=0)

        # Přidej nové pozice
        for _, row in entries.iterrows():
            ric = row['RIC']
            entry_day = row['date_t'] + timedelta(days=1)  # t+1
            if ric not in ric_dates:
                continue
            idx = ric_pos_of_date[ric].get(entry_day, None)
            if idx is None:
                # když entry_day není obchodní den, vezmi nejbližší následující
                dates_arr = ric_dates[ric]
                ins = dates_arr.searchsorted(entry_day)
                if ins >= len(dates_arr):
                    continue
                idx = int(ins)
            entry_open, entry_high, entry_low, entry_close = ric_ohlc_np[ric][idx]
            entry_date_eff = ric_dates[ric][idx]
            entry_price = float(entry_open)
            pos = Position(ric=ric, direction=int(row['direction']), entry_date=entry_date_eff,
                           entry_price=entry_price, tp=tp, sl=sl)
            pos.last_price = float(entry_close)
            positions.append(pos)

        # 2) M2M a kontroly bariér budou řešeny v denní smyčce přes all_dates níže.
        pass


    # Mapa: pro každý den vybereme z pred_df nové vstupy, které mají signál včerejška (protože vstup je dnes)
    pred_by_signal = pred_df.groupby('date_t')

    daily_pnl_list = []
    for d in all_dates:
        # Heartbeat log
        if len(daily_pnl_list) % heartbeat_every == 0 and len(daily_pnl_list) > 0:
            log(f"  ...simulace {phase or ''}: zpracováno {len(daily_pnl_list)} dní")
        pnl_today = []

        # 1) Nejprve zkontroluj bariéry a M2M pro existující pozice na den d
        to_remove = []
        for i, pos in enumerate(positions):
            ric = pos.ric
            idx = ric_pos_of_date[ric].get(d, None)
            if idx is None:
                continue
            o, h, l, c = ric_ohlc_np[ric][idx]
            hit_today, realized = pos.check_barriers(
                date=d,
                high=float(h),
                low=float(l),
                close=float(c),
                priority=priority
            )
            pnl_today.append(realized)
            if hit_today and not pos.is_open:
                to_remove.append(i)

        # odstraň uzavřené (od konce, aby se indexy neposunuly)
        for i in reversed(to_remove):
            positions.pop(i)

        # 2) Poté založ nové pozice z dne d-1 (signál včera → vstup dnes na Open)
        yesterday = d - timedelta(days=1)
        if yesterday in pred_by_signal.groups:
            day_preds = pred_by_signal.get_group(yesterday).sort_values('pred', ascending=False)
            entries = pd.concat([
                day_preds.head(top_n).assign(direction=+1),
                day_preds.tail(bottom_n).assign(direction=-1)
            ], axis=0)
            for _, row in entries.iterrows():
                ric = row['RIC']
                if ric not in ric_dates:
                    continue
                idx = ric_pos_of_date[ric].get(d, None)
                if idx is None:
                    continue
                o, h, l, c = ric_ohlc_np[ric][idx]
                entry_price = float(o)
                pos = Position(ric=ric, direction=int(row['direction']), entry_date=d,
                               entry_price=entry_price, tp=tp, sl=sl)
                # první M2M bude vůči Close dne d
                pos.last_price = float(c)
                positions.append(pos)

        # Denní PnL je průměr napříč (top_n + bottom_n) pozicemi? V M2M portfoliu je počet pozic proměnný;
        # použijeme equal-weighted průměr z dnešních příspěvků (pokud dnes není žádná pozice → 0).
        if len(pnl_today) > 0:
            daily_pnl_list.append((d, float(np.mean(pnl_today))))
        else:
            daily_pnl_list.append((d, 0.0))

    pnl_series = pd.Series({d: v for d, v in daily_pnl_list}).sort_index()
    return pnl_series

def make_ohlc_map(df):
    cols = ['Date','OpenAdj','HighAdj','LowAdj','CloseAdj']
    ohlc_map = {}
    for ric, g in df.groupby('RIC'):
        gg = g[cols].dropna().sort_values('Date').reset_index(drop=True)
        ohlc_map[ric] = gg
    return ohlc_map

def sharpe_ratio(daily_returns, risk_free=0.0, periods_per_year=252):
    ex = daily_returns - risk_free/periods_per_year
    mu = ex.mean()
    sd = ex.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0, 0.0
    sr_pd = mu / sd
    sr_pa = sr_pd * np.sqrt(periods_per_year)
    return float(sr_pd), float(sr_pa)

def realized_alpha(strategy_ret, benchmark_ret):
    # OLS: strategy = alpha + beta*benchmark + e
    df = pd.concat([strategy_ret, benchmark_ret], axis=1, keys=['strategy','benchmark']).dropna()
    if len(df) < 10:
        return np.nan, np.nan, np.nan
    X = sm.add_constant(df['benchmark'].values)
    y = df['strategy'].values
    model = sm.OLS(y, X).fit()
    alpha = model.params[0]
    beta = model.params[1]
    # t-stat pro alpha
    try:
        alpha_t = model.tvalues[0]
    except Exception:
        alpha_t = np.nan
    return float(alpha), float(beta), float(alpha_t)

def plot_and_save(strategy_cum, benchmark_cum, title, out_paths):
    plt.figure(figsize=(11,6))
    plt.plot(strategy_cum.index, strategy_cum.values, label='Strategy (cum)')
    plt.plot(benchmark_cum.index, benchmark_cum.values, label='Benchmark (cum)')
    plt.legend()
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)

    saved = None
    for p in out_paths:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            plt.savefig(p, bbox_inches='tight', dpi=150)
            saved = p
            break
        except Exception as e:
            continue
    plt.close()
    return saved

# =========================
# ========= MAIN ==========
# =========================

def main():
    t0 = time.time()
    log("Startuji LSTM SP100 pipeline...")

    # 1) Načtení dat + IDContIndex
    df, vix = load_data()
    df = compute_IDContIndex(df)

    # 2) TA indikátory per IDContIndex
    df = compute_ta_per_idcont(df)

    # 3) Stavba subsekvencí a split
    all_seqs, feat_names = build_all_sequences(df, vix)
    dev, test = split_dev_test(all_seqs)

    X_dev, y_dev, dates_dev, ids_dev, idcont_dev, meta_dev = stack_Xy(dev)
    X_test, y_test, dates_test, ids_test, idcont_test, meta_test = stack_Xy(test)

    log(f"Shapes: X_dev={X_dev.shape}, X_test={X_test.shape} ; počet kanálů={X_dev.shape[-1]} ; feat_names={feat_names + ['VIX_CHG']}")

    # 4) CV trénink (pro získání rozumné inicializace a sanity checku)
    model_cv, mu_cv, sigma_cv = cv_train_lstm(X_dev, y_dev, dates_dev)

    # 5) Finální trénink na celém developmentu
    model, mu_final, sigma_final = final_train_lstm(X_dev, y_dev)

    # 6) Predikce na dev i test
    log("Predikuji na development a test setu...")
    preds_dev = predict_with(model, X_dev, mu_final, sigma_final)
    preds_test = predict_with(model, X_test, mu_final, sigma_final)

    # 7) Sestavení DataFrame s predikcemi (pro strategii)
    pred_dev_df = pd.DataFrame({
        'date_t': dates_dev,
        'RIC': meta_dev['RIC'],
        'pred': preds_dev,
        'OpenAdj_t1': meta_dev['OpenAdj_t1'],
        'HighAdj_t1': meta_dev['HighAdj_t1'],
        'LowAdj_t1': meta_dev['LowAdj_t1'],
        'CloseAdj_t1': meta_dev['CloseAdj_t1'],
        'y_true': y_dev
    }).sort_values(['date_t','pred'], ascending=[True, False])

    pred_test_df = pd.DataFrame({
        'date_t': dates_test,
        'RIC': meta_test['RIC'],
        'pred': preds_test,
        'OpenAdj_t1': meta_test['OpenAdj_t1'],
        'HighAdj_t1': meta_test['HighAdj_t1'],
        'LowAdj_t1': meta_test['LowAdj_t1'],
        'CloseAdj_t1': meta_test['CloseAdj_t1'],
        'y_true': y_test
    }).sort_values(['date_t','pred'], ascending=[True, False])

    # 8) OHLC map pro strategii a benchmark
    ohlc_map = make_ohlc_map(df)
    benchmark = compute_benchmark(df)  # denní benchmark přes celé období

    # 9) Strategie (M2M, bariéry) — zvlášť na development a test (použijeme odpovídající podmnožinu kalendáře)
    # Development signálové dny:
    dev_dates_ordered = np.unique(pred_dev_df['date_t'])
    test_dates_ordered = np.unique(pred_test_df['date_t'])

    strat_cfg = HYPERPARAMS['strategy']
    # Development
    pnl_dev = run_strategy(
        pred_dev_df, ohlc_map, dev_dates_ordered,
        top_n=strat_cfg['top_n'], bottom_n=strat_cfg['bottom_n'],
        tp=strat_cfg['tp'], sl=strat_cfg['sl'], priority=strat_cfg['priority'],
        phase='dev'
    )
    # Test
    pnl_test = run_strategy(
        pred_test_df, ohlc_map, test_dates_ordered,
        top_n=strat_cfg['top_n'], bottom_n=strat_cfg['bottom_n'],
        tp=strat_cfg['tp'], sl=strat_cfg['sl'], priority=strat_cfg['priority'],
        phase='test'
    )

    # 10) Vyhodnocení (Sharpe)
    sr_pd_dev, sr_pa_dev = sharpe_ratio(pnl_dev)
    sr_pd_test, sr_pa_test = sharpe_ratio(pnl_test)

    log(f"Sharpe_pd (dev) = {sr_pd_dev:.4f}, Sharpe_pa (dev) = {sr_pa_dev:.4f}")
    log(f"Sharpe_pd (test) = {sr_pd_test:.4f}, Sharpe_pa (test) = {sr_pa_test:.4f}")

    # 11) Realizovaná alfa (na test sample)
    # Sladíme benchmark s horizontem pnl_test (jejich průnik indexů)
    bench_test = benchmark.reindex(pnl_test.index).fillna(0.0)
    alpha, beta, alpha_t = realized_alpha(pnl_test, bench_test)
    log(f"Realizovaná alfa na testu: alpha={alpha:.6f} (t={alpha_t:.2f}), beta={beta:.4f}")

    # 12) Kumulativní výnosy a graf
    strat_cum_dev = (1 + pnl_dev).cumprod() - 1.0
    strat_cum_test = (1 + pnl_test).cumprod() - 1.0
    bench_cum = (1 + benchmark).cumprod() - 1.0

    # spojíme strategii dev+test pro souvislý graf strategie; benchmark je přes celé období
    strat_cum = pd.concat([strat_cum_dev, strat_cum_test[~strat_cum_test.index.isin(strat_cum_dev.index)]], axis=0).sort_index()

    saved_png = plot_and_save(
        strategy_cum=strat_cum,
        benchmark_cum=bench_cum.reindex(strat_cum.index).fillna(method='ffill').fillna(0.0),
        title="LSTM Strategy vs. Benchmark (Cumulative Returns)",
        out_paths=HYPERPARAMS['output']['png_paths']
    )
    if saved_png:
        log(f"Uloženo PNG: {saved_png}")
    else:
        log("Nepodařilo se uložit PNG (zkontroluj cesty v HYPERPARAMS['output']['png_paths']).")

    # 13) Finální summary
    t1 = time.time()
    elapsed = t1 - t0
    log("====== SUMMARY ======")
    log(f"Vzorky: dev={len(pred_dev_df):,}, test={len(pred_test_df):,}, okno h={HYPERPARAMS['features']['window']}, kanály={X_dev.shape[-1]}")
    log(f"Sharpe_dev pd={sr_pd_dev:.4f}, pa={sr_pa_dev:.4f}")
    log(f"Sharpe_test pd={sr_pd_test:.4f}, pa={sr_pa_test:.4f}")
    log(f"Alpha_test={alpha:.6f} (t={alpha_t:.2f}), Beta_test={beta:.4f}")
    log(f"Doba běhu: {elapsed/60:.2f} min")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Chyba: {repr(e)}")
        raise
