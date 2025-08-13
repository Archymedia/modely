# RNN_mac.py — 3rd generation
# ------------------------------------------------------------
# Cíl: Predikce z-score simple returnu následujícího dne pro akcie v SP100
# Model: RNN (Elman)
# Pipeline: zachovává princip MLP (data loading → feature engineering →
#           tvorba sekvencí → trénink s CV → trénink finálního modelu → backtest
#           self-financing L/S s PT/SL bariérami)
# Pozn.: drtivá většina je shodná s MLP, mění se pouze část "model build + data do sekvencí".
# Validace dle kalendáře (posledních X % dnů), gradient clipping, bez leakáže mezi tickery.
# ------------------------------------------------------------

import os
import math
import json
import time
import random
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# Logging and Timer helpers
# ------------------------------------------------------------
from contextlib import contextmanager
import datetime

def log(msg: str):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{now}] {msg}")

@contextmanager
def Timer(name: str):
    t0 = time.perf_counter()
    log(f"{name}…")
    yield
    dt = time.perf_counter() - t0
    log(f"{name} hotovo za {int(dt//60)} min {int(dt%60)} s")

# ------------------------------------------------------------
# HYPERPARAMS (část společná s MLP nahoře beze změn + část specifická pro RNN)
# ------------------------------------------------------------
HYPERPARAMS = {
    'data': {
        'data_path': [
            "/Users/lindawaisova/Desktop/DP/data/SP_100/READY_DATA/9DATA_FINAL.csv",
            r"C:\\Users\\david\\Desktop\\SP100\\9DATA_FINAL.csv",
            "/mnt/data/9DATA_FINAL.csv"
        ],
        'vix_path': [
            "/Users/lindawaisova/Desktop/DP/data/SP_100/READY_DATA/VIX_2005_2023.csv",
            r"C:\\Users\\david\\Desktop\\SP100\\VIX_2005_2023.csv",
            "/mnt/data/VIX_2005_2023.csv"
        ],
        'train_end_date': '2020-12-31',
        'test_start_date': '2021-01-01',
        'output_png': "/Users/lindawaisova/Desktop/DP/Git/DP/modely/3rd generation/RNN_dashboard_v1.png",
        'output_fallback_png': r"C:\\Users\\david\\Desktop\\lindaPy\\RNN_dashboard_v2.png",
    },
    'final_model_epochs': 2,
    'cv_epochs': 2,
    'patience': 1,
    'cv_folds': 2,
    'n_iter': 1,
    'search_space': {
        'layers': [1, 2],
        'units': [128, 256],
        'learning_rate': [1e-4, 3e-4, 1e-3]
    },
    'fixed_params': {
        'batch_size': 256,
        'dropout_rate': 0.1,
        'l2_reg': 1e-4,
        'random_seed': 42,
        'val_fraction_for_final': 0.15,
        'max_grad_norm': 1.0,
    },
    # Strategie – stejná jako v MLP
    'strategy': {
        'n_long': 10,
        'n_short': 10,
        'pt_pct': 0.02,
        'sl_pct': 0.02,
        'rf_annual': 0.02,
        'trading_days': 252,
        'heartbeat_days': 50,
    },
    # Specifické pro RNN
    'rnn_specific': {
        'seq_len': 20,
        'bidirectional': False # nechat takto, jinak bude leakage
    }
}

# ------------------------------------------------------------
# Pomocné funkce
# ------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def first_existing_path(paths: List[str]) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    msg = f"Soubor nenalezen na žádné z cest: {paths}"
    log(msg)
    raise FileNotFoundError(msg)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    dp = first_existing_path(HYPERPARAMS['data']['data_path'])
    vp = first_existing_path(HYPERPARAMS['data']['vix_path'])

    sp = pd.read_csv(dp)
    vix = pd.read_csv(vp)

    sp['Date'] = pd.to_datetime(sp['Date'])
    vix['Date'] = pd.to_datetime(vix['Date'])

    # Sloupce dle zadání (SP100: ID RIC Name Date TotRet SimpleReturn OpenAdj HighAdj LowAdj CloseAdj Close Volume VolumeAdj VolumeUSDadj)
    # Ujistíme se o správných typech
    sp = sp.sort_values(['ID', 'Date']).reset_index(drop=True)
    vix = vix.sort_values('Date').reset_index(drop=True)

    return sp, vix


def add_vix(sp: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    vix_small = vix[['Date', 'Close', 'VIX_Change', 'VIX_Change_Abs']].rename(
        columns={'Close': 'VIX_Close'})
    df = sp.merge(vix_small, on='Date', how='left')
    df[['VIX_Close', 'VIX_Change', 'VIX_Change_Abs']] = df[['VIX_Close', 'VIX_Change', 'VIX_Change_Abs']].ffill()
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Základní prediktory – "nejužitečnější" a přitom jednoduché a rychlé
    # 1) Minulá návratnost a volatilita
    df['ret_1'] = df.groupby('ID')['SimpleReturn'].shift(1)
    df['ret_5'] = df.groupby('ID')['SimpleReturn'].rolling(5).mean().reset_index(level=0, drop=True).shift(1)
    df['vol_20'] = df.groupby('ID')['SimpleReturn'].rolling(20).std().reset_index(level=0, drop=True).shift(1)

    # 2) Range a objemové informace
    df['hl_range'] = (df['HighAdj'] - df['LowAdj']) / df['CloseAdj']
    df['hl_range_5'] = df.groupby('ID')['hl_range'].rolling(5).mean().reset_index(level=0, drop=True).shift(1)
    # Correct rolling window for volchg_5 to avoid cross-ID leakage and use only past data
    df['volchg_5'] = (
        df.groupby('ID')['VolumeAdj']
          .apply(lambda s: s.pct_change().rolling(5).mean())
          .reset_index(level=0, drop=True)
          .shift(1)
    )

    # 3) VIX informace (současný stav trhu)
    df['VIX_Close_5'] = df['VIX_Close'].rolling(5).mean().shift(1)
    df['VIX_Change_5'] = df['VIX_Change'].rolling(5).mean().shift(1)

    # Leakage guard: all rolling features use only past information via shift(1)
    # (effectively a left-closed window). Computations are done per-ID.
    # Target: z-score SimpleReturn následujícího dne (per-ID)
    grp = df.groupby('ID', group_keys=False)
    next_ret = grp['SimpleReturn'].shift(-1)
    mu = grp['SimpleReturn'].rolling(252).mean().reset_index(level=0, drop=True)
    sd = grp['SimpleReturn'].rolling(252).std().reset_index(level=0, drop=True)
    df['target_z'] = (next_ret - mu) / sd

    # Odstranit řádky bez targetu nebo bez featur
    feature_cols = [
        'ret_1', 'ret_5', 'vol_20', 'hl_range_5', 'volchg_5',
        'VIX_Close_5', 'VIX_Change_5'
    ]
    df = df.dropna(subset=feature_cols + ['target_z', 'OpenAdj', 'HighAdj', 'LowAdj', 'CloseAdj'])
    return df


def scale_features_per_id(df: pd.DataFrame, feature_cols: List[str], vix_cols: List[str], train_end: pd.Timestamp) -> Tuple[pd.DataFrame, Dict[str, StandardScaler], StandardScaler]:
    """
    Per-ID standardizace hlavních featur (fit jen na train část daného ID).
    VIX featury se škálují globálně scalerem fitnutým jen na train datech (≤ train_end).
    Vrací (df_scaled, scalers_per_id, vix_scaler).
    """
    df = df.sort_values(['ID', 'Date']).reset_index(drop=True).copy()
    # 1) VIX scaler (globální, jen z train)
    if vix_cols:
        vix_scaler = StandardScaler().fit(df.loc[df['Date'] <= train_end, vix_cols].values)
        df.loc[:, vix_cols] = vix_scaler.transform(df.loc[:, vix_cols].values)
    else:
        vix_scaler = None
    # 2) Per-ID scalery pro hlavní featury (fit jen na train část daného ID)
    scalers: Dict[str, StandardScaler] = {}
    out = []
    for sec_id, g in df.groupby('ID', sort=False):
        g = g.sort_values('Date').copy()
        tr_mask = g['Date'] <= train_end
        sc = StandardScaler().fit(g.loc[tr_mask, feature_cols].values)
        g.loc[:, feature_cols] = sc.transform(g.loc[:, feature_cols].values)
        scalers[sec_id] = sc
        out.append(g)
    df_scaled = pd.concat(out, axis=0).sort_values(['ID', 'Date']).reset_index(drop=True)
    return df_scaled, scalers, vix_scaler

# ------------------------------------------------------------
# Dataset se sekvencemi pro RNN
# ------------------------------------------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_sequences(df: pd.DataFrame, feature_cols: List[str], seq_len: int,
                    start_date: Optional[pd.Timestamp], end_date: Optional[pd.Timestamp]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Vytvoří sekvence pro RNN. Očekává, že feature_cols jsou již škálované (per-ID/VIX global).
    """
    # Filtrace dle dat
    if start_date is not None:
        df = df[df['Date'] >= start_date]
    if end_date is not None:
        df = df[df['Date'] <= end_date]

    # Seřazení
    df = df.sort_values(['ID', 'Date'])

    # Očekáváme, že feature_cols jsou již škálované předem (per-ID / VIX global)
    df_feat = df[['ID', 'Date', 'target_z'] + feature_cols].copy()

    X_list, y_list, rows_meta = [], [], []

    # Tvorba sekvencí per-ID, aby se nepřelévaly mezi akciemi (řeší survivorship logiku)
    for _, g in df_feat.groupby('ID'):
        g = g.sort_values('Date').reset_index(drop=True)
        F = g[feature_cols].values
        y = g['target_z'].values
        for t in range(len(g) - seq_len):
            X_list.append(F[t:t+seq_len, :])
            y_list.append(y[t+seq_len])
            rows_meta.append(g.iloc[t+seq_len])

    X = np.stack(X_list) if X_list else np.empty((0, seq_len, len(feature_cols)))
    y = np.array(y_list) if y_list else np.empty((0,))
    meta = pd.DataFrame(rows_meta)
    return X, y, meta


# ------------------------------------------------------------
# RNN model
# ------------------------------------------------------------
class RNNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float,
                 bidirectional: bool = False):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity='tanh',
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1)
        )

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


# ------------------------------------------------------------
# Trénink + validace (s gradient clipping)
# ------------------------------------------------------------

def train_one_model(X_train, y_train, X_val, y_val, params: Dict, epochs: int, patience: int,
                    weight_decay: float, max_grad_norm: float, batch_size: int, device: str = 'cpu') -> Tuple[nn.Module, float, Dict[str, List[float]]]:
    train_ds = SeqDataset(X_train, y_train)
    val_ds = SeqDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = RNNRegressor(
        input_dim=X_train.shape[2],
        hidden_dim=params['units'],
        num_layers=params['layers'],
        dropout=HYPERPARAMS['fixed_params']['dropout_rate'],
        bidirectional=HYPERPARAMS['rnn_specific']['bidirectional']
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    for ep in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            epoch_losses.append(loss.item())

        epoch_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')

        # Val
        model.eval()
        with torch.no_grad():
            vs = []
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                vs.append(loss_fn(pred, yb).item())
            val_loss = float(np.mean(vs)) if vs else float('inf')

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val, history





# ------------------------------------------------------------
# Backtest: self-financing L/S s PT/SL bariérami (2%)
# ------------------------------------------------------------

def compute_barrier_return(next_open, next_high, next_low, next_close, direction: int, pt: float, sl: float):
    # direction: +1 long, -1 short
    # Pracujeme relativně k open: PT/SL se vztahuje k open ceny následujícího dne
    if np.isnan(next_open) or np.isnan(next_high) or np.isnan(next_low) or np.isnan(next_close):
        return np.nan

    if direction == 1:  # long
        # PT dosažen, pokud high >= open*(1+pt)
        pt_price = next_open * (1 + pt)
        sl_price = next_open * (1 - sl)
        if next_high >= pt_price:
            return pt  # přibližně (open->pt)/open
        if next_low <= sl_price:
            return -sl
        return (next_close - next_open) / next_open
    else:  # short
        pt_price = next_open * (1 - pt)  # u short je profit když cena klesne o pt
        sl_price = next_open * (1 + sl)
        if next_low <= pt_price:
            return pt
        if next_high >= sl_price:
            return -sl
        return (next_open - next_close) / next_open


def backtest_daily(meta: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    # meta: sloupce [ID, Date, OpenAdj, HighAdj, LowAdj, CloseAdj] přidáme ještě next-day hodnoty
    stg = HYPERPARAMS['strategy']
    nL, nS = stg['n_long'], stg['n_short']
    pt, sl = stg['pt_pct'], stg['sl_pct']
    rf_daily = (1 + stg['rf_annual']) ** (1 / stg['trading_days']) - 1

    dfp = meta.copy()
    dfp['pred'] = preds

    # Sloupcová posunutí na další den pro výpočet realizace
    dfp['Open_next'] = dfp.groupby('ID')['OpenAdj'].shift(-1)
    dfp['High_next'] = dfp.groupby('ID')['HighAdj'].shift(-1)
    dfp['Low_next'] = dfp.groupby('ID')['LowAdj'].shift(-1)
    dfp['Close_next'] = dfp.groupby('ID')['CloseAdj'].shift(-1)

    # Každý den vybereme top/bottom podle pred, obchodujeme následující den od Open
    results = []
    for d, g in dfp.groupby('Date'):
        g = g.dropna(subset=['Open_next', 'High_next', 'Low_next', 'Close_next'])
        if len(g) < (nL + nS):
            continue
        long_ids = g.nlargest(nL, 'pred')
        short_ids = g.nsmallest(nS, 'pred')

        long_ret = long_ids.apply(lambda r: compute_barrier_return(r['Open_next'], r['High_next'], r['Low_next'], r['Close_next'], 1, pt, sl), axis=1).mean()
        short_ret = short_ids.apply(lambda r: compute_barrier_return(r['Open_next'], r['High_next'], r['Low_next'], r['Close_next'], -1, pt, sl), axis=1).mean()
        # self-financing: průměr long + průměr short (obě strany stejně velké)
        port_ret = 0.5 * (long_ret + short_ret)
        # excess vs rf
        excess = port_ret - rf_daily
        results.append({'Date': d, 'ret': port_ret, 'excess': excess})

    bt = pd.DataFrame(results).sort_values('Date').reset_index(drop=True)
    bt['cum_ret'] = (1 + bt['ret']).cumprod() - 1
    bt['cum_excess'] = (1 + bt['excess']).cumprod() - 1
    return bt


#
# ------------------------------------------------------------
# Dashboard (MLP-like PNG)
# ------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def max_drawdown_curve(r: pd.Series):
    curve = (1 + r.fillna(0)).cumprod()
    peak = curve.cummax()
    dd = (curve/peak) - 1.0
    return float(dd.min()), curve

def table(ax, rows, title, pad=6):
    ax.axis('off')
    tbl = ax.table(cellText=rows, loc='center')
    for (row, col), cell in tbl.get_celld().items():
        cell.set_text_props(ha='center', va='center')
        cell.set_height(cell.get_height() * 1.2)
        cell.set_width(cell.get_width() * 0.85)
        txt = cell.get_text().get_text()
        # šedé pozadí pro nenumerické
        try:
            float(str(txt).replace('%','').replace(',',''))
            is_num = True
        except Exception:
            is_num = False
        if not is_num:
            cell.set_facecolor('#f0f0f0')
    ax.set_title(title, pad=pad)

def make_png(path, bt_test: pd.DataFrame, cv_histories: List[Dict], final_history: Dict[str, List[float]], summary: Dict[str, float]):
    fig = plt.figure(figsize=(18,12), dpi=150, constrained_layout=True)
    # Grid 8x4 (zjednodušený, ale rozložení jako v MLP)
    ax1 = plt.subplot2grid((8,4),(0,0), colspan=4, rowspan=2)  # Cum PnL (test)
    _, curve_test = max_drawdown_curve(bt_test['ret'])
    ax1.plot(bt_test['Date'], curve_test.values, label='Strategie (test)')
    ax1.set_title("Kumulativní výnos — TEST")
    ax1.legend(loc='best')

    # CV MSE křivky (průměr přes kandidáty)
    ax2 = plt.subplot2grid((8,4),(2,0), colspan=2, rowspan=1)
    if cv_histories:
        max_len = max(len(h['val_loss']) for h in cv_histories if 'val_loss' in h)
        mean_val = np.zeros(max_len); cnt = np.zeros(max_len)
        for h in cv_histories:
            vl = np.array(h.get('val_loss', []), dtype=float)
            for i in range(len(vl)):
                mean_val[i] += vl[i]; cnt[i] += 1
        mean_val = np.divide(mean_val, np.maximum(cnt, 1))
        ax2.plot(range(1, len(mean_val)+1), mean_val, label='CV Val MSE (průměr)')
    ax2.set_title("CV — MSE (epocha)"); ax2.set_xlabel("Epocha"); ax2.set_ylabel("MSE"); ax2.legend(loc='best')

    # Finální trénink — MSE křivky
    ax3 = plt.subplot2grid((8,4),(2,2), colspan=2, rowspan=1)
    if final_history and final_history.get('val_loss'):
        ax3.plot(range(1, len(final_history['val_loss'])+1), final_history['val_loss'], label='Val MSE')
    if final_history and final_history.get('train_loss'):
        ax3.plot(range(1, len(final_history['train_loss'])+1), final_history['train_loss'], label='Train MSE')
    ax3.set_title("Finální model — MSE (epocha)"); ax3.set_xlabel("Epocha"); ax3.set_ylabel("MSE"); ax3.legend(loc='best')

    # Tabulky metrik
    ax4 = plt.subplot2grid((8,4),(3,0), colspan=2, rowspan=2)
    reg_tbl = [["", "Test"],
               ["RMSE",  f"{summary.get('rmse', np.nan):.4f}"],
               ["Ann. Return", f"{summary.get('ann_ret', np.nan):.4f}"],
               ["Sharpe", f"{summary.get('sharpe', np.nan):.4f}"],
               ["Ann. Volatility", f"{summary.get('ann_vol', np.nan):.4f}"]]
    table(ax4, reg_tbl, "Výnosové metriky")

    # Rezervní tabulka pro info o CV/konfiguraci
    ax5 = plt.subplot2grid((8,4),(3,2), colspan=2, rowspan=2)
    best_cfg = summary.get('best_cfg', {})
    cfg_rows = [["Hyperparametry", ""]] + [[str(k), str(v)] for k,v in (best_cfg.items() if isinstance(best_cfg, dict) else [])]
    if len(cfg_rows)==1: cfg_rows.append(["—","—"])
    table(ax5, cfg_rows, "Nejlepší konfigurace")

    # Poznámka/drobnosti
    ax6 = plt.subplot2grid((8,4),(5,0), colspan=4, rowspan=3)
    ax6.axis('off')
    ax6.text(0.01, 0.9, "Pozn.: Dashboard harmonizován s MLP (časově bezpečné splitování, per‑ID škálování).", fontsize=10)
    ax6.text(0.01, 0.75, f"N test dnů: {summary.get('n_test_days', 0)}", fontsize=10)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    set_seed(HYPERPARAMS['fixed_params']['random_seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with Timer("Načítám data"):
        sp, vix = load_data()
        df = add_vix(sp, vix)
        df = feature_engineering(df)

    # Stejné škálování jako v MLP: VIX globálně (fit na train), hlavní featury per-ID (fit na train-ID)
    feature_cols = ['ret_1', 'ret_5', 'vol_20', 'hl_range_5']
    vix_cols = ['VIX_Close_5', 'VIX_Change_5']
    train_end = pd.to_datetime(HYPERPARAMS['data']['train_end_date'])
    df, per_id_scalers, vix_scaler = scale_features_per_id(df, feature_cols, vix_cols, train_end)

    # Featury a data split
    feature_cols = ['ret_1', 'ret_5', 'vol_20', 'hl_range_5', 'volchg_5', 'VIX_Close_5', 'VIX_Change_5']
    seq_len = HYPERPARAMS['rnn_specific']['seq_len']

    train_end = pd.to_datetime(HYPERPARAMS['data']['train_end_date'])
    test_start = pd.to_datetime(HYPERPARAMS['data']['test_start_date'])

    # Train sekvence
    X_train, y_train, meta_train = build_sequences(
        df, feature_cols, seq_len,
        start_date=None, end_date=train_end
    )

    # Pro CV potřebujeme datum na konci sekvence (meta_train['Date'])
    dates_train = meta_train['Date'].values.astype('datetime64[D]')

    # Calendar-based validation split (globálně podle data) pro výběr hyperparametrů – stejně jako v MLP
    param_grid = {
        'layers': HYPERPARAMS['search_space']['layers'],
        'units': HYPERPARAMS['search_space']['units'],
        'learning_rate': HYPERPARAMS['search_space']['learning_rate'],
    }
    sampler = list(ParameterSampler(param_grid, n_iter=HYPERPARAMS['n_iter'], random_state=HYPERPARAMS['fixed_params']['random_seed']))

    val_frac = HYPERPARAMS['fixed_params']['val_fraction_for_final']
    dates_np = meta_train['Date'].values.astype('datetime64[D]')
    uniq_dates = np.array(sorted(np.unique(dates_np)))
    cutoff_idx = int(np.floor((1 - val_frac) * len(uniq_dates)))
    cutoff_date = uniq_dates[max(0, min(len(uniq_dates) - 1, cutoff_idx))]
    tr_mask_cv = dates_np < cutoff_date
    va_mask_cv = dates_np >= cutoff_date
    log(f"CV split cutoff date: {pd.to_datetime(str(cutoff_date)).date()}")

    best_cfg, best_score = None, float('inf')
    cv_histories = []

    with Timer("Random search na jednom time-safe CV splitu (MLP styl)"):
        for i, cfg in enumerate(sampler, 1):
            X_tr, y_tr = X_train[tr_mask_cv], y_train[tr_mask_cv]
            X_va, y_va = X_train[va_mask_cv], y_train[va_mask_cv]
            model, vloss, history = train_one_model(
                X_tr, y_tr, X_va, y_va,
                params=cfg,
                epochs=HYPERPARAMS['cv_epochs'],
                patience=HYPERPARAMS['patience'],
                weight_decay=HYPERPARAMS['fixed_params']['l2_reg'],
                max_grad_norm=HYPERPARAMS['fixed_params']['max_grad_norm'],
                batch_size=HYPERPARAMS['fixed_params']['batch_size'],
                device=device
            )
            cv_histories.append({'cfg': cfg, 'train_loss': history['train_loss'], 'val_loss': history['val_loss']})
            if vloss < best_score:
                best_score = vloss
                best_cfg = cfg

    log(f"Best CV cfg: {best_cfg}, score: {best_score:.6f}")

    # Finální train/val split (stejné cutoff_date jako u CV)
    tr_mask_final = dates_np < cutoff_date
    va_mask_final = dates_np >= cutoff_date
    log(f"Final train/val split cutoff date: {pd.to_datetime(str(cutoff_date)).date()}")

    X_tr, y_tr = X_train[tr_mask_final], y_train[tr_mask_final]
    X_va, y_va = X_train[va_mask_final], y_train[va_mask_final]

    with Timer("Trénink finálního modelu"):
        final_model, _, final_history = train_one_model(
            X_tr, y_tr, X_va, y_va,
            params=best_cfg,
            epochs=HYPERPARAMS['final_model_epochs'],
            patience=HYPERPARAMS['patience'],
            weight_decay=HYPERPARAMS['fixed_params']['l2_reg'],
            max_grad_norm=HYPERPARAMS['fixed_params']['max_grad_norm'],
            batch_size=HYPERPARAMS['fixed_params']['batch_size'],
            device=device
        )

    # Test sekvence
    X_test, y_test, meta_test = build_sequences(
        df, feature_cols, seq_len,
        start_date=test_start, end_date=None
    )

    with Timer("Predikce na testu"):
        final_model.eval()
        with torch.no_grad():
            test_pred = []
            bs = HYPERPARAMS['fixed_params']['batch_size']
            for i in range(0, len(X_test), bs):
                xb = torch.tensor(X_test[i:i+bs]).to(device)
                yp = final_model(xb).cpu().numpy()
                test_pred.append(yp)
            test_pred = np.concatenate(test_pred) if len(test_pred) else np.array([])

    # Backtest
    # Přidej do meta_test nutné sloupce z původního df (OpenAdj, HighAdj, LowAdj, CloseAdj)
    meta_test = meta_test.merge(
        df[['ID', 'Date', 'OpenAdj', 'HighAdj', 'LowAdj', 'CloseAdj']],
        on=['ID', 'Date'], how='left'
    )

    with Timer("Backtest strategie"):
        bt = backtest_daily(meta_test[['ID','Date','OpenAdj','HighAdj','LowAdj','CloseAdj']], test_pred)

    with Timer("Výpočet metrik"):
        # Shrnutí metrik
        rmse = float(np.sqrt(mean_squared_error(y_test, test_pred))) if len(test_pred) else float('nan')
        ann_ret = (1 + bt['ret']).prod() ** (252 / max(1, len(bt))) - 1 if len(bt) else np.nan
        ann_vol = np.std(bt['ret']) * np.sqrt(252) if len(bt) else np.nan
        sharpe = ann_ret / ann_vol if (ann_vol and ann_vol != 0 and not np.isnan(ann_vol)) else np.nan

        summary = {
            'rmse': rmse,
            'ann_ret': ann_ret,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'n_test_days': int(len(bt))
        }
        summary['best_cfg'] = best_cfg
        final_history_local = final_history  # from final training
        log(json.dumps(summary, indent=2, default=float))

    try:
        outp = HYPERPARAMS['data']['output_png']
        make_png(outp, bt, cv_histories, final_history_local, summary)
        log(f"Dashboard uložen do: {outp}")
    except Exception as e:
        log(f"Chyba při tvorbě dashboardu: {e}")
        try:
            outp = HYPERPARAMS['data']['output_fallback_png']
            make_png(outp, bt, cv_histories, final_history_local, summary)
            log(f"Dashboard uložen do (fallback): {outp}")
        except Exception as e2:
            log(f"Chyba i ve fallbacku: {e2}")


if __name__ == '__main__':
    main()
