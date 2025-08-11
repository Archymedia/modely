# -*- coding: utf-8 -*-
"""
S&P100 DAILY — MLP PIPELINE (golden baseline, v3)
Changes in v3:
- Cumulative returns plotted for FULL period (train+test), with split line.
- Metrics tables now include TRAIN and TEST (regression, trading, realized alpha).
- PNG saved to a user-defined absolute path (with safe fallback).

Run on Python 3.11 with:
  numpy==1.24.3 pandas==2.0.3 matplotlib==3.7.2 scikit-learn==1.3.0
  tensorflow-macos==2.13.0 tensorflow-metal==1.1.0 scikeras==0.12.0
  scipy==1.11.1 statsmodels==0.14.2
"""
import os, sys, time, math, warnings
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import statsmodels.api as sm
warnings.filterwarnings("ignore")

# ===== HYPERPARAMETER CONFIGURATION =====
HYPERPARAMS = {
    'data': {
        'data_path': ["/Users/lindawaisova/Desktop/DP/data/SP_100/READY_DATA/9DATA_FINAL.csv",
                      r"C:\Users\david\Desktop\SP100\9DATA_FINAL.csv",
                      "/mnt/data/9DATA_FINAL.csv"],
        'vix_path':  ["/Users/lindawaisova/Desktop/DP/data/SP_100/READY_DATA/VIX_2005_2023.csv",
                      r"C:\Users\david\Desktop\SP100\VIX_2005_2023.csv",
                      "/mnt/data/VIX_2005_2023.csv"],
        'train_end_date': '2020-12-31',
        'test_start_date': '2021-01-01',
    },
    'final_model_epochs': 2,
    'cv_epochs': 2,
    'patience': 1,
    'cv_folds': 2,
    'n_iter': 1,
    'search_space': {
        'layers': [1, 2],
        'units': [64, 128, 256],
        'learning_rate': [1e-4, 3e-4, 1e-3]
    },
    'fixed_params': {
        'batch_size': 128,
        'dropout_rate': 0.15,
        'l2_reg': 3e-4,
        'random_seed': 42,
    },
    'strategy': {
        'n_long': 10,
        'n_short': 10,
        'pt_pct': 0.02,
        'sl_pct': 0.02,
        'rf_annual': 0.02,
        'fast_backtest': True
    },
    'output_png': "/Users/lindawaisova/Desktop/DP/Git/DP/modely/3rd generation/MLP_dashboard.png"
}

def _resolve_first_existing(paths: List[str]) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the provided paths exist: {paths}")

def _parse_dates(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors='coerce', dayfirst=True).dt.tz_localize(None)

def _print_section(title: str):
    print("\n" + "="*26 + f" {title} " + "="*26)

# ===== START =====
_t0 = time.time()
np.random.seed(HYPERPARAMS['fixed_params']['random_seed'])
keras.utils.set_random_seed(HYPERPARAMS['fixed_params']['random_seed'])

# 1) Load
_print_section("1. Data Loading and Basic Checks")
stocks_csv = _resolve_first_existing(HYPERPARAMS['data']['data_path'])
vix_csv    = _resolve_first_existing(HYPERPARAMS['data']['vix_path'])
print(f"Stocks CSV: {stocks_csv}")
print(f"VIX    CSV: {vix_csv}")
stocks = pd.read_csv(stocks_csv)
vix    = pd.read_csv(vix_csv)
stocks['Date'] = _parse_dates(stocks, 'Date')
vix_date_col = 'Date' if 'Date' in vix.columns else 'Date.1'
vix[vix_date_col] = pd.to_datetime(vix[vix_date_col], errors='coerce').dt.tz_localize(None)
print("Stocks date range:", stocks['Date'].min().date(), "->", stocks['Date'].max().date())

# 2) Features
_print_section("2. Feature Engineering")
stocks = stocks.sort_values(['ID','Date']).reset_index(drop=True)

def compute_indicators(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    for L in [1,2,3,4,5]:
        g[f'ret_lag_{L}'] = g['SimpleReturn'].shift(L)
    for n in [5,10,20]:
        g[f'sma_{n}'] = g['CloseAdj'].rolling(n, min_periods=n).mean()
        g[f'ema_{n}'] = g['CloseAdj'].ewm(span=n, adjust=False, min_periods=n).mean()
    ema12 = g['CloseAdj'].ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = g['CloseAdj'].ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    g['macd'] = macd
    g['macd_signal'] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    def rsi(series, period=14):
        d = series.diff()
        up = d.clip(lower=0); down = -d.clip(upper=0)
        ma_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        ma_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        rs = ma_up / (ma_down + 1e-12); return 100 - (100/(1+rs))
    g['rsi_14'] = rsi(g['CloseAdj'],14); g['rsi_7'] = rsi(g['CloseAdj'],7)
    win=20; sma20=g['CloseAdj'].rolling(win, min_periods=win).mean(); std20=g['CloseAdj'].rolling(win, min_periods=win).std()
    g['bb_upper']=sma20+2*std20; g['bb_lower']=sma20-2*std20
    g['bb_bw']=(g['bb_upper']-g['bb_lower'])/(sma20+1e-12)
    g['bb_percB']=(g['CloseAdj']-g['bb_lower'])/((g['bb_upper']-g['bb_lower'])+1e-12)
    pc=g['CloseAdj']; prev=pc.shift(1)
    tr=pd.concat([(g['HighAdj']-g['LowAdj']), (g['HighAdj']-prev).abs(), (g['LowAdj']-prev).abs()],axis=1).max(axis=1)
    g['atr_14']=tr.rolling(14, min_periods=14).mean()
    for n in [10,20]: g[f'hv_{n}']=g['SimpleReturn'].rolling(n, min_periods=n).std()
    sign=np.sign(g['CloseAdj'].diff()); g['obv']=(sign.fillna(0)*g['Volume']).cumsum()
    for n in [5,10]:
        g[f'vroc_{n}']=(g['Volume']-g['Volume'].shift(n))/(g['Volume'].shift(n)+1e-12)
        g[f'roc_{n}']=(g['CloseAdj']-g['CloseAdj'].shift(n))/(g['CloseAdj'].shift(n)+1e-12)
    for n in [20,50,250]:
        s=g['CloseAdj'].rolling(n, min_periods=n).mean()
        g[f'dist_sma_{n}']= (g['CloseAdj']/(s+1e-12))-1.0
        for k in [4,3,2,1,0]:
            g[f'close_div_sma{n}_tminus{k}']=(g['CloseAdj'].shift(k)/(s.shift(k)+1e-12))-1.0
    return g

stocks = stocks.groupby('ID', group_keys=False).apply(compute_indicators)
vix_small = vix[[vix_date_col, 'Close']].rename(columns={vix_date_col:'Date','Close':'VIX_Close'})
stocks = stocks.merge(vix_small, on='Date', how='left')
stocks['VIX_Close_ffill']=stocks['VIX_Close'].ffill()
stocks['VIX_SMA5']=stocks['VIX_Close_ffill'].rolling(5, min_periods=1).mean()
stocks['is_vix_missing']=(stocks['VIX_Close'].isna()).astype(int)
stocks['volume_raw']=stocks['Volume']

# 3) Splits
_print_section("3. Time Splits")
train_end = pd.to_datetime(HYPERPARAMS['data']['train_end_date'])
test_start= pd.to_datetime(HYPERPARAMS['data']['test_start_date'])
stocks['is_train']=(stocks['Date']<=train_end)
stocks['is_test']=(stocks['Date']>=test_start)
unique_train_dates = np.sort(stocks.loc[stocks['is_train'],'Date'].unique())
k=HYPERPARAMS['cv_folds']
fold_bounds = np.array_split(unique_train_dates, k)
cv_date_folds=[]
for i in range(k):
    val_dates = fold_bounds[i]
    if len(val_dates)==0: continue
    tr_dates = unique_train_dates[unique_train_dates < val_dates.min()]
    if len(tr_dates)==0: continue
    cv_date_folds.append((tr_dates, val_dates))
print(f"Constructed {len(cv_date_folds)} folds.")
for i,(trd, vald) in enumerate(cv_date_folds,1):
    print(f"  Fold {i}: train {pd.Timestamp(trd[0]).date()} -> {pd.Timestamp(trd[-1]).date()} | val {pd.Timestamp(vald[0]).date()} -> {pd.Timestamp(vald[-1]).date()}")

# 4) Target
_print_section("4. Target = zscore(SimpleReturn_{t+1}) per-ID (train μ,σ)")
stocks['ret_tplus1']=stocks.groupby('ID')['SimpleReturn'].shift(-1)
id_stats=(stocks[stocks['is_train']].groupby('ID')['ret_tplus1'].agg(['mean','std']).rename(columns={'mean':'mu_train','std':'sigma_train'}))
id_stats['sigma_train']=id_stats['sigma_train'].replace(0,np.nan)
stocks=stocks.merge(id_stats,on='ID',how='left')
stocks['target_z']=(stocks['ret_tplus1']-stocks['mu_train'])/(stocks['sigma_train']+1e-12)
stocks=stocks.dropna(subset=['target_z'])

# 5) Features & scaling
_print_section("5. Feature Matrix & Scaling")
feature_cols=[*[f'ret_lag_{L}' for L in [1,2,3,4,5]], *[f'sma_{n}' for n in [5,10,20]], *[f'ema_{n}' for n in [5,10,20]],
             'macd','macd_signal','rsi_14','rsi_7','bb_upper','bb_lower','bb_bw','bb_percB','atr_14','hv_10','hv_20',
             'obv','vroc_5','vroc_10','roc_5','roc_10','dist_sma_20','dist_sma_50','dist_sma_250',
             *[f'close_div_sma20_tminus{k}' for k in [4,3,2,1,0]],
             *[f'close_div_sma50_tminus{k}' for k in [4,3,2,1,0]],
             *[f'close_div_sma250_tminus{k}' for k in [4,3,2,1,0]],
             'VIX_Close_ffill','VIX_SMA5','is_vix_missing','volume_raw']
binary_cols=['is_vix_missing']
scale_cols=[c for c in feature_cols if c not in binary_cols]

parts=[]
from sklearn.exceptions import NotFittedError
for sid,g in stocks.groupby('ID'):
    g=g.copy()
    scaler=StandardScaler()
    fitX=g.loc[g['is_train'],scale_cols].astype(float).replace([np.inf,-np.inf],np.nan).dropna()
    if len(fitX)==0:
        g[[f's_{c}' for c in scale_cols]]=g[scale_cols]
    else:
        scaler.fit(fitX)
        X_all=g[scale_cols].astype(float).replace([np.inf,-np.inf],np.nan)
        X_tr = scaler.transform(X_all.fillna(method='ffill').fillna(0))
        g=pd.concat([g,pd.DataFrame(X_tr,index=g.index,columns=[f's_{c}' for c in scale_cols])],axis=1)
    for b in binary_cols: g[f's_{b}']=g[b].astype(float)
    parts.append(g)
stocks=pd.concat(parts,axis=0).sort_values(['ID','Date'])
scaled_cols=[f's_{c}' for c in feature_cols]
stocks_model=stocks.dropna(subset=scaled_cols+['target_z','OpenAdj','HighAdj','LowAdj'])

# 6) Matrices
X_all = stocks_model[scaled_cols].values.astype(np.float32)
y_all = stocks_model['target_z'].values.astype(np.float32)
dates_all = stocks_model['Date'].values.astype('datetime64[D]')
print("Design matrix:", X_all.shape)

fold_indices=[]
for trd,vald in cv_date_folds:
    tr_mask=np.isin(dates_all,trd); va_mask=np.isin(dates_all,vald)
    tr_idx=np.where(tr_mask)[0]; va_idx=np.where(va_mask)[0]
    if len(tr_idx) and len(va_idx): fold_indices.append((tr_idx,va_idx))
print("Usable folds:", len(fold_indices))

train_mask_full = dates_all <= np.datetime64(pd.to_datetime(HYPERPARAMS['data']['train_end_date']).date())
test_mask_full  = dates_all >= np.datetime64(pd.to_datetime(HYPERPARAMS['data']['test_start_date']).date())

# 7) Model & tuning
def make_mlp(layers_n=1, units=128, learning_rate=3e-4, input_dim=X_all.shape[1],
             l2_reg=HYPERPARAMS['fixed_params']['l2_reg'], dropout_rate=HYPERPARAMS['fixed_params']['dropout_rate'],
             seed=HYPERPARAMS['fixed_params']['random_seed']):
    keras.utils.set_random_seed(seed)
    m=keras.Sequential([layers.Input(shape=(input_dim,))])
    for _ in range(layers_n):
        m.add(layers.Dense(units,activation='relu',kernel_regularizer=regularizers.l2(l2_reg)))
        m.add(layers.Dropout(dropout_rate))
    m.add(layers.Dense(1,activation='linear'))
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return m

reg=KerasRegressor(model=make_mlp, epochs=HYPERPARAMS['cv_epochs'], batch_size=HYPERPARAMS['fixed_params']['batch_size'],
                   verbose=0, callbacks=[callbacks.EarlyStopping(monitor='val_loss',patience=HYPERPARAMS['patience'],restore_best_weights=True)],
                   validation_split=0.0)

param_distributions={'model__layers_n':HYPERPARAMS['search_space']['layers'],
                     'model__units':HYPERPARAMS['search_space']['units'],
                     'model__learning_rate':HYPERPARAMS['search_space']['learning_rate']}

class PredefinedTimeCV:
    def __init__(self, folds): self.folds=folds
    def split(self,X,y=None,groups=None):
        for tr,va in self.folds: yield tr,va
    def get_n_splits(self,X=None,y=None,groups=None): return len(self.folds)

cv_obj=PredefinedTimeCV(fold_indices)
try:
    tuner=RandomizedSearchCV(estimator=reg,param_distributions=param_distributions,n_iter=HYPERPARAMS['n_iter'],
                             cv=cv_obj,n_jobs=1,random_state=HYPERPARAMS['fixed_params']['random_seed'],
                             scoring='neg_mean_squared_error',refit=False,verbose=2)
    tuner.fit(X_all,y_all,validation_data=None)
except Exception as e:
    print("[FATAL] RandomizedSearchCV failed:", repr(e)); sys.exit(1)

best_params=tuner.best_params_; print("Best params:", best_params)

# 8) Final fit with val = union of CV vals
if len(cv_date_folds)>0:
    val_dates_union=np.unique(np.concatenate([v for _,v in cv_date_folds]))
else:
    trd=np.sort(stocks_model.loc[train_mask_full,'Date'].values.astype('datetime64[D]'))
    cutoff=max(5,int(0.1*len(trd))); val_dates_union=trd[-cutoff:]

val_mask=np.isin(dates_all,val_dates_union)
train_mask=(train_mask_full)&(~val_mask)
X_tr,y_tr = X_all[train_mask], y_all[train_mask]
X_va,y_va = X_all[val_mask], y_all[val_mask]

final_model=make_mlp(layers_n=best_params['model__layers_n'], units=best_params['model__units'],
                     learning_rate=best_params['model__learning_rate'])
hist=final_model.fit(X_tr,y_tr,validation_data=(X_va,y_va),
                     epochs=HYPERPARAMS['final_model_epochs'],batch_size=HYPERPARAMS['fixed_params']['batch_size'],
                     verbose=2,callbacks=[callbacks.EarlyStopping(monitor='val_loss',patience=HYPERPARAMS['patience'],restore_best_weights=True)])

# 9) Predictions & signals for FULL period (trade on D+1 open)
stocks_model=stocks_model.copy()
stocks_model['y_pred']=final_model.predict(X_all,verbose=0).reshape(-1)
daily_mean_ret = stocks_model.groupby('Date')['ret_tplus1'].mean()

sig = stocks_model[['ID','Date','OpenAdj','HighAdj','LowAdj','CloseAdj','ret_tplus1','y_pred']].copy()
for col in ['OpenAdj','HighAdj','LowAdj','CloseAdj']:
    sig[f'{col}_tplus1']=sig.groupby('ID')[col].shift(-1)
sig['trade_date']=sig.groupby('ID')['Date'].shift(-1)
sig=sig.dropna(subset=['OpenAdj_tplus1','HighAdj_tplus1','LowAdj_tplus1','trade_date'])

# Portfolios each day
n_long=HYPERPARAMS['strategy']['n_long']; n_short=HYPERPARAMS['strategy']['n_short']
pt=HYPERPARAMS['strategy']['pt_pct']; sl=HYPERPARAMS['strategy']['sl_pct']

def pick_portfolios(g: pd.DataFrame) -> pd.DataFrame:
    g=g.sort_values('y_pred',ascending=False)
    n_side=min(n_long, len(g)//2)
    return pd.concat([g.head(n_side).assign(side=1), g.tail(n_side).assign(side=-1)],axis=0)

ports = sig.groupby('trade_date', group_keys=False).apply(pick_portfolios)

openp=ports['OpenAdj_tplus1'].values; highp=ports['HighAdj_tplus1'].values; lowp=ports['LowAdj_tplus1'].values
closep=ports['CloseAdj_tplus1'].values; side=ports['side'].values
ptL=openp*(1+pt); slL=openp*(1-sl)
ptS=openp*(1-pt); slS=openp*(1+sl)
is_long=(side==1)
ret_long=np.where(lowp<=slL, -sl, np.where(highp>=ptL, pt, (closep/openp-1.0)))
ret_short=np.where(highp>=slS, -sl, np.where(lowp<=ptS, pt, (openp/closep-1.0)))
ports['position_ret']=np.where(is_long, ret_long, ret_short)

daily_pnl_full = ports.groupby('trade_date')['position_ret'].mean().rename('strategy_ret')

# 10) Metrics helper
def sharpe_ratio(returns, rf_annual=0.02):
    rf_daily=rf_annual/252.0
    ex=returns - rf_daily
    return ex.mean()/(ex.std(ddof=1)+1e-12)*math.sqrt(252) if len(returns)>1 else np.nan
def max_drawdown(cum):
    rm=np.maximum.accumulate(cum); dd=(cum/rm)-1.0; return dd.min() if len(dd) else np.nan
def sortino_ratio(returns, rf_annual=0.02):
    rf_daily=rf_annual/252.0; ex=returns-rf_daily; dn=ex[ex<0]
    return ex.mean()/((dn.std(ddof=1)+1e-12))*math.sqrt(252) if len(ex)>1 else np.nan
def realized_alpha_vs_bench(ret, bench):
    y=ret.values; x=bench.reindex(ret.index).values; X=sm.add_constant(x)
    ols=sm.OLS(y,X,missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags':5})
    a,b=ols.params[0],ols.params[1]; at,bt=ols.tvalues[0],ols.tvalues[1]; ap,bp=ols.pvalues[0],ols.pvalues[1]
    return a,b,at,bt,ap,bp

# Split train/test on trade_date
train_mask_ret = daily_pnl_full.index < np.datetime64(pd.to_datetime(HYPERPARAMS['data']['test_start_date']))
test_mask_ret  = daily_pnl_full.index >= np.datetime64(pd.to_datetime(HYPERPARAMS['data']['test_start_date']))

str_train = daily_pnl_full[train_mask_ret]
str_test  = daily_pnl_full[test_mask_ret]
bench_full = daily_mean_ret
bench_train = bench_full.reindex(str_train.index).fillna(0.0)
bench_test  = bench_full.reindex(str_test.index).fillna(0.0)

def metrics_block(ret, bench):
    cum=(1+ret).cumprod()
    ann_vol = ret.std(ddof=1)*math.sqrt(252)
    ann_ret = (1+ret).prod()**(252/max(len(ret),1)) - 1.0 if len(ret)>0 else np.nan
    sr=sharpe_ratio(ret, HYPERPARAMS['strategy']['rf_annual'])
    so=sortino_ratio(ret, HYPERPARAMS['strategy']['rf_annual'])
    mdd=max_drawdown(cum.values)
    win=(ret>0).mean() if len(ret)>0 else np.nan
    pf = ret[ret>0].sum()/(-(ret[ret<0].sum())+1e-12) if (ret[ret<0].sum()!=0) else np.nan
    a,b,at,bt,ap,bp = realized_alpha_vs_bench(ret, bench)
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sr=sr, so=so, mdd=mdd, win=win, pf=pf,
                alpha=a, alpha_t=at, alpha_p=ap, beta=b, beta_t=bt, beta_p=bp, cum=cum)

mb_train = metrics_block(str_train, bench_train)
mb_test  = metrics_block(str_test, bench_test)

# Regression metrics for train/test (on target_z vs y_pred by pred_date)
pred_dates = stocks_model['Date'].values.astype('datetime64[D]')
mask_train_pred = pred_dates < np.datetime64(pd.to_datetime(HYPERPARAMS['data']['test_start_date']).date())
mask_test_pred  = pred_dates >= np.datetime64(pd.to_datetime(HYPERPARAMS['data']['test_start_date']).date())

def reg_metrics(mask):
    y_true = stocks_model.loc[mask, 'target_z'].values
    y_pred = stocks_model.loc[mask, 'y_pred'].values
    if len(y_true)==0:
        return dict(rmse=np.nan,mae=np.nan,mse=np.nan,r2=np.nan)
    rmse = math.sqrt(mean_squared_error(y_true,y_pred))
    mae  = mean_absolute_error(y_true,y_pred)
    mse  = mean_squared_error(y_true,y_pred)
    r2   = r2_score(y_true,y_pred)
    return dict(rmse=rmse,mae=mae,mse=mse,r2=r2)
reg_train = reg_metrics(mask_train_pred)
reg_test  = reg_metrics(mask_test_pred)

# 11) Visualization
_print_section("11. Visualization")
fig = plt.figure(figsize=(15, 11))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1,1])

# (A) Loss
ax1 = fig.add_subplot(gs[0,0])
trL=np.array(hist.history.get('loss',[])); vaL=np.array(hist.history.get('val_loss',[]))
ax1.plot(trL*100.0,label='Train MSE x100'); ax1.plot(vaL*100.0,label='Val MSE x100')
ax1.plot(np.sqrt(trL),label='Train RMSE');   ax1.plot(np.sqrt(vaL),label='Val RMSE')
ax1.set_title('Loss vs Epochs'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()

# (B) Cumulative FULL period + split
ax2 = fig.add_subplot(gs[0,1])
bench_cum_full=(1+bench_full.reindex(daily_pnl_full.index).fillna(0.0)).cumprod()
ax2.plot((1+daily_pnl_full).cumprod().values, label='Strategy (full)')
ax2.plot(bench_cum_full.values, label='EW benchmark (full)')
# vertical split
dates_idx=pd.Series(daily_pnl_full.index)
split_ix=np.where(dates_idx.values >= np.datetime64(pd.to_datetime(HYPERPARAMS['data']['test_start_date'])))[0]
if len(split_ix): ax2.axvline(split_ix[0], linestyle='--')
ax2.set_title('Cumulative Returns (Train+Test)'); ax2.set_xlabel('Days'); ax2.set_ylabel('Equity'); ax2.legend()

# (C) Hyperparams
ax3=fig.add_subplot(gs[1,0]); ax3.axis('off')
fixed_tbl={'batch_size':HYPERPARAMS['fixed_params']['batch_size'],
           'dropout_rate':HYPERPARAMS['fixed_params']['dropout_rate'],
           'l2_reg':HYPERPARAMS['fixed_params']['l2_reg'],
           'patience':HYPERPARAMS['patience'],
           'final_epochs(max)':HYPERPARAMS['final_model_epochs'],
           'cv_epochs(max)':HYPERPARAMS['cv_epochs']}
param_tbl={k.replace('model__',''):v for k,v in best_params.items()}
cell_text=[[k,str(v)] for k,v in {**fixed_tbl,**param_tbl}.items()]
ax3.table(cellText=cell_text,colLabels=['Hyperparameter','Value'],loc='center')
ax3.set_title('Chosen Hyperparameters & Fixed')

# (D) Metrics table (TRAIN + TEST)
ax4=fig.add_subplot(gs[1,1]); ax4.axis('off')
rows=[
 ['Reg (TRAIN) RMSE', f"{reg_train['rmse']:.4f}"], ['Reg (TRAIN) MAE', f"{reg_train['mae']:.4f}"],
 ['Reg (TRAIN) MSE', f"{reg_train['mse']:.6f}"], ['Reg (TRAIN) R2', f"{reg_train['r2']:.4f}"],
 ['AnnRet (TRAIN)', f"{mb_train['ann_ret']:.2%}"], ['AnnVol (TRAIN)', f"{mb_train['ann_vol']:.2%}"],
 ['Sharpe (TRAIN)', f"{mb_train['sr']:.3f}"], ['Sortino (TRAIN)', f"{mb_train['so']:.3f}"],
 ['MaxDD (TRAIN)', f"{mb_train['mdd']:.2%}"], ['WinRate (TRAIN)', f"{mb_train['win']:.2%}"],
 ['ProfitFactor (TRAIN)', f"{mb_train['pf']:.3f}"],
 ['Alpha (TRAIN) vs EW', f"{mb_train['alpha']:.5f} (t={mb_train['alpha_t']:.2f}, p={mb_train['alpha_p']:.3f})"],
 ['Beta  (TRAIN) vs EW', f"{mb_train['beta']:.3f} (t={mb_train['beta_t']:.2f}, p={mb_train['beta_p']:.3f})"],
 ['Reg (TEST) RMSE', f"{reg_test['rmse']:.4f}"], ['Reg (TEST) MAE', f"{reg_test['mae']:.4f}"],
 ['Reg (TEST) MSE', f"{reg_test['mse']:.6f}"], ['Reg (TEST) R2', f"{reg_test['r2']:.4f}"],
 ['AnnRet (TEST)', f"{mb_test['ann_ret']:.2%}"], ['AnnVol (TEST)', f"{mb_test['ann_vol']:.2%}"],
 ['Sharpe (TEST)', f"{mb_test['sr']:.3f}"], ['Sortino (TEST)', f"{mb_test['so']:.3f}"],
 ['MaxDD (TEST)', f"{mb_test['mdd']:.2%}"], ['WinRate (TEST)', f"{mb_test['win']:.2%}"],
 ['ProfitFactor (TEST)', f"{mb_test['pf']:.3f}"],
 ['Alpha (TEST) vs EW', f"{mb_test['alpha']:.5f} (t={mb_test['alpha_t']:.2f}, p={mb_test['alpha_p']:.3f})"],
 ['Beta  (TEST) vs EW', f"{mb_test['beta']:.3f} (t={mb_test['beta_t']:.2f}, p={mb_test['beta_p']:.3f})"],
]
ax4.table(cellText=rows,colLabels=['Metric','Value'],loc='center')
ax4.set_title('Key Metrics (TRAIN & TEST)')

plt.tight_layout()
out_path = HYPERPARAMS.get('output_png') or os.path.join(os.path.dirname(stocks_csv),'MLP_dashboard.png')
try:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved dashboard to: {out_path}")
except Exception as e:
    print("[WARN] Could not save to preferred path. Reason:", e)
    fallback = os.path.join(os.path.dirname(stocks_csv),'MLP_dashboard.png')
    plt.savefig(fallback, dpi=200)
    print(f"Saved dashboard to fallback: {fallback}")

# Elapsed
_print_section("Elapsed")
elapsed=time.time()-_t0; print(f"Total elapsed: {int(elapsed//60)}m {int(elapsed%60)}s")
