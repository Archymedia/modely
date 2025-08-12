
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP_final_v15.py
----------------
- CV loss x epochs: vykreslí jednotlivé foldy (ne průměr). Každý fold má vlastní barvu; train = plná, val = světlejší.
- Finální model loss x epochs v samostatném grafu.
- PNG název ladí s verzí (MLP_dashboard_v16.png).
- Ostatní: stejné jako v13/14 (bez CSV/JSON; tabulky Train/Test; heartbeat v backtestu).
"""

import os, time, math, random, warnings
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- Table styling helper for PNG ----
def _style_table(tbl):
    # Center text, tweak size/scale, grey background for non-numeric cells
    try:
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        # shrink width a bit (x<1), increase height (y>1)
        tbl.scale(0.85, 1.20)
        for (row, col), cell in tbl.get_celld().items():
            # center
            cell.set_text_props(ha='center', va='center')
            txt = cell.get_text().get_text()
            # detect numeric
            is_numeric = False
            try:
                # consider numbers like 1.23, -0.5, 1e-3
                float(str(txt).replace('%','').replace(',',''))
                is_numeric = True
            except Exception:
                is_numeric = False
            if not is_numeric:
                cell.set_facecolor('#f0f0f0')
    except Exception as _e:
        pass
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, optimizers, Model, Input

warnings.filterwarnings("ignore", category=UserWarning)

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
        'output_png': "/Users/lindawaisova/Desktop/DP/Git/DP/modely/3rd generation/MLP_dashboard_v16.png",
        'output_fallback_png': r"C:\Users\david\Desktop\SP100\MLP_dashboard_v16.png",
    },
    'final_model_epochs': 5,
    'cv_epochs': 3,
    'patience': 2,
    'cv_folds': 3,
    'n_iter': 1,  # search je deterministický (náš malý prostor)
    'search_space': {
        'layers': [1],
        'units': [64],
        'learning_rate': [1e-3]
    },
    'fixed_params': {
        'batch_size': 128,
        'dropout_rate': 0.15,
        'l2_reg': 3e-4,
        'random_seed': 42,
        'val_fraction_for_final': 0.15,
    },
    'strategy': {
        'n_long': 10,
        'n_short': 10,
        'pt_pct': 0.02,
        'sl_pct': 0.02,
        'rf_annual': 0.02,
        'trading_days': 252,
        'heartbeat_days': 50,
    }
}

def set_seed(seed:int):
    os.environ["PYTHONHASHSEED"]=str(seed); random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def log(msg:str):
    print(f"[MLP] {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} — {msg}", flush=True)

class Timer:
    def __init__(self, label:str): self.label=label
    def __enter__(self):
        log(f"▶️ {self.label} — start"); self.t0=time.perf_counter(); return self
    def __exit__(self, *args):
        dt=time.perf_counter()-self.t0; log(f"✅ {self.label} — hotovo za {int(dt//60)}m {int(dt%60)}s")

def pick_first(paths:List[str])->Optional[str]:
    for p in paths:
        if p and os.path.exists(p): return p
    return None

def ensure_png_path(path:str, fallback:str)->str:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True); return path
    except Exception:
        os.makedirs(os.path.dirname(fallback), exist_ok=True); return fallback

# ---------- Data ----------
def load_sp100(path:str)->pd.DataFrame:
    log(f"Načítám SP100 data z: {path}")
    df = pd.read_csv(path, low_memory=False)
    need = ['ID','Date','SimpleReturn','OpenAdj','HighAdj','LowAdj','CloseAdj']
    for c in need:
        if c not in df.columns: raise ValueError(f"Chybí sloupec: {c}")
    df['Date']=pd.to_datetime(df['Date']); df['ID']=df['ID'].astype(str)
    return df.sort_values(['ID','Date']).reset_index(drop=True)

def load_vix(path:str)->pd.DataFrame:
    log(f"Načítám VIX data z: {path}")
    v=pd.read_csv(path)
    need=['Date','Open','High','Low','Close','VIX_Change','VIX_Change_Abs']
    for c in need:
        if c not in v.columns: raise ValueError(f"Chybí VIX sloupec: {c}")
    v['Date']=pd.to_datetime(v['Date']); v=v.sort_values('Date').set_index('Date').ffill().reset_index()
    return v

def per_id_features(df:pd.DataFrame)->pd.DataFrame:
    log("Vypočítávám per‑ID prediktory…")
    def f(g:pd.DataFrame)->pd.DataFrame:
        g=g.sort_values('Date').copy()
        g['ret_1']=g['SimpleReturn'].shift(1)
        g['ret_5']=g['SimpleReturn'].rolling(5).sum().shift(1)
        g['ret_10']=g['SimpleReturn'].rolling(10).sum().shift(1)
        g['vol_10']=g['SimpleReturn'].rolling(10).std(ddof=0).shift(1)
        g['vol_20']=g['SimpleReturn'].rolling(20).std(ddof=0).shift(1)
        g['ma_5']=g['CloseAdj'].rolling(5).mean().shift(1)
        g['ma_20']=g['CloseAdj'].rolling(20).mean().shift(1)
        std5=g['CloseAdj'].rolling(5).std(ddof=0); std20=g['CloseAdj'].rolling(20).std(ddof=0)
        z5=(g['CloseAdj']-g['ma_5'].shift(-1))/std5.replace(0,np.nan)
        z20=(g['CloseAdj']-g['ma_20'].shift(-1))/std20.replace(0,np.nan)
        g['z_close_ma5']=z5.shift(1); g['z_close_ma20']=z20.shift(1)
        g['hl_range']=((g['HighAdj']-g['LowAdj'])/g['OpenAdj']).shift(1)
        g['oc_intraday']=((g['CloseAdj']-g['OpenAdj'])/g['OpenAdj']).shift(1)
        g['ewm_ret_10']=g['SimpleReturn'].ewm(span=10, adjust=False).mean().shift(1)
        g['ewm_vol_10']=g['SimpleReturn'].ewm(span=10, adjust=False).std().shift(1)
        if 'VolumeAdj' in g.columns:
            volstd=g['VolumeAdj'].rolling(20).std(ddof=0)
            g['vol_z_20']=((g['VolumeAdj']-g['VolumeAdj'].rolling(20).mean())/volstd.replace(0,np.nan)).shift(1)
        else:
            g['vol_z_20']=np.nan
        return g
    out=df.groupby('ID', group_keys=False).apply(f)
    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    return out

def merge_vix(df:pd.DataFrame, v:pd.DataFrame)->pd.DataFrame:
    log("Přidávám VIX prediktory (globálně, ffill)…")
    vv=v.copy()
    vv['vix_level']=vv['Close']; vv['vix_chg']=vv['VIX_Change']; vv['vix_abs']=vv['VIX_Change_Abs']
    vv['vix_ma5']=vv['Close'].rolling(5).mean(); vv['vix_ma20']=vv['Close'].rolling(20).mean()
    vv['vix_vol20']=vv['Close'].rolling(20).std(ddof=0)
    vv=vv[['Date','vix_level','vix_chg','vix_abs','vix_ma5','vix_ma20','vix_vol20']]
    out=df.merge(vv, on='Date', how='left').sort_values(['ID','Date'])
    for c in ['vix_level','vix_chg','vix_abs','vix_ma5','vix_ma20','vix_vol20']:
        out[c]=out[c].shift(1)
    out.sort_values('Date', inplace=True)
    out[['vix_level','vix_chg','vix_abs','vix_ma5','vix_ma20','vix_vol20']]= \
        out[['vix_level','vix_chg','vix_abs','vix_ma5','vix_ma20','vix_vol20']].ffill()
    return out

def compute_target(df:pd.DataFrame, train_end:pd.Timestamp)->pd.DataFrame:
    log("Počítám target (mode=cross_sectional)…")
    df=df.copy()
    df['future_return']=df.groupby('ID')['SimpleReturn'].shift(-1)
    def z_by_date(g):
        mu=g['future_return'].mean(); sd=g['future_return'].std(ddof=0)
        return (g['future_return']-mu) if (sd==0 or np.isnan(sd)) else (g['future_return']-mu)/sd
    df['target_z']=df.groupby('Date', group_keys=False).apply(z_by_date)
    return df

def impute_scale(df:pd.DataFrame, train_end:pd.Timestamp,
                 feature_cols:List[str], vix_cols:List[str]):
    log("Imputuji a škáluji featury (bez leakage)…")
    df=df.copy(); df.replace([np.inf,-np.inf], np.nan, inplace=True)
    parts=[]
    for sec_id, g in df.groupby('ID'):
        tr=g.loc[g['Date']<=train_end, feature_cols]
        if tr.shape[0]==0:
            parts.append(g.assign(_drop_me_=True)); continue
        means=tr.mean(numeric_only=True); g[feature_cols]=g[feature_cols].fillna(means.to_dict()); parts.append(g)
    df=pd.concat(parts, axis=0).sort_values(['Date','ID']).reset_index(drop=True)
    if '_drop_me_' in df.columns:
        for bid in df.loc[df['_drop_me_']==True,'ID'].unique().tolist():
            log(f"Varování: ID {bid} nemá žádné TRAIN řádky po featuringu. ID vyřazuji.")
        df=df[df.get('_drop_me_',False)!=True].drop(columns=['_drop_me_'])
    train_mask = df['Date']<=train_end
    vix_means=df.loc[train_mask, vix_cols].mean(numeric_only=True)
    df.loc[:, vix_cols]=df.loc[:, vix_cols].fillna(vix_means.to_dict())
    vix_scaler=StandardScaler().fit(df.loc[train_mask, vix_cols].values)
    df.loc[:, vix_cols]=vix_scaler.transform(df.loc[:, vix_cols].values)
    keep=[]; scalers={}
    for sec_id, g in df.groupby('ID'):
        tr=g.loc[g['Date']<=train_end, feature_cols]; sc=StandardScaler().fit(tr.values)
        g.loc[:, feature_cols]=sc.transform(g[feature_cols].values); scalers[sec_id]=sc; keep.append(g)
    df=pd.concat(keep, axis=0).sort_values(['Date','ID']).reset_index(drop=True)
    return df, scalers, vix_scaler

# =========================== CV & Model =============================== #
def build_mlp(input_dim:int, units:int, layers_n:int, dropout:float, l2_reg:float, lr:float)->Model:
    inp=Input(shape=(input_dim,)); x=inp
    for _ in range(layers_n):
        x=layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
        if dropout>0: x=layers.Dropout(dropout)(x)
    out=layers.Dense(1, activation='linear')(x)
    m=Model(inp,out)
    try: opt=optimizers.legacy.Adam(learning_rate=lr)
    except Exception: opt=optimizers.Adam(learning_rate=lr)
    m.compile(optimizer=opt, loss='mse', metrics=['mse']); return m

def make_calendar_folds(dates:pd.Series, n_folds:int):
    uniq=np.array(sorted(dates.unique())); idx=np.linspace(0,len(uniq),n_folds+1,dtype=int)
    return [(pd.Timestamp(uniq[idx[i]]), pd.Timestamp(uniq[idx[i+1]-1])) for i in range(n_folds)]

def random_search_cv(X,y,dates,cfg)->Dict:
    folds=make_calendar_folds(dates, cfg['cv_folds'])
    best=None; best_cv_hist=None
    for hp in [{'layers':l,'units':u,'learning_rate':lr}
               for l in cfg['search_space']['layers']
               for u in cfg['search_space']['units']
               for lr in cfg['search_space']['learning_rate']]:
        fold_losses=[]; cv_histories=[]
        for (vs,ve) in folds:
            vmask=(dates>=vs)&(dates<=ve); tmask=dates<vs
            if vmask.sum()==0 or tmask.sum()==0: continue
            model=build_mlp(X.shape[1], hp['units'], hp['layers'],
                            cfg['fixed_params']['dropout_rate'], cfg['fixed_params']['l2_reg'], hp['learning_rate'])
            es=callbacks.EarlyStopping(monitor='val_loss', patience=cfg['patience'], mode='min', restore_best_weights=True, verbose=0)
            hist=model.fit(X[tmask],y[tmask], validation_data=(X[vmask],y[vmask]),
                           epochs=cfg['cv_epochs'], batch_size=cfg['fixed_params']['batch_size'], verbose=0, callbacks=[es])
            fold_losses.append(np.min(hist.history['val_loss']))
            cv_histories.append({'loss':hist.history.get('loss',[]), 'val_loss':hist.history.get('val_loss',[])})
        avg=float(np.mean(fold_losses)) if fold_losses else np.inf
        log(f"  → průměrná VAL MSE přes foldy: {avg:.6f}")
        if best is None or avg<best['avg']:
            best={'hp':hp,'avg':avg}; best_cv_hist=cv_histories
    log(f"Nejlepší nalezené HP: {best}")
    return {'hp': best['hp'], 'cv_histories': best_cv_hist}

def fit_final(X,y,dates,hp,cfg):
    val_frac=cfg['fixed_params']['val_fraction_for_final']
    uniq=sorted(dates.unique()); cut=int((1-val_frac)*len(uniq)); vstart=uniq[cut]
    vmask=dates>=vstart; tmask=dates<vstart
    log(f"🏁 Trénuji finální model na všech train datech s HP={hp}")
    log(f"Finální fit: trénink {int(tmask.sum())} vzorků, validace {int(vmask.sum())} vzorků.")
    m=build_mlp(X.shape[1], hp['units'], hp['layers'], cfg['fixed_params']['dropout_rate'], cfg['fixed_params']['l2_reg'], hp['learning_rate'])
    es=callbacks.EarlyStopping(monitor='val_loss', patience=cfg['patience'], mode='min', restore_best_weights=True, verbose=1)
    hist=m.fit(X[tmask],y[tmask], validation_data=(X[vmask],y[vmask]),
               epochs=cfg['final_model_epochs'], batch_size=cfg['fixed_params']['batch_size'], verbose=1, callbacks=[es])
    return m, {'loss':hist.history.get('loss',[]), 'val_loss':hist.history.get('val_loss',[])}

# ---------- Strategy / Backtest ----------
@dataclass
class Position:
    sec_id:str; side:str; entry_date:pd.Timestamp; entry_px:float
    exit_date:Optional[pd.Timestamp]=None; exit_px:Optional[float]=None
    active:bool=True; daily_returns:Dict[pd.Timestamp,float]=field(default_factory=dict); hit:Optional[str]=None

def simulate_strategy(df:pd.DataFrame, preds_col:str, params:Dict,
                      train_end:pd.Timestamp, test_start:pd.Timestamp):
    n_long=params['n_long']; n_short=params['n_short']; pt=params['pt_pct']; sl=params['sl_pct']; hb=params.get('heartbeat_days',50)
    px=df[['Date','ID','OpenAdj','HighAdj','LowAdj','CloseAdj']].copy().set_index(['Date','ID']).sort_index()
    preds=df[['Date','ID',preds_col]].dropna().sort_values(['Date','ID']); dates=sorted(preds['Date'].unique())
    open_pos={}; all_pos=[]; daily_long={}; daily_short={}; valid_preds={}
    for d_idx, D in enumerate(dates[:-1]):
        if hb and d_idx%hb==0: log(f"… backtest {d_idx+1}/{len(dates)} — {D.date()}, otevřených: {sum(p.active for p in all_pos)}")
        day=preds[preds['Date']==D]; valid_preds[D]=int(day.shape[0]); avail=day[~day['ID'].isin(open_pos.keys())]
        top=avail.nlargest(n_long, preds_col); bottom=avail.nsmallest(n_short, preds_col)
        Dp1=dates[d_idx+1]
        for _,r in top.iterrows():
            tid=r['ID']; key=(Dp1,tid)
            if key in px.index and tid not in open_pos:
                entry=float(px.loc[key,'OpenAdj']); pos=Position(tid,'long',Dp1,entry); open_pos[tid]=pos; all_pos.append(pos)
        for _,r in bottom.iterrows():
            tid=r['ID']; key=(Dp1,tid)
            if key in px.index and tid not in open_pos:
                entry=float(px.loc[key,'OpenAdj']); pos=Position(tid,'short',Dp1,entry); open_pos[tid]=pos; all_pos.append(pos)
        for tid,pos in list(open_pos.items()):
            if D<pos.entry_date: continue
            key=(D,tid)
            if key not in px.index: continue
            o,h,l,c=px.loc[key,['OpenAdj','HighAdj','LowAdj','CloseAdj']].astype(float)
            if D==pos.entry_date: last_price=pos.entry_px
            else:
                prev=dates[d_idx-1] if d_idx>0 else None
                last_price=float(px.loc[(prev,tid),'CloseAdj']) if (prev is not None and (prev,tid) in px.index) else o
            if pos.side=='long':
                pt_px=pos.entry_px*(1+pt); sl_px=pos.entry_px*(1-sl)
                hit_sl=(not np.isnan(l)) and (l<=sl_px); hit_pt=(not np.isnan(h)) and (h>=pt_px)
                if hit_sl or hit_pt:
                    fill=sl_px if hit_sl else pt_px; day_ret=(fill/last_price)-1.0; daily_long.setdefault(D,[]).append(day_ret); pos.daily_returns[D]=float(day_ret)
                    pos.active=False; pos.exit_date=D; pos.exit_px=float(fill); pos.hit='SL' if hit_sl else 'PT'; del open_pos[tid]
                else:
                    day_ret=(c/last_price)-1.0; daily_long.setdefault(D,[]).append(day_ret); pos.daily_returns[D]=float(day_ret)
            else:
                pt_px=pos.entry_px*(1-pt); sl_px=pos.entry_px*(1+sl)
                hit_sl=(not np.isnan(h)) and (h>=sl_px); hit_pt=(not np.isnan(l)) and (l<=pt_px)
                if hit_sl or hit_pt:
                    fill=sl_px if hit_sl else pt_px; day_ret=(last_price/fill)-1.0; daily_short.setdefault(D,[]).append(day_ret); pos.daily_returns[D]=float(day_ret)
                    pos.active=False; pos.exit_date=D; pos.exit_px=float(fill); pos.hit='SL' if hit_sl else 'PT'; del open_pos[tid]
                else:
                    day_ret=(last_price/c)-1.0; daily_short.setdefault(D,[]).append(day_ret); pos.daily_returns[D]=float(day_ret)
    # aggregate
    all_days=sorted(set(daily_long)|set(daily_short))
    r_all=pd.Series({d:(0.5*(np.mean(daily_long.get(d,[])) if daily_long.get(d) else 0.0) + 0.5*(np.mean(daily_short.get(d,[])) if daily_short.get(d) else 0.0) if (bool(daily_long.get(d))+bool(daily_short.get(d))==2) else 0.5*((np.mean(daily_long.get(d,[])) if daily_long.get(d) else 0.0)+(np.mean(daily_short.get(d,[])) if daily_short.get(d) else 0.0))) for d in all_days}).sort_index()
    # test-only
    test_pos=[p for p in all_pos if p.entry_date>=test_start]; dl_t={}; ds_t={}
    for p in test_pos:
        for d,ret in p.daily_returns.items():
            (dl_t if p.side=='long' else ds_t).setdefault(d,[]).append(ret)
    test_days=sorted(set(dl_t)|set(ds_t))
    r_test_only=pd.Series({d:(0.5*(np.mean(dl_t.get(d,[])) if dl_t.get(d) else 0.0) + 0.5*(np.mean(ds_t.get(d,[])) if ds_t.get(d) else 0.0) if (bool(dl_t.get(d))+bool(ds_t.get(d))==2) else 0.5*((np.mean(dl_t.get(d,[])) if dl_t.get(d) else 0.0)+(np.mean(ds_t.get(d,[])) if ds_t.get(d) else 0.0))) for d in test_days}).sort_index()
    # train-only
    train_pos=[p for p in all_pos if p.entry_date<=train_end]; dl_tr={}; ds_tr={}
    for p in train_pos:
        for d,ret in p.daily_returns.items():
            (dl_tr if p.side=='long' else ds_tr).setdefault(d,[]).append(ret)
    train_days=sorted(set(dl_tr)|set(ds_tr))
    r_train_only=pd.Series({d:(0.5*(np.mean(dl_tr.get(d,[])) if dl_tr.get(d) else 0.0) + 0.5*(np.mean(ds_tr.get(d,[])) if ds_tr.get(d) else 0.0) if (bool(dl_tr.get(d))+bool(ds_tr.get(d))==2) else 0.5*((np.mean(dl_tr.get(d,[])) if dl_tr.get(d) else 0.0)+(np.mean(ds_tr.get(d,[])) if ds_tr.get(d) else 0.0))) for d in train_days}).sort_index()
    trades=[]
    for p in all_pos:
        if p.exit_date is None: continue
        ret=(p.exit_px/p.entry_px-1.0) if p.side=='long' else (p.entry_px/p.exit_px-1.0)
        trades.append({'ID':p.sec_id,'Side':p.side,'EntryDate':p.entry_date,'ExitDate':p.exit_date,'Return':ret,'Hit':p.hit})
    trades_df=pd.DataFrame(trades).sort_values('EntryDate')
    diag={'n_trades':len(trades_df)}
    return r_all, r_test_only, r_train_only, trades_df, diag

# ---------- Metrics & Viz ----------
def cumulative_return(r:pd.Series)->float:
    return float(np.prod(1+r.dropna().values)-1.0) if len(r) else 0.0
def annualized_return(r:pd.Series, td:int)->float:
    if len(r)==0: return 0.0
    years=max(1e-9, len(r)/td); return float((np.prod(1+r.dropna().values))**(1/years)-1.0)
def sharpe_ratio(r:pd.Series, rf_daily:float, td:int)->float:
    if len(r)==0: return 0.0
    ex=r-rf_daily; sd=ex.std(ddof=0); return float((ex.mean()/sd)*np.sqrt(td)) if sd>0 and not np.isnan(sd) else 0.0
def max_drawdown_curve(r:pd.Series):
    curve=(1+r.fillna(0)).cumprod(); peak=curve.cummax(); dd=(curve/peak)-1.0; return float(dd.min()), curve
def alpha_beta_ols(r_s:pd.Series, r_b:pd.Series)->Dict[str,float]:
    df=pd.concat([r_s,r_b],axis=1,keys=['s','b']).dropna()
    if df.empty:
        return {'alpha_daily':0.0,'alpha_annual':0.0,'alpha_t':np.nan,'alpha_p':np.nan,'beta':0.0,'beta_t':np.nan,'beta_p':np.nan,'r2':0.0}
    X=sm.add_constant(df['b'].values); y=df['s'].values; m=sm.OLS(y,X).fit()
    return {'alpha_daily':float(m.params[0]), 'alpha_annual':float(m.params[0]*252.0),
            'alpha_t':float(m.tvalues[0]), 'alpha_p':float(m.pvalues[0]),
            'beta':float(m.params[1]), 'beta_t':float(m.tvalues[1]), 'beta_p':float(m.pvalues[1]), 'r2':float(m.rsquared)}

def make_png(path, train_end, r_all, r_test, r_train, ew_all, ew_test,
             final_hist, reg_train, reg_test, ret_train, ret_test, trade_m, ab, cv_histories):
    log(f"Ukládám dashboard PNG: {path}")
    plt.figure(figsize=(18,12), dpi=150)
    # Horní řádek: kumulativní výnosy all/test
    ax1 = plt.subplot2grid((6,4),(0,0), colspan=4, rowspan=2)
    _, curve_all=max_drawdown_curve(r_all); _, curve_b_all=max_drawdown_curve(ew_all)
    ax1.plot(curve_all.index, curve_all.values, label='Strategie (all)')
    ax1.plot(curve_b_all.index, curve_b_all.values, label='Benchmark EW (all)')
    ax1.axvline(pd.Timestamp(train_end)+pd.Timedelta(days=1), color='gray', linestyle='--', label='2021-01-01')
    ax1.set_title("Kumulativní výnos — celé období (train + test)")
    ax1.set_ylabel("Equity (index, start=1.0)")
    ax1.legend(loc='best')

    ax2 = plt.subplot2grid((6,4),(2,0), colspan=4, rowspan=1)
    curve_t=(1+r_test.fillna(0)).cumprod(); curve_tb=(1+ew_test.fillna(0)).cumprod()
    ax2.plot(curve_t.index, curve_t.values, label='Strategie (test)')
    ax2.plot(curve_tb.index, curve_tb.values, label='Benchmark EW (test)')
    ax2.axvline(pd.Timestamp(train_end)+pd.Timedelta(days=1), color='gray', linestyle='--', label='2021‑01‑01')
    ax2.set_title("Kumulativní výnos — reset v 2021‑01‑01 (test část)")
    ax2.set_ylabel("Equity (index, start=1.0)")
    ax2.legend(loc='best')

    # Levý spodní sloupec: dva grafy MSE (CV per-fold + Final)
    ax3 = plt.subplot2grid((6,4),(3,0), colspan=2, rowspan=1)
    # CV per-fold
    if cv_histories:
        import itertools
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, hist in enumerate(cv_histories):
            c = colors[i % len(colors)]
            ax3.plot(hist.get('loss',[]), color=c, alpha=0.9, linewidth=1.5)
            ax3.plot(hist.get('val_loss',[]), color=c, alpha=0.4, linewidth=1.5)
        ax3.set_title("CV MSE (per fold) — train (tmavě), val (světle)")
    ax3.set_xlabel("Epocha"); ax3.set_ylabel("MSE")

    ax4 = plt.subplot2grid((6,4),(4,0), colspan=2, rowspan=2)
    ax4.plot(final_hist.get('loss',[]), label='Final Train MSE')
    ax4.plot(final_hist.get('val_loss',[]), label='Final Val MSE')
    ax4.set_title("Finální model — MSE (epocha)"); ax4.set_xlabel("Epocha"); ax4.set_ylabel("MSE"); ax4.legend(loc='best')

    # Pravý spodní sloupec: 4 tabulky v mřížce 2x2
    def table(ax, rows, title, pad=6):
        ax.axis('off')
        tbl = ax.table(cellText=rows, loc='center')
        # Style: center text, increase height, decrease width, grey bg for non-numeric
        for (row, col), cell in tbl.get_celld().items():
            cell.set_text_props(ha='center', va='center')
            cell.set_height(cell.get_height() * 1.2)
            cell.set_width(cell.get_width() * 0.85)
            txt = cell.get_text().get_text()
            is_numeric = False
            try:
                float(str(txt).replace('%','').replace(',',''))
                is_numeric = True
            except Exception:
                is_numeric = False
            if not is_numeric:
                cell.set_facecolor('#f0f0f0')
        ax.set_title(title, pad=pad)

    ax5 = plt.subplot2grid((6,4),(3,2), colspan=2, rowspan=1)
    reg_tbl = [["", "Train", "Test"],
               ["mse",  f"{reg_train['mse']:.4f}",  f"{reg_test['mse']:.4f}"],
               ["mae",  f"{reg_train['mae']:.4f}",  f"{reg_test['mae']:.4f}"],
               ["rmse", f"{reg_train['rmse']:.4f}", f"{reg_test['rmse']:.4f}"],
               ["r2",   f"{reg_train['r2']:.4f}",   f"{reg_test['r2']:.4f}"],]
    table(ax5, reg_tbl, "Regresní metriky (Train/Test)")

    ax6 = plt.subplot2grid((6,4),(4,2), colspan=2, rowspan=1)
    ret_tbl = [["", "Train", "Test"],
               ["cum",     f"{ret_train['cum']:.4f}",     f"{ret_test['cum']:.4f}"],
               ["ann",     f"{ret_train['ann']:.4f}",     f"{ret_test['ann']:.4f}"],
               ["sharpe",  f"{ret_train['sharpe']:.4f}",  f"{ret_test['sharpe']:.4f}"],
               ["maxdd",   f"{ret_train['maxdd']:.4f}",   f"{ret_test['maxdd']:.4f}"],
               ["vola_ann",f"{ret_train['vola_ann']:.4f}",f"{ret_test['vola_ann']:.4f}"],]
    table(ax6, ret_tbl, "Výnosové metriky (Train/Test)")

    ax7 = plt.subplot2grid((6,4),(5,2), colspan=1, rowspan=1)
    trades_tbl = [["win_rate",   f"{trade_m['win_rate']:.4f}"],
                  ["profit_factor", f"{trade_m['profit_factor']:.4f}"],
                  ["avg_holding_days", f"{trade_m['avg_holding_days']:.2f}"],
                  ["pt_hit_pct", f"{trade_m['pt_hit_pct']:.4f}"],
                  ["sl_hit_pct", f"{trade_m['sl_hit_pct']:.4f}"],]
    table(ax7, trades_tbl, "Metriky obchodů")

    ax8 = plt.subplot2grid((6,4),(5,3), colspan=1, rowspan=1)
    ab_tbl = [["alpha_daily",  f"{ab['alpha_daily']:.6f}"],
              ["alpha_annual", f"{ab['alpha_annual']:.4f}"],
              ["alpha_t",      f"{ab['alpha_t']:.4f}"],
              ["alpha_p",      f"{ab['alpha_p']:.4f}"],
              ["beta",         f"{ab['beta']:.4f}"],
              ["beta_t",       f"{ab['beta_t']:.4f}"],
              ["beta_p",       f"{ab['beta_p']:.4f}"],
              ["r2",           f"{ab['r2']:.4f}"],]
    table(ax8, ab_tbl, "Realizovaná alfa/beta (Test)", pad=20)

    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()
    try:
        nbytes=os.path.getsize(path); log(f"PNG uložen: {path} ({nbytes} B)")
    except Exception as e:
        log(f"Varování: PNG existuje, ale nelze změřit velikost: {e}")

# ---------- Main ----------
def main():
    t0=time.perf_counter(); set_seed(HYPERPARAMS['fixed_params']['random_seed'])
    out_png=ensure_png_path(HYPERPARAMS['data']['output_png'], HYPERPARAMS['data']['output_fallback_png'])
    sp_path=pick_first(HYPERPARAMS['data']['data_path']); vix_path=pick_first(HYPERPARAMS['data']['vix_path'])
    df=load_sp100(sp_path); vix=load_vix(vix_path)
    df=per_id_features(df); df=merge_vix(df,vix)
    train_end=pd.Timestamp(HYPERPARAMS['data']['train_end_date']); test_start=pd.Timestamp(HYPERPARAMS['data']['test_start_date'])
    df=compute_target(df, train_end)
    id_counts=df.loc[df['Date']<=train_end].groupby('ID').size()
    log(f"Po featuringu má trénink {len(id_counts)} unikátních ID.")
    if not id_counts.empty:
        log(f"Minimum řádků pro některé ID v train: {int(id_counts.min())}")
        log(f"Počet ID s méně než 5 řádky v train: {int((id_counts<5).sum())}")
    feature_cols=['ret_1','ret_5','ret_10','vol_10','vol_20','z_close_ma5','z_close_ma20','hl_range','oc_intraday','ewm_ret_10','ewm_vol_10','vol_z_20']
    vix_cols=['vix_level','vix_chg','vix_abs','vix_ma5','vix_ma20','vix_vol20']
    all_feats=feature_cols+vix_cols
    df_scaled, _, _ = impute_scale(df, train_end, feature_cols, vix_cols)
    df_scaled=df_scaled.dropna(subset=['target_z']).copy()
    train_mask=df_scaled['Date']<=train_end; test_mask=df_scaled['Date']>=test_start
    X=df_scaled[all_feats].values; y=df_scaled['target_z'].values; dates=df_scaled['Date']
    Xtr, ytr, dtr = X[train_mask], y[train_mask], dates[train_mask]
    Xte, yte, dte = X[test_mask], y[test_mask], dates[test_mask]
    log(f"Train size: {Xtr.shape}, Test size: {Xte.shape}")
    log("Spouštím random search + časové CV…")
    rs = random_search_cv(Xtr,ytr,dtr,HYPERPARAMS)
    hp = rs['hp']; cv_histories = rs['cv_histories']
    model, final_history = fit_final(Xtr,ytr,dtr,hp,HYPERPARAMS)
    with Timer("Predikce na celé období"):
        preds=model.predict(X, batch_size=HYPERPARAMS['fixed_params']['batch_size'], verbose=0).reshape(-1)
        df_scaled['pred']=preds
    with Timer("Backtest strategie"):
        r_all, r_test_only, r_train_only, trades_df, diag = simulate_strategy(df_scaled,'pred',HYPERPARAMS['strategy'],train_end,test_start)
    with Timer("Výpočet benchmarku EW"):
        bench=df[['Date','ID','SimpleReturn']].dropna().copy()
        ew=bench.groupby('Date')['SimpleReturn'].mean().sort_index()
        ew_all=ew.reindex(r_all.index).fillna(0); ew_test=ew.reindex(r_test_only.index).fillna(0)
    with Timer("Výpočet metrik"):
        preds_train=model.predict(Xtr, batch_size=HYPERPARAMS['fixed_params']['batch_size'], verbose=0).reshape(-1)
        preds_test=model.predict(Xte, batch_size=HYPERPARAMS['fixed_params']['batch_size'], verbose=0).reshape(-1)
        reg_train={'mse':float(mean_squared_error(ytr,preds_train)),
                   'mae':float(mean_absolute_error(ytr,preds_train)),
                   'rmse':float(np.sqrt(mean_squared_error(ytr,preds_train))),
                   'r2':float(r2_score(ytr,preds_train)) if len(ytr)>1 else 0.0}
        reg_test={'mse':float(mean_squared_error(yte,preds_test)),
                  'mae':float(mean_absolute_error(yte,preds_test)),
                  'rmse':float(np.sqrt(mean_squared_error(yte,preds_test))),
                  'r2':float(r2_score(yte,preds_test)) if len(yte)>1 else 0.0}
        td=HYPERPARAMS['strategy']['trading_days']; rf_daily=(1.0+HYPERPARAMS['strategy']['rf_annual'])**(1/td)-1.0
        def ret_metrics(rser):
            mdd,_=max_drawdown_curve(rser)
            return {'cum':cumulative_return(rser), 'ann':annualized_return(rser,td),
                    'sharpe':sharpe_ratio(rser,rf_daily,td), 'maxdd':mdd,
                    'vola_ann':float(rser.std(ddof=0)*np.sqrt(td)) if len(rser)>1 else 0.0}
        ret_train=ret_metrics(r_train_only); ret_test=ret_metrics(r_test_only)
        trade_m={'win_rate': float((trades_df['Return']>0).mean()) if len(trades_df)>0 else 0.0,
                 'profit_factor': float(trades_df.loc[trades_df['Return']>0,'Return'].sum() / max(1e-12, -trades_df.loc[trades_df['Return']<0,'Return'].sum())) if len(trades_df)>0 else 0.0,
                 'avg_holding_days': float((trades_df['ExitDate']-trades_df['EntryDate']).dt.days.mean()) if len(trades_df)>0 else 0.0,
                 'pt_hit_pct': float((trades_df['Hit']=='PT').mean()) if len(trades_df)>0 else 0.0,
                 'sl_hit_pct': float((trades_df['Hit']=='SL').mean()) if len(trades_df)>0 else 0.0}
        ab=alpha_beta_ols(r_test_only, ew_test)
    with Timer("Render PNG"):
        make_png(out_png, train_end, r_all, r_test_only, r_train_only, ew_all, ew_test,
                 final_history, reg_train, reg_test, ret_train, ret_test, trade_m, ab, cv_histories)
    dt=time.perf_counter()-t0; log(f"Celkový čas běhu: {int(dt//60)} min {int(dt%60)} s")

if __name__=="__main__":
    try: main()
    except Exception as e:
        log(f"FATAL ERROR: {e}"); raise
