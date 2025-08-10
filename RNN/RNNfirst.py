# RNN_vanilla.py
"""
Vanilla SimpleRNN model for next-day simple return prediction and trading strategy.
Designed to mirror the MLP workflow (same features, same CV/random-search/visualization).
Author: Assistant (adapted for user's MLP setup)
Date: 2025-08-10
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from tensorflow.keras.regularizers import l2
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping

# -------------------- CONFIG (match MLP defaults) --------------------
DATA_PATH = "/Users/lindawaisova/Desktop/DP/data/SP_100/READY_DATA/9DATA_FINAL.csv"
VIX_PATH  = "/Users/lindawaisova/Desktop/DP/data/SP_100/VIX/VIX_2005_2023.csv"

# Train/test split dates should match MLP.py
TRAIN_END_DATE = "2020-12-31"   # Match MLP.py
TEST_START_DATE = "2021-01-01"  # Match MLP.py

HYPERPARAMS = {
    'final_model_epochs': 5,
    'cv_epochs': 3,
    'patience': 2,
    'cv_folds': 2,
    'n_iter': 1,
    'search_space': {
        'rnn_layers': [1, 2, 3],
        'neurons_per_layer': [32, 64, 128],
        'learning_rate': [0.0001, 0.001, 0.01],
    },
    'fixed_params': {
        'batch_size': 64,
        'dropout_rate': 0.2,
        'l2_reg': 0.001
    },
    # sequence length (timesteps) — adjust/tune if chceš; 5 used as rozumný začátek
    'timesteps': 5
}

# Trading params (ke srovnání s MLP)
TRADING_PARAMS = {
    'long_positions': 10,
    'short_positions': 10
}

# -------------------- Utilities: indicators & preprocessing --------------------
def load_data():
    print("Loading datasets...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Main data not found at {DATA_PATH}")
    if not os.path.exists(VIX_PATH):
        raise FileNotFoundError(f"VIX data not found at {VIX_PATH}")
    df = pd.read_csv(DATA_PATH)
    vix = pd.read_csv(VIX_PATH)
    # Ensure datetime
    df['Date'] = pd.to_datetime(df['Date'])
    vix['Date'] = pd.to_datetime(vix['Date'])
    # Merge VIX into df by Date (VIX is in 'Close' column)
    df = df.merge(vix[['Date','Close']].rename(columns={'Close': 'VIX'}), on='Date', how='left')
    return df

def calculate_technical_indicators(df):
    """
    Compute technical indicators in a look-back safe way:
    For each stock, use .shift(1) when computing indicators (so only T-1 and earlier).
    We mirror main indicators used in MLP.py (SMA, EMA, MACD, RSI, Bollinger, ATR, HV, OBV, ROC).
    """
    print("Calculating technical indicators (historical-only)...")
    df = df.sort_values(['ID','Date']).reset_index(drop=True)
    out_frames = []
    for stock_id in df['ID'].unique():
        s = df[df['ID']==stock_id].copy().sort_values('Date').reset_index(drop=True)
        # lagged returns
        for lag in [1,2,3,5]:
            s[f'SimpleReturn_lag_{lag}'] = s['SimpleReturn'].shift(lag)
        # SMA / EMA ratios (use close shifted by 1 to avoid look-ahead)
        for period in [5,10,20]:
            close_hist = s['CloseAdj'].shift(1)
            s[f'SMA_{period}'] = close_hist.rolling(window=period).mean()
            s[f'Price_SMA_{period}_ratio'] = close_hist / s[f'SMA_{period}']
            s[f'EMA_{period}'] = close_hist.ewm(span=period).mean()
            s[f'Price_EMA_{period}_ratio'] = close_hist / s[f'EMA_{period}']
        # MACD
        close_hist = s['CloseAdj'].shift(1)
        ema12 = close_hist.ewm(span=12).mean()
        ema26 = close_hist.ewm(span=26).mean()
        s['MACD'] = ema12 - ema26
        s['MACD_signal'] = s['MACD'].ewm(span=9).mean()
        s['MACD_hist'] = s['MACD'] - s['MACD_signal']
        # RSI
        for period in [7,14]:
            close_hist = s['CloseAdj'].shift(1)
            delta = close_hist.diff()
            gain = (delta.where(delta>0,0)).rolling(window=period).mean()
            loss = (-delta.where(delta<0,0)).rolling(window=period).mean()
            rs = gain / loss
            s[f'RSI_{period}'] = 100 - (100/(1+rs))
        # Bollinger
        close_hist = s['CloseAdj'].shift(1)
        sma20 = close_hist.rolling(20).mean()
        std20 = close_hist.rolling(20).std()
        s['BB_upper'] = sma20 + 2*std20
        s['BB_lower'] = sma20 - 2*std20
        s['BB_pos'] = (close_hist - s['BB_lower']) / (s['BB_upper'] - s['BB_lower'])
        # ATR (approx)
        high_hist = s['HighAdj'].shift(1)
        low_hist  = s['LowAdj'].shift(1)
        close_prev = s['CloseAdj'].shift(1)
        high_low = high_hist - low_hist
        high_close = np.abs(high_hist - close_prev)
        low_close = np.abs(low_hist - close_prev)
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        s['ATR_14'] = true_range.rolling(window=14).mean()
        # Historical volatility
        for period in [10,20]:
            r = s['SimpleReturn'].shift(1)
            s[f'HV_{period}'] = r.rolling(window=period).std() * np.sqrt(252)
        # OBV
        volume_hist = s['Volume'].shift(1)
        returns_hist = s['SimpleReturn'].shift(1)
        s['OBV'] = (volume_hist * np.sign(returns_hist)).cumsum()
        # VIX features (shifted)
        s['VIX_SMA_5'] = s['VIX'].shift(1).rolling(5).mean()
        s['VIX_change'] = s['VIX'].shift(1).pct_change()
        out_frames.append(s)
    df2 = pd.concat(out_frames, axis=0).reset_index(drop=True)
    return df2

def prepare_features_and_target(df):
    """
    Prepare features and target consistent with MLP:
    - target = next-day SimpleReturn (grouped per ID -> shift(-1))
    - select only lagged/historical features (exclude T current-day raw columns)
    - return DataFrames (train/test) so we can build per-stock sequences
    """
    print("Preparing features and target (no look-ahead bias)...")
    df = df.sort_values(['ID','Date']).reset_index(drop=True)
    df['target'] = df.groupby('ID')['SimpleReturn'].shift(-1)  # T -> T+1
    # Split by date to keep same train/test as MLP
    train_df = df[df['Date'] <= TRAIN_END_DATE].copy()
    test_df  = df[df['Date'] >= TEST_START_DATE].copy()

    # select feature columns: include ones containing keywords produced above
    exclude = ['ID','RIC','Name','Date','target','TotRet','SimpleReturn','Close','Volume','VolumeAdj','VolumeUSDadj',
               'OpenAdj','HighAdj','LowAdj','CloseAdj','VIX']
    feature_cols = []
    for c in df.columns:
        if c in exclude: continue
        if df[c].dtype not in [np.float64, np.float32, np.int64, np.int32]: continue
        if any(k in c for k in ['_lag','SMA_','EMA_','MACD','RSI_','BB_','ATR_','HV_','OBV','VROC_','ROC_','VIX_','Price_']):
            feature_cols.append(c)
    print(f"Selected {len(feature_cols)} features")
    # drop rows with NaN in features/target
    train_df = train_df.dropna(subset=feature_cols + ['target'])
    test_df  = test_df.dropna(subset=feature_cols + ['target'])
    return train_df, test_df, feature_cols

# -------------------- Sequence builder for RNN --------------------
def build_sequences_per_stock(df, feature_cols, timesteps):
    """
    Build sliding window sequences per stock (no cross-stock mixing).
    Returns X (n_samples, timesteps, n_features), y (n_samples,)
    """
    X_list, y_list = [], []
    ids = df['ID'].unique()
    for stock in ids:
        s = df[df['ID']==stock].sort_values('Date')
        arr = s[feature_cols].values
        targ = s['target'].values
        n = len(s)
        for i in range(n - timesteps):
            X_list.append(arr[i:i+timesteps])
            # target aligned with last timestep -> that is the future return for day after last time step
            y_list.append(targ[i+timesteps-1])
    if len(X_list) == 0:
        return np.empty((0, timesteps, len(feature_cols))), np.empty((0,))
    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y

# -------------------- Model builder (vanilla SimpleRNN) --------------------
def create_rnn_model(input_timesteps, input_dim, rnn_layers=1, neurons_per_layer=64, learning_rate=0.001, meta=None, compile_kwargs=None):
    """
    Create and compile a vanilla SimpleRNN model.
    - activation for recurrent units: tanh (default)
    - output: linear (regression)
    Added meta and compile_kwargs parameters for scikeras compatibility.
    """
    from tensorflow.keras.optimizers import Adam
    model = Sequential()
    # First RNN layer: return_sequences True if stacking
    for i in range(rnn_layers):
        return_seq = (i < rnn_layers - 1)
        if i == 0:
            model.add(SimpleRNN(
                units=neurons_per_layer,
                activation='tanh',
                return_sequences=return_seq,
                input_shape=(input_timesteps, input_dim),
                kernel_regularizer=l2(HYPERPARAMS['fixed_params']['l2_reg'])
            ))
        else:
            model.add(SimpleRNN(
                units=neurons_per_layer,
                activation='tanh',
                return_sequences=return_seq,
                kernel_regularizer=l2(HYPERPARAMS['fixed_params']['l2_reg'])
            ))
        # dropout on inputs to layer
        model.add(Dropout(HYPERPARAMS['fixed_params']['dropout_rate']))
    # final dense regressor
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

# -------------------- Training / Tuning pipeline --------------------
def tune_hyperparameters(X_train_seq, y_train):
    """
    Randomized search using KerasRegressor wrapper.
    X_train_seq shape: (n_samples, timesteps, features)
    We'll pass input_timesteps and input_dim to the builder.
    """
    print("Starting RandomizedSearchCV for RNN hyperparams...")
    timesteps = X_train_seq.shape[1]
    feat_dim  = X_train_seq.shape[2]

    # KerasRegressor for scikeras - updated interface
    model_wrapper = KerasRegressor(
        model=create_rnn_model,
        input_timesteps=timesteps,
        input_dim=feat_dim,
        epochs=HYPERPARAMS['cv_epochs'],
        batch_size=HYPERPARAMS['fixed_params']['batch_size'],
        verbose=1
    )

    # Build param dist mapping to builder args (add model__ prefix for scikeras)
    param_dist = {
        'model__rnn_layers': HYPERPARAMS['search_space']['rnn_layers'],
        'model__neurons_per_layer': HYPERPARAMS['search_space']['neurons_per_layer'],
        'model__learning_rate': HYPERPARAMS['search_space']['learning_rate']
    }

    cv = KFold(n_splits=HYPERPARAMS['cv_folds'], shuffle=True, random_state=42)
    rnd = RandomizedSearchCV(
        estimator=model_wrapper,
        param_distributions=param_dist,
        n_iter=HYPERPARAMS['n_iter'],
        scoring='neg_mean_squared_error',
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=1
    )

    # Fit (note: KerasRegressor will accept X_train_seq as-is)
    rnd.fit(X_train_seq, y_train)
    print("Best params:", rnd.best_params_)
    print("Best score (neg MSE):", rnd.best_score_)
    return rnd.best_params_

def train_final_model(X_train_seq, y_train, X_val_seq, y_val, best_params):
    print("Training final model with best hyperparameters...")
    
    # Remove model__ prefix from parameters
    clean_params = {}
    for key, value in best_params.items():
        if key.startswith('model__'):
            clean_key = key.replace('model__', '')
            clean_params[clean_key] = value
        else:
            clean_params[key] = value
    
    model = create_rnn_model(
        input_timesteps=X_train_seq.shape[1],
        input_dim=X_train_seq.shape[2],
        rnn_layers=clean_params.get('rnn_layers',1),
        neurons_per_layer=clean_params.get('neurons_per_layer',64),
        learning_rate=clean_params.get('learning_rate',0.001)
    )
    es = EarlyStopping(monitor='val_loss', patience=HYPERPARAMS['patience'], restore_best_weights=True, verbose=1)
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=HYPERPARAMS['final_model_epochs'],
        batch_size=HYPERPARAMS['fixed_params']['batch_size'],
        callbacks=[es],
        verbose=2
    )
    return model, history

# -------------------- Trading strategy (same logic as MLP) --------------------
def generate_signals_and_strategy(df_original, feature_cols, scaler, model, timesteps):
    """
    Given the trained model, generate predictions on test set, create signals (rank top long/short),
    and compute daily P&L and cumulative returns. We'll reconstruct sequences per stock in test period,
    then aggregate daily portfolio returns as done in MLP.
    """
    # Build sequences per stock but keep date & ID mapping to know which prediction corresponds to which day & ID
    X_list, y_list, meta = [], [], []  # meta: (Date, ID)
    ids = df_original['ID'].unique()
    for stock in ids:
        s = df_original[df_original['ID']==stock].sort_values('Date').reset_index(drop=True)
        arr = s[feature_cols].values
        dates = s['Date'].values
        n = len(s)
        for i in range(n - timesteps):
            X_list.append(arr[i:i+timesteps])
            # prediction corresponds to the last timestep's date (the day we make the decision for next day)
            meta.append((dates[i+timesteps-1], stock))
    if len(X_list)==0:
        return None
    X_seq = np.stack(X_list)
    # scale features (scaler expects 2D; reshape and transform then reshape back)
    n_samples = X_seq.shape[0]
    n_steps = X_seq.shape[1]
    n_feat  = X_seq.shape[2]
    X_2d = X_seq.reshape(-1, n_feat)
    X_2d_scaled = scaler.transform(X_2d)
    X_seq_scaled = X_2d_scaled.reshape(n_samples, n_steps, n_feat)

    preds = model.predict(X_seq_scaled).reshape(-1)

    # collect into DataFrame
    df_preds = pd.DataFrame(meta, columns=['Date','ID'])
    df_preds['pred'] = preds
    # Need actual next-day returns for P&L: we find target by aligning original df
    # Build mapping for (ID, Date) -> next day return (target)
    df_map = df_original.set_index(['ID','Date'])
    next_returns = []
    for (date, stock) in zip(df_preds['Date'], df_preds['ID']):
        try:
            next_returns.append(df_map.loc[(stock, pd.to_datetime(date))]['target'])
        except Exception:
            next_returns.append(np.nan)
    df_preds['next_return'] = next_returns
    df_preds.dropna(subset=['next_return'], inplace=True)

    # For each date, rank preds and pick top/bottom N
    daily_pl = []
    grouped = df_preds.groupby('Date')
    for date, g in grouped:
        g = g.copy()
        longs = g.nlargest(TRADING_PARAMS['long_positions'], 'pred')
        shorts = g.nsmallest(TRADING_PARAMS['short_positions'], 'pred')
        # portfolio return = average of selected positions' actual next returns
        if len(longs)>0:
            long_ret = longs['next_return'].mean()
        else:
            long_ret = 0.0
        if len(shorts)>0:
            short_ret = -shorts['next_return'].mean()  # short profit if next_return negative
        else:
            short_ret = 0.0
        port_ret = 0.5*(long_ret + short_ret)  # equally weight long and short side
        daily_pl.append((date, port_ret))
    df_pl = pd.DataFrame(daily_pl, columns=['Date','strategy_return']).sort_values('Date')
    # compute cumulative returns
    df_pl['cum_strategy'] = (1 + df_pl['strategy_return']).cumprod()
    # buy & hold baseline: average next_return across all stocks each day
    # we compute average true next_return per date from df_original
    avg_next = df_original.groupby('Date')['target'].mean().reset_index().rename(columns={'target':'avg_next'})
    df_pl = df_pl.merge(avg_next, on='Date', how='left')
    df_pl['cum_bh'] = (1 + df_pl['avg_next']).cumprod()
    return df_pl

# -------------------- Visualization helpers --------------------
def plot_history(history):
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title("Training history (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_strategy(df_pl):
    plt.figure(figsize=(10,6))
    plt.plot(df_pl['Date'], df_pl['cum_strategy'], label='RNN strategy (cumulative)')
    plt.plot(df_pl['Date'], df_pl['cum_bh'], label='Buy & Hold (avg) cumulative')
    plt.title("Strategy vs Buy & Hold (cumulative returns)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------- Main pipeline --------------------
def main():
    t0 = time.time()
    df_raw = load_data()
    df_feat = calculate_technical_indicators(df_raw)
    train_df, test_df, feature_cols = prepare_features_and_target(df_feat)

    # scale features using train set (fit scaler on all rows but features only)
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    # build sequences per stock for RNN
    timesteps = HYPERPARAMS['timesteps']
    X_train_seq, y_train = build_sequences_per_stock(train_df, feature_cols, timesteps)
    X_test_seq,  y_test  = build_sequences_per_stock(test_df, feature_cols, timesteps)

    print(f"Training sequences: {X_train_seq.shape}, Training targets: {y_train.shape}")
    print(f"Test sequences:     {X_test_seq.shape}, Test targets:     {y_test.shape}")

    # scale sequences (fit scaler already on train_df features; just transform)
    # reshape 3D -> 2D, transform, back to 3D
    def scale_seq(X_seq, scaler):
        if X_seq.size == 0:
            return X_seq
        n_s, n_t, n_f = X_seq.shape
        X2 = X_seq.reshape(-1, n_f)
        X2s = scaler.transform(X2)
        return X2s.reshape(n_s, n_t, n_f)

    X_train_seq_scaled = scale_seq(X_train_seq, scaler)
    X_test_seq_scaled  = scale_seq(X_test_seq, scaler)

    # If dataset small, create small validation set from training sequences
    if len(X_train_seq_scaled) > 1000:
        X_train_for_search, X_val_for_search, y_train_for_search, y_val_for_search = train_test_split(
            X_train_seq_scaled, y_train, test_size=0.15, random_state=42)
    else:
        # use entire train for search (cv will split)
        X_train_for_search, y_train_for_search = X_train_seq_scaled, y_train

    # Hyperparameter tuning (RandomizedSearch)
    best_params = tune_hyperparameters(X_train_for_search, y_train_for_search)

    # Train final model using best_params (use validation from test split or small val)
    # create small holdout val from training if possible
    if len(X_train_seq_scaled) > 200:
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_seq_scaled, y_train, test_size=0.1, random_state=42)
    else:
        X_tr, y_tr = X_train_seq_scaled, y_train
        X_val, y_val = X_test_seq_scaled, y_test  # fallback

    model, history = train_final_model(X_tr, y_tr, X_val, y_val, best_params)
    plot_history(history)

    # Evaluate on test sequences (MSE)
    if X_test_seq_scaled.size > 0:
        preds_test = model.predict(X_test_seq_scaled).reshape(-1)
        mse = mean_squared_error(y_test, preds_test)
        print(f"Test MSE: {mse:.6e}")

    # Generate strategy P&L on test_df using full test set (not only samples used in sequences)
    df_pl = generate_signals_and_strategy(test_df, feature_cols, scaler, model, timesteps)
    if df_pl is not None:
        plot_strategy(df_pl)
    else:
        print("No predictions / strategy could be generated on test set (maybe too short sequences).")

    t1 = time.time()
    print(f"Total runtime: {(t1-t0)/60:.2f} minutes")

if __name__ == "__main__":
    main()
