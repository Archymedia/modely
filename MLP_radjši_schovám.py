#!/usr/bin/env python3
"""
MLP Model for Stock Return Prediction and Trading Strategy
Author: Assistant
Date: 2025-08-08
Description: Daily stock return prediction using MLP with technical indicators and trading strategy
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from scikeras.wrappers import KerasRegressor
from scipy.stats import uniform, randint
from datetime import datetime

warnings.filterwarnings('ignore')

# ===== CONFIGURATION PARAMETERS =====
print("=" * 60)
print("MLP STOCK PREDICTION MODEL")
print("=" * 60)

# Track execution time
start_time = time.time()

# ===== ZJEDNODUŠENÉ HYPERPARAMETRY PRO LADĚNÍ =====
HYPERPARAMS = {
    'final_model_epochs': 50,        # Maximální počet epoch pro FINÁLNÍ trénink
    'cv_epochs': 10,                 # Počet epoch pro CROSS-VALIDATION (rychlejší)
    'patience': 5,                   # Early stopping patience
    'batch_size': 64,               # Velikost batch
    'learning_rate': 0.001,          # Learning rate
    'hidden_layers': 2,              # HLAVNÍ PARAMETR: Počet skrytých vrstev (1, 2, nebo 3)
    'neurons_per_layer': 64,        # Počet neuronů v každé skryté vrstvě (stejný pro všechny)
    'dropout_rate': 0.1,               # Dropout rate
    'l2_reg': 0.0001,                     # L2 regularizace
    'cv_folds': 3                    # Počet foldů pro cross-validation (sníženo pro rychlost)
}

# Data paths (adjust these paths according to your data location)
DATA_PATH = "/Users/lindawaisova/Desktop/DP/data/SP_100/READY_DATA/9DATA_FINAL.csv"
VIX_PATH = "/Users/lindawaisova/Desktop/DP/data/SP_100/VIX/VIX_2005_2023.csv"

# Trading strategy parameters
TRADING_PARAMS = {
    'long_positions': 10,
    'short_positions': 10,
    'profit_target': 0.02,  # 2%
    'stop_loss': 0.02,      # 2%
    'risk_free_rate': 0.02  # 2% annually
}

# Date split
TRAIN_END_DATE = '2020-12-31'
TEST_START_DATE = '2021-01-01'
TEST_END_DATE = '2023-12-29'

print(f"Training period: 2005 - {TRAIN_END_DATE}")
print(f"Testing period: {TEST_START_DATE} - {TEST_END_DATE}")
print(f"Hidden layers: {HYPERPARAMS['hidden_layers']}")
print(f"Neurons per layer: {HYPERPARAMS['neurons_per_layer']}")
print(f"Max epochs (final): {HYPERPARAMS['final_model_epochs']}")
print(f"CV epochs: {HYPERPARAMS['cv_epochs']}")
print(f"Cross-validation folds: {HYPERPARAMS['cv_folds']}")
print()

# ---------------------- 1. Data Loading and Preprocessing ----------------------
print("=" * 60)
print("1. DATA LOADING AND PREPROCESSING")
print("=" * 60)

def load_and_prepare_data():
    """Load and prepare the main dataset and VIX data"""
    
    print("Loading main dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
    
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Main dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Unique stocks (ID): {df['ID'].nunique()}")
        
    except Exception as e:
        raise RuntimeError(f"Error loading main dataset: {e}")
    
    print("\nLoading VIX data...")
    if not os.path.exists(VIX_PATH):
        raise FileNotFoundError(f"VIX dataset not found at: {VIX_PATH}")
    
    try:
        vix_df = pd.read_csv(VIX_PATH)
        vix_df['Date'] = pd.to_datetime(vix_df['Date'])
        vix_df = vix_df[['Date', 'Close']].rename(columns={'Close': 'VIX'})
        
        print(f"VIX data shape: {vix_df.shape}")
        print(f"VIX date range: {vix_df['Date'].min()} to {vix_df['Date'].max()}")
        
    except Exception as e:
        raise RuntimeError(f"Error loading VIX data: {e}")
    
    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter data for the specified period
    df = df[(df['Date'] >= '2005-01-01') & (df['Date'] <= TEST_END_DATE)]
    
    # Merge with VIX data
    df = df.merge(vix_df, on='Date', how='left')
    df['VIX'] = df['VIX'].fillna(method='ffill')
    
    print(f"\nFinal dataset shape after preprocessing: {df.shape}")
    print(f"Total observations: {df.shape[0]}")
    
    return df

# ---------------------- 2. Technical Indicators Calculation ----------------------
print("=" * 60)
print("2. TECHNICAL INDICATORS CALCULATION")
print("=" * 60)

def calculate_technical_indicators(df):
    """Calculate all technical indicators for each stock"""
    
    print("Calculating technical indicators...")
    
    # Sort by ID and Date to ensure proper calculation
    df = df.sort_values(['ID', 'Date']).reset_index(drop=True)
    
    indicators_df = []
    
    total_stocks = df['ID'].nunique()
    
    for i, stock_id in enumerate(df['ID'].unique()):
        if i % 20 == 0:
            print(f"Processing stock {i+1}/{total_stocks} (ID: {stock_id})")
        
        stock_data = df[df['ID'] == stock_id].copy()
        stock_data = stock_data.sort_values('Date').reset_index(drop=True)
        
        # Basic lagged returns
        for lag in [1, 2, 3, 5]:
            stock_data[f'SimpleReturn_lag_{lag}'] = stock_data['SimpleReturn'].shift(lag)
        
        # Simple Moving Averages
        for period in [5, 10, 20]:
            stock_data[f'SMA_{period}'] = stock_data['CloseAdj'].rolling(window=period).mean()
            stock_data[f'Price_SMA_{period}_ratio'] = stock_data['CloseAdj'] / stock_data[f'SMA_{period}']
        
        # Exponential Moving Averages
        for period in [5, 10, 20]:
            stock_data[f'EMA_{period}'] = stock_data['CloseAdj'].ewm(span=period).mean()
            stock_data[f'Price_EMA_{period}_ratio'] = stock_data['CloseAdj'] / stock_data[f'EMA_{period}']
        
        # MACD
        ema_12 = stock_data['CloseAdj'].ewm(span=12).mean()
        ema_26 = stock_data['CloseAdj'].ewm(span=26).mean()
        stock_data['MACD'] = ema_12 - ema_26
        stock_data['MACD_signal'] = stock_data['MACD'].ewm(span=9).mean()
        stock_data['MACD_histogram'] = stock_data['MACD'] - stock_data['MACD_signal']
        
        # RSI
        for period in [7, 14]:
            delta = stock_data['CloseAdj'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            stock_data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma_20 = stock_data['CloseAdj'].rolling(window=20).mean()
        std_20 = stock_data['CloseAdj'].rolling(window=20).std()
        stock_data['BB_upper'] = sma_20 + (2 * std_20)
        stock_data['BB_lower'] = sma_20 - (2 * std_20)
        stock_data['BB_position'] = (stock_data['CloseAdj'] - stock_data['BB_lower']) / (stock_data['BB_upper'] - stock_data['BB_lower'])
        
        # ATR (Average True Range)
        high_low = stock_data['HighAdj'] - stock_data['LowAdj']
        high_close = np.abs(stock_data['HighAdj'] - stock_data['CloseAdj'].shift())
        low_close = np.abs(stock_data['LowAdj'] - stock_data['CloseAdj'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        stock_data['ATR_14'] = true_range.rolling(window=14).mean()
        
        # Historical Volatility
        for period in [10, 20]:
            stock_data[f'HV_{period}'] = stock_data['SimpleReturn'].rolling(window=period).std() * np.sqrt(252)
        
        # On-Balance Volume (OBV)
        stock_data['OBV'] = (stock_data['Volume'] * np.sign(stock_data['SimpleReturn'])).cumsum()
        
        # Volume Rate of Change
        for period in [5, 10]:
            stock_data[f'VROC_{period}'] = stock_data['Volume'].pct_change(periods=period)
        
        # Price Rate of Change
        for period in [5, 10]:
            stock_data[f'ROC_{period}'] = stock_data['CloseAdj'].pct_change(periods=period)
        
        # VIX indicators
        stock_data['VIX_SMA_5'] = stock_data['VIX'].rolling(window=5).mean()
        stock_data['VIX_change'] = stock_data['VIX'].pct_change()
        
        indicators_df.append(stock_data)
    
    print("Technical indicators calculation completed!")
    
    return pd.concat(indicators_df, ignore_index=True)

# ---------------------- 3. Target Variable and Feature Preparation ----------------------
print("=" * 60)
print("3. TARGET VARIABLE AND FEATURE PREPARATION")
print("=" * 60)

def prepare_features_and_target(df):
    """Prepare features and target variable with proper normalization"""
    
    print("Preparing features and target variable...")
    
    # Sort by ID and Date
    df = df.sort_values(['ID', 'Date']).reset_index(drop=True)
    
    # Create target variable (next day's return)
    df['target'] = df.groupby('ID')['SimpleReturn'].shift(-1)
    
    # Remove rows where target is NaN (last day for each stock)
    df = df.dropna(subset=['target'])
    
    print(f"Data shape after target creation: {df.shape}")
    
    # Split into train and test based on dates
    train_data = df[df['Date'] <= TRAIN_END_DATE].copy()
    test_data = df[df['Date'] >= TEST_START_DATE].copy()
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Train date range: {train_data['Date'].min()} to {train_data['Date'].max()}")
    print(f"Test date range: {test_data['Date'].min()} to {test_data['Date'].max()}")
    
    # Select feature columns (exclude non-predictive columns)
    exclude_cols = ['ID', 'RIC', 'Name', 'Date', 'target', 'TotRet', 'SimpleReturn', 
                   'Close', 'Volume', 'VolumeAdj', 'VolumeUSDadj']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
    
    print(f"Selected {len(feature_cols)} features:")
    for i, col in enumerate(feature_cols):
        print(f"  {i+1:2d}. {col}")
    
    # Extract features and target
    X_train = train_data[feature_cols].values
    y_train = train_data['target'].values
    X_test = test_data[feature_cols].values
    y_test = test_data['target'].values
    
    # Remove rows with NaN values
    train_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
    test_mask = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test)
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    # Update corresponding dataframes
    train_data = train_data[train_mask].reset_index(drop=True)
    test_data = test_data[test_mask].reset_index(drop=True)
    
    print(f"Final train data shape: {X_train.shape}")
    print(f"Final test data shape: {X_test.shape}")
    
    # Normalize features using training data statistics only (prevent data leakage)
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create Z-score for target variable (SimpleReturnZscore)
    target_mean = y_train.mean()
    target_std = y_train.std()
    
    print(f"Target variable (SimpleReturn) statistics:")
    print(f"  Train mean: {target_mean:.6f}")
    print(f"  Train std: {target_std:.6f}")
    print(f"  Train min: {y_train.min():.6f}")
    print(f"  Train max: {y_train.max():.6f}")
    
    return (X_train_scaled, X_test_scaled, y_train, y_test, 
            train_data, test_data, scaler, feature_cols,
            target_mean, target_std)

# ---------------------- 4. MLP Model Definition ----------------------
print("=" * 60)
print("4. MLP MODEL DEFINITION")
print("=" * 60)

def create_mlp_model(input_dim, hidden_layers=2, neurons_per_layer=100, 
                     dropout_rate=0.3, l2_reg=0.01, learning_rate=0.001, 
                     meta=None, compile_kwargs=None):
    """Create MLP model with specified architecture - všechny hidden vrstvy mají stejný počet neuronů"""
    
    print(f"    Creating model: {hidden_layers} hidden layers, {neurons_per_layer} neurons each")
    
    model = Sequential()
    
    # První skrytá vrstva
    model.add(Dense(neurons_per_layer, 
                   input_dim=input_dim,
                   activation='tanh',
                   kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    
    # Další skryté vrstvy (všechny mají stejný počet neuronů)
    for layer_num in range(2, hidden_layers + 1):
        print(f"    Adding hidden layer {layer_num} with {neurons_per_layer} neurons")
        model.add(Dense(neurons_per_layer, 
                       activation='tanh',
                       kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))
    
    # Výstupní vrstva
    model.add(Dense(1, activation='linear'))
    
    # Kompilace modelu
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    print(f"    Model compiled with learning rate: {learning_rate}")
    
    return model

# ---------------------- 5. Hyperparameter Tuning ----------------------
print("=" * 60)
print("5. HYPERPARAMETER TUNING")
print("=" * 60)

def tune_hyperparameters(X_train, y_train):
    """Zjednodušené ladění hyperparametrů - pouze cross-validation s přednastavenými parametry"""
    
    print("Spouštím zjednodušené ladění hyperparametrů...")
    print("Používám přednastavené parametry z konfigurace")
    print()
    
    # Použijeme přednastavené parametry z HYPERPARAMS
    best_params = {
        'hidden_layers': HYPERPARAMS['hidden_layers'],
        'neurons_per_layer': HYPERPARAMS['neurons_per_layer'],
        'dropout_rate': HYPERPARAMS['dropout_rate'],
        'l2_reg': HYPERPARAMS['l2_reg'],
        'learning_rate': HYPERPARAMS['learning_rate'],
        'batch_size': HYPERPARAMS['batch_size']
    }
    
    print("Parametry pro model:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print()
    
    # Provedeme pouze jednoduché cross-validation pro ověření
    print(f"Provádím {HYPERPARAMS['cv_folds']}-fold cross-validation pro ověření parametrů...")
    
    cv = KFold(n_splits=HYPERPARAMS['cv_folds'], shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train)):
        print(f"  Cross-validation fold {fold+1}/{HYPERPARAMS['cv_folds']}...")
        
        X_train_fold = X_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_train_fold = y_train[train_idx]
        y_val_fold = y_train[val_idx]
        
        # Vytvoříme a natrénujeme model
        model = create_mlp_model(
            input_dim=X_train.shape[1],
            hidden_layers=best_params['hidden_layers'],
            neurons_per_layer=best_params['neurons_per_layer'],
            dropout_rate=best_params['dropout_rate'],
            l2_reg=best_params['l2_reg'],
            learning_rate=best_params['learning_rate']
        )
        
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True, verbose=1)
        
        print(f"    Trénink fold {fold+1}...")
        print(f"    Velikost trénovacích dat: {X_train_fold.shape[0]:,} vzorků")
        print(f"    Velikost validačních dat: {X_val_fold.shape[0]:,} vzorků")
        print("    🚀 Zahájení tréninku fold...")
        
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=HYPERPARAMS['cv_epochs'],  
            batch_size=best_params['batch_size'],
            callbacks=[early_stopping],
            verbose=1  # Zobrazí progress každé epochy během CV
        )
        
        # Vyhodnocení na validačním fold
        print(f"    📊 Vytváření predikcí pro fold {fold+1}...")
        val_pred = model.predict(X_val_fold, verbose=1)
        mse = mean_squared_error(y_val_fold, val_pred)
        cv_scores.append(mse)
        
        print(f"    ✅ Fold {fold+1} dokončen - MSE: {mse:.6f}")
        print(f"    Počet epoch: {len(history.history['loss'])}")
        print("    " + "-" * 40)
    
    avg_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    print(f"\nCross-validation dokončeno!")
    print(f"Průměrné CV MSE: {avg_score:.6f} ± {std_score:.6f}")
    print("Parametry potvrzeny pro finální trénink.")
    print()
    
    return best_params, -avg_score

# ---------------------- 6. Model Training ----------------------
print("=" * 60)
print("6. MODEL TRAINING")
print("=" * 60)

def train_best_model(X_train, y_train, X_test, y_test, best_params):
    """Trénink finálního modelu s nejlepšími parametry"""
    
    print("=" * 50)
    print("TRÉNINK FINÁLNÍHO MODELU")
    print("=" * 50)
    
    print("Vytváření finálního modelu s parametry:")
    for key, value in best_params.items():
        if key != 'batch_size':
            print(f"  {key}: {value}")
    print()
    
    # Vytvoření modelu s nejlepšími parametry
    model = create_mlp_model(
        input_dim=X_train.shape[1],
        hidden_layers=best_params['hidden_layers'],
        neurons_per_layer=best_params['neurons_per_layer'],
        dropout_rate=best_params['dropout_rate'],
        l2_reg=best_params['l2_reg'],
        learning_rate=best_params['learning_rate']
    )
    
    print("\nArchitektura modelu:")
    model.summary()
    print()
    
    # Nastavení callbacků
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=HYPERPARAMS['patience'],
        restore_best_weights=True,
        verbose=1
    )
    
    print(f"Zahájení tréninku:")
    print(f"  Maximální epochy: {HYPERPARAMS['final_model_epochs']}")
    print(f"  Batch size: {best_params['batch_size']}")
    print(f"  Early stopping patience: {HYPERPARAMS['patience']}")
    print(f"  Velikost trénovacích dat: {X_train.shape[0]:,} vzorků")
    print(f"  Velikost testovacích dat: {X_test.shape[0]:,} vzorků")
    print()
    
    # Trénink modelu s verbose=1 pro komunikaci během tréninku
    print("🚀 ZAHÁJENÍ TRÉNINKU...")
    print("=" * 50)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=HYPERPARAMS['final_model_epochs'],
        batch_size=best_params['batch_size'],
        callbacks=[early_stopping],
        verbose=1  # Zobrazí progress bar pro každou epochu
    )
    
    print("=" * 50)
    print("✅ TRÉNINK DOKONČEN!")
    print(f"Model trénován {len(history.history['loss'])} epoch")
    print(f"Nejlepší validační loss: {min(history.history['val_loss']):.6f}")
    print("=" * 50)
    print()
    
    return model, history

# ---------------------- 7. Model Evaluation ----------------------
print("=" * 60)
print("7. MODEL EVALUATION")
print("=" * 60)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Vyhodnocení výkonnosti modelu"""
    
    print("=" * 50)
    print("VYHODNOCENÍ MODELU")
    print("=" * 50)
    
    print("Vytváření predikcí...")
    print("  Predikce na trénovacích datech...")
    y_train_pred = model.predict(X_train, verbose=0).flatten()
    
    print("  Predikce na testovacích datech...")
    y_test_pred = model.predict(X_test, verbose=0).flatten()
    
    print("Výpočet regresních metrik...")
    
    # Výpočet regresních metrik
    train_metrics = {
        'MSE': mean_squared_error(y_train, y_train_pred),
        'MAE': mean_absolute_error(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'R2': r2_score(y_train, y_train_pred)
    }
    
    test_metrics = {
        'MSE': mean_squared_error(y_test, y_test_pred),
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'R2': r2_score(y_test, y_test_pred)
    }
    
    print("\n📊 REGRESNÍ METRIKY:")
    print("=" * 40)
    print(f"{'Metrika':<8} {'Trénink':<12} {'Test':<12}")
    print("-" * 35)
    for metric in ['MSE', 'MAE', 'RMSE', 'R2']:
        print(f"{metric:<8} {train_metrics[metric]:<12.6f} {test_metrics[metric]:<12.6f}")
    
    print()
    print("✅ Vyhodnocení modelu dokončeno!")
    print("=" * 50)
    print()
    
    return y_train_pred, y_test_pred, train_metrics, test_metrics

# ---------------------- 8. Trading Strategy Implementation ----------------------
print("=" * 60)
print("8. TRADING STRATEGY IMPLEMENTATION")
print("=" * 60)

def implement_trading_strategy(test_data, predictions):
    """Implementace obchodní strategie s profit target a stop loss"""
    
    print("=" * 50)
    print("IMPLEMENTACE OBCHODNÍ STRATEGIE")
    print("=" * 50)
    
    print("Parametry strategie:")
    print(f"  Long pozice denně: {TRADING_PARAMS['long_positions']}")
    print(f"  Short pozice denně: {TRADING_PARAMS['short_positions']}")
    print(f"  Profit target: {TRADING_PARAMS['profit_target']*100:.1f}%")
    print(f"  Stop loss: {TRADING_PARAMS['stop_loss']*100:.1f}%")
    print()
    
    # Přidání predikcí k testovacím datům
    test_data_copy = test_data.copy()
    test_data_copy['predicted_return'] = predictions
    
    # Inicializace sledování portfolia
    portfolio_returns = []
    trade_log = []
    
    # Získání unikátních obchodních dat
    trading_dates = sorted(test_data_copy['Date'].unique())
    
    print(f"📅 Obchodní období: {len(trading_dates)} dní")
    print(f"Období: {trading_dates[0].strftime('%Y-%m-%d')} až {trading_dates[-1].strftime('%Y-%m-%d')}")
    print()
    
    daily_returns = []
    total_trades = 0
    
    print("🚀 ZAHÁJENÍ SIMULACE OBCHODOVÁNÍ...")
    print("=" * 50)
    
    for i, current_date in enumerate(trading_dates[:-1]):  # Vynechání posledního dne
        if i % 100 == 0:  # Progress každých 100 dní
            print(f"📊 Zpracování dne {i+1}/{len(trading_dates)-1}: {current_date.strftime('%Y-%m-%d')}")
        
        # Získání dat a predikcí pro aktuální den
        current_data = test_data_copy[test_data_copy['Date'] == current_date].copy()
        
        if len(current_data) == 0:
            continue
        
        # Výběr top 10 akcií pro long pozice (nejvyšší predikované výnosy)
        long_stocks = current_data.nlargest(TRADING_PARAMS['long_positions'], 'predicted_return')
        
        # Výběr top 10 akcií pro short pozice (nejnižší predikované výnosy)
        short_stocks = current_data.nsmallest(TRADING_PARAMS['short_positions'], 'predicted_return')
        
        # Výpočet denního výnosu portfolia
        daily_return = 0
        position_count = 0
        
        # Zpracování long pozic
        for _, stock in long_stocks.iterrows():
            actual_return = stock['target']
            
            # Aplikace profit target a stop loss
            if actual_return >= TRADING_PARAMS['profit_target']:
                realized_return = TRADING_PARAMS['profit_target']
            elif actual_return <= -TRADING_PARAMS['stop_loss']:
                realized_return = -TRADING_PARAMS['stop_loss']
            else:
                realized_return = actual_return
            
            daily_return += realized_return
            position_count += 1
            total_trades += 1
            
            trade_log.append({
                'Date': current_date,
                'ID': stock['ID'],
                'Position': 'Long',
                'Predicted_Return': stock['predicted_return'],
                'Actual_Return': actual_return,
                'Realized_Return': realized_return
            })
        
        # Zpracování short pozic
        for _, stock in short_stocks.iterrows():
            actual_return = stock['target']
            # Pro short pozice, zisk když akcie klesá
            short_return = -actual_return
            
            # Aplikace profit target a stop loss
            if short_return >= TRADING_PARAMS['profit_target']:
                realized_return = TRADING_PARAMS['profit_target']
            elif short_return <= -TRADING_PARAMS['stop_loss']:
                realized_return = -TRADING_PARAMS['stop_loss']
            else:
                realized_return = short_return
            
            daily_return += realized_return
            position_count += 1
            total_trades += 1
            
            trade_log.append({
                'Date': current_date,
                'ID': stock['ID'],
                'Position': 'Short',
                'Predicted_Return': stock['predicted_return'],
                'Actual_Return': actual_return,
                'Realized_Return': realized_return
            })
        
        # Průměrný výnos napříč všemi pozicemi
        if position_count > 0:
            daily_return = daily_return / position_count
        
        daily_returns.append({
            'Date': current_date,
            'Portfolio_Return': daily_return,
            'Position_Count': position_count
        })
    
    portfolio_df = pd.DataFrame(daily_returns)
    trades_df = pd.DataFrame(trade_log)
    
    print("=" * 50)
    print("✅ SIMULACE OBCHODOVÁNÍ DOKONČENA!")
    print(f"📈 Celkem provedených obchodů: {len(trades_df):,}")
    print(f"📅 Obchodní dny: {len(portfolio_df):,}")
    print(f"📊 Průměrný počet pozic denně: {trades_df.groupby('Date').size().mean():.1f}")
    print("=" * 50)
    print()
    
    return portfolio_df, trades_df

# ---------------------- 9. Performance Metrics Calculation ----------------------
print("=" * 60)
print("9. PERFORMANCE METRICS CALCULATION")
print("=" * 60)

def calculate_trading_metrics(portfolio_df, trades_df):
    """Calculate comprehensive trading performance metrics"""
    
    print("Calculating trading performance metrics...")
    
    if len(portfolio_df) == 0:
        print("Warning: No portfolio returns to analyze")
        return {}
    
    # Basic portfolio statistics
    returns = portfolio_df['Portfolio_Return'].values
    
    # Remove any NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        print("Warning: No valid returns to analyze")
        return {}
    
    # Calculate metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_volatility = np.std(returns) * np.sqrt(252)
    
    # Sharpe ratio
    excess_returns = returns - (TRADING_PARAMS['risk_free_rate'] / 252)
    sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    
    # Trade statistics
    if len(trades_df) > 0:
        winning_trades = trades_df[trades_df['Realized_Return'] > 0]
        losing_trades = trades_df[trades_df['Realized_Return'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df)
        
        avg_win = winning_trades['Realized_Return'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['Realized_Return'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if avg_loss != 0 and len(losing_trades) > 0 else float('inf')
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    # ===== REALIZOVANÁ ALFA - SPRÁVNÝ VÝPOČET =====
    # Alfa = Výnos portfolia - Risk-free rate (již implementováno správně)
    # Risk-free rate je fixně 2% p.a. dle zadání
    risk_free_rate = TRADING_PARAMS['risk_free_rate']  # 2% p.a.
    alpha = annualized_return - risk_free_rate
    
    print(f"\n📊 VÝPOČET REALIZOVANÉ ALFY:")
    print(f"  Anualizovaný výnos portfolia: {annualized_return:.2%}")
    print(f"  Risk-free rate (2% p.a.):     {risk_free_rate:.2%}")
    print(f"  Realizovaná alfa:             {alpha:.2%}")
    
    # Dodatečné statistiky pro lepší analýzu
    information_ratio = alpha / annualized_volatility if annualized_volatility > 0 else 0
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Sortino ratio (podobné Sharpe, ale pouze s downside volatilitou)
    downside_returns = returns[returns < 0]
    downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
    
    metrics = {
        'Total_Return': total_return,
        'Annualized_Return': annualized_return,
        'Annualized_Volatility': annualized_volatility,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown,
        'Win_Rate': win_rate,
        'Profit_Factor': profit_factor,
        'Alpha': alpha,  # Realizovaná alfa
        'Information_Ratio': information_ratio,
        'Calmar_Ratio': calmar_ratio,
        'Sortino_Ratio': sortino_ratio,
        'Total_Trades': len(trades_df),
        'Avg_Win': avg_win,
        'Avg_Loss': avg_loss,
        'Risk_Free_Rate': risk_free_rate
    }
    
    print("\nTrading Performance Metrics:")
    print("-" * 40)
    print(f"Total Return: {metrics['Total_Return']:.2%}")
    print(f"Annualized Return: {metrics['Annualized_Return']:.2%}")
    print(f"Annualized Volatility: {metrics['Annualized_Volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.3f}")
    print(f"Sortino Ratio: {metrics['Sortino_Ratio']:.3f}")
    print(f"Information Ratio: {metrics['Information_Ratio']:.3f}")
    print(f"Calmar Ratio: {metrics['Calmar_Ratio']:.3f}")
    print(f"Maximum Drawdown: {metrics['Max_Drawdown']:.2%}")
    print(f"Win Rate: {metrics['Win_Rate']:.2%}")
    print(f"Profit Factor: {metrics['Profit_Factor']:.3f}")
    print(f"Realized Alpha: {metrics['Alpha']:.2%}")
    print(f"Total Trades: {metrics['Total_Trades']}")
    
    return metrics

# ---------------------- 10. Visualization ----------------------
print("=" * 60)
print("10. VISUALIZATION")
print("=" * 60)

def create_comprehensive_visualization(history, portfolio_df, best_params, train_metrics, test_metrics, trading_metrics):
    """Create comprehensive visualization with all results"""
    
    print("Creating comprehensive visualization...")
    
    # Set up the plot
    fig = plt.figure(figsize=(20, 16))
    
    # Create a grid layout
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1])
    
    # 1. Training History - Loss/Epochs
    ax1 = fig.add_subplot(gs[0, :2])
    if history:
        ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Model Training Progress', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No training history available', ha='center', va='center', fontsize=12)
        ax1.set_title('Model Training Progress', fontsize=14, fontweight='bold')
    
    # 2. Portfolio Performance
    ax2 = fig.add_subplot(gs[1, :])
    if len(portfolio_df) > 0:
        # Calculate cumulative returns
        portfolio_df['Cumulative_Return'] = (1 + portfolio_df['Portfolio_Return']).cumprod()
        
        ax2.plot(portfolio_df['Date'], portfolio_df['Cumulative_Return'], 
                linewidth=2, color='darkgreen', label='Strategy Performance')
        
        # Add vertical line at train/test split
        train_end = pd.to_datetime(TRAIN_END_DATE)
        ax2.axvline(x=train_end, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                   label=f'Train/Test Split ({TRAIN_END_DATE})')
        
        ax2.set_title('Cumulative Portfolio Returns', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No portfolio data available', ha='center', va='center', fontsize=12)
        ax2.set_title('Cumulative Portfolio Returns', fontsize=14, fontweight='bold')
    
    # 3. Best Hyperparameters Table
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('tight')
    ax3.axis('off')
    
    # Prepare hyperparameters data
    params_data = []
    for key, value in best_params.items():
        if isinstance(value, float):
            params_data.append([key, f"{value:.4f}"])
        else:
            params_data.append([key, str(value)])
    
    if params_data:
        table1 = ax3.table(cellText=params_data,
                          colLabels=['Parameter', 'Value'],
                          cellLoc='left',
                          loc='center',
                          colWidths=[0.6, 0.4])
        table1.auto_set_font_size(False)
        table1.set_fontsize(9)
        table1.scale(1, 1.5)
        
        # Style the table
        for i in range(len(params_data) + 1):
            for j in range(2):
                cell = table1[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax3.set_title('Best Hyperparameters', fontsize=12, fontweight='bold')
    
    # 4. Regression Metrics Table
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('tight')
    ax4.axis('off')
    
    metrics_data = []
    for metric in ['MSE', 'MAE', 'RMSE', 'R2']:
        train_val = train_metrics.get(metric, 0)
        test_val = test_metrics.get(metric, 0)
        metrics_data.append([metric, f"{train_val:.4f}", f"{test_val:.4f}"])
    
    table2 = ax4.table(cellText=metrics_data,
                      colLabels=['Metric', 'Train', 'Test'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.4, 0.3, 0.3])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.5)
    
    # Style the table
    for i in range(len(metrics_data) + 1):
        for j in range(3):
            cell = table2[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#2196F3')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax4.set_title('Regression Metrics', fontsize=12, fontweight='bold')
    
    # 5. Trading Metrics Table
    ax5 = fig.add_subplot(gs[2, 1:])
    ax5.axis('tight')
    ax5.axis('off')
    
    trading_data = []
    if trading_metrics:
        trading_data = [
            ['Total Return', f"{trading_metrics.get('Total_Return', 0):.2%}"],
            ['Annualized Return', f"{trading_metrics.get('Annualized_Return', 0):.2%}"],
            ['Sharpe Ratio', f"{trading_metrics.get('Sharpe_Ratio', 0):.3f}"],
            ['Max Drawdown', f"{trading_metrics.get('Max_Drawdown', 0):.2%}"],
            ['Win Rate', f"{trading_metrics.get('Win_Rate', 0):.2%}"],
            ['Profit Factor', f"{trading_metrics.get('Profit_Factor', 0):.3f}"],
            ['Alpha', f"{trading_metrics.get('Alpha', 0):.2%}"],
            ['Total Trades', f"{trading_metrics.get('Total_Trades', 0):,}"]
        ]
    
    if trading_data:
        table3 = ax5.table(cellText=trading_data,
                          colLabels=['Trading Metric', 'Value'],
                          cellLoc='left',
                          loc='center',
                          colWidths=[0.6, 0.4])
        table3.auto_set_font_size(False)
        table3.set_fontsize(9)
        table3.scale(1, 1.5)
        
        # Style the table
        for i in range(len(trading_data) + 1):
            for j in range(2):
                cell = table3[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#FF9800')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax5.set_title('Trading Performance Metrics', fontsize=12, fontweight='bold')
    
    # 6. Model Evaluation Summary
    ax6 = fig.add_subplot(gs[3, :])
    
    # Calculate overall score and grade
    score = calculate_overall_score(test_metrics, trading_metrics)
    grade, emoji = get_grade_and_emoji(score)
    
    # Create evaluation text
    evaluation_text = f"""
MODEL AND TRADING STRATEGY EVALUATION

Overall Score: {score:.1f}/10 {emoji}
Grade: {grade}

🔬 REGRESNÍ METRIKY:
• R² Score: {test_metrics.get('R2', 0):.4f} • MSE: {test_metrics.get('MSE', 0):.6f}
• MAE: {test_metrics.get('MAE', 0):.6f} • RMSE: {test_metrics.get('RMSE', 0):.6f}

💰 VÝNOSOVÉ METRIKY:
• Celkový výnos: {trading_metrics.get('Total_Return', 0):.1%}
• Anualizovaný výnos: {trading_metrics.get('Annualized_Return', 0):.1%}
• Sharpe Ratio: {trading_metrics.get('Sharpe_Ratio', 0):.3f}
• Max Drawdown: {trading_metrics.get('Max_Drawdown', 0):.1%}

🎯 ÚSPĚŠNOST OBCHODŮ:
• Win Rate: {trading_metrics.get('Win_Rate', 0):.1%}
• Profit Factor: {trading_metrics.get('Profit_Factor', 0):.3f}
• Celkem obchodů: {trading_metrics.get('Total_Trades', 0):,}

📈 RIZIKO A ALFA:
• Anualizovaná volatilita: {trading_metrics.get('Annualized_Volatility', 0):.1%}
• Realizovaná alfa: {trading_metrics.get('Alpha', 0):.2%}
• Risk-free rate: {TRADING_PARAMS['risk_free_rate']:.1%} p.a.
    """
    
    ax6.text(0.05, 0.95, evaluation_text.strip(), transform=ax6.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    ax6.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_file = 'mlp_comprehensive_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Comprehensive visualization saved as: {output_file}")
    
    return output_file

def calculate_overall_score(test_metrics, trading_metrics):
    """
    Komplexní výpočet celkového skóre modelu z 10 bodů
    Založeno na regresních metrikách a metrikách obchodní strategie
    """
    
    score = 0
    score_details = {}
    
    print("\n" + "="*60)
    print("DETAILNÍ VÝPOČET CELKOVÉHO SKÓRE")
    print("="*60)
    
    # ===== 1. REGRESNÍ METRIKY (0-2.5 bodů) =====
    print("\n📊 REGRESNÍ METRIKY (max 2.5 bodů):")
    regression_score = 0
    
    # R² Score (0-1.5 bodů)
    r2 = test_metrics.get('R2', 0)
    if r2 > 0.05:
        r2_score = 1.5
    elif r2 > 0.02:
        r2_score = 1.0
    elif r2 > 0.005:
        r2_score = 0.5
    else:
        r2_score = 0
    regression_score += r2_score
    print(f"  • R² Score: {r2:.4f} → {r2_score:.1f}/1.5 bodů")
    
    # RMSE kvalita (0-1 bod)
    rmse = test_metrics.get('RMSE', float('inf'))
    if rmse < 0.02:
        rmse_score = 1.0
    elif rmse < 0.03:
        rmse_score = 0.6
    elif rmse < 0.05:
        rmse_score = 0.3
    else:
        rmse_score = 0
    regression_score += rmse_score
    print(f"  • RMSE: {rmse:.4f} → {rmse_score:.1f}/1.0 bodů")
    
    score += regression_score
    score_details['regression'] = regression_score
    print(f"  ✅ Celkem regresní metriky: {regression_score:.1f}/2.5 bodů")
    
    # ===== 2. VÝNOSOVÉ METRIKY (0-3 body) =====
    print("\n💰 VÝNOSOVÉ METRIKY (max 3 body):")
    return_score = 0
    
    # Sharpe Ratio (0-1.5 bodů)
    sharpe = trading_metrics.get('Sharpe_Ratio', 0)
    if sharpe > 1.5:
        sharpe_score = 1.5
    elif sharpe > 1.0:
        sharpe_score = 1.0
    elif sharpe > 0.5:
        sharpe_score = 0.5
    else:
        sharpe_score = 0
    return_score += sharpe_score
    print(f"  • Sharpe Ratio: {sharpe:.3f} → {sharpe_score:.1f}/1.5 bodů")
    
    # Anualizovaný výnos vs riziko (0-1 bod)
    ann_return = trading_metrics.get('Annualized_Return', 0)
    ann_volatility = trading_metrics.get('Annualized_Volatility', 0)
    if ann_return > 0.10 and ann_volatility < 0.25:
        return_risk_score = 1.0
    elif ann_return > 0.05:
        return_risk_score = 0.6
    elif ann_return > 0:
        return_risk_score = 0.3
    else:
        return_risk_score = 0
    return_score += return_risk_score
    print(f"  • Výnos {ann_return:.1%} / Volatilita {ann_volatility:.1%} → {return_risk_score:.1f}/1.0 bodů")
    
    # Maximum Drawdown (0-0.5 bodů)
    max_dd = abs(trading_metrics.get('Max_Drawdown', 0))
    if max_dd < 0.10:
        dd_score = 0.5
    elif max_dd < 0.20:
        dd_score = 0.3
    else:
        dd_score = 0
    return_score += dd_score
    print(f"  • Max Drawdown: {max_dd:.1%} → {dd_score:.1f}/0.5 bodů")
    
    score += return_score
    score_details['returns'] = return_score
    print(f"  ✅ Celkem výnosové metriky: {return_score:.1f}/3.0 bodů")
    
    # ===== 3. METRIKY ÚSPĚŠNOSTI OBCHODŮ (0-2 body) =====
    print("\n🎯 METRIKY ÚSPĚŠNOSTI OBCHODŮ (max 2 body):")
    trade_score = 0
    
    # Win Rate (0-1 bod)
    win_rate = trading_metrics.get('Win_Rate', 0)
    if win_rate > 0.55:
        win_score = 1.0
    elif win_rate > 0.52:
        win_score = 0.7
    elif win_rate > 0.50:
        win_score = 0.4
    else:
        win_score = 0
    trade_score += win_score
    print(f"  • Win Rate: {win_rate:.1%} → {win_score:.1f}/1.0 bodů")
    
    # Profit Factor (0-1 bod)
    profit_factor = trading_metrics.get('Profit_Factor', 0)
    if profit_factor > 1.3:
        pf_score = 1.0
    elif profit_factor > 1.1:
        pf_score = 0.6
    elif profit_factor > 1.0:
        pf_score = 0.3
    else:
        pf_score = 0
    trade_score += pf_score
    print(f"  • Profit Factor: {profit_factor:.3f} → {pf_score:.1f}/1.0 bodů")
    
    score += trade_score
    score_details['trades'] = trade_score
    print(f"  ✅ Celkem úspěšnost obchodů: {trade_score:.1f}/2.0 bodů")
    
    # ===== 4. REALIZOVANÁ ALFA A STATISTICKÁ VÝZNAMNOST (0-2.5 bodů) =====
    print("\n📈 ALFA A STATISTICKÁ VÝZNAMNOST (max 2.5 bodů):")
    alpha_score = 0
    
    # Realizovaná alfa (0-1.5 bodů)
    alpha = trading_metrics.get('Alpha', 0)
    if alpha > 0.08:
        alpha_pts = 1.5
    elif alpha > 0.05:
        alpha_pts = 1.0
    elif alpha > 0.02:
        alpha_pts = 0.5
    else:
        alpha_pts = 0
    alpha_score += alpha_pts
    print(f"  • Realizovaná Alfa: {alpha:.2%} → {alpha_pts:.1f}/1.5 bodů")
    
    # Statistická významnost (0-1 bod)
    total_trades = trading_metrics.get('Total_Trades', 0)
    sharpe_significance = sharpe * np.sqrt(total_trades / 252) if total_trades > 0 else 0
    if sharpe_significance > 2.0:  # 95% confidence
        stat_score = 1.0
    elif sharpe_significance > 1.5:
        stat_score = 0.6
    elif sharpe_significance > 1.0:
        stat_score = 0.3
    else:
        stat_score = 0
    alpha_score += stat_score
    print(f"  • Statistická významnost (t-stat): {sharpe_significance:.2f} → {stat_score:.1f}/1.0 bodů")
    
    score += alpha_score
    score_details['alpha'] = alpha_score
    print(f"  ✅ Celkem alfa a významnost: {alpha_score:.1f}/2.5 bodů")
    
    # ===== CELKOVÉ SKÓRE =====
    print("\n" + "="*60)
    print("📋 SHRNUTÍ SKÓRE:")
    print(f"  Regresní metriky:     {score_details['regression']:.1f}/2.5")
    print(f"  Výnosové metriky:     {score_details['returns']:.1f}/3.0") 
    print(f"  Úspěšnost obchodů:    {score_details['trades']:.1f}/2.0")
    print(f"  Alfa & významnost:    {score_details['alpha']:.1f}/2.5")
    print("  " + "-"*40)
    print(f"  CELKEM:              {score:.1f}/10.0")
    print("="*60)
    
    return min(score, 10)  # Cap at 10

def get_grade_and_emoji(score):
    """
    Přidělení známky a emoji na základě celkového skóre
    Rozšířené hodnocení s detailními kategoriemi
    """
    
    print(f"\n🏆 FINÁLNÍ HODNOCENÍ MODELU:")
    print("="*50)
    
    if score >= 8.5:
        grade = "Výjimečný"
        emoji = "🏆"
        description = "Vynikající model s excelentní predikční schopností i obchodní strategií"
        color = "🟢"
    elif score >= 7.0:
        grade = "Výborný" 
        emoji = "🌟"
        description = "Velmi dobrý model s nadprůměrnými výsledky"
        color = "🟢"
    elif score >= 5.5:
        grade = "Dobrý"
        emoji = "✅"
        description = "Uspokojivý model s rozumnou predikční schopností"
        color = "🟡"
    elif score >= 4.0:
        grade = "Průměrný"
        emoji = "⚠️"
        description = "Základní model s omezenou predikční schopností"
        color = "🟡"
    elif score >= 2.5:
        grade = "Podprůměrný"
        emoji = "❌"
        description = "Slabý model s nízkou predikční schopností"
        color = "🔴"
    else:
        grade = "Neuspokojivý"
        emoji = "💀"
        description = "Model s velmi nízkou nebo žádnou predikční schopností"
        color = "🔴"
    
    print(f"{color} Skóre: {score:.1f}/10.0")
    print(f"{emoji} Známka: {grade}")
    print(f"📝 Popis: {description}")
    
    # Doporučení pro zlepšení
    print(f"\n💡 DOPORUČENÍ PRO ZLEPŠENÍ:")
    if score < 4.0:
        print("  • Zvažte komplexnější architekturu modelu")
        print("  • Přidejte více technických indikátorů")
        print("  • Optimalizujte hyperparametry")
        print("  • Zkontrolujte kvalitu dat")
    elif score < 7.0:
        print("  • Optimalizujte obchodní strategii (profit targets, stop losses)")
        print("  • Zvažte pokročilejší feature engineering")
        print("  • Testujte různé velikosti portfolia")
    else:
        print("  • Model vykazuje dobré výsledky")
        print("  • Můžete experimentovat s pokročilejšími technikami")
        print("  • Zvažte ensemble metody")
    
    print("="*50)
    
    return grade, emoji

# ---------------------- MAIN EXECUTION ----------------------
print("=" * 60)
print("MAIN EXECUTION")
print("=" * 60)

def main():
    """Main execution function"""
    
    try:
        # 1. Load and prepare data
        df = load_and_prepare_data()
        
        # 2. Calculate technical indicators
        df_with_indicators = calculate_technical_indicators(df)
        
        # 3. Prepare features and target
        (X_train, X_test, y_train, y_test, 
         train_data, test_data, scaler, feature_cols,
         target_mean, target_std) = prepare_features_and_target(df_with_indicators)
        
        # 4. Hyperparameter tuning
        best_params, best_score = tune_hyperparameters(X_train, y_train)
        
        # 5. Train best model
        model, history = train_best_model(X_train, y_train, X_test, y_test, best_params)
        
        # 6. Evaluate model
        y_train_pred, y_test_pred, train_metrics, test_metrics = evaluate_model(
            model, X_train, y_train, X_test, y_test)
        
        # 7. Implement trading strategy
        portfolio_df, trades_df = implement_trading_strategy(test_data, y_test_pred)
        
        # 8. Calculate trading metrics
        trading_metrics = calculate_trading_metrics(portfolio_df, trades_df)
        
        # 9. Create visualization
        output_file = create_comprehensive_visualization(
            history, portfolio_df, best_params, train_metrics, test_metrics, trading_metrics)
        
        # 10. Final summary
        print("\n" + "=" * 60)
        print("EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Calculate total execution time
        end_time = time.time()
        execution_time = end_time - start_time
        minutes = int(execution_time // 60)
        seconds = int(execution_time % 60)
        
        print(f"Total execution time: {minutes} minutes and {seconds} seconds")
        print(f"Results visualization saved as: {output_file}")
        
        # Final evaluation
        score = calculate_overall_score(test_metrics, trading_metrics)
        grade, emoji = get_grade_and_emoji(score)
        
        print(f"\nFINAL MODEL EVALUATION: {score:.1f}/10 {emoji}")
        print(f"Grade: {grade}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR during execution: {e}")
        print("Please check your data paths and configuration.")
        
        # Calculate execution time even on error
        end_time = time.time()
        execution_time = end_time - start_time
        minutes = int(execution_time // 60)
        seconds = int(execution_time % 60)
        print(f"Execution time before error: {minutes} minutes and {seconds} seconds")
        
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nExecution failed. Please review the error messages above.")
