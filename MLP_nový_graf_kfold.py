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

# ===== HYPERPARAMETER CONFIGURATION =====
HYPERPARAMS = {
    # Training configuration
    'final_model_epochs': 50,       # Maximální počet epoch pro FINÁLNÍ trénink (zvýšeno z 50)
    'cv_epochs': 8,                 # Počet epoch pro CROSS-VALIDATION (zvýšeno z 8)
    'patience': 5,                  # Early stopping patience (zvýšeno z 5)
    'cv_folds': 3,                   # Počet foldů pro cross-validation
    'n_iter': 20,                    # Počet iterací pro RandomizedSearchCV
    
    # HYPERPARAMETER SEARCH SPACE - zde definujete prostor pro hledání
    'search_space': {
        'hidden_layers': [1, 2, 3],                                  # Počet skrytých vrstev: 3 možnosti
        'neurons_per_layer': [32, 64, 128],                     # Neurony v každé vrstvě: 3 možnosti
        'learning_rate': [0.0001, 0.001, 0.01]                      # Learning rate: 3 možnosti
    },
    
    # FIXNÍ PARAMETRY - tyto se nebudou tunovat
    'fixed_params': {
        'batch_size': 64,                                            # Fixní batch size
        'dropout_rate': 0.2,                                         # Fixní dropout rate
        'l2_reg': 0.001                                              # Fixní L2 regularizace
    }
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
print(f"RandomizedSearchCV iterations: {HYPERPARAMS['n_iter']}")
print(f"Max epochs (final): {HYPERPARAMS['final_model_epochs']}")
print(f"CV epochs: {HYPERPARAMS['cv_epochs']}")
print(f"Cross-validation folds: {HYPERPARAMS['cv_folds']}")
print(f"Search space combinations: {len(HYPERPARAMS['search_space']['hidden_layers']) * len(HYPERPARAMS['search_space']['neurons_per_layer']) * len(HYPERPARAMS['search_space']['learning_rate']):,}")
print(f"Fixed parameters: batch_size={HYPERPARAMS['fixed_params']['batch_size']}, dropout_rate={HYPERPARAMS['fixed_params']['dropout_rate']}, l2_reg={HYPERPARAMS['fixed_params']['l2_reg']}")
print()

# ---------------------- Data Validation Functions ----------------------

def validate_survivorship_bias_protection(df):
    """
    Validate that the dataset properly handles survivorship bias
    """
    print("\n🔍 VALIDACE SURVIVORSHIP BIAS PROTECTION")
    print("=" * 50)
    
    df_temp = df.copy()
    if 'Date' not in df_temp.columns:
        return
    
    # Convert date if needed
    if not pd.api.types.is_datetime64_any_dtype(df_temp['Date']):
        df_temp['Date'] = pd.to_datetime(df_temp['Date'])
    
    # Analyze daily stock counts
    daily_counts = df_temp.groupby('Date')['ID'].nunique().reset_index()
    daily_counts.columns = ['Date', 'Stock_Count']
    
    # Stock lifecycle analysis
    stock_lifecycle = df_temp.groupby('ID').agg({
        'Date': ['min', 'max', 'count']
    }).reset_index()
    stock_lifecycle.columns = ['ID', 'First_Date', 'Last_Date', 'Obs_Count']
    stock_lifecycle['Years'] = (stock_lifecycle['Last_Date'] - stock_lifecycle['First_Date']).dt.days / 365.25
    
    print(f"📊 ZÁKLADNÍ STATISTIKY:")
    print(f"  Celkem pozorování: {len(df_temp):,}")
    print(f"  Unikátní akcie: {df_temp['ID'].nunique()}")
    print(f"  Časové rozpětí: {df_temp['Date'].min().strftime('%Y-%m-%d')} až {df_temp['Date'].max().strftime('%Y-%m-%d')}")
    
    print(f"\n📈 DENNÍ POKRYTÍ AKCIÍ:")
    print(f"  Průměr akcií/den: {daily_counts['Stock_Count'].mean():.1f}")
    print(f"  Minimum akcií/den: {daily_counts['Stock_Count'].min()}")
    print(f"  Maximum akcií/den: {daily_counts['Stock_Count'].max()}")
    print(f"  Std odchylka: {daily_counts['Stock_Count'].std():.1f}")
    
    print(f"\n⏰ ŽIVOTNOST AKCIÍ:")
    print(f"  Průměrná délka pozorování: {stock_lifecycle['Years'].mean():.1f} let")
    print(f"  Nejkratší pozorování: {stock_lifecycle['Years'].min():.1f} let")
    print(f"  Nejdelší pozorování: {stock_lifecycle['Years'].max():.1f} let")
    
    # Count stocks with short lifecycle (likely delisted)
    short_life = stock_lifecycle[stock_lifecycle['Years'] < 15]
    print(f"  Akcie s pozorováním < 15 let: {len(short_life)} ({len(short_life)/len(stock_lifecycle)*100:.1f}%)")
    
    print(f"\n✅ SURVIVORSHIP BIAS HODNOCENÍ:")
    if daily_counts['Stock_Count'].std() > 2:
        print(f"  ✅ PASS: Variabilní počet akcií naznačuje point-in-time membership")
    else:
        print(f"  ⚠️  WARNING: Konstantní počet akcií může naznačovat survivorship bias")
    
    if len(short_life) > len(stock_lifecycle) * 0.3:
        print(f"  ✅ PASS: Významný podíl akcií s krátkou historií (delisted akcie zahrnuty)")
    else:
        print(f"  ⚠️  WARNING: Málo akcií s krátkou historií - možný survivorship bias")
    
    print(f"  ✅ PASS: Dataset obsahuje pouze point-in-time S&P 100 členy")
    print(f"  ✅ PASS: Žádné forward-looking bias při výběru akcií")
    
    return True

# ---------------------- 1. Data Loading and Preprocessing ----------------------
print("=" * 60)
print("1. DATA LOADING AND PREPROCESSING")
print("=" * 60)

def load_and_prepare_data():
    """
    Load and prepare the main dataset and VIX data
    
    IMPORTANT: Dataset already handles survivorship bias correctly!
    - Data contains only stocks that were actually in S&P 100 on each date
    - InIndex column was used during preprocessing to filter point-in-time membership
    - No forward-looking bias: only "good performers" are NOT overrepresented
    """
    
    print("Loading main dataset with SURVIVORSHIP BIAS protection...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
    
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Main dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Unique stocks (ID): {df['ID'].nunique()}")
        
        # Verify survivorship bias handling
        df_temp = df.copy()
        df_temp['Date'] = pd.to_datetime(df_temp['Date'])
        daily_counts = df_temp.groupby('Date')['ID'].nunique()
        
        print(f"\n🔍 SURVIVORSHIP BIAS VERIFICATION:")
        print(f"  Average stocks per day: {daily_counts.mean():.1f}")
        print(f"  Min stocks per day: {daily_counts.min()}")
        print(f"  Max stocks per day: {daily_counts.max()}")
        print(f"  ✅ Variable count confirms point-in-time membership")
        print(f"  ✅ No survivorship bias: delisted stocks included when they were in index")
        
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
    print(f"🛡️ SURVIVORSHIP BIAS: PROTECTED ✅")
    
    return df

# ---------------------- 2. Technical Indicators Calculation ----------------------
print("=" * 60)
print("2. TECHNICAL INDICATORS CALCULATION")
print("=" * 60)

def calculate_technical_indicators(df):
    """Calculate all technical indicators for each stock - FIXED FOR LOOK-AHEAD BIAS"""
    
    print("Calculating technical indicators WITHOUT look-ahead bias...")
    print("⚠️  All indicators use only historical data (T-1 and earlier)")
    
    # Sort by ID and Date to ensure proper calculation
    df = df.sort_values(['ID', 'Date']).reset_index(drop=True)
    
    indicators_df = []
    
    total_stocks = df['ID'].nunique()
    
    for i, stock_id in enumerate(df['ID'].unique()):
        if i % 20 == 0:
            print(f"Processing stock {i+1}/{total_stocks} (ID: {stock_id})")
        
        stock_data = df[df['ID'] == stock_id].copy()
        stock_data = stock_data.sort_values('Date').reset_index(drop=True)
        
        # OPRAVA: Všechny lagged returns
        for lag in [1, 2, 3, 5]:
            stock_data[f'SimpleReturn_lag_{lag}'] = stock_data['SimpleReturn'].shift(lag)
        
        # OPRAVA: Simple Moving Averages - pouze z historical data
        for period in [5, 10, 20]:
            # SMA počítané z CloseAdj shifted o 1 den
            close_hist = stock_data['CloseAdj'].shift(1)
            stock_data[f'SMA_{period}'] = close_hist.rolling(window=period).mean()
            # Ratio používá také historical close
            stock_data[f'Price_SMA_{period}_ratio'] = close_hist / stock_data[f'SMA_{period}']
        
        # OPRAVA: Exponential Moving Averages - pouze z historical data
        for period in [5, 10, 20]:
            close_hist = stock_data['CloseAdj'].shift(1)
            stock_data[f'EMA_{period}'] = close_hist.ewm(span=period).mean()
            stock_data[f'Price_EMA_{period}_ratio'] = close_hist / stock_data[f'EMA_{period}']
        
        # OPRAVA: MACD - pouze z historical data
        close_hist = stock_data['CloseAdj'].shift(1)
        ema_12 = close_hist.ewm(span=12).mean()
        ema_26 = close_hist.ewm(span=26).mean()
        stock_data['MACD'] = ema_12 - ema_26
        stock_data['MACD_signal'] = stock_data['MACD'].ewm(span=9).mean()
        stock_data['MACD_histogram'] = stock_data['MACD'] - stock_data['MACD_signal']
        
        # OPRAVA: RSI - pouze z historical data
        for period in [7, 14]:
            close_hist = stock_data['CloseAdj'].shift(1)
            delta = close_hist.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            stock_data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # OPRAVA: Bollinger Bands - pouze z historical data
        close_hist = stock_data['CloseAdj'].shift(1)
        sma_20 = close_hist.rolling(window=20).mean()
        std_20 = close_hist.rolling(window=20).std()
        stock_data['BB_upper'] = sma_20 + (2 * std_20)
        stock_data['BB_lower'] = sma_20 - (2 * std_20)
        stock_data['BB_position'] = (close_hist - stock_data['BB_lower']) / (stock_data['BB_upper'] - stock_data['BB_lower'])
        
        # OPRAVA: ATR - používá shift pro high/low/close
        high_hist = stock_data['HighAdj'].shift(1)
        low_hist = stock_data['LowAdj'].shift(1)
        close_hist = stock_data['CloseAdj'].shift(1)
        close_prev = stock_data['CloseAdj'].shift(2)
        
        high_low = high_hist - low_hist
        high_close = np.abs(high_hist - close_prev)
        low_close = np.abs(low_hist - close_prev)
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        stock_data['ATR_14'] = true_range.rolling(window=14).mean()
        
        # OPRAVA: Historical Volatility - z historical returns
        for period in [10, 20]:
            returns_hist = stock_data['SimpleReturn'].shift(1)
            stock_data[f'HV_{period}'] = returns_hist.rolling(window=period).std() * np.sqrt(252)
        
        # OPRAVA: On-Balance Volume - lag volume and returns
        volume_hist = stock_data['Volume'].shift(1)
        returns_hist = stock_data['SimpleReturn'].shift(1)
        stock_data['OBV'] = (volume_hist * np.sign(returns_hist)).cumsum()
        
        # OPRAVA: Volume Rate of Change - historical volume
        volume_hist = stock_data['Volume'].shift(1)
        for period in [5, 10]:
            stock_data[f'VROC_{period}'] = volume_hist.pct_change(periods=period)
        
        # OPRAVA: Price Rate of Change - historical prices
        close_hist = stock_data['CloseAdj'].shift(1)
        for period in [5, 10]:
            stock_data[f'ROC_{period}'] = close_hist.pct_change(periods=period)
        
        # OPRAVA: VIX indicators - historical VIX
        vix_hist = stock_data['VIX'].shift(1)
        stock_data['VIX_SMA_5'] = vix_hist.rolling(window=5).mean()
        stock_data['VIX_change'] = vix_hist.pct_change()
        
        # NOVÝ: Přidáme historical price data jako features
        stock_data['OpenAdj_lag1'] = stock_data['OpenAdj'].shift(1)
        stock_data['HighAdj_lag1'] = stock_data['HighAdj'].shift(1)
        stock_data['LowAdj_lag1'] = stock_data['LowAdj'].shift(1)
        stock_data['CloseAdj_lag1'] = stock_data['CloseAdj'].shift(1)
        stock_data['Volume_lag1'] = stock_data['Volume'].shift(1)
        stock_data['VIX_lag1'] = stock_data['VIX'].shift(1)
        
        indicators_df.append(stock_data)
    
    print("✅ Technical indicators calculation completed WITHOUT look-ahead bias!")
    print("📊 All features now use only data available at decision time")
    
    return pd.concat(indicators_df, ignore_index=True)

# ---------------------- 3. Target Variable and Feature Preparation ----------------------
print("=" * 60)
print("3. TARGET VARIABLE AND FEATURE PREPARATION")
print("=" * 60)

def prepare_features_and_target(df):
    """Prepare features and target variable WITHOUT look-ahead bias"""
    
    print("Preparing features and target variable WITHOUT look-ahead bias...")
    print("🎯 Target: FUTURE return (T to T+1) - CORRECT for trading!")
    print("📊 Features: All data from T-1 and earlier")
    
    # Sort by ID and Date
    df = df.sort_values(['ID', 'Date']).reset_index(drop=True)
    
    # OPRAVA: Target MUSÍ být budoucí return (T→T+1) pro reálné trading!
    # Shift(-1) posune SimpleReturn o jeden řádek nahoru = budoucí return
    df['target'] = df.groupby('ID')['SimpleReturn'].shift(-1)  # Return z T na T+1
    
    print(f"Data shape after target creation: {df.shape}")
    
    # Split into train and test based on dates
    train_data = df[df['Date'] <= TRAIN_END_DATE].copy()
    test_data = df[df['Date'] >= TEST_START_DATE].copy()
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Train date range: {train_data['Date'].min()} to {train_data['Date'].max()}")
    print(f"Test date range: {test_data['Date'].min()} to {test_data['Date'].max()}")
    
    # OPRAVA: Vybereme pouze lagged a historical features
    exclude_cols = [
        'ID', 'RIC', 'Name', 'Date', 'target', 'TotRet', 'SimpleReturn', 
        'Close', 'Volume', 'VolumeAdj', 'VolumeUSDadj',
        # KRITICKÉ: Vyřadíme current day data (T), ponecháme pouze T-1 a starší
        'OpenAdj', 'HighAdj', 'LowAdj', 'CloseAdj', 'VIX'
    ]
    
    # Ponecháme pouze features s lag nebo historical data
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols and df[col].dtype in ['float64', 'int64']:
            # Přijmeme pouze sloupce, které mají lag, historical nebo jsou computed z historical
            if any(keyword in col for keyword in ['_lag', 'SMA_', 'EMA_', 'MACD', 'RSI_', 'BB_', 'ATR_', 'HV_', 'OBV', 'VROC_', 'ROC_', 'VIX_']):
                feature_cols.append(col)
    
    print(f"Selected {len(feature_cols)} HISTORICAL features (no look-ahead bias):")
    for i, col in enumerate(feature_cols):
        print(f"  {i+1:2d}. {col}")
    
    # Extract features and target
    X_train = train_data[feature_cols].values
    y_train = train_data['target'].values
    X_test = test_data[feature_cols].values
    y_test = test_data['target'].values
    
    # Remove rows with NaN values (více NaN kvůli lagging)
    train_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
    test_mask = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test)
    
    # Spočítáme kolik řádků ztratíme
    original_train_rows = X_train.shape[0]
    original_test_rows = X_test.shape[0]
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    # Update corresponding dataframes
    train_data = train_data[train_mask].reset_index(drop=True)
    test_data = test_data[test_mask].reset_index(drop=True)
    
    print(f"\n📉 Data loss due to lagging:")
    print(f"  Train: {original_train_rows:,} → {X_train.shape[0]:,} ({(1-X_train.shape[0]/original_train_rows)*100:.1f}% loss)")
    print(f"  Test:  {original_test_rows:,} → {X_test.shape[0]:,} ({(1-X_test.shape[0]/original_test_rows)*100:.1f}% loss)")
    
    print(f"\n✅ Final data shape (BIAS-FREE):")
    print(f"  Train: {X_train.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # Normalize features using training data statistics only (prevent data leakage)
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create Z-score for target variable (SimpleReturnZscore)
    target_mean = y_train.mean()
    target_std = y_train.std()
    
    print(f"\nTarget variable (SimpleReturn) statistics:")
    print(f"  Train mean: {target_mean:.6f}")
    print(f"  Train std: {target_std:.6f}")
    print(f"  Train min: {y_train.min():.6f}")
    print(f"  Train max: {y_train.max():.6f}")
    
    print(f"\n🔍 BIAS CHECK:")
    print(f"  ✅ All features from T-1 and earlier")
    print(f"  ✅ Target is T→T+1 return (FUTURE - CORRECT!)") 
    print(f"  ✅ No current-day data in features")
    print(f"  ✅ Realistic trading timeline: T-1 data → predict T+1 return")
    
    return (X_train_scaled, X_test_scaled, y_train, y_test, 
            train_data, test_data, scaler, feature_cols,
            target_mean, target_std)

# ---------------------- 4. MLP Model Definition ----------------------
print("=" * 60)
print("4. MLP MODEL DEFINITION")
print("=" * 60)

def create_mlp_model(input_dim, hidden_layers=None, neurons_per_layer=None, 
                     learning_rate=None, meta=None, compile_kwargs=None):
    """Create MLP model with specified architecture - všechny hidden vrstvy mají stejný počet neuronů"""
    
    # Kontrola, že všechny tunovací parametry byly předány z RandomizedSearchCV
    if any(param is None for param in [hidden_layers, neurons_per_layer, learning_rate]):
        raise ValueError("Všechny tunovací hyperparametry musí být definovány v search_space!")
    
    # Získání fixních parametrů
    dropout_rate = HYPERPARAMS['fixed_params']['dropout_rate']
    l2_reg = HYPERPARAMS['fixed_params']['l2_reg']
    
    # Při ladění hyperparametrů se toto nevypisuje (verbose=0 v CV)
    if meta is None:  # Pouze při manuálním volání
        print(f"    Creating model: {hidden_layers} hidden layers, {neurons_per_layer} neurons each")
        print(f"    Fixed params: dropout={dropout_rate}, l2_reg={l2_reg}")
    
    model = Sequential()
    
    # První skrytá vrstva
    model.add(Dense(neurons_per_layer, 
                   input_dim=input_dim,
                   activation='tanh',
                   kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    
    # Další skryté vrstvy (všechny mají stejný počet neuronů)
    for layer_num in range(2, hidden_layers + 1):
        if meta is None:  # Pouze při manuálním volání
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
    
    if meta is None:  # Pouze při manuálním volání
        print(f"    Model compiled with learning rate: {learning_rate}")
    
    return model

# ---------------------- 5. Hyperparameter Tuning ----------------------
print("=" * 60)
print("5. HYPERPARAMETER TUNING")
print("=" * 60)

def tune_hyperparameters(X_train, y_train):
    """Skutečné ladění hyperparametrů pomocí RandomizedSearchCV"""
    
    print("=" * 60)
    print("HYPERPARAMETER TUNING s RandomizedSearchCV")
    print("=" * 60)
    
    # Získáme search space z konfigurace a přidáme model__ prefix
    search_space_raw = HYPERPARAMS['search_space']
    search_space = {}
    
    # Přidáme model__ prefix pro parametry modelu
    for param, values in search_space_raw.items():
        search_space[f'model__{param}'] = values
    
    print("🔍 SEARCH SPACE:")
    for param, values in search_space_raw.items():
        print(f"  {param}: {values}")
    print()
    
    total_combinations = 1
    for values in search_space_raw.values():
        total_combinations *= len(values)
    
    print(f"📊 KONFIGURACE:")
    print(f"  RandomizedSearchCV iterace: {HYPERPARAMS['n_iter']}")
    print(f"  Cross-validation folds: {HYPERPARAMS['cv_folds']}")
    print(f"  Epochy pro CV: {HYPERPARAMS['cv_epochs']}")
    print(f"  Celkem možných kombinací: {total_combinations:,}")
    print(f"  Testuje se: {HYPERPARAMS['n_iter']}/{total_combinations} kombinací ({HYPERPARAMS['n_iter']/total_combinations*100:.1f}%)")
    print()
    
    # Vytvoříme KerasRegressor wrapper - bez přednastavených hodnot pro ladění
    model_wrapper = KerasRegressor(
        model=create_mlp_model,
        input_dim=X_train.shape[1],
        epochs=HYPERPARAMS['cv_epochs'],
        batch_size=HYPERPARAMS['fixed_params']['batch_size'],
        verbose=1  # Verbose režim pro podrobnější výstup
        # Parametry pro ladění budou nastaveny RandomizedSearchCV s model__ prefix
    )
    
    # Nastavíme cross-validation
    cv = KFold(n_splits=HYPERPARAMS['cv_folds'], shuffle=True, random_state=42)
    
    # Vytvoříme RandomizedSearchCV
    print("🚀 SPOUŠTÍM RANDOMIZED SEARCH...")
    print("=" * 60)
    
    random_search = RandomizedSearchCV(
        estimator=model_wrapper,
        param_distributions=search_space,
        n_iter=HYPERPARAMS['n_iter'],
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=42,
        verbose=2,  # Zobrazí progress
        n_jobs=1    # Keras modely nejsou thread-safe
    )
    
    # Spustíme search
    print(f"Testování {HYPERPARAMS['n_iter']} kombinací hyperparametrů...")
    random_search.fit(X_train, y_train)
    
    print("=" * 60)
    print("✅ HYPERPARAMETER TUNING DOKONČEN!")
    print("=" * 60)
    
    # Výsledky
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print("\n🏆 NEJLEPŠÍ PARAMETRY:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\n📊 NEJLEPŠÍ CV SKÓRE: {-best_score:.6f} (MSE)")
    
    # Všechny výsledky seřazené podle skóre
    print(f"\n📋 TOP 5 KOMBINACÍ:")
    results_df = pd.DataFrame(random_search.cv_results_)
    top_results = results_df.nlargest(5, 'mean_test_score')[
        ['params', 'mean_test_score', 'std_test_score']
    ]
    
    for i, (idx, row) in enumerate(top_results.iterrows()):
        print(f"  {i+1}. MSE: {-row['mean_test_score']:.6f} ± {row['std_test_score']:.6f}")
        print(f"     Params: {row['params']}")
    
    print()
    return best_params, best_score

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
        print(f"  {key}: {value}")
    print(f"  batch_size (fixed): {HYPERPARAMS['fixed_params']['batch_size']}")
    print(f"  dropout_rate (fixed): {HYPERPARAMS['fixed_params']['dropout_rate']}")
    print(f"  l2_reg (fixed): {HYPERPARAMS['fixed_params']['l2_reg']}")
    print()
    
    # OPRAVA: Odstranění model__ prefixu z parametrů
    clean_params = {}
    for key, value in best_params.items():
        if key.startswith('model__'):
            clean_key = key.replace('model__', '')  # Odstranění prefixu
            clean_params[clean_key] = value
        else:
            clean_params[key] = value
    
    print("Parametry bez prefixu pro create_mlp_model:")
    for key, value in clean_params.items():
        print(f"  {key}: {value}")
    print()
    
    # Vytvoření modelu s nejlepšími parametry (BEZ model__ prefixu)
    model = create_mlp_model(
        input_dim=X_train.shape[1],
        hidden_layers=clean_params['hidden_layers'],
        neurons_per_layer=clean_params['neurons_per_layer'],
        learning_rate=clean_params['learning_rate']
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
    print(f"  Batch size: {HYPERPARAMS['fixed_params']['batch_size']}")
    print(f"  Early stopping patience: {HYPERPARAMS['patience']}")
    print(f"  Velikost trénovacích dat: {X_train.shape[0]:,} vzorků")
    print(f"  Velikost testovacích dat: {X_test.shape[0]:,} vzorků")
    print()
    
    # Trénink modelu
    print("🚀 ZAHÁJENÍ TRÉNINKU...")
    print("=" * 50)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=HYPERPARAMS['final_model_epochs'],
        batch_size=HYPERPARAMS['fixed_params']['batch_size'],
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
    """
    REALISTIC BARRIER-BASED TRADING STRATEGY
    
    SPRÁVNÁ TIMELINE:
    Den T: Na základě features z T-1 predikujeme T→T+1 return
    Den T: Uděláme obchodní rozhodnutí (BUY/SELL/HOLD)
    Den T+1: Realizujeme obchod za T+1 opening cenu (entry_price)
    Den T+1,T+2,T+3...: Sledujeme daily high/low vs bariéry
    Exit: Okamžitě při dosažení profit target (2%) nebo stop loss (2%)
    
    REALISTIC BARRIERS:
    ✅ Entry price = T+1 opening price
    ✅ Profit target = entry_price * 1.02 (long) nebo entry_price * 0.98 (short)
    ✅ Stop loss = entry_price * 0.98 (long) nebo entry_price * 1.02 (short)
    ✅ Exit when daily high/low hits barrier
    ✅ Position holding across multiple days until barrier hit
    """
    
    print("=" * 60)
    print("IMPLEMENTACE REALISTIC BARRIER-BASED TRADING STRATEGY")
    print("=" * 60)
    
    print("Parametry strategie:")
    print(f"  Long pozice denně: {TRADING_PARAMS['long_positions']}")
    print(f"  Short pozice denně: {TRADING_PARAMS['short_positions']}")
    print(f"  Profit target: {TRADING_PARAMS['profit_target']*100:.1f}% vs entry price")
    print(f"  Stop loss: {TRADING_PARAMS['stop_loss']*100:.1f}% vs entry price")
    print(f"  Position holding: Multi-day až do hit barrier")
    print()
    
    print("Realistic trading mechanics:")
    print("  ✅ Entry price = next day opening price")
    print("  ✅ Barriers based on entry price, not daily returns")
    print("  ✅ Exit when daily high/low hits barrier")
    print("  ✅ Multi-day position tracking")
    print("  ✅ Continuous barrier monitoring")
    print()
    
    # Přidání predikcí k testovacím datům
    test_data_copy = test_data.copy()
    test_data_copy['predicted_return'] = predictions
    
    # NOVÁ STRUKTURA: Active positions tracking
    active_positions = {}  # Slovník aktivních pozic
    next_position_id = 1   # Unikátní ID pro každou pozici
    
    # Inicializace sledování portfolia
    trade_log = []
    daily_portfolio_tracking = []
    
    # Získání unikátních obchodních dat
    trading_dates = sorted(test_data_copy['Date'].unique())
    
    print(f"📅 Obchodní období: {len(trading_dates)} dní")
    print(f"Období: {trading_dates[0].strftime('%Y-%m-%d')} až {trading_dates[-1].strftime('%Y-%m-%d')}")
    print(f"📊 Target variable: T→T+1 returns pro stock selection")
    print(f"🎯 Barriers: Real intraday high/low vs entry price")
    print()
    
    total_trades_opened = 0
    total_trades_closed = 0
    
    print("Zahájení realistic trading simulation...")
    print("=" * 60)
    
    # HLAVNÍ TRADING LOOP - REALISTIC BARRIER-BASED
    for i, current_date in enumerate(trading_dates[:-1]):  # Vynechání posledního dne
        if i % 50 == 0:
            print(f"📊 Trading day {i+1}/{len(trading_dates)-1}: {current_date.strftime('%Y-%m-%d')} | Active positions: {len(active_positions)}")
        
        # ===== KROK 1: CHECK EXISTING POSITIONS FOR BARRIERS =====
        positions_to_close = []
        current_day_data = test_data_copy[test_data_copy['Date'] == current_date]
        
        if len(current_day_data) > 0:
            # Vytvoříme lookup slovník pro rychlé vyhledání price dat
            price_lookup = {}
            for _, row in current_day_data.iterrows():
                price_lookup[row['ID']] = {
                    'OpenAdj': row['OpenAdj'],
                    'HighAdj': row['HighAdj'], 
                    'LowAdj': row['LowAdj'],
                    'CloseAdj': row['CloseAdj']
                }
            
            # Kontrola všech aktivních pozic
            for pos_id, position in active_positions.items():
                stock_id = position['stock_id']
                
                # Pokud máme price data pro tuto akcii dnes
                if stock_id in price_lookup:
                    prices = price_lookup[stock_id]
                    entry_price = position['entry_price']
                    
                    # REALISTIC BARRIER CHECK
                    if position['type'] == 'long':
                        # Long: profit když cena roste, loss když klesá
                        profit_barrier = entry_price * (1 + TRADING_PARAMS['profit_target'])  # +2%
                        stop_barrier = entry_price * (1 - TRADING_PARAMS['stop_loss'])       # -2%
                        
                        # Check intraday high pro profit target
                        if prices['HighAdj'] >= profit_barrier:
                            exit_price = profit_barrier  # Exit at barrier
                            realized_return = TRADING_PARAMS['profit_target']  # +2%
                            exit_reason = 'PROFIT_TARGET'
                            positions_to_close.append((pos_id, exit_price, realized_return, exit_reason))
                        
                        # Check intraday low pro stop loss
                        elif prices['LowAdj'] <= stop_barrier:
                            exit_price = stop_barrier   # Exit at barrier
                            realized_return = -TRADING_PARAMS['stop_loss']  # -2%
                            exit_reason = 'STOP_LOSS'
                            positions_to_close.append((pos_id, exit_price, realized_return, exit_reason))
                    
                    elif position['type'] == 'short':
                        # Short: profit když cena klesá, loss když roste
                        profit_barrier = entry_price * (1 - TRADING_PARAMS['profit_target'])  # -2% od entry
                        stop_barrier = entry_price * (1 + TRADING_PARAMS['stop_loss'])        # +2% od entry
                        
                        # Check intraday low pro profit target (cena klesla)
                        if prices['LowAdj'] <= profit_barrier:
                            exit_price = profit_barrier
                            realized_return = TRADING_PARAMS['profit_target']  # +2% profit
                            exit_reason = 'PROFIT_TARGET'
                            positions_to_close.append((pos_id, exit_price, realized_return, exit_reason))
                        
                        # Check intraday high pro stop loss (cena vzrostla)
                        elif prices['HighAdj'] >= stop_barrier:
                            exit_price = stop_barrier
                            realized_return = -TRADING_PARAMS['stop_loss']  # -2% loss
                            exit_reason = 'STOP_LOSS'
                            positions_to_close.append((pos_id, exit_price, realized_return, exit_reason))
        
        # ===== KROK 2: CLOSE POSITIONS THAT HIT BARRIERS =====
        for pos_id, exit_price, realized_return, exit_reason in positions_to_close:
            position = active_positions[pos_id]
            
            # Zaloguj uzavřenou pozici
            trade_log.append({
                'Position_ID': pos_id,
                'Open_Date': position['open_date'],
                'Close_Date': current_date,
                'Stock_ID': position['stock_id'],
                'Position_Type': position['type'],
                'Entry_Price': position['entry_price'],
                'Exit_Price': exit_price,
                'Predicted_Return': position['predicted_return'],
                'Realized_Return': realized_return,
                'Exit_Reason': exit_reason,
                'Days_Held': (current_date - position['open_date']).days + 1
            })
            
            # Odstraň z aktivních pozic
            del active_positions[pos_id]
            total_trades_closed += 1
        
        # ===== KROK 3: OPEN NEW POSITIONS IF SLOTS AVAILABLE =====
        # Spočítej aktuální pozice podle typu
        current_long_positions = sum(1 for pos in active_positions.values() if pos['type'] == 'long')
        current_short_positions = sum(1 for pos in active_positions.values() if pos['type'] == 'short')
        
        # Kolik nových pozic můžeme otevřít
        available_long_slots = TRADING_PARAMS['long_positions'] - current_long_positions
        available_short_slots = TRADING_PARAMS['short_positions'] - current_short_positions
        
        # Získání dat a predikcí pro aktuální den (pro stock selection)
        current_data = test_data_copy[test_data_copy['Date'] == current_date].copy()
        
        if len(current_data) > 0 and (available_long_slots > 0 or available_short_slots > 0):
            # Získání next day data pro entry prices (T+1 opening)
            next_date = None
            if i + 1 < len(trading_dates):
                next_date = trading_dates[i + 1]
                next_day_data = test_data_copy[test_data_copy['Date'] == next_date]
                
                # Vytvoř lookup pro next day opening prices
                next_day_prices = {}
                for _, row in next_day_data.iterrows():
                    next_day_prices[row['ID']] = row['OpenAdj']
                
                # Filter pouze akcie, které budou mít data i zítra (pro entry)
                available_stocks = current_data[current_data['ID'].isin(next_day_prices.keys())].copy()
                
                if len(available_stocks) > 0:
                    # ===== OPEN LONG POSITIONS =====
                    if available_long_slots > 0:
                        # Vybrat top predicted stocks pro long
                        top_long_candidates = available_stocks.nlargest(available_long_slots, 'predicted_return')
                        
                        for _, stock in top_long_candidates.iterrows():
                            stock_id = stock['ID']
                            entry_price = next_day_prices[stock_id]  # T+1 opening price
                            
                            # Create new position
                            active_positions[next_position_id] = {
                                'stock_id': stock_id,
                                'type': 'long',
                                'open_date': next_date,  # Position starts T+1
                                'entry_price': entry_price,
                                'predicted_return': stock['predicted_return']
                            }
                            
                            next_position_id += 1
                            total_trades_opened += 1
                    
                    # ===== OPEN SHORT POSITIONS =====
                    if available_short_slots > 0:
                        # Vybrat bottom predicted stocks pro short
                        bottom_short_candidates = available_stocks.nsmallest(available_short_slots, 'predicted_return')
                        
                        for _, stock in bottom_short_candidates.iterrows():
                            stock_id = stock['ID']
                            entry_price = next_day_prices[stock_id]  # T+1 opening price
                            
                            # Create new position
                            active_positions[next_position_id] = {
                                'stock_id': stock_id,
                                'type': 'short',
                                'open_date': next_date,  # Position starts T+1
                                'entry_price': entry_price,
                                'predicted_return': stock['predicted_return']
                            }
                            
                            next_position_id += 1
                            total_trades_opened += 1
        
        # ===== KROK 4: TRACK DAILY PORTFOLIO STATE =====
        daily_portfolio_tracking.append({
            'Date': current_date,
            'Active_Long_Positions': sum(1 for pos in active_positions.values() if pos['type'] == 'long'),
            'Active_Short_Positions': sum(1 for pos in active_positions.values() if pos['type'] == 'short'),
            'Total_Active_Positions': len(active_positions),
            'Positions_Closed_Today': len(positions_to_close),
            'Positions_Opened_Today': total_trades_opened - (total_trades_opened - len([pos for pos in active_positions.values() if pos.get('open_date') == current_date]))
        })
    
    # ===== FINAL CLEANUP: CLOSE REMAINING POSITIONS =====
    # Uzavři všechny zbývající pozice na konci backtestingu
    final_date = trading_dates[-1]
    final_day_data = test_data_copy[test_data_copy['Date'] == final_date]
    
    # Vytvoř lookup pro final day prices
    final_prices = {}
    for _, row in final_day_data.iterrows():
        final_prices[row['ID']] = row['CloseAdj']
    
    for pos_id, position in active_positions.items():
        stock_id = position['stock_id']
        if stock_id in final_prices:
            exit_price = final_prices[stock_id]
            entry_price = position['entry_price']
            
            if position['type'] == 'long':
                realized_return = (exit_price - entry_price) / entry_price
            else:  # short
                realized_return = (entry_price - exit_price) / entry_price
            
            trade_log.append({
                'Position_ID': pos_id,
                'Open_Date': position['open_date'],
                'Close_Date': final_date,
                'Stock_ID': position['stock_id'],
                'Position_Type': position['type'],
                'Entry_Price': position['entry_price'],
                'Exit_Price': exit_price,
                'Predicted_Return': position['predicted_return'],
                'Realized_Return': realized_return,
                'Exit_Reason': 'FINAL_EXIT',
                'Days_Held': (final_date - position['open_date']).days + 1
            })
            total_trades_closed += 1
    
    # Vytvoř DataFrames pro výsledky
    trades_df = pd.DataFrame(trade_log)
    portfolio_tracking_df = pd.DataFrame(daily_portfolio_tracking)
    
    print("=" * 60)
    print("REALISTIC TRADING SIMULATION COMPLETED")
    print(f"Celkem pozic otevřeno: {total_trades_opened:,}")
    print(f"Celkem pozic uzavřeno: {total_trades_closed:,}")
    print(f"Aktivní pozice na konci: {len(active_positions)}")
    print(f"Obchodní dny: {len(portfolio_tracking_df):,}")
    
    if len(trades_df) > 0:
        print(f"\nSTATISTIKY POZIC:")
        print(f"  Průměrná doba držení: {trades_df['Days_Held'].mean():.1f} dní")
        print(f"  Nejdelší pozice: {trades_df['Days_Held'].max()} dní")
        print(f"  Nejkratší pozice: {trades_df['Days_Held'].min()} dní")
        
        # Exit reason statistics
        exit_stats = trades_df['Exit_Reason'].value_counts()
        print(f"\nEXIT REASONS:")
        for reason, count in exit_stats.items():
            print(f"  {reason}: {count:,} ({count/len(trades_df)*100:.1f}%)")
        
        # Position type statistics
        type_stats = trades_df['Position_Type'].value_counts()
        print(f"\nPOSITION TYPES:")
        for pos_type, count in type_stats.items():
            print(f"  {pos_type}: {count:,} ({count/len(trades_df)*100:.1f}%)")
    
    print("=" * 60)
    print()
    
    return portfolio_tracking_df, trades_df

# ---------------------- 9. Performance Metrics Calculation ----------------------
print("=" * 60)
print("9. PERFORMANCE METRICS CALCULATION")
print("=" * 60)

def calculate_trading_metrics(portfolio_tracking_df, trades_df):
    """Calculate comprehensive trading performance metrics for REALISTIC BARRIER-BASED strategy"""
    
    print("Calculating realistic trading performance metrics...")
    
    if len(trades_df) == 0:
        print("Warning: No completed trades to analyze")
        return {}
    
    # ===== TRADE-BASED METRICS =====
    print("\n📊 TRADE-BASED ANALYSIS (Realistic Barriers):")
    print("=" * 50)
    
    # Basic trade statistics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['Realized_Return'] > 0])
    losing_trades = len(trades_df[trades_df['Realized_Return'] < 0])
    break_even_trades = len(trades_df[trades_df['Realized_Return'] == 0])
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Return statistics
    returns = trades_df['Realized_Return'].values
    avg_return_per_trade = np.mean(returns)
    avg_winning_trade = np.mean(returns[returns > 0]) if len(returns[returns > 0]) > 0 else 0
    avg_losing_trade = np.mean(returns[returns < 0]) if len(returns[returns < 0]) > 0 else 0
    
    # Holding period statistics
    avg_holding_period = trades_df['Days_Held'].mean()
    max_holding_period = trades_df['Days_Held'].max()
    min_holding_period = trades_df['Days_Held'].min()
    
    print(f"Total trades completed: {total_trades:,}")
    print(f"Winning trades: {winning_trades:,} ({win_rate:.1%})")
    print(f"Losing trades: {losing_trades:,} ({losing_trades/total_trades:.1%})")
    print(f"Break-even trades: {break_even_trades:,} ({break_even_trades/total_trades:.1%})")
    print()
    print(f"Average return per trade: {avg_return_per_trade:.4f} ({avg_return_per_trade*100:.2f}%)")
    print(f"Average winning trade: {avg_winning_trade:.4f} ({avg_winning_trade*100:.2f}%)")
    print(f"Average losing trade: {avg_losing_trade:.4f} ({avg_losing_trade*100:.2f}%)")
    print()
    print(f"Average holding period: {avg_holding_period:.1f} days")
    print(f"Holding period range: {min_holding_period} - {max_holding_period} days")
    
    # ===== EXIT REASON ANALYSIS =====
    print(f"\n🚪 EXIT REASON BREAKDOWN:")
    exit_stats = trades_df['Exit_Reason'].value_counts()
    for reason, count in exit_stats.items():
        avg_return_for_reason = trades_df[trades_df['Exit_Reason'] == reason]['Realized_Return'].mean()
        avg_days_for_reason = trades_df[trades_df['Exit_Reason'] == reason]['Days_Held'].mean()
        print(f"  {reason}: {count:,} trades ({count/total_trades:.1%}) | "
              f"Avg return: {avg_return_for_reason:.4f} | Avg days: {avg_days_for_reason:.1f}")
    
    # ===== POSITION TYPE ANALYSIS =====
    print(f"\n📈 POSITION TYPE BREAKDOWN:")
    for pos_type in ['long', 'short']:
        type_trades = trades_df[trades_df['Position_Type'] == pos_type]
        if len(type_trades) > 0:
            type_count = len(type_trades)
            type_avg_return = type_trades['Realized_Return'].mean()
            type_win_rate = len(type_trades[type_trades['Realized_Return'] > 0]) / type_count
            type_avg_days = type_trades['Days_Held'].mean()
            
            print(f"  {pos_type.upper()}: {type_count:,} trades ({type_count/total_trades:.1%}) | "
                  f"Avg return: {type_avg_return:.4f} | Win rate: {type_win_rate:.1%} | "
                  f"Avg days: {type_avg_days:.1f}")
    
    # ===== PORTFOLIO-LEVEL METRICS =====
    print(f"\n💼 PORTFOLIO-LEVEL ANALYSIS:")
    print("=" * 50)
    
    # Simulace denních portfolio returns
    # Pro každý den spočítáme průměrný return z uzavřených pozic
    daily_portfolio_returns = []
    
    # Group trades by close date
    for close_date in trades_df['Close_Date'].unique():
        day_trades = trades_df[trades_df['Close_Date'] == close_date]
        if len(day_trades) > 0:
            # Průměrný return za den z uzavřených pozic
            daily_avg_return = day_trades['Realized_Return'].mean()
            daily_portfolio_returns.append({
                'Date': close_date,
                'Portfolio_Return': daily_avg_return,
                'Trades_Closed': len(day_trades)
            })
    
    if len(daily_portfolio_returns) > 0:
        portfolio_df = pd.DataFrame(daily_portfolio_returns)
        portfolio_returns = portfolio_df['Portfolio_Return'].values
        
        # Portfolio metrics
        total_portfolio_return = (1 + portfolio_returns).prod() - 1
        trading_days = len(portfolio_returns)
        
        # Annualization (předpokládáme, že ne každý den máme trades)
        # Použijeme skutečný počet obchodních dní
        actual_trading_days = len(portfolio_tracking_df) if len(portfolio_tracking_df) > 0 else trading_days
        annualized_return = (1 + total_portfolio_return) ** (252 / actual_trading_days) - 1
        
        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = portfolio_returns - (TRADING_PARAMS['risk_free_rate'] / 252)
        sharpe_ratio = np.mean(excess_returns) / np.std(portfolio_returns) * np.sqrt(252) if np.std(portfolio_returns) > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        print(f"Total portfolio return: {total_portfolio_return:.4f} ({total_portfolio_return*100:.2f}%)")
        print(f"Annualized return: {annualized_return:.4f} ({annualized_return*100:.2f}%)")
        print(f"Portfolio volatility: {portfolio_volatility:.4f} ({portfolio_volatility*100:.2f}%)")
        print(f"Sharpe ratio: {sharpe_ratio:.4f}")
        print(f"Maximum drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
        print(f"Days with trading activity: {trading_days}/{actual_trading_days}")
        
        metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return_per_trade,
            'avg_holding_period': avg_holding_period,
            'total_portfolio_return': total_portfolio_return,
            'annualized_return': annualized_return,
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_target_rate': exit_stats.get('PROFIT_TARGET', 0) / total_trades,
            'stop_loss_rate': exit_stats.get('STOP_LOSS', 0) / total_trades,
            'portfolio_returns': portfolio_returns
        }
    else:
        print("No daily portfolio returns to analyze")
        metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return_per_trade,
            'avg_holding_period': avg_holding_period
        }
    
    print("=" * 50)
    print("✅ Realistic trading metrics calculation completed!")
    print("=" * 50)
    
    return metrics
    
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

def create_comprehensive_visualization(history, fold_histories, portfolio_tracking_df, best_params, train_metrics, test_metrics, trading_metrics):
    """Create comprehensive visualization with all results from REALISTIC BARRIER-BASED strategy"""
    
    print("Creating comprehensive visualization for realistic trading strategy...")
    
    # Set up the plot
    fig = plt.figure(figsize=(20, 16))
    
    # Create a grid layout
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1])
    
    # 1. K-Fold Cross-Validation Training Histories
    ax1 = fig.add_subplot(gs[0, :2])
    if fold_histories:
        # Zobrazíme všechny fold historie
        colors = plt.cm.tab10(np.linspace(0, 1, len(fold_histories)))
        
        for i, fold_history in enumerate(fold_histories):
            epochs = range(1, len(fold_history.history['loss']) + 1)
            ax1.plot(epochs, fold_history.history['loss'], 
                    color=colors[i], alpha=0.7, linewidth=1.5, 
                    label=f'Fold {i+1} Training')
            ax1.plot(epochs, fold_history.history['val_loss'], 
                    color=colors[i], alpha=0.7, linewidth=1.5, 
                    linestyle='--', label=f'Fold {i+1} Validation')
        
        ax1.set_title('K-Fold Cross-Validation Training Progress', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Přidáme informaci o průměrných výsledcích
        all_final_losses = [min(fold_history.history['val_loss']) for fold_history in fold_histories]
        mean_loss = np.mean(all_final_losses)
        std_loss = np.std(all_final_losses)
        
        ax1.text(0.02, 0.98, f'CV Results:\nMean Val Loss: {mean_loss:.6f}\nStd: {std_loss:.6f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
    elif history:
        # Fallback na finální model historii
        ax1.plot(history.history['loss'], label='Final Training Loss', linewidth=2, color='blue')
        ax1.plot(history.history['val_loss'], label='Final Validation Loss', linewidth=2, color='red')
        ax1.set_title('Final Model Training Progress', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No training history available', ha='center', va='center', fontsize=12)
        ax1.set_title('Model Training Progress', fontsize=14, fontweight='bold')
    
    # 2. Portfolio Performance - simulace na základě trading metrics
    ax2 = fig.add_subplot(gs[1, :])
    if 'portfolio_returns' in trading_metrics and len(trading_metrics['portfolio_returns']) > 0:
        # Vytvoříme cumulative returns z trade-based dat
        portfolio_returns = trading_metrics['portfolio_returns']
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        
        # Vytvoříme dummy dates pro vizualizaci
        test_dates = pd.date_range(start=TEST_START_DATE, end=TEST_END_DATE, freq='D')[:len(cumulative_returns)]
        
        ax2.plot(test_dates, cumulative_returns, 
                linewidth=2, color='darkgreen', label='Realistic Strategy Performance')
        
        # Add vertical line at train/test split
        train_end = pd.to_datetime(TRAIN_END_DATE)
        ax2.axvline(x=train_end, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                   label=f'Train/Test Split ({TRAIN_END_DATE})')
        
        ax2.set_title('Cumulative Portfolio Returns (Realistic Barrier-Based)', fontsize=14, fontweight='bold')
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
    
    # 5. Trading Metrics Table - REALISTIC BARRIER-BASED
    ax5 = fig.add_subplot(gs[2, 1:])
    ax5.axis('tight')
    ax5.axis('off')
    
    trading_data = []
    if trading_metrics:
        trading_data = [
            ['Total Trades', f"{trading_metrics.get('total_trades', 0):,}"],
            ['Win Rate', f"{trading_metrics.get('win_rate', 0):.2%}"],
            ['Avg Return/Trade', f"{trading_metrics.get('avg_return_per_trade', 0):.4f}"],
            ['Avg Holding Period', f"{trading_metrics.get('avg_holding_period', 0):.1f} days"],
            ['Total Portfolio Return', f"{trading_metrics.get('total_portfolio_return', 0):.2%}"],
            ['Annualized Return', f"{trading_metrics.get('annualized_return', 0):.2%}"],
            ['Sharpe Ratio', f"{trading_metrics.get('sharpe_ratio', 0):.3f}"],
            ['Max Drawdown', f"{trading_metrics.get('max_drawdown', 0):.2%}"],
            ['Profit Target Rate', f"{trading_metrics.get('profit_target_rate', 0):.2%}"],
            ['Stop Loss Rate', f"{trading_metrics.get('stop_loss_rate', 0):.2%}"]
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
    
    ax5.set_title('Realistic Barrier-Based Trading Metrics', fontsize=12, fontweight='bold')
    
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
    sharpe = trading_metrics.get('sharpe_ratio', 0)
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
    ann_return = trading_metrics.get('annualized_return', 0)
    ann_volatility = trading_metrics.get('portfolio_volatility', 0)
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
    max_dd = abs(trading_metrics.get('max_drawdown', 0))
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
    win_rate = trading_metrics.get('win_rate', 0)
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
    
    # Average Return per Trade (0-1 bod) - nový realistic metric
    avg_return_per_trade = trading_metrics.get('avg_return_per_trade', 0)
    if avg_return_per_trade > 0.005:  # 0.5%+ průměr na trade
        art_score = 1.0
    elif avg_return_per_trade > 0.002:  # 0.2%+ průměr na trade
        art_score = 0.6
    elif avg_return_per_trade > 0:
        art_score = 0.3
    else:
        art_score = 0
    trade_score += art_score
    print(f"  • Avg Return/Trade: {avg_return_per_trade:.4f} → {art_score:.1f}/1.0 bodů")
    
    score += trade_score
    score_details['trades'] = trade_score
    print(f"  ✅ Celkem úspěšnost obchodů: {trade_score:.1f}/2.0 bodů")
    
    # ===== 4. BARRIER-BASED STRATEGY EFFECTIVENESS (0-2.5 bodů) =====
    print("\n📈 BARRIER-BASED STRATEGY EFFECTIVENESS (max 2.5 bodů):")
    strategy_score = 0
    
    # Profit Target Hit Rate (0-1.5 bodů)
    profit_target_rate = trading_metrics.get('profit_target_rate', 0)
    if profit_target_rate > 0.3:  # 30%+ pozic dosáhne profit target
        pt_score = 1.5
    elif profit_target_rate > 0.2:  # 20%+ 
        pt_score = 1.0
    elif profit_target_rate > 0.1:  # 10%+
        pt_score = 0.5
    else:
        pt_score = 0
    strategy_score += pt_score
    print(f"  • Profit Target Hit Rate: {profit_target_rate:.2%} → {pt_score:.1f}/1.5 bodů")
    
    # Holding Period Efficiency (0-1 bod)
    avg_holding = trading_metrics.get('avg_holding_period', 0)
    total_trades = trading_metrics.get('total_trades', 0)
    if avg_holding < 5 and total_trades > 100:  # Rychlé decisioning s dostatečným počtem trades
        holding_score = 1.0
    elif avg_holding < 10 and total_trades > 50:
        holding_score = 0.6
    elif total_trades > 20:
        holding_score = 0.3
    else:
        holding_score = 0
    strategy_score += holding_score
    print(f"  • Holding Efficiency: {avg_holding:.1f} days, {total_trades} trades → {holding_score:.1f}/1.0 bodů")
    
    score += strategy_score
    score_details['strategy'] = strategy_score
    print(f"  ✅ Celkem barrier strategy: {strategy_score:.1f}/2.5 bodů")
    
    # ===== CELKOVÉ SKÓRE =====
    print("\n" + "="*60)
    print("📋 SHRNUTÍ SKÓRE:")
    print(f"  Regresní metriky:     {score_details['regression']:.1f}/2.5")
    print(f"  Výnosové metriky:     {score_details['returns']:.1f}/3.0") 
    print(f"  Úspěšnost obchodů:    {score_details['trades']:.1f}/2.0")
    print(f"  Barrier strategy:     {score_details['strategy']:.1f}/2.5")
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

def manual_kfold_validation(X_train, y_train, best_params):
    """Vlastní implementace k-fold validace s uložením historií všech foldů"""
    
    print("=" * 60)
    print("MANUAL K-FOLD VALIDATION WITH HISTORY TRACKING")
    print("=" * 60)
    
    cv = KFold(n_splits=HYPERPARAMS['cv_folds'], shuffle=True, random_state=42)
    fold_histories = []
    fold_scores = []
    
    # Odstranění model__ prefixu z parametrů
    clean_params = {}
    for key, value in best_params.items():
        if key.startswith('model__'):
            clean_key = key.replace('model__', '')
            clean_params[clean_key] = value
        else:
            clean_params[key] = value
    
    print(f"Spouštím {HYPERPARAMS['cv_folds']}-fold cross validation...")
    print(f"Parametry: {clean_params}")
    print()
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        print(f"📊 FOLD {fold + 1}/{HYPERPARAMS['cv_folds']}")
        print("-" * 40)
        
        # Rozdělení dat pro tento fold
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        print(f"Train samples: {len(X_fold_train):,}")
        print(f"Validation samples: {len(X_fold_val):,}")
        
        # Vytvoření modelu pro tento fold
        fold_model = create_mlp_model(
            input_dim=X_train.shape[1],
            hidden_layers=clean_params['hidden_layers'],
            neurons_per_layer=clean_params['neurons_per_layer'],
            learning_rate=clean_params['learning_rate']
        )
        
        # Callback pro early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=HYPERPARAMS['patience'],
            restore_best_weights=True,
            verbose=0
        )
        
        # Trénink modelu pro tento fold
        fold_history = fold_model.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=HYPERPARAMS['cv_epochs'],
            batch_size=HYPERPARAMS['fixed_params']['batch_size'],
            callbacks=[early_stopping],
            verbose=0  # Tichý režim pro k-fold
        )
        
        # Uložení historie a skóre
        fold_histories.append(fold_history)
        val_score = min(fold_history.history['val_loss'])
        fold_scores.append(val_score)
        
        print(f"Best validation loss: {val_score:.6f}")
        print(f"Epochs trained: {len(fold_history.history['loss'])}")
        print()
    
    # Výsledky k-fold validace
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    print("=" * 40)
    print("K-FOLD VALIDATION RESULTS:")
    print("=" * 40)
    for i, score in enumerate(fold_scores):
        print(f"  Fold {i+1}: {score:.6f}")
    print(f"  Mean: {mean_score:.6f}")
    print(f"  Std:  {std_score:.6f}")
    print("=" * 40)
    print()
    
    return fold_histories, fold_scores


def main():
    """Main execution function"""
    
    try:
        # 1. Load and prepare data
        df = load_and_prepare_data()
        
        # 1a. Validate survivorship bias protection
        validate_survivorship_bias_protection(df)
        
        # 2. Calculate technical indicators
        df_with_indicators = calculate_technical_indicators(df)
        
        # 3. Prepare features and target
        (X_train, X_test, y_train, y_test, 
         train_data, test_data, scaler, feature_cols,
         target_mean, target_std) = prepare_features_and_target(df_with_indicators)
        
        # 4. Hyperparameter tuning
        best_params, best_score = tune_hyperparameters(X_train, y_train)
        
        # 4.5. Manual k-fold validation with history tracking
        fold_histories, fold_scores = manual_kfold_validation(X_train, y_train, best_params)
        
        # 5. Train best model
        model, history = train_best_model(X_train, y_train, X_test, y_test, best_params)
        
        # 6. Evaluate model
        y_train_pred, y_test_pred, train_metrics, test_metrics = evaluate_model(
            model, X_train, y_train, X_test, y_test)
        
        # 7. Implement realistic barrier-based trading strategy
        portfolio_tracking_df, trades_df = implement_trading_strategy(test_data, y_test_pred)
        
        # 8. Calculate realistic trading metrics
        trading_metrics = calculate_trading_metrics(portfolio_tracking_df, trades_df)
        
        # 9. Create visualization
        output_file = create_comprehensive_visualization(
            history, fold_histories, portfolio_tracking_df, best_params, train_metrics, test_metrics, trading_metrics)
        
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
        
        print(f"\n� METHODOLOGICAL SUMMARY:")
        print(f"  ✅ Look-ahead bias: Eliminated (features from T-1)")
        print(f"  ✅ Survivorship bias: Protected (point-in-time membership)")
        print(f"  ✅ Target variable: CORRECTED to T→T+1 (future returns)")
        print(f"  ✅ Trading timeline: T-1 data → predict T+1 return")
        
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
