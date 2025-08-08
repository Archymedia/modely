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
import matplotlib.dates as mdates
from datetime import datetime

warnings.filterwarnings('ignore')

# ===== CONFIGURATION PARAMETERS =====
print("=" * 60)
print("           MLP STOCK PREDICTION MODEL")
print("=" * 60)

# Track execution time
start_time = time.time()

# Hyperparameters for tuning
HYPERPARAMS = {
    'epochs': 1000,
    'batch_size': [32, 64, 128, 256],
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_layers': [1, 2, 3],
    'neurons_layer1': [50, 100, 200, 300],
    'neurons_layer2': [25, 50, 100, 150],
    'neurons_layer3': [10, 25, 50],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'l2_reg': [0.001, 0.01, 0.1],
    'patience': [5, 10, 20],
    'random_search_iter': 50,
    'cv_folds': 10
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
print(f"K-Fold CV: {HYPERPARAMS['cv_folds']} folds")
print(f"Random Search iterations: {HYPERPARAMS['random_search_iter']}")
print()

# ---------------------- 1. Data Loading and Preprocessing ----------------------
print("=" * 60)
print("1. DATA LOADING AND PREPROCESSING")
print("=" * 60)

def load_and_prepare_data():
    """Load and prepare the main dataset and VIX data"""
    
    print("Loading main dataset...")
    try:
        # Try to load the data from the specified path
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
        else:
            # If the file doesn't exist, create sample data for demonstration
            print(f"Warning: {DATA_PATH} not found. Creating sample data for demonstration.")
            df = create_sample_data()
        
        print(f"Main dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Unique stocks (ID): {df['ID'].nunique()}")
        
    except Exception as e:
        print(f"Error loading main dataset: {e}")
        print("Creating sample data for demonstration...")
        df = create_sample_data()
    
    print("\nLoading VIX data...")
    try:
        if os.path.exists(VIX_PATH):
            vix_df = pd.read_csv(VIX_PATH)
            vix_df['Date'] = pd.to_datetime(vix_df['Date'])
            vix_df = vix_df[['Date', 'Close']].rename(columns={'Close': 'VIX'})
        else:
            print(f"Warning: {VIX_PATH} not found. Using sample VIX data.")
            vix_df = create_sample_vix_data()
            
        print(f"VIX data shape: {vix_df.shape}")
        print(f"VIX date range: {vix_df['Date'].min()} to {vix_df['Date'].max()}")
        
    except Exception as e:
        print(f"Error loading VIX data: {e}")
        print("Creating sample VIX data...")
        vix_df = create_sample_vix_data()
    
    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter data for the specified period
    df = df[(df['Date'] >= '2005-01-01') & (df['Date'] <= TEST_END_DATE)]
    
    # Merge with VIX data
    df = df.merge(vix_df, on='Date', how='left')
    df['VIX'] = df['VIX'].fillna(method='ffill')
    
    print(f"\nFinal dataset shape after preprocessing: {df.shape}")
    print(f"Stocks with InIndex=1: {df[df['InIndex']==1].shape[0]} observations")
    
    return df

def create_sample_data():
    """Create sample data for demonstration if real data is not available"""
    print("Creating sample dataset...")
    
    dates = pd.date_range('2005-01-01', '2023-12-29', freq='D')
    dates = dates[dates.weekday < 5]  # Only weekdays
    
    stocks = list(range(1, 101))  # 100 stocks
    
    data = []
    for stock_id in stocks:
        np.random.seed(stock_id)  # For reproducibility
        
        for date in dates:
            # Simulate stock data
            base_price = 50 + stock_id * 0.5
            volatility = 0.02
            
            simple_return = np.random.normal(0, volatility)
            
            data.append({
                'ID': stock_id,
                'RIC': f'STOCK{stock_id:03d}.N',
                'Name': f'Stock {stock_id}',
                'Date': date,
                'TotRet': np.random.normal(0, volatility),
                'SimpleReturn': simple_return,
                'OpenAdj': base_price,
                'HighAdj': base_price * (1 + abs(np.random.normal(0, 0.01))),
                'LowAdj': base_price * (1 - abs(np.random.normal(0, 0.01))),
                'CloseAdj': base_price * (1 + simple_return),
                'Close': base_price * (1 + simple_return),
                'Volume': np.random.randint(1000000, 10000000),
                'VolumeAdj': np.random.randint(1000000, 10000000),
                'VolumeUSDadj': np.random.randint(50000000, 500000000),
                'InIndex': 1 if np.random.random() > 0.1 else 0  # 90% chance of being in index
            })
    
    return pd.DataFrame(data)

def create_sample_vix_data():
    """Create sample VIX data"""
    dates = pd.date_range('2005-01-01', '2023-12-29', freq='D')
    dates = dates[dates.weekday < 5]  # Only weekdays
    
    # Simulate VIX data (typically ranges from 10-80)
    np.random.seed(42)
    vix_values = []
    vix = 20  # Starting value
    
    for _ in dates:
        vix += np.random.normal(0, 1)
        vix = max(10, min(80, vix))  # Keep within reasonable bounds
        vix_values.append(vix)
    
    return pd.DataFrame({'Date': dates, 'VIX': vix_values})

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
    
    # Filter only stocks that were in index (InIndex = 1)
    df = df[df['InIndex'] == 1].copy()
    
    print(f"Data shape after filtering InIndex=1: {df.shape}")
    
    # Split into train and test based on dates
    train_data = df[df['Date'] <= TRAIN_END_DATE].copy()
    test_data = df[df['Date'] >= TEST_START_DATE].copy()
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Train date range: {train_data['Date'].min()} to {train_data['Date'].max()}")
    print(f"Test date range: {test_data['Date'].min()} to {test_data['Date'].max()}")
    
    # Select feature columns (exclude non-predictive columns)
    exclude_cols = ['ID', 'RIC', 'Name', 'Date', 'target', 'InIndex', 'TotRet', 'SimpleReturn', 
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

def create_mlp_model(input_dim, hidden_layers=2, neurons_layer1=100, neurons_layer2=50, 
                     neurons_layer3=25, dropout_rate=0.2, l2_reg=0.01, learning_rate=0.001, 
                     meta=None, compile_kwargs=None):
    """Create MLP model with specified architecture"""
    
    model = Sequential()
    
    # First hidden layer
    model.add(Dense(neurons_layer1, 
                   input_dim=input_dim,
                   activation='tanh',
                   kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    
    # Additional hidden layers
    if hidden_layers >= 2:
        model.add(Dense(neurons_layer2, 
                       activation='tanh',
                       kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))
    
    if hidden_layers >= 3:
        model.add(Dense(neurons_layer3, 
                       activation='tanh',
                       kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1, activation='linear'))
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# ---------------------- 5. Hyperparameter Tuning ----------------------
print("=" * 60)
print("5. HYPERPARAMETER TUNING")
print("=" * 60)

def tune_hyperparameters(X_train, y_train):
    """Perform hyperparameter tuning using RandomizedSearchCV"""
    
    print("Starting hyperparameter tuning with RandomizedSearchCV...")
    print(f"Search space size: {HYPERPARAMS['random_search_iter']} iterations")
    print(f"Cross-validation folds: {HYPERPARAMS['cv_folds']}")
    
    # Define the parameter distribution for random search
    param_dist = {
        'model__hidden_layers': HYPERPARAMS['hidden_layers'],
        'model__neurons_layer1': HYPERPARAMS['neurons_layer1'],
        'model__neurons_layer2': HYPERPARAMS['neurons_layer2'],
        'model__neurons_layer3': HYPERPARAMS['neurons_layer3'],
        'model__dropout_rate': HYPERPARAMS['dropout_rate'],
        'model__l2_reg': HYPERPARAMS['l2_reg'],
        'model__learning_rate': HYPERPARAMS['learning_rate'],
        'batch_size': HYPERPARAMS['batch_size']
    }
    
    # Create KerasRegressor
    model = KerasRegressor(
        model=lambda **kwargs: create_mlp_model(input_dim=X_train.shape[1], **kwargs),
        epochs=HYPERPARAMS['epochs'],
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0
    )
    
    # Setup cross-validation
    cv = KFold(n_splits=HYPERPARAMS['cv_folds'], shuffle=True, random_state=42)
    
    # Perform randomized search
    try:
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=HYPERPARAMS['random_search_iter'],
            cv=cv,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=1,  # Use single job for keras models
            verbose=1
        )
        
        print("Fitting RandomizedSearchCV...")
        random_search.fit(X_train, y_train)
        
        print("Hyperparameter tuning completed successfully!")
        print(f"Best CV score: {-random_search.best_score_:.6f}")
        print("Best parameters:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
        
        return random_search.best_params_, random_search.best_score_
        
    except Exception as e:
        print(f"ERROR in RandomizedSearchCV: {e}")
        print("Stopping execution due to hyperparameter tuning failure.")
        raise Exception("RandomizedSearchCV failed. Please check your configuration and try again.")

# ---------------------- 6. Model Training ----------------------
print("=" * 60)
print("6. MODEL TRAINING")
print("=" * 60)

def train_best_model(X_train, y_train, X_test, y_test, best_params):
    """Train the final model with best parameters"""
    
    print("Training final model with best parameters...")
    
    # Extract model parameters
    model_params = {}
    training_params = {}
    
    for key, value in best_params.items():
        if key.startswith('model__'):
            model_params[key[7:]] = value  # Remove 'model__' prefix
        else:
            training_params[key] = value
    
    # Create model with best parameters
    model = create_mlp_model(input_dim=X_train.shape[1], **model_params)
    
    print("Model architecture:")
    model.summary()
    
    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=best_params.get('model__patience', 20),
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print(f"Training for max {HYPERPARAMS['epochs']} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=HYPERPARAMS['epochs'],
        batch_size=training_params.get('batch_size', 64),
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("Model training completed!")
    
    return model, history

# ---------------------- 7. Model Evaluation ----------------------
print("=" * 60)
print("7. MODEL EVALUATION")
print("=" * 60)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance"""
    
    print("Evaluating model performance...")
    
    # Make predictions
    y_train_pred = model.predict(X_train, verbose=0).flatten()
    y_test_pred = model.predict(X_test, verbose=0).flatten()
    
    # Calculate regression metrics
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
    
    print("Regression Metrics:")
    print(f"{'Metric':<8} {'Train':<12} {'Test':<12}")
    print("-" * 35)
    for metric in ['MSE', 'MAE', 'RMSE', 'R2']:
        print(f"{metric:<8} {train_metrics[metric]:<12.6f} {test_metrics[metric]:<12.6f}")
    
    return y_train_pred, y_test_pred, train_metrics, test_metrics

# ---------------------- 8. Trading Strategy Implementation ----------------------
print("=" * 60)
print("8. TRADING STRATEGY IMPLEMENTATION")
print("=" * 60)

def implement_trading_strategy(test_data, predictions):
    """Implement trading strategy with profit target and stop loss"""
    
    print("Implementing trading strategy...")
    print(f"Long positions per day: {TRADING_PARAMS['long_positions']}")
    print(f"Short positions per day: {TRADING_PARAMS['short_positions']}")
    print(f"Profit target: {TRADING_PARAMS['profit_target']*100:.1f}%")
    print(f"Stop loss: {TRADING_PARAMS['stop_loss']*100:.1f}%")
    
    # Add predictions to test data
    test_data_copy = test_data.copy()
    test_data_copy['predicted_return'] = predictions
    
    # Initialize portfolio tracking
    portfolio_returns = []
    trade_log = []
    
    # Get unique trading dates
    trading_dates = sorted(test_data_copy['Date'].unique())
    
    print(f"Trading period: {len(trading_dates)} days")
    
    daily_returns = []
    
    for i, current_date in enumerate(trading_dates[:-1]):  # Exclude last date
        if i % 60 == 0:  # Print progress every 60 days
            print(f"Processing day {i+1}/{len(trading_dates)-1}: {current_date.strftime('%Y-%m-%d')}")
        
        # Get current day's data and predictions
        current_data = test_data_copy[test_data_copy['Date'] == current_date].copy()
        
        if len(current_data) == 0:
            continue
        
        # Select top 10 stocks for long positions (highest predicted returns)
        long_stocks = current_data.nlargest(TRADING_PARAMS['long_positions'], 'predicted_return')
        
        # Select top 10 stocks for short positions (lowest predicted returns)
        short_stocks = current_data.nsmallest(TRADING_PARAMS['short_positions'], 'predicted_return')
        
        # Calculate daily portfolio return
        daily_return = 0
        position_count = 0
        
        # Process long positions
        for _, stock in long_stocks.iterrows():
            actual_return = stock['target']
            
            # Apply profit target and stop loss
            if actual_return >= TRADING_PARAMS['profit_target']:
                realized_return = TRADING_PARAMS['profit_target']
            elif actual_return <= -TRADING_PARAMS['stop_loss']:
                realized_return = -TRADING_PARAMS['stop_loss']
            else:
                realized_return = actual_return
            
            daily_return += realized_return
            position_count += 1
            
            trade_log.append({
                'Date': current_date,
                'ID': stock['ID'],
                'Position': 'Long',
                'Predicted_Return': stock['predicted_return'],
                'Actual_Return': actual_return,
                'Realized_Return': realized_return
            })
        
        # Process short positions
        for _, stock in short_stocks.iterrows():
            actual_return = stock['target']
            # For short positions, profit when stock goes down
            short_return = -actual_return
            
            # Apply profit target and stop loss
            if short_return >= TRADING_PARAMS['profit_target']:
                realized_return = TRADING_PARAMS['profit_target']
            elif short_return <= -TRADING_PARAMS['stop_loss']:
                realized_return = -TRADING_PARAMS['stop_loss']
            else:
                realized_return = short_return
            
            daily_return += realized_return
            position_count += 1
            
            trade_log.append({
                'Date': current_date,
                'ID': stock['ID'],
                'Position': 'Short',
                'Predicted_Return': stock['predicted_return'],
                'Actual_Return': actual_return,
                'Realized_Return': realized_return
            })
        
        # Average return across all positions
        if position_count > 0:
            daily_return = daily_return / position_count
        
        daily_returns.append({
            'Date': current_date,
            'Portfolio_Return': daily_return,
            'Position_Count': position_count
        })
    
    portfolio_df = pd.DataFrame(daily_returns)
    trades_df = pd.DataFrame(trade_log)
    
    print(f"Total trades executed: {len(trades_df)}")
    print(f"Trading days: {len(portfolio_df)}")
    
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
    
    # Calculate alpha (simplified - comparing to risk-free rate)
    benchmark_return = TRADING_PARAMS['risk_free_rate']
    alpha = annualized_return - benchmark_return
    
    metrics = {
        'Total_Return': total_return,
        'Annualized_Return': annualized_return,
        'Annualized_Volatility': annualized_volatility,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown,
        'Win_Rate': win_rate,
        'Profit_Factor': profit_factor,
        'Alpha': alpha,
        'Total_Trades': len(trades_df),
        'Avg_Win': avg_win,
        'Avg_Loss': avg_loss
    }
    
    print("Trading Performance Metrics:")
    print("-" * 40)
    print(f"Total Return: {metrics['Total_Return']:.2%}")
    print(f"Annualized Return: {metrics['Annualized_Return']:.2%}")
    print(f"Annualized Volatility: {metrics['Annualized_Volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.3f}")
    print(f"Maximum Drawdown: {metrics['Max_Drawdown']:.2%}")
    print(f"Win Rate: {metrics['Win_Rate']:.2%}")
    print(f"Profit Factor: {metrics['Profit_Factor']:.3f}")
    print(f"Alpha: {metrics['Alpha']:.2%}")
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
        clean_key = key.replace('model__', '')
        if isinstance(value, float):
            params_data.append([clean_key, f"{value:.4f}"])
        else:
            params_data.append([clean_key, str(value)])
    
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

Key Insights:
• R² Score: {test_metrics.get('R2', 0):.4f} (Test Set)
• Sharpe Ratio: {trading_metrics.get('Sharpe_Ratio', 0):.3f}
• Win Rate: {trading_metrics.get('Win_Rate', 0):.1%}
• Alpha: {trading_metrics.get('Alpha', 0):.2%}

Strategy Summary:
• Portfolio generated {trading_metrics.get('Total_Return', 0):.1%} total return
• Maximum drawdown: {trading_metrics.get('Max_Drawdown', 0):.1%}
• Executed {trading_metrics.get('Total_Trades', 0):,} trades over testing period
• Risk-adjusted performance: {trading_metrics.get('Sharpe_Ratio', 0):.3f}
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
    """Calculate overall model score out of 10"""
    
    score = 0
    
    # R² component (0-3 points)
    r2 = test_metrics.get('R2', 0)
    if r2 > 0.1:
        score += 3
    elif r2 > 0.05:
        score += 2
    elif r2 > 0.01:
        score += 1
    
    # Sharpe ratio component (0-3 points)
    sharpe = trading_metrics.get('Sharpe_Ratio', 0)
    if sharpe > 1.5:
        score += 3
    elif sharpe > 1.0:
        score += 2
    elif sharpe > 0.5:
        score += 1
    
    # Win rate component (0-2 points)
    win_rate = trading_metrics.get('Win_Rate', 0)
    if win_rate > 0.55:
        score += 2
    elif win_rate > 0.50:
        score += 1
    
    # Alpha component (0-2 points)
    alpha = trading_metrics.get('Alpha', 0)
    if alpha > 0.05:
        score += 2
    elif alpha > 0.02:
        score += 1
    
    return min(score, 10)  # Cap at 10

def get_grade_and_emoji(score):
    """Get grade and emoji based on score"""
    
    if score >= 7:
        grade = "Výborný"
        emoji = "🌟"
    elif score >= 5:
        grade = "Dobrý"
        emoji = "✅"
    elif score >= 3:
        grade = "Průměrný"
        emoji = "⚠️"
    else:
        grade = "Slabý"
        emoji = "❌"
    
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
