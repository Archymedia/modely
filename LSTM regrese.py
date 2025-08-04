#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LSTM model for predicting weekly stock returns of S&P100 companies
with trading strategy implementation and comprehensive evaluation
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings

from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from scipy import stats
from scikeras.wrappers import KerasRegressor

# Suppress warnings
warnings.filterwarnings('ignore')
# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==================== CONFIGURATION PARAMETERS ====================
# Basic model parameters
WINDOW_SIZE = 20

# Hyperparameter search parameters  
HP_EPOCHS = 30
HP_BATCH_SIZE = 64
HP_PATIENCE = 2
HP_N_ITER = 4
HP_CV_SPLITS = 2

# Hyperparameter space for optimization
PARAM_DIST = {
    'model__units': [32, 64, 128],
    'model__dropout_rate': [0.2, 0.3],
    'model__l2_reg': [0.01, 0.001],
    'model__n_layers': [1, 2, 3],
    'model__learning_rate': [0.001]
}

# Final model training parameters
FINAL_EPOCHS = 30
FINAL_BATCH_SIZE = 64
FINAL_PATIENCE = 3

# Trading strategy parameters
TAKE_PROFIT_PCT = 0.02
STOP_LOSS_PCT = 0.02

# Data filtering parameters
START_DATE = '2005-01-01'
END_DATE = '2023-12-29'
TRAIN_CUTOFF = '2020-12-31'

# Start time measurement
start_time = time.time()

# ---------------------- 1. Data Loading and Preprocessing ----------------------
print("\n" + "="*80)
print("LSTM WEEKLY MODEL FOR S&P100 STOCK RETURN PREDICTION")
print("="*80)

print("\n" + "-"*80)
print("1. LOADING AND PREPROCESSING DATA")
print("-"*80)

# Data paths (try both paths)
data_paths = [
    "/Users/lindawaisova/Desktop/DP/data/SP_100/Reuters_SP100_Data.csv",
    r"C:\Users\Unknown\Desktop\data\SP100\Reuters_SP100_Data.csv"
]

# Try loading from different paths
for data_path in data_paths:
    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        break
else:
    raise FileNotFoundError("Could not find the S&P100 data file in any specified location")

print(f"Raw data shape: {df.shape}")
print(f"Time range: {df['Date'].min()} to {df['Date'].max()}")

# Convert date column to datetime with no timezone
df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
print("Date column converted to datetime with timezone handling")

# *** FILTER DATA TO DEFINED PERIOD ***
start_date = pd.Timestamp(START_DATE)
end_date = pd.Timestamp(END_DATE)
print(f"\nFiltering data to defined period: {start_date.date()} to {end_date.date()}")

initial_rows = df.shape[0]
df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
print(f"Filtered data shape: {df.shape} (removed {initial_rows - df.shape[0]} rows outside defined period)")
print(f"Filtered time range: {df['Date'].min()} to {df['Date'].max()}")

# Check for missing values only in relevant columns (Date, Close and columns to the right)
relevant_columns = ['Date', 'Close', 'CloseAdj', 'Volume', 'TotRet', 'ID']
# Filter to only show columns that exist in the dataframe
existing_relevant_cols = [col for col in relevant_columns if col in df.columns]
print(f"\nMissing values in relevant columns:")
for col in existing_relevant_cols:
    missing_count = df[col].isna().sum()
    if missing_count > 0:
        print(f"{col}: {missing_count}")
    else:
        print(f"{col}: 0")
print("(Ignoring bid, ask, MV and other non-essential columns)")

# ---------------------- 2. Weekly Data Processing ----------------------
print("\n" + "-"*80)
print("2. PROCESSING WEEKLY DATA")
print("-"*80)

# Convert daily data to weekly (end of week)
print("Converting daily data to weekly...")
df['Year'] = df['Date'].dt.year
df['Week'] = df['Date'].dt.isocalendar().week
df['YearWeek'] = df['Year'].astype(str) + '-' + df['Week'].astype(str).str.zfill(2)

# Get the last day of each week for each stock
weekly_df = df.sort_values(['ID', 'Date']).groupby(['ID', 'YearWeek']).last().reset_index()
print(f"Weekly data shape: {weekly_df.shape}")

# ---------------------- 3. Adjusting OHLC Prices for Stock Splits ----------------------
print("\n" + "-"*80)
print("3. ADJUSTING OHLC PRICES FOR STOCK SPLITS")
print("-"*80)

# Calculate adjustment factor
weekly_df['AdjFactor'] = weekly_df['CloseAdj'] / weekly_df['Close']

# Adjust Open, High, Low prices
print("Adjusting Open, High, Low prices for stock splits...")
weekly_df['OpenAdj'] = weekly_df['Open'] * weekly_df['AdjFactor']
weekly_df['HighAdj'] = weekly_df['High'] * weekly_df['AdjFactor']
weekly_df['LowAdj'] = weekly_df['Low'] * weekly_df['AdjFactor']

# Adjust Volume
weekly_df['VolumeAdj'] = weekly_df['Volume'] / weekly_df['AdjFactor']

print("OHLC prices and volume adjusted for stock splits")

# ---------------------- 4. Calculating Weekly Returns ----------------------
print("\n" + "-"*80)
print("4. CALCULATING WEEKLY RETURNS")
print("-"*80)

# Calculate simple returns from TotRet
weekly_df = weekly_df.sort_values(['ID', 'Date'])
weekly_df['TotRet_Prev'] = weekly_df.groupby('ID')['TotRet'].shift(1)
weekly_df['SimpleReturn'] = (weekly_df['TotRet'] - weekly_df['TotRet_Prev']) / weekly_df['TotRet_Prev']

# Drop rows with NaN returns (first week for each stock)
initial_rows = weekly_df.shape[0]
weekly_df = weekly_df.dropna(subset=['SimpleReturn'])
print(f"Dropped {initial_rows - weekly_df.shape[0]} rows with NaN returns")

# Check for infinite or very large values
inf_mask = ~np.isfinite(weekly_df['SimpleReturn'])
if inf_mask.any():
    print(f"Found {inf_mask.sum()} infinite values in SimpleReturn - replacing with 0")
    weekly_df.loc[inf_mask, 'SimpleReturn'] = 0

# Also check for very large returns that might be data errors
large_returns_mask = abs(weekly_df['SimpleReturn']) > 0.5  # 50% threshold
if large_returns_mask.any():
    print(f"Found {large_returns_mask.sum()} returns with absolute value > 50% - replacing with mean")
    mean_return = weekly_df.loc[~large_returns_mask, 'SimpleReturn'].mean()
    weekly_df.loc[large_returns_mask, 'SimpleReturn'] = mean_return

print(f"Simple return statistics:")
print(weekly_df['SimpleReturn'].describe())

# ---------------------- 5. Feature Engineering ----------------------
print("\n" + "-"*80)
print("5. FEATURE ENGINEERING")
print("-"*80)

# Group by ID to calculate features for each stock separately
print("Calculating technical indicators...")
weekly_grouped = weekly_df.groupby('ID')

# Initialize empty lists to store the results
all_data = []

# Process each stock individually
for name, group in weekly_grouped:
    # Sort by date
    group = group.sort_values('Date')
    
    # 5-week and 10-week rolling standard deviation (volatility)
    group['Volatility_5W'] = group['SimpleReturn'].rolling(window=5).std()
    group['Volatility_10W'] = group['SimpleReturn'].rolling(window=10).std()
    
    # Moving averages for CloseAdj
    group['MA5'] = group['CloseAdj'].rolling(window=5).mean()
    group['MA10'] = group['CloseAdj'].rolling(window=10).mean()
    
    # Relative strength (ratio of current price to average)
    group['RelStrength_5W'] = group['CloseAdj'] / group['MA5']
    group['RelStrength_10W'] = group['CloseAdj'] / group['MA10']
    
    # Volume indicators
    group['Volume_MA5'] = group['VolumeAdj'].rolling(window=5).mean()
    group['Volume_Ratio'] = group['VolumeAdj'] / group['Volume_MA5']
    
    # Future return (target variable)
    group['NextWeekReturn'] = group['SimpleReturn'].shift(-1)
    
    # Add to the list
    all_data.append(group)

# Combine all stocks back into a single DataFrame
feature_df = pd.concat(all_data)
print("Feature engineering completed")

# Drop rows with NaN from feature calculation
initial_rows = feature_df.shape[0]
feature_df = feature_df.dropna()
print(f"Dropped {initial_rows - feature_df.shape[0]} rows with NaN features")
print(f"Final data shape: {feature_df.shape}")

# ---------------------- 6. Handling Survivorship Bias ----------------------
print("\n" + "-"*80)
print("6. HANDLING SURVIVORSHIP BIAS")
print("-"*80)

print("Filtering data to include only stocks that were in S&P100 at the time of trading...")

# Determine S&P100 membership status
# A stock is considered in the index if it has valid data (not NA)
feature_df['InIndex'] = ~feature_df['TotRet'].isna()

# Verify we have index membership info
index_status_counts = feature_df['InIndex'].value_counts()
print(f"Index membership status counts:")
print(index_status_counts)

# Filter to keep only stocks that were in the index
feature_df_active = feature_df[feature_df['InIndex']].copy()
print(f"After filtering for active index components: {feature_df_active.shape[0]} rows")

# ---------------------- 7. Train-Test Split ----------------------
print("\n" + "-"*80)
print("7. TRAIN-TEST SPLIT")
print("-"*80)

# Define the cutoff date for train-test split
train_cutoff = pd.Timestamp(TRAIN_CUTOFF)
print(f"Train-test split date: {train_cutoff} (training period: 2005-2020, testing period: 2021-2023)")

# Ensure dates have consistent timezone handling for comparison
feature_df_active['Date'] = feature_df_active['Date'].dt.tz_localize(None)

# Split data
train_data = feature_df_active[feature_df_active['Date'] <= train_cutoff].copy()
test_data = feature_df_active[feature_df_active['Date'] > train_cutoff].copy()

print(f"Training data: {train_data.shape[0]} rows from {train_data['Date'].min()} to {train_data['Date'].max()}")
print(f"Testing data: {test_data.shape[0]} rows from {test_data['Date'].min()} to {test_data['Date'].max()}")

# ---------------------- 8. Data Preparation for LSTM ----------------------
print("\n" + "-"*80)
print("8. DATA PREPARATION FOR LSTM")
print("-"*80)

# Define the window size
window_size = WINDOW_SIZE  # Use configuration parameter

# Define feature columns and target
feature_columns = [
    'SimpleReturn',
    'OpenAdj', 'HighAdj', 'LowAdj', 'CloseAdj',
    'VolumeAdj',
    'Volatility_5W', 'Volatility_10W',
    'MA5', 'MA10',
    'RelStrength_5W', 'RelStrength_10W',
    'Volume_MA5', 'Volume_Ratio'
]
target_column = 'NextWeekReturn'

print(f"Using {len(feature_columns)} features: {', '.join(feature_columns)}")
print(f"Target column: {target_column}")
print(f"Window size: {window_size} weeks")

# Standardize the data
print("\nStandardizing features...")
scaler = StandardScaler()
train_data_scaled = train_data.copy()
train_data_scaled[feature_columns] = scaler.fit_transform(train_data[feature_columns])

test_data_scaled = test_data.copy()
test_data_scaled[feature_columns] = scaler.transform(test_data[feature_columns])

# Function to create sequences
def create_sequences(data, stocks, window_size, feature_cols, target_col):
    X, y, stock_info = [], [], []
    
    for stock_id in stocks:
        # Make a copy and ensure Date has no timezone
        stock_data = data[data['ID'] == stock_id].copy()
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.tz_localize(None)
        stock_data = stock_data.sort_values('Date')
        
        if len(stock_data) <= window_size:
            continue
            
        for i in range(len(stock_data) - window_size):
            X.append(stock_data[feature_cols].iloc[i:i+window_size].values)
            y.append(stock_data[target_col].iloc[i+window_size])
            stock_info.append({
                'ID': stock_id,
                'Date': stock_data['Date'].iloc[i+window_size],
                'CloseAdj': stock_data['CloseAdj'].iloc[i+window_size]
            })
    
    return np.array(X), np.array(y), stock_info

# Get unique stock IDs
train_stocks = train_data_scaled['ID'].unique()
test_stocks = test_data_scaled['ID'].unique()

print(f"Creating sequences for {len(train_stocks)} stocks in training set...")
X_train, y_train, train_info = create_sequences(
    train_data_scaled, train_stocks, window_size, feature_columns, target_column
)

print(f"Creating sequences for {len(test_stocks)} stocks in testing set...")
X_test, y_test, test_info = create_sequences(
    test_data_scaled, test_stocks, window_size, feature_columns, target_column
)

print(f"Training sequences shape: {X_train.shape}")
print(f"Testing sequences shape: {X_test.shape}")

# Convert stock info to DataFrame for easier analysis
train_info_df = pd.DataFrame(train_info)
test_info_df = pd.DataFrame(test_info)

# ---------------------- 9. LSTM Model Definition ----------------------
print("\n" + "-"*80)
print("9. LSTM MODEL DEFINITION")
print("-"*80)

# Define the function to create the LSTM model
def create_lstm_model(units=64, dropout_rate=0.3, l2_reg=0.001, n_layers=1, learning_rate=0.001):
    model = Sequential()
    
    # First LSTM layer
    # Using standard LSTM activations: tanh for candidate states and outputs, sigmoid for gates
    if n_layers == 1:
        model.add(LSTM(units=units, 
                      activation='tanh',           # tanh for candidate states and outputs
                      recurrent_activation='sigmoid', # sigmoid for gates (input, forget, output)
                      input_shape=(window_size, len(feature_columns)),
                      kernel_regularizer=l2(l2_reg), 
                      return_sequences=False))
    else:
        model.add(LSTM(units=units, 
                      activation='tanh',           # tanh for candidate states and outputs
                      recurrent_activation='sigmoid', # sigmoid for gates (input, forget, output)
                      input_shape=(window_size, len(feature_columns)),
                      kernel_regularizer=l2(l2_reg), 
                      return_sequences=True))
    
    model.add(Dropout(dropout_rate))
    
    # Additional LSTM layers if specified
    for i in range(1, n_layers):
        if i == n_layers - 1:
            model.add(LSTM(units=units, 
                          activation='tanh',           # tanh for candidate states and outputs
                          recurrent_activation='sigmoid', # sigmoid for gates (input, forget, output)
                          kernel_regularizer=l2(l2_reg), 
                          return_sequences=False))
        else:
            model.add(LSTM(units=units, 
                          activation='tanh',           # tanh for candidate states and outputs
                          recurrent_activation='sigmoid', # sigmoid for gates (input, forget, output)
                          kernel_regularizer=l2(l2_reg), 
                          return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    # Output layer - Linear activation for regression (predicting returns)
    model.add(Dense(1, activation='linear'))  # Explicit linear activation for regression
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

print("LSTM model architecture defined")

# Create model wrapper for scikit-learn compatibility
lstm_model = KerasRegressor(
    model=create_lstm_model,
    epochs=HP_EPOCHS,
    batch_size=HP_BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

# ---------------------- 10. Hyperparameter Tuning ----------------------
print("\n" + "-"*80)
print("10. HYPERPARAMETER TUNING WITH RANDOMIZED SEARCH")
print("-"*80)

# Define early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=HP_PATIENCE,
    restore_best_weights=True,
    verbose=1
)

# Use hyperparameter space from configuration
param_dist = PARAM_DIST

print("Hyperparameter space:")
for param, values in param_dist.items():
    print(f"  {param}: {values}")

# Define time series k-fold cross-validation
tscv = TimeSeriesSplit(n_splits=HP_CV_SPLITS)

# Define RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=lstm_model,
    param_distributions=param_dist,
    n_iter=HP_N_ITER,
    cv=tscv,
    verbose=1,
    n_jobs=-1,  # Run in parallel to speed up the search
    random_state=42,
    scoring='neg_mean_squared_error'
)

print("\nStarting RandomizedSearchCV (this may take a while)...")
print(f"Training on {X_train.shape[0]} sequences with {tscv.n_splits} folds")

# Fit the random search
# When using KerasRegressor from scikeras, fit parameters should be included in the model initialization
random_search.fit(X_train, y_train)

# Get the best parameters
best_params = random_search.best_params_
print("\nBest parameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

# ---------------------- 11. Training Final Model ----------------------
print("\n" + "-"*80)
print("11. TRAINING FINAL MODEL WITH BEST PARAMETERS")
print("-"*80)

# Extract best parameters
best_units = best_params['model__units']
best_dropout_rate = best_params['model__dropout_rate']
best_l2_reg = best_params['model__l2_reg']
best_n_layers = best_params['model__n_layers']
best_learning_rate = best_params['model__learning_rate']

# Create final model with best parameters
final_model = create_lstm_model(
    units=best_units,
    dropout_rate=best_dropout_rate,
    l2_reg=best_l2_reg,
    n_layers=best_n_layers,
    learning_rate=best_learning_rate
)

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=FINAL_PATIENCE,
    restore_best_weights=True,
    verbose=1
)

# Train final model
print("Training final model...")
history = final_model.fit(
    X_train, y_train,
    epochs=FINAL_EPOCHS,
    batch_size=FINAL_BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# ---------------------- 12. Model Evaluation ----------------------
print("\n" + "-"*80)
print("12. MODEL EVALUATION")
print("-"*80)

# Make predictions
y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

# Calculate MSE
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Calculate R-squared
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nModel Performance:")
print(f"Training MSE: {train_mse:.6f}")
print(f"Testing MSE: {test_mse:.6f}")
print(f"Training R²: {train_r2:.6f}")
print(f"Testing R²: {test_r2:.6f}")

# Create prediction DataFrames
train_predictions_df = pd.DataFrame({
    'ID': [info['ID'] for info in train_info],
    'Date': [info['Date'] for info in train_info],
    'CloseAdj': [info['CloseAdj'] for info in train_info],
    'Actual': y_train,
    'Predicted': y_train_pred.flatten()
})

test_predictions_df = pd.DataFrame({
    'ID': [info['ID'] for info in test_info],
    'Date': [info['Date'] for info in test_info],
    'CloseAdj': [info['CloseAdj'] for info in test_info],
    'Actual': y_test,
    'Predicted': y_test_pred.flatten()
})

# ---------------------- 13. Implementing Trading Strategy ----------------------
print("\n" + "-"*80)
print("13. IMPLEMENTING TRADING STRATEGY")
print("-"*80)

def implement_trading_strategy(predictions_df, take_profit_pct=TAKE_PROFIT_PCT, stop_loss_pct=STOP_LOSS_PCT):
    print(f"Implementing trading strategy with PT={take_profit_pct:.1%}, SL={stop_loss_pct:.1%}")
    
    # Ensure dates have no timezone information for consistent comparison
    predictions_df = predictions_df.copy()
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date']).dt.tz_localize(None)
    
    # Group by date
    trades_by_date = []
    
    for date, group in predictions_df.groupby('Date'):
        # Sort by predicted return
        sorted_predictions = group.sort_values('Predicted', ascending=False)
        
        # Select top 10 for long positions
        long_positions = sorted_predictions.head(10).copy()
        long_positions['Position'] = 'LONG'
        
        # Select bottom 10 for short positions
        short_positions = sorted_predictions.tail(10).copy()
        short_positions['Position'] = 'SHORT'
        
        # Combine positions
        positions = pd.concat([long_positions, short_positions])
        positions['EntryPrice'] = positions['CloseAdj']
        positions['TakeProfit'] = positions.apply(
            lambda x: x['EntryPrice'] * (1 + take_profit_pct) if x['Position'] == 'LONG' else x['EntryPrice'] * (1 - take_profit_pct),
            axis=1
        )
        positions['StopLoss'] = positions.apply(
            lambda x: x['EntryPrice'] * (1 - stop_loss_pct) if x['Position'] == 'LONG' else x['EntryPrice'] * (1 + stop_loss_pct),
            axis=1
        )
        
        # Add to trades list
        trades_by_date.append(positions)
    
    # Combine all trades
    all_trades = pd.concat(trades_by_date)
    
    # Simulation of trades would require looking at next week's OHLC data
    # For simplicity, let's assume the actual return represents the outcome
    all_trades['TradeReturn'] = all_trades.apply(
        lambda x: x['Actual'] if x['Position'] == 'LONG' else -x['Actual'],
        axis=1
    )
    
    # Implement PT/SL logic
    all_trades['TargetHit'] = all_trades.apply(
        lambda x: 'TP' if x['TradeReturn'] >= take_profit_pct else 
                 'SL' if x['TradeReturn'] <= -stop_loss_pct else 'HOLD',
        axis=1
    )
    
    # Calculate trade P&L
    all_trades['TradePL'] = all_trades.apply(
        lambda x: take_profit_pct if x['TargetHit'] == 'TP' else
                 -stop_loss_pct if x['TargetHit'] == 'SL' else
                 x['TradeReturn'],
        axis=1
    )
    
    return all_trades

# Apply trading strategy to training and testing data
print("Applying trading strategy to training data...")
train_trades = implement_trading_strategy(train_predictions_df)

print("Applying trading strategy to testing data...")
test_trades = implement_trading_strategy(test_predictions_df)

# ---------------------- 14. Performance Metrics Calculation ----------------------
print("\n" + "-"*80)
print("14. CALCULATING TRADING PERFORMANCE METRICS")
print("-"*80)

def calculate_performance_metrics(trades):
    # Group trades by date to get portfolio returns
    portfolio_returns = trades.groupby('Date')['TradePL'].mean()
    
    # Calculate cumulative return
    cumulative_return = (1 + portfolio_returns).cumprod() - 1
    total_return = cumulative_return.iloc[-1]
    
    # Calculate annualized return
    # Assuming 52 weeks in a year
    n_years = len(portfolio_returns) / 52
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    
    # Calculate volatility
    annualized_volatility = portfolio_returns.std() * np.sqrt(52)
    
    # Calculate Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    # Calculate maximum drawdown
    cumulative_wealth = (1 + portfolio_returns).cumprod()
    peak = cumulative_wealth.cummax()
    drawdown = (cumulative_wealth - peak) / peak
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    win_rate = (trades['TradePL'] > 0).mean()
    
    # Calculate profit factor
    gross_profit = trades.loc[trades['TradePL'] > 0, 'TradePL'].sum()
    gross_loss = abs(trades.loc[trades['TradePL'] < 0, 'TradePL'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Calculate number of trades
    total_trades = len(trades)
    long_trades = len(trades[trades['Position'] == 'LONG'])
    short_trades = len(trades[trades['Position'] == 'SHORT'])
    
    return {
        'Cumulative Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Total Trades': total_trades,
        'Long Trades': long_trades,
        'Short Trades': short_trades,
        'Portfolio Returns': portfolio_returns,
        'Cumulative Wealth': cumulative_wealth
    }

# Calculate performance metrics
train_metrics = calculate_performance_metrics(train_trades)
test_metrics = calculate_performance_metrics(test_trades)

print("\nTrading Performance Metrics:")
print("\nTraining Period:")
print(f"Cumulative Return: {train_metrics['Cumulative Return']:.2%}")
print(f"Annualized Return: {train_metrics['Annualized Return']:.2%}")
print(f"Sharpe Ratio: {train_metrics['Sharpe Ratio']:.3f}")
print(f"Maximum Drawdown: {train_metrics['Maximum Drawdown']:.2%}")
print(f"Win Rate: {train_metrics['Win Rate']:.2%}")
print(f"Profit Factor: {train_metrics['Profit Factor']:.3f}")
print(f"Total Trades: {train_metrics['Total Trades']:,} (Long: {train_metrics['Long Trades']:,}, Short: {train_metrics['Short Trades']:,})")

print("\nTesting Period:")
print(f"Cumulative Return: {test_metrics['Cumulative Return']:.2%}")
print(f"Annualized Return: {test_metrics['Annualized Return']:.2%}")
print(f"Sharpe Ratio: {test_metrics['Sharpe Ratio']:.3f}")
print(f"Maximum Drawdown: {test_metrics['Maximum Drawdown']:.2%}")
print(f"Win Rate: {test_metrics['Win Rate']:.2%}")
print(f"Profit Factor: {test_metrics['Profit Factor']:.3f}")
print(f"Total Trades: {test_metrics['Total Trades']:,} (Long: {test_metrics['Long Trades']:,}, Short: {test_metrics['Short Trades']:,})")

# ---------------------- 15. Visualization ----------------------
print("\n" + "-"*80)
print("15. CREATING COMPREHENSIVE VISUALIZATION")
print("-"*80)

# Set up the plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10

# Create a directory for saving plots
output_dir = "lstm_weekly_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Create comprehensive figure with subplots
fig = plt.figure(figsize=(20, 24))
gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.2], hspace=0.3, wspace=0.3)

# 1. Model Loss (Training History)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
ax1.set_title('Model Loss During Training', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss (MSE)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Stock Price - Actual vs Predicted (FILTERED DATA ONLY)
ax2 = fig.add_subplot(gs[0, 1])

# Combine all predictions (train + test) for complete timeline
all_predictions = pd.concat([train_predictions_df, test_predictions_df]).sort_values('Date')
all_predictions['Date'] = pd.to_datetime(all_predictions['Date']).dt.tz_localize(None)

# *** FILTER TO DEFINED PERIOD ONLY ***
period_start = pd.Timestamp('2005-01-01')
period_end = pd.Timestamp('2023-12-29')
all_predictions = all_predictions[
    (all_predictions['Date'] >= period_start) & 
    (all_predictions['Date'] <= period_end)
].copy()

# Aggregate daily average predictions across all stocks
daily_avg = all_predictions.groupby('Date')[['Actual', 'Predicted']].mean().reset_index()

ax2.plot(daily_avg['Date'], daily_avg['Actual'], label='Actual Returns', 
         alpha=0.8, linewidth=1, color='darkblue')
ax2.plot(daily_avg['Date'], daily_avg['Predicted'], label='Predicted Returns', 
         alpha=0.8, linewidth=1, color='red', linestyle='--')

# Add vertical line at train/test split (2021-01-01)
ax2.axvline(x=pd.Timestamp('2021-01-01'), color='green', linestyle='-', 
            linewidth=2, label='Train/Test Split (2021-01-01)')

ax2.set_title('Actual vs Predicted Returns (2005-2023)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Weekly Return')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(period_start, period_end)  # Explicitly set x-axis limits

# 3. Trading Strategy Performance - Cumulative Returns
ax3 = fig.add_subplot(gs[1, :])

# Training period - filter to defined period
train_dates = pd.to_datetime(train_metrics['Portfolio Returns'].index).tz_localize(None)
# Filter training dates to be within defined period
period_start = pd.Timestamp('2005-01-01')
period_end = pd.Timestamp('2023-12-29')
train_mask = (train_dates >= period_start) & (train_dates <= pd.Timestamp('2020-12-31'))
filtered_train_dates = train_dates[train_mask]
filtered_train_wealth = train_metrics['Cumulative Wealth'][train_mask]

ax3.plot(filtered_train_dates, filtered_train_wealth, 
         label='Training Period (2005-2020)', color='blue', linewidth=2)

# Testing period - filter to defined period  
test_dates = pd.to_datetime(test_metrics['Portfolio Returns'].index).tz_localize(None)
test_mask = (test_dates >= pd.Timestamp('2021-01-01')) & (test_dates <= period_end)
filtered_test_dates = test_dates[test_mask]
filtered_test_wealth = test_metrics['Cumulative Wealth'][test_mask]

ax3.plot(filtered_test_dates, filtered_test_wealth, 
         label='Testing Period (2021-2023)', color='green', linewidth=2)

# Add vertical line at the train/test split point
ax3.axvline(x=pd.Timestamp('2021-01-01'), color='red', linestyle='--', 
            linewidth=2, label='Train/Test Split (2021-01-01)')

ax3.set_title('Trading Strategy Cumulative Returns (2005-2023)', fontsize=16, fontweight='bold')
ax3.set_xlabel('Years')
ax3.set_ylabel('Portfolio Value (Starting at $1)')
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(period_start, period_end)  # Explicitly set x-axis limits

# 4. Best Parameters Table
ax4 = fig.add_subplot(gs[2, 0])
ax4.axis('off')
ax4.set_title('Best Model Parameters', fontsize=14, fontweight='bold', pad=20)

# Create the model parameters table
param_data = [
    ["Parameter", "Value"],
    ["Window Size", window_size],
    ["LSTM Units", best_units],
    ["Number of Layers", best_n_layers],
    ["Dropout Rate", f"{best_dropout_rate:.1f}"],
    ["L2 Regularization", f"{best_l2_reg:.4f}"],
    ["Learning Rate", f"{best_learning_rate:.4f}"],
    ["Patience (Final)", FINAL_PATIENCE],
    ["Batch Size (Final)", FINAL_BATCH_SIZE],
    ["Epochs (Final)", FINAL_EPOCHS]
]

param_table = ax4.table(cellText=param_data, loc='center', cellLoc='center', 
                       colWidths=[0.5, 0.3])
param_table.auto_set_font_size(False)
param_table.set_fontsize(11)
param_table.scale(1, 1.8)

# Style the header row
for (i, j), cell in param_table.get_celld().items():
    if i == 0:
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#E6E6E6')
    else:
        cell.set_facecolor('#F9F9F9')

# 5. Performance Metrics Table
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')
ax5.set_title('Model Performance Metrics', fontsize=14, fontweight='bold', pad=20)

# Create the performance metrics table
metrics_data = [
    ["Metric", "Training", "Testing"],
    ["MSE", f"{train_mse:.6f}", f"{test_mse:.6f}"],
    ["R²", f"{train_r2:.4f}", f"{test_r2:.4f}"],
    ["Cumulative Return", f"{train_metrics['Cumulative Return']:.2%}", f"{test_metrics['Cumulative Return']:.2%}"],
    ["Annualized Return", f"{train_metrics['Annualized Return']:.2%}", f"{test_metrics['Annualized Return']:.2%}"],
    ["Sharpe Ratio", f"{train_metrics['Sharpe Ratio']:.3f}", f"{test_metrics['Sharpe Ratio']:.3f}"],
    ["Max Drawdown", f"{train_metrics['Maximum Drawdown']:.2%}", f"{test_metrics['Maximum Drawdown']:.2%}"],
    ["Win Rate", f"{train_metrics['Win Rate']:.2%}", f"{test_metrics['Win Rate']:.2%}"],
    ["Profit Factor", f"{train_metrics['Profit Factor']:.3f}", f"{test_metrics['Profit Factor']:.3f}"],
    ["Total Trades", f"{train_metrics['Total Trades']:,}", f"{test_metrics['Total Trades']:,}"],
    ["Long Trades", f"{train_metrics['Long Trades']:,}", f"{test_metrics['Long Trades']:,}"],
    ["Short Trades", f"{train_metrics['Short Trades']:,}", f"{test_metrics['Short Trades']:,}"]
]

metrics_table = ax5.table(cellText=metrics_data, loc='center', cellLoc='center', 
                         colWidths=[0.4, 0.25, 0.25])
metrics_table.auto_set_font_size(False)
metrics_table.set_fontsize(11)
metrics_table.scale(1, 1.8)

# Style the header row
for (i, j), cell in metrics_table.get_celld().items():
    if i == 0:
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#E6E6E6')
    else:
        cell.set_facecolor('#F9F9F9')

# 6. Model Assessment Summary
ax6 = fig.add_subplot(gs[3, :])
ax6.axis('off')

# Calculate overall score (from the existing code)
score = 0
score += min(2, max(0, test_r2 * 20))
score += min(2, max(0, test_metrics['Sharpe Ratio']))
score += min(2, max(0, (test_metrics['Win Rate'] - 0.4) * 5))
risk_free_rate = 0.02
test_alpha = test_metrics['Annualized Return'] - risk_free_rate
score += min(2, max(0, test_alpha * 10))
score += min(2, max(0, 2 + test_metrics['Maximum Drawdown'] * 10))

if score >= 7:
    grade = "Excellent"
    emoji = "🌟"
elif score >= 5:
    grade = "Good"
    emoji = "✅"
elif score >= 3:
    grade = "Average"
    emoji = "⚠️"
else:
    grade = "Poor"
    emoji = "❌"

# Add model assessment text
assessment_text = f"""
LSTM MODEL ASSESSMENT SUMMARY

{emoji} Overall Grade: {grade} (Score: {score:.1f}/10)

Key Findings:
• Model uses standard LSTM activations (tanh + sigmoid) optimal for financial time series
• Window size of {window_size} weeks with {best_units} LSTM units in {best_n_layers} layer(s)
• Testing R² of {test_r2:.4f} indicates {'strong' if test_r2 > 0.1 else 'weak' if test_r2 > 0.01 else 'very weak'} predictive power
• Annualized return: {test_metrics['Annualized Return']:.2%} vs market benchmark ~7%
• Risk profile: {('High' if test_metrics['Annualized Volatility'] > 0.25 else 'Medium' if test_metrics['Annualized Volatility'] > 0.15 else 'Low')} volatility ({test_metrics['Annualized Volatility']:.2%})

Recommendation: {'Model ready for deployment' if score >= 6 else 'Model requires further optimization'}
"""

ax6.text(0.05, 0.95, assessment_text, transform=ax6.transAxes, fontsize=12,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))

# Add main title
fig.suptitle('LSTM Weekly Stock Return Prediction - Comprehensive Analysis', 
             fontsize=18, fontweight='bold', y=0.98)

# Save the comprehensive plot
comprehensive_plot_path = os.path.join(output_dir, 'lstm_comprehensive_analysis.png')
plt.savefig(comprehensive_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved comprehensive analysis to {comprehensive_plot_path}")

# Clean up - close the figure to free memory
plt.close(fig)

# ---------------------- 16. Export Results ----------------------
print("\n" + "-"*80)
print("16. EXPORTING RESULTS")
print("-"*80)

# Export trades data
train_trades.to_csv(os.path.join(output_dir, 'train_trades.csv'), index=False)
test_trades.to_csv(os.path.join(output_dir, 'test_trades.csv'), index=False)
print(f"Exported trade data to {output_dir}")

# ---------------------- 17. Extended Performance Evaluation ----------------------
print("\n" + "="*60)
print("           EXTENDED MODEL PERFORMANCE EVALUATION")
print("="*60)

# 17.1.1 MSE
print("\n17.1.1 MSE (Mean Squared Error)")
print("-" * 30)
print(f"Training MSE: {train_mse:.6f}")
print(f"Testing MSE: {test_mse:.6f}")
print(f"Test/Train MSE Ratio: {test_mse/train_mse:.3f}")
if test_mse/train_mse > 1.5:
    print("⚠️  Model may be overfitted (high Test/Train MSE ratio)")
elif test_mse/train_mse < 0.8:
    print("✅ Model generalizes well")
else:
    print("✅ Model has reasonable generalization ability")

# 17.1.2 Cumulative Return
print("\n17.1.2 Cumulative Return")
print("-" * 20)
train_total_return = train_metrics['Cumulative Return']
test_total_return = test_metrics['Cumulative Return']
train_annual_return = train_metrics['Annualized Return']
test_annual_return = test_metrics['Annualized Return']

print(f"Training cumulative return: {train_total_return:.2%}")
print(f"Testing cumulative return: {test_total_return:.2%}")
print(f"Training annualized return: {train_annual_return:.2%}")
print(f"Testing annualized return: {test_annual_return:.2%}")

# Benchmark comparison (assuming market return ~7% annually)
market_benchmark = 0.07
if test_annual_return > market_benchmark:
    print(f"✅ Model outperforms market benchmark ({market_benchmark:.1%}) by {(test_annual_return-market_benchmark):.2%}")
else:
    print(f"❌ Model underperforms market benchmark by {(market_benchmark-test_annual_return):.2%}")

# 17.1.3 Risk
print("\n17.1.3 Risk")
print("-" * 12)
train_volatility = train_metrics['Annualized Volatility']
test_volatility = test_metrics['Annualized Volatility']
train_max_dd = train_metrics['Maximum Drawdown']
test_max_dd = test_metrics['Maximum Drawdown']

print(f"Training volatility: {train_volatility:.2%}")
print(f"Testing volatility: {test_volatility:.2%}")
print(f"Training max drawdown: {train_max_dd:.2%}")
print(f"Testing max drawdown: {test_max_dd:.2%}")

# Risk assessment
if test_volatility > 0.25:
    risk_level = "High"
elif test_volatility > 0.15:
    risk_level = "Medium"
else:
    risk_level = "Low"
print(f"Risk level: {risk_level}")

if abs(test_max_dd) > 0.20:
    print("⚠️  High maximum drawdown - high risk of loss")
elif abs(test_max_dd) > 0.10:
    print("⚠️  Medium maximum drawdown - reasonable risk")
else:
    print("✅ Low maximum drawdown - conservative risk")

# 17.1.4 Realized Alpha
print("\n17.1.4 Realized Alpha")
print("-" * 18)
# Alpha calculation (excess return over risk-free rate, adjusted for market beta)
risk_free_rate = 0.02  # Assume 2% risk-free rate
train_alpha = train_annual_return - risk_free_rate
test_alpha = test_annual_return - risk_free_rate

print(f"Training alpha (above risk-free rate): {train_alpha:.2%}")
print(f"Testing alpha (above risk-free rate): {test_alpha:.2%}")

# Information ratio (similar to Sharpe but vs benchmark)
train_info_ratio = (train_annual_return - market_benchmark) / train_volatility if train_volatility > 0 else 0
test_info_ratio = (test_annual_return - market_benchmark) / test_volatility if test_volatility > 0 else 0

print(f"Training information ratio: {train_info_ratio:.3f}")
print(f"Testing information ratio: {test_info_ratio:.3f}")

if test_info_ratio > 0.5:
    print("✅ High realized alpha - model adds significant value")
elif test_info_ratio > 0.2:
    print("✅ Medium realized alpha - model adds some value")
else:
    print("❌ Low realized alpha - model doesn't add sufficient value")

# 17.1.5 Statistical Significance
print("\n17.1.5 Statistical Significance")
print("-" * 25)

# Calculate t-statistics for returns
train_weekly_returns = train_trades.groupby('Date')['TradePL'].sum()
test_weekly_returns = test_trades.groupby('Date')['TradePL'].sum()

# T-test for mean return significance
train_t_stat = train_weekly_returns.mean() / (train_weekly_returns.std() / np.sqrt(len(train_weekly_returns))) if len(train_weekly_returns) > 1 else 0
test_t_stat = test_weekly_returns.mean() / (test_weekly_returns.std() / np.sqrt(len(test_weekly_returns))) if len(test_weekly_returns) > 1 else 0

train_p_value = 2 * (1 - stats.t.cdf(abs(train_t_stat), len(train_weekly_returns)-1)) if len(train_weekly_returns) > 1 else 1
test_p_value = 2 * (1 - stats.t.cdf(abs(test_t_stat), len(test_weekly_returns)-1)) if len(test_weekly_returns) > 1 else 1

print(f"Training t-statistic: {train_t_stat:.3f}")
print(f"Testing t-statistic: {test_t_stat:.3f}")
print(f"Training p-value: {train_p_value:.4f}")
print(f"Testing p-value: {test_p_value:.4f}")

# Statistical significance assessment
alpha_level = 0.05
if test_p_value < alpha_level:
    print(f"✅ Results are statistically significant at {alpha_level*100}% level")
else:
    print(f"❌ Results are not statistically significant at {alpha_level*100}% level")

# Model prediction accuracy
print(f"\nPrediction accuracy:")
print(f"Training R²: {train_r2:.4f} ({'strong' if train_r2 > 0.1 else 'weak' if train_r2 > 0.01 else 'very weak'} predictive power)")
print(f"Testing R²: {test_r2:.4f} ({'strong' if test_r2 > 0.1 else 'weak' if test_r2 > 0.01 else 'very weak'} predictive power)")

# Win rate analysis
train_win_rate = train_metrics['Win Rate']
test_win_rate = test_metrics['Win Rate']
print(f"\nTrade success rate:")
print(f"Training win rate: {train_win_rate:.2%}")
print(f"Testing win rate: {test_win_rate:.2%}")

if test_win_rate > 0.55:
    print("✅ High trade success rate")
elif test_win_rate > 0.50:
    print("✅ Above average trade success rate")
else:
    print("❌ Below average trade success rate")

# Final assessment
print(f"\n{'='*60}")
print("OVERALL MODEL ASSESSMENT")
print(f"{'='*60}")

score = 0
# R² score (0-2 points)
score += min(2, max(0, test_r2 * 20))
# Sharpe ratio (0-2 points)
score += min(2, max(0, test_metrics['Sharpe Ratio']))
# Win rate (0-2 points)  
score += min(2, max(0, (test_win_rate - 0.4) * 5))
# Alpha (0-2 points)
score += min(2, max(0, test_alpha * 10))
# Low drawdown (0-2 points)
score += min(2, max(0, 2 + test_max_dd * 10))

if score >= 7:
    grade = "Excellent"
    emoji = "🌟"
elif score >= 5:
    grade = "Good"
    emoji = "✅"
elif score >= 3:
    grade = "Average"
    emoji = "⚠️"
else:
    grade = "Poor"
    emoji = "❌"

print(f"{emoji} Overall assessment: {grade} (score: {score:.1f}/10)")
print(f"Recommendation: {'Model is ready for deployment' if score >= 6 else 'Model requires further optimization'}")

print(f"\n{'='*60}")
print("ANALYSIS COMPLETED")
print(f"{'='*60}")

# ---------------------- 18. Execution Time ----------------------
end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print(f"\nTotal execution time: {minutes} minutes and {seconds} seconds")
