import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from scikeras.wrappers import KerasRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
warnings.filterwarnings('ignore')

# Start time measurement
start_time = time.time()
print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load the data
print("Loading data...")
data_path = '/Users/lindawaisova/Desktop/DP/data/SP_100/Reuters_SP100_Data.csv'
df = pd.read_csv(data_path)

# Check basic information about the dataset
print(f"Dataset shape: {df.shape}")
print("Dataset columns:", df.columns.tolist())

# Convert date column to datetime and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['ID', 'Date'])

# Function to adjust OHLC data for stock splits
def adjust_ohlc(df):
    """Adjust Open, High, Low prices and Volume for stock splits using Close and CloseAdj"""
    adjustment_ratio = df['CloseAdj'] / df['Close']
    
    df['OpenAdj'] = df['Open'] * adjustment_ratio
    df['HighAdj'] = df['High'] * adjustment_ratio
    df['LowAdj'] = df['Low'] * adjustment_ratio
    df['VolumeAdj'] = df['Volume'] / adjustment_ratio
    
    return df

print("Adjusting OHLC data...")
df = adjust_ohlc(df)

# Convert daily data to weekly data
print("Converting to weekly data...")
df['Year_Week'] = df['Date'].dt.isocalendar().year.astype(str) + '_' + df['Date'].dt.isocalendar().week.astype(str).str.zfill(2)
weekly_df = df.groupby(['ID', 'Year_Week']).agg({
    'Date': 'last',  # Last day of the week
    'OpenAdj': 'first',  # First day's open
    'HighAdj': 'max',    # Week's high
    'LowAdj': 'min',     # Week's low
    'CloseAdj': 'last',  # Last day's close
    'VolumeAdj': 'sum',  # Week's total volume
}).reset_index()

# Sort by ID and Date
weekly_df = weekly_df.sort_values(['ID', 'Date'])

# Calculate weekly returns
print("Calculating weekly returns...")
weekly_df['Return'] = weekly_df.groupby('ID')['CloseAdj'].pct_change()

# Create features using sliding windows of returns
def create_features(df, window_size=10):
    """Create features based on sliding window of past returns"""
    features = []
    targets = []
    dates = []
    stock_ids = []
    
    grouped = df.groupby('ID')
    
    for stock_id, stock_data in grouped:
        # Skip if too few data points
        if len(stock_data) <= window_size:
            continue
        
        # Create sliding windows
        for i in range(window_size, len(stock_data)):
            # Past window_size returns as features
            X = stock_data['Return'].iloc[i-window_size:i].values
            
            # Next period return as target
            y = stock_data['Return'].iloc[i]
            
            # Skip rows with NaN values
            if np.isnan(X).any() or np.isnan(y):
                continue
            
            features.append(X)
            targets.append(y)
            dates.append(stock_data['Date'].iloc[i])
            stock_ids.append(stock_id)
    
    return np.array(features), np.array(targets), np.array(dates), np.array(stock_ids)

# Create features with a 10-week lookback period
print("Creating features with sliding window...")
window_size = 10
X, y, dates, stock_ids = create_features(weekly_df, window_size)

# Split into training and testing sets based on dates
print("Splitting into training and testing sets...")
train_end_date = pd.Timestamp('2020-12-31')
train_mask = dates <= train_end_date
test_mask = dates > train_end_date

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
dates_train, dates_test = dates[train_mask], dates[test_mask]
stock_ids_train, stock_ids_test = stock_ids[train_mask], stock_ids[test_mask]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Standardize the features
print("Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the MLP model creation function
def create_mlp_model(units=64, hidden_layers=2, dropout_rate=0.2, l2_reg=0.001, learning_rate=0.001):
    """Create MLP model with specified architecture"""
    model = Sequential()
    
    # Input layer
    model.add(Dense(units, activation='relu', kernel_regularizer=l2(l2_reg), 
                  input_shape=(window_size,)))
    model.add(Dropout(dropout_rate))
    
    # Hidden layers
    for _ in range(hidden_layers - 1):
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    return model

# Define the parameter grid for random search
param_grid = {
    'model__units': [32, 64, 128],
    'model__hidden_layers': [1, 2, 3],
    'model__dropout_rate': [0.1, 0.2, 0.3],
    'model__l2_reg': [0.0001, 0.001, 0.01],
    'model__learning_rate': [0.0001, 0.001, 0.01]
}

# Set up early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Create TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=10)

# Define the model for sklearn wrapper
model = KerasRegressor(
    model=create_mlp_model,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)

# Perform random search for hyperparameter tuning
print("Performing random search for hyperparameter tuning...")
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=10,  # Number of parameter settings sampled
    cv=tscv,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_scaled, y_train)

# Get the best parameters and model
best_params = random_search.best_params_
print(f"Best parameters: {best_params}")

# Retrain the model with best parameters
print("Retraining model with best parameters...")
best_model = KerasRegressor(
    model=lambda: create_mlp_model(
        units=best_params['model__units'],
        hidden_layers=best_params['model__hidden_layers'],
        dropout_rate=best_params['model__dropout_rate'],
        l2_reg=best_params['model__l2_reg'],
        learning_rate=best_params['model__learning_rate']
    ),
    epochs=1000,
    batch_size=32,
    verbose=0
)

# Save history for plotting
history = best_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Make predictions
print("Making predictions...")
y_train_pred = best_model.predict(X_train_scaled)
y_test_pred = best_model.predict(X_test_scaled)

# Calculate R-squared
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE: {test_rmse:.4f}")

# Implement trading strategy
print("Implementing trading strategy...")

# Function to simulate weekly trading
def simulate_weekly_trading(dates, stock_ids, predictions, actual_returns, profit_target=0.02, stop_loss=0.02):
    """
    Simulates trading strategy based on predictions:
    - Long top 10 stocks with highest predicted returns
    - Short top 10 stocks with lowest predicted returns
    - Apply profit target and stop loss of 2%
    """
    # Group by week
    date_df = pd.DataFrame({
        'Date': dates,
        'StockID': stock_ids,
        'Prediction': predictions.flatten(),
        'ActualReturn': actual_returns.flatten()
    })
    
    # Convert to weekly format
    date_df['Year_Week'] = pd.to_datetime(date_df['Date']).dt.strftime('%Y_%W')
    
    # Results storage
    weekly_returns = []
    long_positions = []
    short_positions = []
    
    # Process each week
    for week, week_data in date_df.groupby('Year_Week'):
        if len(week_data) < 20:  # Need at least 20 stocks to select 10 for long and 10 for short
            continue
            
        # Rank stocks by prediction
        week_data = week_data.sort_values('Prediction', ascending=False)
        
        # Select top 10 for long
        long_stocks = week_data.head(10)
        long_return = long_stocks['ActualReturn'].mean()
        
        # Select bottom 10 for short
        short_stocks = week_data.tail(10)
        short_return = -short_stocks['ActualReturn'].mean()  # Negative because shorting
        
        # Apply profit target and stop loss (simplified approach)
        # In real implementation, we would check daily prices during the week
        # Here we'll just cap/floor the returns
        long_return = min(max(long_return, -stop_loss), profit_target)
        short_return = min(max(short_return, -stop_loss), profit_target)
        
        # Calculate combined return
        combined_return = (long_return + short_return) / 2
        
        weekly_returns.append({
            'Week': week,
            'Date': week_data['Date'].iloc[0],
            'LongReturn': long_return,
            'ShortReturn': short_return,
            'CombinedReturn': combined_return
        })
        
        # Store positions for analysis
        long_positions.append(long_stocks)
        short_positions.append(short_stocks)
    
    return pd.DataFrame(weekly_returns), pd.concat(long_positions), pd.concat(short_positions)

# Run trading simulation for test period
weekly_returns_df, long_positions_df, short_positions_df = simulate_weekly_trading(
    dates_test, stock_ids_test, y_test_pred, y_test
)

# Calculate cumulative returns
weekly_returns_df['CumReturn'] = (1 + weekly_returns_df['CombinedReturn']).cumprod() - 1

# Calculate Sharpe ratio (assuming weekly returns)
annual_factor = 52  # 52 weeks in a year
risk_free_rate = 0  # Simplified assumption
mean_return = weekly_returns_df['CombinedReturn'].mean()
std_return = weekly_returns_df['CombinedReturn'].std()
sharpe_ratio = (mean_return - risk_free_rate) * np.sqrt(annual_factor) / std_return

# Win rate
win_rate = (weekly_returns_df['CombinedReturn'] > 0).mean()

# Maximum drawdown
cum_returns = (1 + weekly_returns_df['CombinedReturn']).cumprod()
running_max = cum_returns.cummax()
drawdown = (cum_returns / running_max) - 1
max_drawdown = drawdown.min()

# Print strategy metrics
print("\nTrading Strategy Metrics:")
print(f"Total Return: {weekly_returns_df['CumReturn'].iloc[-1]:.4f}")
print(f"Annualized Return: {(1 + weekly_returns_df['CumReturn'].iloc[-1]) ** (1 / (len(weekly_returns_df) / 52)) - 1:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Win Rate: {win_rate:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.4f}")

# Visualizations
print("\nCreating visualizations...")
plt.figure(figsize=(20, 16))

# Plot 1: Training and validation loss
plt.subplot(3, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axvline(x=len(history.history['loss']) - early_stopping.patience, color='r', linestyle='--', 
           label=f'Early Stopping Patience ({early_stopping.patience})')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

# Plot 2: Actual vs Predicted Returns (Test Set)
plt.subplot(3, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.3)
plt.plot([-0.2, 0.2], [-0.2, 0.2], color='red', linestyle='--')
plt.title(f'Actual vs Predicted Returns (Test Set, R² = {test_r2:.4f})')
plt.xlabel('Actual Weekly Return')
plt.ylabel('Predicted Weekly Return')
plt.grid(True)

# Plot 3: Cumulative Strategy Returns
plt.subplot(3, 2, 3)
plt.plot(pd.to_datetime(weekly_returns_df['Date']), weekly_returns_df['CumReturn'], marker='o', markersize=4)
plt.title(f'Cumulative Strategy Returns (Sharpe = {sharpe_ratio:.2f})')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)

# Plot 4: Weekly Strategy Returns
plt.subplot(3, 2, 4)
plt.bar(range(len(weekly_returns_df)), weekly_returns_df['CombinedReturn'])
plt.title(f'Weekly Strategy Returns (Win Rate = {win_rate:.2f})')
plt.xlabel('Week')
plt.ylabel('Weekly Return')
plt.axhline(y=0, color='r', linestyle='-')
plt.grid(True)

# Plot 5: Long vs Short Returns
plt.subplot(3, 2, 5)
plt.plot(pd.to_datetime(weekly_returns_df['Date']), (1 + weekly_returns_df['LongReturn']).cumprod() - 1, label='Long')
plt.plot(pd.to_datetime(weekly_returns_df['Date']), (1 + weekly_returns_df['ShortReturn']).cumprod() - 1, label='Short')
plt.plot(pd.to_datetime(weekly_returns_df['Date']), weekly_returns_df['CumReturn'], label='Combined')
plt.title('Cumulative Returns: Long vs Short vs Combined')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)

# Plot 6: Return Distribution
plt.subplot(3, 2, 6)
sns.histplot(weekly_returns_df['CombinedReturn'], kde=True)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Distribution of Weekly Returns')
plt.xlabel('Weekly Return')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()
plt.savefig('weekly_mlp_trading_results.png', dpi=300, bbox_inches='tight')

# End time measurement
end_time = time.time()
execution_time = end_time - start_time
print(f"\nScript completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")

plt.show()