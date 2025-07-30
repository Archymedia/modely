import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor

warnings.filterwarnings('ignore')

# Record start time to measure execution time
start_time = time.time()

# Load data
print("Loading data...")
file_path = '/Users/lindawaisova/Desktop/DP/data/SP_100/Reuters_SP100_Data.csv'
data = pd.read_csv(file_path)
print(f"Data loaded: {data.shape} rows")

# Convert date to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Filter for the required time period (Jan 1, 2005 to Dec 29, 2023)
data = data[(data['Date'] >= '2005-01-01') & (data['Date'] <= '2023-12-29')]
print(f"Data filtered to 2005-2023: {data.shape} rows")

# Adjust OHLC prices for stock splits
print("Adjusting prices for stock splits...")
data['OpenAdj'] = data['Open'] * (data['CloseAdj'] / data['Close'])
data['HighAdj'] = data['High'] * (data['CloseAdj'] / data['Close'])
data['LowAdj'] = data['Low'] * (data['CloseAdj'] / data['Close'])
data['VolumeAdj'] = data['Volume'] / (data['CloseAdj'] / data['Close'])

# Replace inf and NaN values
for col in ['OpenAdj', 'HighAdj', 'LowAdj', 'CloseAdj', 'VolumeAdj']:
    data[col] = data[col].replace([np.inf, -np.inf], np.nan)
    data[col] = data[col].fillna(method='ffill')
    data[col] = data[col].fillna(method='bfill')

# Create weekly data
print("Creating weekly data...")
data['Year'] = data['Date'].dt.year
data['Week'] = data['Date'].dt.isocalendar().week

# Group by stock ID, year and week to get weekly data
weekly_data = data.groupby(['ID', 'Year', 'Week']).agg({
    'Date': 'last',  # Last trading day of the week
    'OpenAdj': 'first',  # First day's open
    'HighAdj': 'max',    # Highest high of the week
    'LowAdj': 'min',     # Lowest low of the week
    'CloseAdj': 'last',  # Last day's close
    'VolumeAdj': 'sum',  # Sum of volume for the week
    'SP100_Component': 'last'  # Use the last day's component status
}).reset_index()

# Sort by ID and date
weekly_data = weekly_data.sort_values(['ID', 'Date'])

# Calculate weekly returns
weekly_data['Return'] = weekly_data.groupby('ID')['CloseAdj'].pct_change()

# Drop rows with NaN returns (first row for each stock)
weekly_data = weekly_data.dropna(subset=['Return'])
print(f"Weekly data created: {weekly_data.shape} rows")

# Create sliding window features
def create_features(df, window_size=10):
    """Create sliding window features and filter for stocks in the index"""
    features = []
    
    for stock_id, stock_data in df.groupby('ID'):
        stock_data = stock_data.sort_values('Date')
        
        # Skip stocks with insufficient history
        if len(stock_data) <= window_size:
            continue
            
        for i in range(window_size, len(stock_data)):
            # Only include if the stock was in SP100 at that time
            if pd.notna(stock_data.iloc[i]['SP100_Component']):
                feature_row = {
                    'ID': stock_id,
                    'Date': stock_data.iloc[i]['Date'],
                    'Target': stock_data.iloc[i]['Return']
                }
                
                # Add past returns as features
                for j in range(window_size):
                    feature_row[f'Return_t-{window_size-j}'] = stock_data.iloc[i-window_size+j]['Return']
                
                # Add price info for PT/ST simulation
                feature_row['CloseAdj'] = stock_data.iloc[i]['CloseAdj']
                feature_row['HighAdj'] = stock_data.iloc[i+1]['HighAdj'] if i+1 < len(stock_data) else np.nan
                feature_row['LowAdj'] = stock_data.iloc[i+1]['LowAdj'] if i+1 < len(stock_data) else np.nan
                
                features.append(feature_row)
                
    return pd.DataFrame(features)

# Create features with a 10-week window
print("Creating sliding window features...")
window_size = 10
features_df = create_features(weekly_data, window_size)
print(f"Features created: {features_df.shape} rows")

# Split into training and testing sets
print("Splitting data into training and testing sets...")
train_df = features_df[features_df['Date'] <= '2020-12-31']
test_df = features_df[features_df['Date'] > '2020-12-31']
print(f"Training set: {train_df.shape} rows")
print(f"Testing set: {test_df.shape} rows")

# Prepare X and y
feature_cols = [f'Return_t-{i}' for i in range(window_size, 0, -1)]
X_train = train_df[feature_cols].values
y_train = train_df['Target'].values
X_test = test_df[feature_cols].values
y_test = test_df['Target'].values

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create test data by date for trading simulation
test_by_date = {}
for date, group in test_df.groupby('Date'):
    X = scaler.transform(group[feature_cols].values)
    test_by_date[date] = {
        'X': X,
        'y': group['Target'].values,
        'IDs': group['ID'].values,
        'CloseAdj': group['CloseAdj'].values,
        'HighAdj': group['HighAdj'].values,
        'LowAdj': group['LowAdj'].values
    }

# Define the MLP model
def create_mlp_model(hidden_layers=2, neurons=64, dropout_rate=0.2, l2_reg=0.001):
    model = Sequential()
    
    # Input layer
    model.add(Dense(neurons, activation='relu', kernel_regularizer=l2(l2_reg), 
                    input_shape=(X_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    
    # Hidden layers
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Create KerasRegressor for hyperparameter tuning
def create_model_wrapper(hidden_layers=2, neurons=64, dropout_rate=0.2, l2_reg=0.001):
    return create_mlp_model(hidden_layers, neurons, dropout_rate, l2_reg)

# Set up the model
print("Setting up the model for hyperparameter tuning...")
model = KerasRegressor(model=create_model_wrapper, verbose=0)

# Define hyperparameter search space
param_dist = {
    'hidden_layers': [1, 2, 3],
    'neurons': [32, 64, 128],
    'dropout_rate': [0.1, 0.2, 0.3],
    'l2_reg': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'epochs': [500]  # Will use early stopping
}

# Define early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Try different patience values (5, 10, 20)
    restore_best_weights=True,
    verbose=1
)

# Set up k-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Random search for hyperparameter tuning
print("Starting random search for hyperparameter tuning...")
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,  # Number of parameter settings to try
    cv=kfold,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Fit the random search
random_search_result = random_search.fit(
    X_train, y_train, 
    callbacks=[early_stopping],
    validation_split=0.2
)

# Get the best parameters
best_params = random_search_result.best_params_
print("Best parameters:", best_params)

# Train the final model with the best parameters
print("Training final model with best parameters...")
final_model = create_mlp_model(
    hidden_layers=best_params['hidden_layers'],
    neurons=best_params['neurons'],
    dropout_rate=best_params['dropout_rate'],
    l2_reg=best_params['l2_reg']
)

# Train history for plotting
history = final_model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=best_params['batch_size'],
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Define PT/ST trading simulation function
def simulate_trading_with_pt_st(model, test_by_date, pt_percentage=0.02, st_percentage=0.02):
    """
    Simulate trading strategy with PT/ST:
    - Buy top 10 stocks, short bottom 10 based on predictions
    - Use PT/ST logic for exit
    """
    results = []
    
    for date, data in sorted(test_by_date.items()):
        X = data['X']
        stock_ids = data['IDs']
        closes = data['CloseAdj']
        highs = data['HighAdj']
        lows = data['LowAdj']
        
        # Skip dates with too few stocks or missing next-week data
        if len(stock_ids) < 20 or np.isnan(highs).any() or np.isnan(lows).any():
            continue
        
        # Predict returns
        predicted_returns = model.predict(X).flatten()
        
        # Create DataFrame for analysis
        predictions_df = pd.DataFrame({
            'ID': stock_ids,
            'PredictedReturn': predicted_returns,
            'CloseAdj': closes,
            'HighAdj': highs,
            'LowAdj': lows
        })
        
        # Sort by predicted return
        predictions_df = predictions_df.sort_values('PredictedReturn', ascending=False)
        
        # Select top 10 for buying and bottom 10 for shorting
        long_stocks = predictions_df.head(10).copy()
        short_stocks = predictions_df.tail(10).copy()
        
        # Apply PT/ST logic for long positions
        long_stocks['PT'] = long_stocks['CloseAdj'] * (1 + pt_percentage)
        long_stocks['ST'] = long_stocks['CloseAdj'] * (1 - st_percentage)
        
        # Determine if PT or ST hit first for long positions
        long_stocks['HitPT'] = long_stocks['HighAdj'] >= long_stocks['PT']
        long_stocks['HitST'] = long_stocks['LowAdj'] <= long_stocks['ST']
        
        # Calculate returns for long positions
        long_stocks['Return'] = 0.0
        
        # If both PT and ST hit, determine which happened first based on distance
        both_hit = long_stocks['HitPT'] & long_stocks['HitST']
        if both_hit.any():
            # For simplicity, we'll assume the one that's exceeded more (percentage-wise) happened first
            pt_distance = (long_stocks['HighAdj'] - long_stocks['PT']) / long_stocks['PT']
            st_distance = (long_stocks['ST'] - long_stocks['LowAdj']) / long_stocks['ST']
            pt_first = pt_distance > st_distance
            
            long_stocks.loc[both_hit & pt_first, 'Return'] = pt_percentage
            long_stocks.loc[both_hit & ~pt_first, 'Return'] = -st_percentage
        
        # If only PT hit
        long_stocks.loc[long_stocks['HitPT'] & ~long_stocks['HitST'], 'Return'] = pt_percentage
        
        # If only ST hit
        long_stocks.loc[~long_stocks['HitPT'] & long_stocks['HitST'], 'Return'] = -st_percentage
        
        # If neither hit, calculate actual return
        neither_hit = ~long_stocks['HitPT'] & ~long_stocks['HitST']
        if neither_hit.any():
            long_stocks.loc[neither_hit, 'Return'] = (
                predictions_df.head(10).loc[neither_hit.index, 'HighAdj'] / 
                predictions_df.head(10).loc[neither_hit.index, 'CloseAdj'] - 1
            )
        
        # Apply PT/ST logic for short positions (inverse logic)
        short_stocks['PT'] = short_stocks['CloseAdj'] * (1 - pt_percentage)  # Lower price = profit for shorts
        short_stocks['ST'] = short_stocks['CloseAdj'] * (1 + st_percentage)  # Higher price = loss for shorts
        
        # Determine if PT or ST hit first for short positions
        short_stocks['HitPT'] = short_stocks['LowAdj'] <= short_stocks['PT']
        short_stocks['HitST'] = short_stocks['HighAdj'] >= short_stocks['ST']
        
        # Calculate returns for short positions
        short_stocks['Return'] = 0.0
        
        # If both PT and ST hit, determine which happened first
        both_hit = short_stocks['HitPT'] & short_stocks['HitST']
        if both_hit.any():
            pt_distance = (short_stocks['PT'] - short_stocks['LowAdj']) / short_stocks['PT']
            st_distance = (short_stocks['HighAdj'] - short_stocks['ST']) / short_stocks['ST']
            pt_first = pt_distance > st_distance
            
            short_stocks.loc[both_hit & pt_first, 'Return'] = pt_percentage
            short_stocks.loc[both_hit & ~pt_first, 'Return'] = -st_percentage
        
        # If only PT hit
        short_stocks.loc[short_stocks['HitPT'] & ~short_stocks['HitST'], 'Return'] = pt_percentage
        
        # If only ST hit
        short_stocks.loc[~short_stocks['HitPT'] & short_stocks['HitST'], 'Return'] = -st_percentage
        
        # If neither hit, calculate actual return (negative because we're shorting)
        neither_hit = ~short_stocks['HitPT'] & ~short_stocks['HitST']
        if neither_hit.any():
            short_stocks.loc[neither_hit, 'Return'] = -(
                predictions_df.tail(10).loc[neither_hit.index, 'HighAdj'] / 
                predictions_df.tail(10).loc[neither_hit.index, 'CloseAdj'] - 1
            )
        
        # Calculate average performance
        long_performance = long_stocks['Return'].mean()
        short_performance = short_stocks['Return'].mean()
        total_performance = (long_performance + short_performance) / 2
        
        # Track PT/ST hits
        long_pt_hits = long_stocks['HitPT'].sum()
        long_st_hits = long_stocks['HitST'].sum()
        short_pt_hits = short_stocks['HitPT'].sum()
        short_st_hits = short_stocks['HitST'].sum()
        
        results.append({
            'Date': date,
            'LongPerformance': long_performance,
            'ShortPerformance': short_performance,
            'TotalPerformance': total_performance,
            'LongPT': long_pt_hits,
            'LongST': long_st_hits,
            'ShortPT': short_pt_hits,
            'ShortST': short_st_hits
        })
    
    return pd.DataFrame(results)

# Run the trading simulation
print("Running trading simulation with PT/ST strategy...")
trading_results = simulate_trading_with_pt_st(final_model, test_by_date)

# Calculate performance metrics
trading_results['CumulativeReturn'] = (1 + trading_results['TotalPerformance']).cumprod() - 1

# Calculate win rate
win_rate = (trading_results['TotalPerformance'] > 0).mean()

# Calculate profit factor
positive_returns = trading_results[trading_results['TotalPerformance'] > 0]['TotalPerformance'].sum()
negative_returns = abs(trading_results[trading_results['TotalPerformance'] < 0]['TotalPerformance'].sum())
profit_factor = positive_returns / negative_returns if negative_returns > 0 else float('inf')

# Calculate drawdown
cumulative_returns = (1 + trading_results['TotalPerformance']).cumprod()
running_max = cumulative_returns.cummax()
drawdown = (cumulative_returns / running_max) - 1
max_drawdown = drawdown.min()

# Calculate Sharpe Ratio (assuming weekly data, so annualize by sqrt(52))
sharpe_ratio = (trading_results['TotalPerformance'].mean() / trading_results['TotalPerformance'].std()) * np.sqrt(52)

# Calculate R-squared on test data
y_pred_test = final_model.predict(X_test).flatten()
r2 = r2_score(y_test, y_pred_test)

# Create visualization
print("Creating visualization...")
plt.figure(figsize=(18, 15))

# 1. Training history (loss)
plt.subplot(3, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axvline(x=np.argmin(history.history['val_loss']), color='r', linestyle='--', 
            label=f'Best epoch: {np.argmin(history.history["val_loss"])+1}')
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

# 2. Actual vs Predicted Returns (Test Set)
plt.subplot(3, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.3)
plt.plot([-0.5, 0.5], [-0.5, 0.5], 'r--')
plt.title('Actual vs Predicted Returns (Test Set)')
plt.xlabel('Actual Weekly Return')
plt.ylabel('Predicted Weekly Return')
plt.grid(True)
plt.text(-0.4, 0.4, f'R²: {r2:.4f}', fontsize=12)

# 3. Cumulative Return of the Strategy
plt.subplot(3, 2, 3)
plt.plot(trading_results['Date'], trading_results['CumulativeReturn'] * 100)
plt.axvline(x=datetime(2020, 12, 31), color='k', linestyle='--', label='Train/Test Split')
plt.title('Cumulative Return of Strategy (%)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (%)')
plt.grid(True)
plt.legend()

# 4. Performance Distribution
plt.subplot(3, 2, 4)
plt.hist(trading_results['TotalPerformance'] * 100, bins=30)
plt.title('Weekly Return Distribution (%)')
plt.xlabel('Weekly Return (%)')
plt.ylabel('Frequency')
plt.grid(True)

# 5. Long vs Short Performance
plt.subplot(3, 2, 5)
plt.plot(trading_results['Date'], trading_results['LongPerformance'] * 100, label='Long')
plt.plot(trading_results['Date'], trading_results['ShortPerformance'] * 100, label='Short')
plt.title('Weekly Long vs Short Performance (%)')
plt.xlabel('Date')
plt.ylabel('Weekly Return (%)')
plt.legend()
plt.grid(True)

# 6. Drawdown
plt.subplot(3, 2, 6)
plt.plot(trading_results['Date'], drawdown * 100)
plt.title('Strategy Drawdown (%)')
plt.xlabel('Date')
plt.ylabel('Drawdown (%)')
plt.grid(True)
plt.tight_layout()

# Create a text box with key metrics
metrics_text = (
    f"Window Size: {window_size} weeks\n"
    f"Best Params: {best_params}\n"
    f"R²: {r2:.4f}\n"
    f"Win Rate: {win_rate:.2%}\n"
    f"Profit Factor: {profit_factor:.2f}\n"
    f"Max Drawdown: {max_drawdown:.2%}\n"
    f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
    f"Total Return: {trading_results['CumulativeReturn'].iloc[-1]:.2%}\n"
    f"Execution Time: {(time.time() - start_time) / 60:.2f} minutes"
)

plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Save the figure
plt.savefig('weekly_mlp_model_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Print key metrics
print("\n=== Model Performance ===")
print(f"Window Size: {window_size} weeks")
print(f"R²: {r2:.4f}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Total Return: {trading_results['CumulativeReturn'].iloc[-1]:.2%}")
print(f"Execution Time: {(time.time() - start_time) / 60:.2f} minutes")