import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from darts import TimeSeries
from darts.models import RNNModel
from darts.metrics import mape, rmse, mae
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Load the dataset
def load_and_prepare_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Filter for the specific stock ID
    stock_data = df[df['ID'] == 891399].copy()
    
    # Convert date to datetime format
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    
    # Sort by date
    stock_data.sort_values('Date', inplace=True)
    
    # Calculate adjusted OHL prices
    stock_data['OpenAdj'] = stock_data['Open'] * (stock_data['CloseAdj'] / stock_data['Close'])
    stock_data['HighAdj'] = stock_data['High'] * (stock_data['CloseAdj'] / stock_data['Close'])
    stock_data['LowAdj'] = stock_data['Low'] * (stock_data['CloseAdj'] / stock_data['Close'])
    stock_data['VolumeAdj'] = stock_data['Volume'] / (stock_data['CloseAdj'] / stock_data['Close'])
    
    # Keep only necessary columns
    stock_data = stock_data[['Date', 'OpenAdj', 'HighAdj', 'LowAdj', 'CloseAdj', 'VolumeAdj']]
    
    # Resample to weekly frequency (end of week)
    weekly_data = stock_data.set_index('Date').resample('W').last()
    
    # Fill missing values if any
    weekly_data.fillna(method='ffill', inplace=True)
    
    return weekly_data

# Calculate technical indicators for MLP input features
def add_technical_indicators(df):
    print("Calculating technical indicators...")
    # Weekly returns
    df['weekly_return'] = df['CloseAdj'].pct_change()
    
    # Calculate returns for different periods
    for period in [1, 2, 3, 4, 8, 12, 26]:
        # Return over the past n weeks
        df[f'return_{period}w'] = df['CloseAdj'].pct_change(period)
    
    # Volatility measures (standard deviation of returns)
    for period in [4, 8, 12, 26]:
        df[f'volatility_{period}w'] = df['weekly_return'].rolling(window=period).std()
    
    # Moving averages
    for period in [4, 8, 12, 26]:
        df[f'ma_{period}w'] = df['CloseAdj'].rolling(window=period).mean()
        # Distance from moving average
        df[f'dist_from_ma_{period}w'] = (df['CloseAdj'] - df[f'ma_{period}w']) / df[f'ma_{period}w']
    
    # Relative strength indicators
    for period in [6, 12, 26]:
        # Simple RSI calculation
        delta = df['CloseAdj'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi_{period}w'] = 100 - (100 / (1 + rs))
    
    # Volume indicators
    df['volume_change'] = df['VolumeAdj'].pct_change()
    for period in [4, 8]:
        df[f'volume_ma_{period}w'] = df['VolumeAdj'].rolling(window=period).mean()
        df[f'rel_volume_{period}w'] = df['VolumeAdj'] / df[f'volume_ma_{period}w']
    
    # Range and volatility indicators
    df['weekly_range'] = (df['HighAdj'] - df['LowAdj']) / df['OpenAdj']
    
    # Clean up NaN values from calculations
    df.dropna(inplace=True)
    
    # Target variable: next week's return sign (1 for positive, -1 for negative)
    df['next_week_return'] = df['weekly_return'].shift(-1)
    df['target_sign'] = np.sign(df['next_week_return'])
    df['target_return'] = df['next_week_return']
    
    # Drop the last row as we don't have the next week's return
    df.dropna(subset=['next_week_return'], inplace=True)
    
    return df

# Prepare data for ML model
def prepare_model_data(df, target_type='sign'):
    print("Preparing data for modeling...")
    # Define features to use
    features = [col for col in df.columns if col not in ['next_week_return', 'target_sign', 'target_return', 'CloseAdj', 'OpenAdj', 'HighAdj', 'LowAdj', 'VolumeAdj']]
    
    # Split data into training (2005-2020) and testing (2021-2025) sets
    train_data = df[df.index < '2021-01-01']
    test_data = df[df.index >= '2021-01-01']
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[features])
    X_test = scaler.transform(test_data[features])
    
    # Prepare target variable
    if target_type == 'sign':
        y_train = train_data['target_sign'].values
        y_test = test_data['target_sign'].values
    else:  # Return value
        y_train = train_data['target_return'].values
        y_test = test_data['target_return'].values
    
    return X_train, y_train, X_test, y_test, train_data, test_data, scaler, features

# Create and train MLP model with random search
def build_and_train_model(X_train, y_train, target_type='sign'):
    print("Building and training MLP model...")
    
    # Create TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=10)
    
    # Define the model
    def create_model(neurons=64, dropout_rate=0.2, l2_reg=0.001, learning_rate=0.001):
        model = Sequential()
        model.add(Dense(neurons, activation='relu', kernel_regularizer=l2(l2_reg), input_shape=(X_train.shape[1],)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(neurons//2, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))
        
        if target_type == 'sign':
            # Classification model
            model.add(Dense(1, activation='tanh'))  # tanh will give values close to -1 or 1
            loss = 'mse'  # Mean squared error works well for this binary task with -1/1 values
        else:
            # Regression model
            model.add(Dense(1, activation='linear'))
            loss = 'mse'
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss
        )
        return model
    
    # Parameter grid for random search 
    param_grid = {
        'neurons': [32, 64],
        'dropout_rate': [0.2, 0.3],
        'l2_reg': [0.0001, 0.001, 0.01],
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [16],
        'patience': [5]
    }
    
    # Random search for hyperparameter tuning
    best_score = float('inf')
    best_params = None
    best_model = None
    
    # Try 20 random combinations
    n_iter = 20
    param_list = []
    
    for _ in range(n_iter):
        params = {
            'neurons': np.random.choice(param_grid['neurons']),
            'dropout_rate': np.random.choice(param_grid['dropout_rate']),
            'l2_reg': np.random.choice(param_grid['l2_reg']),
            'learning_rate': np.random.choice(param_grid['learning_rate']),
            'batch_size': np.random.choice(param_grid['batch_size']),
            'patience': np.random.choice(param_grid['patience'])
        }
        param_list.append(params)
    
    for i, params in enumerate(param_list):
        print(f"\nTraining with parameters set {i+1}/{n_iter}:")
        print(params)
        
        # Cross-validation
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
            
            # Create model
            model = create_model(
                neurons=params['neurons'],
                dropout_rate=params['dropout_rate'],
                l2_reg=params['l2_reg'],
                learning_rate=params['learning_rate']
            )
            
            # Early stopping
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=params['patience'],
                restore_best_weights=True,
                verbose=0
            )
            
            # Train model
            history = model.fit(
                X_cv_train, y_cv_train,
                epochs=10,  # !!! dala jsem 10 epoch, aby to bylo rychlejší, můžeš zvýšit na 1000
                batch_size=params['batch_size'],
                validation_data=(X_cv_val, y_cv_val),
                callbacks=[early_stop],
                verbose=0
            )
            
            # Record validation score
            cv_scores.append(min(history.history['val_loss']))
        
        mean_cv_score = np.mean(cv_scores)
        print(f"Mean CV Score: {mean_cv_score:.4f}")
        
        if mean_cv_score < best_score:
            best_score = mean_cv_score
            best_params = params
            
    print("\nBest parameters:", best_params)
    
    # Train final model with best parameters
    final_model = create_model(
        neurons=best_params['neurons'],
        dropout_rate=best_params['dropout_rate'],
        l2_reg=best_params['l2_reg'],
        learning_rate=best_params['learning_rate']
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=best_params['patience'],
        restore_best_weights=True,
        verbose=1
    )
    
    # Split a small validation set from the training set
    val_size = int(len(X_train) * 0.1)
    X_final_train, X_val = X_train[:-val_size], X_train[-val_size:]
    y_final_train, y_val = y_train[:-val_size], y_train[-val_size:]
    
    # Train the final model
    history = final_model.fit(
        X_final_train, y_final_train,
        epochs=10,  # !!! dala jsem 10 epoch, aby to bylo rychlejší, můžeš zvýšit na 1000
        batch_size=best_params['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    return final_model, history, best_params

# Evaluate model and simulate trading strategy
def evaluate_and_simulate(model, X_train, y_train, X_test, y_test, train_data, test_data, target_type='sign'):
    print("Evaluating model and simulating trading strategy...")
    
    # Get predictions
    train_pred = model.predict(X_train).flatten()
    test_pred = model.predict(X_test).flatten()
    
    # Convert to signs if the target is return value
    if target_type != 'sign':
        train_pred_sign = np.sign(train_pred)
        test_pred_sign = np.sign(test_pred)
        y_train_sign = np.sign(y_train)
        y_test_sign = np.sign(y_test)
    else:
        train_pred_sign = np.sign(train_pred)
        test_pred_sign = np.sign(test_pred)
        y_train_sign = y_train
        y_test_sign = y_test
    
    # Classification accuracy
    train_accuracy = np.mean(train_pred_sign == y_train_sign)
    test_accuracy = np.mean(test_pred_sign == y_test_sign)
    
    # Výpočet R² (koeficient determinace)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Testing accuracy: {test_accuracy:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    
    # Store predictions in DataFrames
    train_data = train_data.copy()
    train_data['prediction'] = train_pred
    train_data['prediction_sign'] = train_pred_sign
    
    test_data = test_data.copy()
    test_data['prediction'] = test_pred
    test_data['prediction_sign'] = test_pred_sign
    
    # Simulate trading strategy with profit target and stop loss
    def simulate_trading(data, pt_sl_pct=0.02):
        """Simulate trading with profit target and stop loss"""
        data = data.copy()
        
        # Initialize result columns
        data['position'] = np.sign(data['prediction'])  # 1 for long, -1 for short
        data['entry_price'] = data['CloseAdj']
        data['profit_target'] = data['entry_price'] * (1 + pt_sl_pct * data['position'])
        data['stop_loss'] = data['entry_price'] * (1 - pt_sl_pct * data['position'])
        data['return'] = 0.0
        
        # Simulate trading
        for i in range(len(data) - 1):
            position = data.iloc[i]['position']
            entry_price = data.iloc[i]['entry_price']
            pt_price = data.iloc[i]['profit_target']
            sl_price = data.iloc[i]['stop_loss']
            
            # Next week's price data
            next_open = data.iloc[i+1]['OpenAdj']
            next_high = data.iloc[i+1]['HighAdj']
            next_low = data.iloc[i+1]['LowAdj']
            next_close = data.iloc[i+1]['CloseAdj']
            
            # Check if profit target or stop loss was hit
            if position > 0:  # Long position
                if next_high >= pt_price and next_low <= sl_price:
                    # Both PT and SL hit - determine which one first
                    # Simple heuristic: if (high-pt) > (sl-low), then SL hit first
                    if (next_high - pt_price) > (sl_price - next_low):
                        data.iloc[i, data.columns.get_loc('return')] = (sl_price - entry_price) / entry_price
                    else:
                        data.iloc[i, data.columns.get_loc('return')] = (pt_price - entry_price) / entry_price
                elif next_high >= pt_price:
                    # PT hit
                    data.iloc[i, data.columns.get_loc('return')] = (pt_price - entry_price) / entry_price
                elif next_low <= sl_price:
                    # SL hit
                    data.iloc[i, data.columns.get_loc('return')] = (sl_price - entry_price) / entry_price
                else:
                    # Exit at close
                    data.iloc[i, data.columns.get_loc('return')] = (next_close - entry_price) / entry_price
            
            elif position < 0:  # Short position
                if next_low <= pt_price and next_high >= sl_price:
                    # Both PT and SL hit - determine which one first
                    # Simple heuristic: if (pt-low) > (high-sl), then SL hit first
                    if (pt_price - next_low) > (next_high - sl_price):
                        data.iloc[i, data.columns.get_loc('return')] = (entry_price - sl_price) / entry_price
                    else:
                        data.iloc[i, data.columns.get_loc('return')] = (entry_price - pt_price) / entry_price
                elif next_low <= pt_price:
                    # PT hit
                    data.iloc[i, data.columns.get_loc('return')] = (entry_price - pt_price) / entry_price
                elif next_high >= sl_price:
                    # SL hit
                    data.iloc[i, data.columns.get_loc('return')] = (entry_price - sl_price) / entry_price
                else:
                    # Exit at close
                    data.iloc[i, data.columns.get_loc('return')] = (entry_price - next_close) / entry_price
        
        # Calculate cumulative returns
        data['cum_return'] = (1 + data['return']).cumprod() - 1
        
        # Calculate performance metrics
        total_trades = len(data) - 1
        winning_trades = sum(data['return'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        profit_trades = data[data['return'] > 0]['return'].sum()
        loss_trades = abs(data[data['return'] < 0]['return'].sum())
        profit_factor = profit_trades / loss_trades if loss_trades > 0 else float('inf')
        
        # Calculate drawdown
        peak = data['cum_return'].expanding().max()
        drawdown = (data['cum_return'] - peak) / (1 + peak)
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        annual_returns = data['return'].mean() * 52  # Annualize based on 52 weeks/year
        annual_volatility = data['return'].std() * np.sqrt(52)
        sharpe_ratio = annual_returns / annual_volatility if annual_volatility > 0 else 0
        
        results = {
            'total_return': data['cum_return'].iloc[-1],
            'annualized_return': annual_returns,
            'annualized_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades
        }
        
        return data, results
    
    # Run simulation
    train_results, train_metrics = simulate_trading(train_data)
    test_results, test_metrics = simulate_trading(test_data)
    
    print("\nTraining period performance:")
    for key, value in train_metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\nTest period performance:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")
    
    return train_results, test_results, train_metrics, test_metrics

# Visualize results using darts
def visualize_results(train_results, test_results, train_metrics, test_metrics, history, best_params):
    print("Creating visualizations...")
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(16, 20))
    
    # 1. Plot loss during training
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss During Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Plot actual vs predicted returns
    ax2 = fig.add_subplot(4, 1, 2)
    combined_dates = train_results.index.tolist() + test_results.index.tolist()
    combined_actual = train_results['next_week_return'].tolist() + test_results['next_week_return'].tolist()
    combined_pred = train_results['prediction'].tolist() + test_results['prediction'].tolist()
    
    ax2.plot(combined_dates, combined_actual, label='Actual Return', alpha=0.7)
    ax2.plot(combined_dates, combined_pred, label='Predicted Return', alpha=0.7)
    
    # Add a vertical line at the train/test split
    split_date = pd.Timestamp('2021-01-01')
    ax2.axvline(x=split_date, color='r', linestyle='--', label='Train/Test Split')
    
    ax2.set_title('Actual vs Predicted Returns')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Return')
    ax2.legend()
    ax2.grid(True)
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Plot cumulative returns from strategy
    ax3 = fig.add_subplot(4, 1, 3)
    combined_cum_returns = train_results['cum_return'].tolist() + test_results['cum_return'].tolist()
    
    ax3.plot(combined_dates, combined_cum_returns)
    ax3.axvline(x=split_date, color='r', linestyle='--', label='Train/Test Split')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax3.set_title('Cumulative Returns from Trading Strategy')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Cumulative Return')
    ax3.grid(True)
    
    # Format x-axis dates
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Display model performance metrics
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.axis('off')
    
    metrics_text = "Model Performance Metrics\n\n"
    
    # Add best hyperparameters
    metrics_text += "Best Hyperparameters:\n"
    for param, value in best_params.items():
        metrics_text += f"- {param}: {value}\n"
    
    metrics_text += "\nTraining Period:\n"
    for key, value in train_metrics.items():
        metrics_text += f"- {key}: {value:.4f}\n"
    
    metrics_text += "\nTest Period:\n"
    for key, value in test_metrics.items():
        metrics_text += f"- {key}: {value:.4f}\n"
    
    ax4.text(0.01, 0.99, metrics_text, va='top', ha='left', fontsize=10, transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig('/Users/lindawaisova/Desktop/DP/figures/weekly_stock_mlp_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Results visualization saved as 'weekly_stock_mlp_results.png'")

# Main execution
def main():
    # Začátek měření času
    start_time = time.time()
    
    # File path
    file_path = '/Users/lindawaisova/Desktop/DP/data/SP_100/Reuters_SP100_Data.csv'
    
    # Load and prepare data
    weekly_data = load_and_prepare_data(file_path)
    
    # Add technical indicators
    data_with_indicators = add_technical_indicators(weekly_data)
    
    # Prepare data for modeling
    X_train, y_train, X_test, y_test, train_data, test_data, scaler, features = prepare_model_data(
        data_with_indicators, target_type='sign'
    )
    
    # Build and train model
    model, history, best_params = build_and_train_model(X_train, y_train, target_type='sign')
    
    # Evaluate model and simulate trading
    train_results, test_results, train_metrics, test_metrics = evaluate_and_simulate(
        model, X_train, y_train, X_test, y_test, train_data, test_data, target_type='sign'
    )
    
    # Visualize results
    visualize_results(train_results, test_results, train_metrics, test_metrics, history, best_params)
    
    # Konec měření času a výpočet celkové doby běhu
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Převod na hodiny, minuty a sekundy pro lepší čitelnost
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    
    print("\n" + "="*50)
    print(f"CELKOVÁ DOBA BĚHU KÓDU: {hours} hodin, {minutes} minut, {seconds} sekund")
    print(f"CELKOVÝ ČAS V SEKUNDÁCH: {execution_time:.2f} s")
    print("="*50)

if __name__ == "__main__":
    main()