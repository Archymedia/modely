
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
import time
import warnings

from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# Potlačení varování
warnings.filterwarnings('ignore')
tf.keras.utils.disable_interactive_logging()

# Nastavení pro reprodukovatelnost
np.random.seed(42)
tf.random.set_seed(42)

# Měření celkového času běhu
start_time = time.time()

# Parametry (DATA_PATH = r"C:\Users\Unknown\Desktop\data\SP100\Reuters_SP100_Data.csv")
DATA_PATH = "/Users/lindawaisova/Desktop/DP/data/SP_100/Reuters_SP100_Data.csv"
WINDOW_SIZE = 10  # Velikost okna pro sliding window
PT_PERCENTAGE = 0.02  # Profit target (2%)
SL_PERCENTAGE = 0.02  # Stop loss (2%)
TRAIN_END_DATE = '2020-12-31'
TEST_START_DATE = '2021-01-01'
TEST_END_DATE = '2023-12-29'

# ---------------------- 1. Načtení a zpracování dat ----------------------
print("Načítání a zpracování dat...")

# Načtení dat
df = pd.read_csv(DATA_PATH)

# Převod datumu na správný formát
df['Date'] = pd.to_datetime(df['Date'])

# Ošetření chybějících a nulových hodnot
# 1. Imputace chybějících hodnot
for col in ['Open', 'High', 'Low', 'Close', 'TotRet']:
    # Nahrazení NaN hodnot předchozími hodnotami (po skupinách podle ID)
    df[col] = df.groupby('ID')[col].fillna(method='ffill')

# 2. Ošetření nulových hodnot (nahrazení předchozí hodnotou)
for col in ['Close', 'TotRet']:
    # Vytvoření masky pro nulové hodnoty
    zero_mask = df[col] == 0
    # Nahrazení nulových hodnot NaN
    df.loc[zero_mask, col] = np.nan
    # Nahrazení NaN hodnot předchozími hodnotami (po skupinách podle ID)
    df[col] = df.groupby('ID')[col].fillna(method='ffill')

# Úprava OHLC cen pro stock splity
# Pokud dělení nulou, ponechej původní hodnoty
df['ratio'] = np.where(df['Close'] > 0, df['CloseAdj'] / df['Close'], 1)
df['OpenAdj'] = df['Open'] * df['ratio']
df['HighAdj'] = df['High'] * df['ratio']
df['LowAdj'] = df['Low'] * df['ratio']
df['VolumeAdj'] = df['Volume'] / df['ratio']

# ---------------------- 2. Výpočet týdenních returnů ----------------------
print("Výpočet týdenních returnů...")

# Převedení denních dat na týdenní (bereme poslední hodnotu týdne)
weekly_df = df.groupby(['ID', pd.Grouper(key='Date', freq='W-FRI')]).last().reset_index()

# Výpočet týdenních returnů z TotRet pro všechny akcie
weekly_df['Return'] = weekly_df.groupby('ID')['TotRet'].pct_change()

# Ošetření NaN a Inf hodnot v returnech
weekly_df['Return'] = weekly_df['Return'].replace([np.inf, -np.inf], np.nan)
weekly_df['Return'] = weekly_df.groupby('ID')['Return'].fillna(0)

# ---------------------- 3. Filtrování období a příprava dat ----------------------
print("Filtrování období a příprava dat...")

# Filtrování období
filtered_df = weekly_df[(weekly_df['Date'] >= '2005-01-01') & (weekly_df['Date'] <= '2023-12-29')]

# Rozdělení na trénovací a testovací období
train_df = filtered_df[filtered_df['Date'] <= TRAIN_END_DATE]
test_df = filtered_df[(filtered_df['Date'] > TEST_START_DATE) & (filtered_df['Date'] <= TEST_END_DATE)]

# ---------------------- 4. Vytvoření prediktorů a sekvencí ----------------------
print("Vytváření prediktorů a sekvencí...")

def create_features_for_stock(df_stock):
    """Vytvoření prediktorů pro jednu akcii"""
    features = pd.DataFrame()
    
    # Základní prediktory: zpožděné returny
    for i in range(1, WINDOW_SIZE + 1):
        features[f'return_lag_{i}'] = df_stock['Return'].shift(i)
    
    # Technické indikátory
    # Jednoduché klouzavé průměry returnů
    features['ma5'] = df_stock['Return'].rolling(window=5).mean()
    features['ma10'] = df_stock['Return'].rolling(window=10).mean()
    
    # Volatilita (standardní odchylka returnů)
    features['volatility_5'] = df_stock['Return'].rolling(window=5).std()
    features['volatility_10'] = df_stock['Return'].rolling(window=10).std()
    
    # Relativní síla - poměr aktuální ceny k průměru
    features['rs_5'] = df_stock['CloseAdj'] / df_stock['CloseAdj'].rolling(window=5).mean()
    features['rs_10'] = df_stock['CloseAdj'] / df_stock['CloseAdj'].rolling(window=10).mean()
    
    # Přidání cílové proměnné - return následujícího týdne
    features['target'] = df_stock['Return'].shift(-1)
    
    # Přidání datumu a ID pro pozdější identifikaci
    features['Date'] = df_stock['Date']
    features['ID'] = df_stock['ID']
    
    # Odstranění řádků s chybějícími hodnotami
    features = features.dropna()
    
    return features

# Aplikace funkce na každou akcii
stock_features = []
for stock_id, stock_data in filtered_df.groupby('ID'):
    stock_features.append(create_features_for_stock(stock_data))

# Spojení všech akcií do jednoho DataFrame
all_features = pd.concat(stock_features, ignore_index=True)

# ---------------------- 5. Standardizace dat ----------------------
print("Standardizace dat...")

# Rozdělení na trénovací a testovací období
train_features = all_features[all_features['Date'] <= TRAIN_END_DATE]
test_features = all_features[(all_features['Date'] > TEST_START_DATE) & (all_features['Date'] <= TEST_END_DATE)]

# Identifikace prediktorů a cílové proměnné
feature_cols = [col for col in all_features.columns if col not in ['Date', 'ID', 'target']]
X_train = train_features[feature_cols]
y_train = train_features['target']
X_test = test_features[feature_cols]
y_test = test_features['target']

# Standardizace prediktorů - fit pouze na trénovacích datech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------- 6. Definice MLP modelu ----------------------
def create_model(neurons=64, hidden_layers=1, dropout_rate=0.2, l2_reg=0.001, learning_rate=0.001):
    """Vytvoření MLP modelu s parametry včetně learning rate"""
    model = Sequential()
    
    # Vstupní vrstva
    model.add(Dense(neurons, activation='relu', kernel_regularizer=l2(l2_reg), 
                   input_shape=(X_train_scaled.shape[1],)))
    model.add(Dropout(dropout_rate))
    
    # Skryté vrstvy
    for _ in range(hidden_layers):
        model.add(Dense(neurons, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))
    
    # Výstupní vrstva - predikce jednoho hodnoty (return)
    model.add(Dense(1))
    
    # Kompilace modelu s learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model

# ---------------------- 7. Random Search pro optimalizaci hyperparametrů ----------------------
print("Optimalizace hyperparametrů...")

# Wrapper pro Keras model
model = KerasRegressor(model=create_model, verbose=0)

# Definice prostoru hyperparametrů včetně learning rate
param_dist = {
    'model__neurons': [32],
    'model__hidden_layers': [1],
    'model__dropout_rate': [0.2],
    'model__l2_reg': [0.01],
    'model__learning_rate': [0.001],
    'batch_size': [64],
    'epochs': [5],  # Používáme early stopping
}

# Early stopping callback
early_stopping = [tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)]

# K-fold CV (10 foldů)
kfold = KFold(n_splits=3, shuffle=True, random_state=42)

# Random search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=2,  # Počet kombinací k testování
    cv=kfold,
    verbose=1,
    n_jobs=-1,  # Použít 1 core (lze zvýšit pro rychlejší trénink) POUŽÍT -1 !!! 
    scoring='neg_mean_squared_error'
)

# Vykonání random search
random_search.fit(
    X_train_scaled, y_train, 
    callbacks=early_stopping,
    validation_split=0.2
)

# Výpis nejlepších hyperparametrů
print("Nejlepší parametry:", random_search.best_params_)
print("Nejlepší skóre:", -random_search.best_score_)

# ---------------------- 8. Trénování modelu s nejlepšími parametry ----------------------
print("Trénování finálního modelu...")

# Získání nejlepších parametrů --> CO DĚLAT KDYŽ V BEST_PARAMS NENÍ NALEZEN ŽÁDNÝ HYPERPARAMETR??
best_params = random_search.best_params_
best_neurons = best_params.get('model__neurons', 64)
best_hidden_layers = best_params.get('model__hidden_layers', 2)
best_dropout_rate = best_params.get('model__dropout_rate', 0.2)
best_l2_reg = best_params.get('model__l2_reg', 0.001)
best_learning_rate = best_params.get('model__learning_rate', 0.001)  # přidáno
best_batch_size = best_params.get('batch_size', 64)

# Vytvoření finálního modelu
final_model = create_model(
    neurons=best_neurons,
    hidden_layers=best_hidden_layers,
    dropout_rate=best_dropout_rate,
    l2_reg=best_l2_reg,
    learning_rate=best_learning_rate
)

# Early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True,
    verbose=1
)

# Trénování finálního modelu
history = final_model.fit(
    X_train_scaled, y_train,
    epochs=5,
    batch_size=best_batch_size,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# ---------------------- 9. Predikce a vyhodnocení modelu ----------------------
print("Predikce a vyhodnocení...")

# Predikce na trénovacích a testovacích datech
y_train_pred = final_model.predict(X_train_scaled)
y_test_pred = final_model.predict(X_test_scaled)

# Metriky modelu
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Trénovací MSE: {train_mse:.6f}")
print(f"Testovací MSE: {test_mse:.6f}")
print(f"Trénovací R²: {train_r2:.6f}")
print(f"Testovací R²: {test_r2:.6f}")

# ---------------------- 10. Simulace obchodní strategie ----------------------
print("Simulace obchodní strategie...")

# Příprava dataframe pro výsledky testovacího období po týdnech
test_results = pd.DataFrame({
    'Date': test_features['Date'],
    'ID': test_features['ID'],
    'Actual_Return': test_features['target'],
    'Predicted_Return': y_test_pred.flatten()
})

# Funkce pro simulaci obchodování v jednom týdnu
def simulate_weekly_trading(df_week, pt_pct=PT_PERCENTAGE, sl_pct=SL_PERCENTAGE):
    """
    Simulace obchodní strategie pro jeden týden:
    1. Top 10 akcií s nejvyšší predikovanou return - long pozice
    2. Bottom 10 akcií s nejnižší predikovanou return - short pozice
    3. Aplikace PT/SL pro každou pozici
    """
    # Seřazení akcií podle predikované return
    sorted_df = df_week.sort_values('Predicted_Return', ascending=False).reset_index(drop=True)
    
    # Výběr top 10 (long) a bottom 10 (short) akcií
    long_positions = sorted_df.head(10).copy()
    long_positions['Position'] = 'LONG'
    
    short_positions = sorted_df.tail(10).copy()
    short_positions['Position'] = 'SHORT'
    
    # Spojení long a short pozic
    positions = pd.concat([long_positions, short_positions], ignore_index=True)
    
    # Lookup pro příští týden k zjištění výsledků
    next_week = (pd.to_datetime(df_week['Date'].iloc[0]) + pd.DateOffset(weeks=1)).strftime('%Y-%m-%d')
    
    # Výsledky obchodů
    trades = []
    
    for _, position in positions.iterrows():
        stock_id = position['ID']
        pos_type = position['Position']
        
        # Vyhledání dat pro následující týden pro tuto akcii
        next_week_data = weekly_df[(weekly_df['ID'] == stock_id) & 
                                   (weekly_df['Date'] > position['Date']) & 
                                   (weekly_df['Date'] <= next_week)]
        
        # Pokud není data pro příští týden, přeskočit
        if next_week_data.empty:
            continue
        
        entry_price = position['CloseAdj']
        next_week_close = next_week_data['CloseAdj'].iloc[0]
        next_week_high = next_week_data['HighAdj'].iloc[0]
        next_week_low = next_week_data['LowAdj'].iloc[0]
        
        # Nastavení PT a SL
        if pos_type == 'LONG':
            pt_price = entry_price * (1 + pt_pct)
            sl_price = entry_price * (1 - sl_pct)
            
            # Určení výsledku obchodu
            if next_week_high >= pt_price and next_week_low <= sl_price:
                # Pokud jsou zasaženy oba PT i SL, rozhodne relativní vzdálenost
                if abs(next_week_close - pt_price) < abs(next_week_close - sl_price):
                    result = 'PT'
                    pnl_pct = pt_pct
                else:
                    result = 'SL'
                    pnl_pct = -sl_pct
            elif next_week_high >= pt_price:
                result = 'PT'
                pnl_pct = pt_pct
            elif next_week_low <= sl_price:
                result = 'SL'
                pnl_pct = -sl_pct
            else:
                result = 'CLOSE'
                pnl_pct = (next_week_close - entry_price) / entry_price
        else:  # SHORT
            pt_price = entry_price * (1 - pt_pct)
            sl_price = entry_price * (1 + sl_pct)
            
            # Určení výsledku obchodu
            if next_week_low <= pt_price and next_week_high >= sl_price:
                # Pokud jsou zasaženy oba PT i SL, rozhodne relativní vzdálenost
                if abs(next_week_close - pt_price) < abs(next_week_close - sl_price):
                    result = 'PT'
                    pnl_pct = pt_pct
                else:
                    result = 'SL'
                    pnl_pct = -sl_pct
            elif next_week_low <= pt_price:
                result = 'PT'
                pnl_pct = pt_pct
            elif next_week_high >= sl_price:
                result = 'SL'
                pnl_pct = -sl_pct
            else:
                result = 'CLOSE'
                pnl_pct = (entry_price - next_week_close) / entry_price
        
        # Uložení výsledku obchodu
        trades.append({
            'Date': position['Date'],
            'ID': stock_id,
            'Position': pos_type,
            'Entry': entry_price,
            'Exit': next_week_close,
            'Result': result,
            'PnL_Pct': pnl_pct
        })
    
    return pd.DataFrame(trades)

# Simulace obchodní strategie po týdnech
all_trades = []

# Získání unikátních týdnů v testovacím období
test_weeks = test_results['Date'].unique()

for week in test_weeks:
    # Filtrování dat pro aktuální týden
    week_data = test_results[test_results['Date'] == week].copy()
    
    # Spojení s original dataframe pro získání potřebných hodnot (Close, High, Low)
    week_data = pd.merge(
        week_data, 
        weekly_df[['Date', 'ID', 'CloseAdj', 'HighAdj', 'LowAdj']], 
        on=['Date', 'ID'], 
        how='left'
    )
    
    # Simulace obchodů pro tento týden
    week_trades = simulate_weekly_trading(week_data)
    all_trades.append(week_trades)

# Spojení všech obchodů
trade_results = pd.concat(all_trades, ignore_index=True)

# ---------------------- 11. Výpočet metrik obchodní strategie ----------------------
print("Výpočet metrik obchodní strategie...")

# Základní statistiky obchodů
total_trades = len(trade_results)
winning_trades = len(trade_results[trade_results['PnL_Pct'] > 0])
losing_trades = len(trade_results[trade_results['PnL_Pct'] <= 0])
win_rate = winning_trades / total_trades if total_trades > 0 else 0

# Profit Factor
gross_profit = trade_results[trade_results['PnL_Pct'] > 0]['PnL_Pct'].sum()
gross_loss = abs(trade_results[trade_results['PnL_Pct'] <= 0]['PnL_Pct'].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

# Kumulativní výnos (předpokládáme rovnoměrné rozdělení kapitálu)
trade_results = trade_results.sort_values('Date')
trade_results['Cumulative_Return'] = (1 + trade_results['PnL_Pct']).cumprod() - 1

# Výpočet Sharpe Ratio (předpokládáme týdenní data)
avg_weekly_return = trade_results.groupby('Date')['PnL_Pct'].mean()
sharpe_ratio = avg_weekly_return.mean() / avg_weekly_return.std() * np.sqrt(52)  # Annualizace (52 týdnů)

# Maximum Drawdown
cumulative_returns = (1 + avg_weekly_return).cumprod()
running_max = cumulative_returns.cummax()
drawdown = (cumulative_returns / running_max) - 1
max_drawdown = drawdown.min()

# Výpis metrik
print(f"Počet obchodů: {total_trades}")
print(f"Win rate: {win_rate:.2%}")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
print(f"Celkový výnos: {trade_results['Cumulative_Return'].iloc[-1] if len(trade_results) > 0 else 0:.2%}")

# ---------------------- 12. Vizualizace výsledků ----------------------
print("Vytváření vizualizací...")

plt.figure(figsize=(18, 12))

# 1. Křivka learning loss
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

# 2. Skutečné vs. předpovídané výnosy (test set)
plt.subplot(2, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.3)
plt.plot([-0.5, 0.5], [-0.5, 0.5], 'r--')
plt.xlabel('Skutečné výnosy')
plt.ylabel('Předpovídané výnosy')
plt.title(f'Skutečné vs. Předpovídané Výnosy (Test Set)\nR² = {test_r2:.4f}')
plt.grid(True)
plt.axis('equal')

# 3. Kumulativní výnos strategie
plt.subplot(2, 2, 3)
trade_results.groupby('Date')['PnL_Pct'].mean().cumsum().plot()
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Kumulativní Výnos Strategie')
plt.xlabel('Datum')
plt.ylabel('Kumulativní výnos')
plt.grid(True)

# 4. Distribuce výnosů z obchodů
plt.subplot(2, 2, 4)
plt.hist(trade_results['PnL_Pct'], bins=50, alpha=0.75)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Distribuce Výnosů z Obchodů')
plt.xlabel('Výnos (%)')
plt.ylabel('Počet obchodů')
plt.grid(True)

# Přidání názvu a metriky
plt.suptitle('MLP Model pro Predikci Týdenních Výnosů S&P100 Akcií', fontsize=16)

# Přidání metrik a nejlepších hyperparametrů do hlavního PNG
metrics_text = (
    f"Testovací R²: {test_r2:.4f} | "
    f"Win Rate: {win_rate:.2%} | "
    f"Profit Factor: {profit_factor:.2f} | "
    f"Sharpe: {sharpe_ratio:.2f} | "
    f"Max Drawdown: {max_drawdown:.2%}"
)
best_params_text = " | ".join([f"{k}: {v}" for k, v in best_params.items()])
plt.figtext(0.5, 0.01, metrics_text + "\nNejlepší hyperparametry: " + best_params_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Uložení grafu
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('sp100_weekly_mlp_results.png', dpi=300)
plt.close()

# Výpis celkového času běhu
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Celkový čas běhu: {elapsed_time:.2f} sekund")

# Uložení výsledků do CSV
if 'trade_results' in locals() and len(trade_results) > 0:
    trade_results.to_csv('trade_results.csv', index=False)
    print("Výsledky obchodů uloženy do 'trade_results.csv'")

# Uložení modelu
final_model.save('mlp_model.h5')
print("Model uložen jako 'mlp_model.h5'")

print("Hotovo! Výsledky uloženy jako 'sp100_weekly_mlp_results.png'")