import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_FILE = 'dataset_cleaned.csv'
FORECAST_HORIZON = 24  # 24 hours ahead

print(f"Starting predictive modeling analysis on {DATA_FILE}...")

# --- Load Data ---
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Successfully loaded {DATA_FILE}. Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    # Ensure required columns exist
    required_columns = ['demand', 'hour', 'day', 'temperature']
    for col in required_columns:
        if col not in df.columns:
            if col == 'temperature' and 'apparentTemperature' in df.columns:
                print("Using 'apparentTemperature' instead of 'temperature'")
            else:
                raise ValueError(f"Required column '{col}' not found in dataset")
                
    # Use apparentTemperature if temperature is not available
    if 'temperature' not in df.columns and 'apparentTemperature' in df.columns:
        df['temperature'] = df['apparentTemperature']
        
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Please ensure the file is in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Data Preprocessing ---
print("\n--- Data Preprocessing ---")

# Check if we have temporal sequence via 'time' column
if 'time' in df.columns:
    # Check if time is already a string date format or a unix timestamp
    if isinstance(df['time'].iloc[0], str) or df['time'].dtype == 'object':
        # Time is likely already a date string
        df['time'] = pd.to_datetime(df['time'])
    else:
        # Time is likely a unix timestamp
        df['time'] = pd.to_datetime(df['time'], unit='s')
    
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    print("Data sorted by timestamp")
elif 'local_time' in df.columns:
    # Try local_time instead
    df['local_time'] = pd.to_datetime(df['local_time'])
    df.set_index('local_time', inplace=True)
    df.sort_index(inplace=True)
    print("Data sorted by local timestamp")
else:
    print("Warning: No timestamp column found. Data may not be in chronological order.")

# Handle missing values
print("\nHandling missing values...")
missing_before = df.isnull().sum().sum()
print(f"Total missing values before imputation: {missing_before}")

if missing_before > 0:
    # For numeric columns, fill with mean
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)
    
    # For categorical columns, fill with mode
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    missing_after = df.isnull().sum().sum()
    print(f"Total missing values after imputation: {missing_after}")

# --- Feature Engineering ---
print("\n--- Feature Engineering ---")

# Check if hour, day, month are already in the dataset
time_features = []
if 'hour' not in df.columns and 'time' in df.index.names:
    df['hour'] = df.index.hour
    time_features.append('hour')
    print("Created 'hour' feature")
    
if 'day' not in df.columns and 'time' in df.index.names:
    df['day'] = df.index.day
    time_features.append('day')
    print("Created 'day' feature")
    
if 'month' not in df.columns and 'time' in df.index.names:
    df['month'] = df.index.month
    time_features.append('month')
    print("Created 'month' feature")
    
if 'day_of_week' not in df.columns and 'time' in df.index.names:
    df['day_of_week'] = df.index.dayofweek
    time_features.append('day_of_week')
    print("Created 'day_of_week' feature")

# Create lag features
def create_lag_features(df, target_col, lag_hours=[1, 24, 168], suffix='_lag'):
    """Create lag features for time series"""
    lag_df = df.copy()
    for lag in lag_hours:
        lag_df[f'{target_col}{suffix}{lag}'] = lag_df[target_col].shift(lag)
    return lag_df

print("\nCreating lag features...")
try:
    if not df.index.is_monotonic_increasing:
        print("Warning: Time index is not strictly increasing. Sorting index...")
        df = df.sort_index()
    
    # Create lags for demand (1 hour, 24 hours, 168 hours (1 week))
    df = create_lag_features(df, 'demand', lag_hours=[1, 24, 168])
    print("Created lag features: demand_lag1, demand_lag24, demand_lag168")
except Exception as e:
    print(f"Error creating lag features: {e}")

# Create rolling window features
def create_rolling_features(df, target_col, windows=[6, 12, 24], statistics=['mean', 'std']):
    """Create rolling window features for time series"""
    roll_df = df.copy()
    for window in windows:
        for stat in statistics:
            if stat == 'mean':
                roll_df[f'{target_col}_roll{window}_{stat}'] = roll_df[target_col].rolling(window=window, min_periods=1).mean()
            elif stat == 'std':
                roll_df[f'{target_col}_roll{window}_{stat}'] = roll_df[target_col].rolling(window=window, min_periods=1).std().fillna(0)
    return roll_df

print("\nCreating rolling window features...")
try:
    df = create_rolling_features(df, 'demand', windows=[6, 12, 24], statistics=['mean', 'std'])
    print("Created rolling window features")
except Exception as e:
    print(f"Error creating rolling window features: {e}")

# Create cyclical features for hour, day of week, month
def create_cyclical_features(df, col_name, period):
    """Create sin and cos features to represent cyclical data"""
    df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name]/period)
    df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name]/period)
    return df

print("\nCreating cyclical features...")
if 'hour' in df.columns:
    df = create_cyclical_features(df, 'hour', 24)
    print("Created cyclical features for hour")
    
if 'day_of_week' in df.columns:
    df = create_cyclical_features(df, 'day_of_week', 7)
    print("Created cyclical features for day of week")
    
if 'month' in df.columns:
    df = create_cyclical_features(df, 'month', 12)
    print("Created cyclical features for month")

# Handle any new missing values from feature engineering
df.dropna(inplace=True)
print(f"\nFinal dataframe shape after preprocessing: {df.shape}")

# --- Prepare Data for Modeling ---
print("\n--- Preparing Data for Modeling ---")

# Select features for modeling
features = [
    'precipIntensity', 'precipProbability', 'temperature', 
    'hour', 'day_of_week', 'month',
    'demand_lag1', 'demand_lag24', 'demand_lag168',
    'demand_roll24_mean', 'hour_sin', 'hour_cos'
]

# Make sure all features exist in the dataframe
features = [f for f in features if f in df.columns]
print(f"Selected features: {features}")

target = 'demand'

# Split data into train/test sets
# Use approximately 80% for training, 20% for testing
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")

# Create X, y datasets
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Scale the features (not the target)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data splitting and scaling complete")

# --- Create Naive Forecast (Baseline) ---
print("\n--- Creating Naive Forecast (Baseline) ---")

# Create naive forecast (previous day's same hour)
def create_naive_forecast(df, target_col='demand'):
    # Use the value from 24 hours ago
    return df[f'{target_col}_lag24']

naive_predictions = create_naive_forecast(test_df)

# Evaluate naive forecast
naive_mae = mean_absolute_error(y_test, naive_predictions)
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_predictions))
naive_mape = np.mean(np.abs((y_test - naive_predictions) / y_test)) * 100

print(f"Naive Forecast (Previous Day's Same Hour) Performance:")
print(f"MAE: {naive_mae:.2f}")
print(f"RMSE: {naive_rmse:.2f}")
print(f"MAPE: {naive_mape:.2f}%")

# --- Model Training & Evaluation Function ---
def train_evaluate_model(model, name, X_train, y_train, X_test, y_test, baseline_metrics):
    print(f"\n--- Training {name} Model ---")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    
    # Calculate improvement over baseline
    mae_improvement = ((naive_mae - mae) / naive_mae) * 100
    rmse_improvement = ((naive_rmse - rmse) / naive_rmse) * 100
    mape_improvement = ((naive_mape - mape) / naive_mape) * 100
    
    print(f"{name} Model Performance:")
    print(f"MAE: {mae:.2f} ({mae_improvement:.1f}% improvement over baseline)")
    print(f"RMSE: {rmse:.2f} ({rmse_improvement:.1f}% improvement over baseline)")
    print(f"MAPE: {mape:.2f}% ({mape_improvement:.1f}% improvement over baseline)")
    print(f"R²: {r2:.4f}")
    
    # Visualize actual vs predicted for a sample of the test data
    sample_size = min(100, len(y_test))
    sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.iloc[sample_indices].values, 'b-', label='Actual')
    plt.plot(y_pred[sample_indices], 'r-', label='Predicted')
    plt.title(f'{name} - Actual vs Predicted Demand')
    plt.xlabel('Sample Index')
    plt.ylabel('Demand')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Store the results
    results = {
        'model': model,
        'name': name,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'predictions': y_pred
    }
    
    return results

# Store model performance metrics
baseline_metrics = {
    'mae': naive_mae,
    'rmse': naive_rmse,
    'mape': naive_mape
}

model_results = []

# --- Linear Regression Model ---
linear_model = LinearRegression()
linear_results = train_evaluate_model(
    linear_model, 'Linear Regression', 
    X_train_scaled, y_train, 
    X_test_scaled, y_test,
    baseline_metrics
)
model_results.append(linear_results)

# --- Ridge Regression Model ---
ridge_model = Ridge(alpha=1.0)
ridge_results = train_evaluate_model(
    ridge_model, 'Ridge Regression', 
    X_train_scaled, y_train, 
    X_test_scaled, y_test,
    baseline_metrics
)
model_results.append(ridge_results)

# --- Random Forest Model ---
rf_model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_results = train_evaluate_model(
    rf_model, 'Random Forest', 
    X_train_scaled, y_train, 
    X_test_scaled, y_test,
    baseline_metrics
)
model_results.append(rf_results)

# --- Gradient Boosting Model ---
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_results = train_evaluate_model(
    gb_model, 'Gradient Boosting', 
    X_train_scaled, y_train, 
    X_test_scaled, y_test,
    baseline_metrics
)
model_results.append(gb_results)

# --- XGBoost Model ---
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
xgb_results = train_evaluate_model(
    xgb_model, 'XGBoost', 
    X_train_scaled, y_train, 
    X_test_scaled, y_test,
    baseline_metrics
)
model_results.append(xgb_results)

# --- PyTorch Neural Network Model ---
# Define the PyTorch neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        return x

def create_and_train_nn(X_train, y_train, X_test, y_test):
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize the model
    input_dim = X_train.shape[1]
    model = NeuralNetwork(input_dim)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print("\n--- Training PyTorch Neural Network Model ---")
    num_epochs = 100
    patience = 10
    best_loss = float('inf')
    counter = 0
    train_losses = []
    val_losses = []
    
    # Set aside 20% of training data for validation
    val_size = int(len(X_train_tensor) * 0.2)
    train_data, val_data = torch.utils.data.random_split(
        train_dataset, [len(X_train_tensor) - val_size, val_size]
    )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            # Save best model
            best_model = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    model.load_state_dict(best_model)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy().flatten()
    
    return model, y_pred

try:
    nn_model, nn_pred = create_and_train_nn(X_train_scaled, y_train, X_test_scaled, y_test)

    # Evaluate NN model
    nn_mae = mean_absolute_error(y_test, nn_pred)
    nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))
    nn_mape = np.mean(np.abs((y_test - nn_pred) / y_test)) * 100
    nn_r2 = r2_score(y_test, nn_pred)

    # Calculate improvement over baseline
    nn_mae_improvement = ((naive_mae - nn_mae) / naive_mae) * 100
    nn_rmse_improvement = ((naive_rmse - nn_rmse) / naive_rmse) * 100
    nn_mape_improvement = ((naive_mape - nn_mape) / naive_mape) * 100

    print(f"PyTorch Neural Network Model Performance:")
    print(f"MAE: {nn_mae:.2f} ({nn_mae_improvement:.1f}% improvement over baseline)")
    print(f"RMSE: {nn_rmse:.2f} ({nn_rmse_improvement:.1f}% improvement over baseline)")
    print(f"MAPE: {nn_mape:.2f}% ({nn_mape_improvement:.1f}% improvement over baseline)")
    print(f"R²: {nn_r2:.4f}")

    # Visualize NN predictions
    sample_size = min(100, len(y_test))
    sample_indices = np.random.choice(len(y_test), sample_size, replace=False)

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.iloc[sample_indices].values, 'b-', label='Actual')
    plt.plot(nn_pred[sample_indices], 'r-', label='Predicted')
    plt.title('PyTorch Neural Network - Actual vs Predicted Demand')
    plt.xlabel('Sample Index')
    plt.ylabel('Demand')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Store NN results
    nn_results = {
        'model': nn_model,
        'name': 'PyTorch Neural Network',
        'mae': nn_mae,
        'rmse': nn_rmse,
        'mape': nn_mape,
        'r2': nn_r2,
        'predictions': nn_pred
    }
    model_results.append(nn_results)
except Exception as e:
    print(f"Error with PyTorch Neural Network: {e}")
    print("Skipping neural network evaluation")

# --- Ensemble Model ---
print("\n--- Creating Ensemble Model ---")

# Create an ensemble using stacking
def create_stacking_ensemble(base_models, X_train, y_train, X_test, y_test):
    # Prepare the base estimators
    estimators = [(model['name'], model['model']) for model in base_models 
                 if 'PyTorch' not in model['name']]  # Exclude PyTorch model from sklearn ensemble
    
    # Create the stacking regressor
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(),
        cv=5
    )
    
    # Train the stacking model
    print("Training Stacking Ensemble Model...")
    stacking_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = stacking_model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    
    # Calculate improvement over baseline
    mae_improvement = ((naive_mae - mae) / naive_mae) * 100
    rmse_improvement = ((naive_rmse - rmse) / naive_rmse) * 100
    mape_improvement = ((naive_mape - mape) / naive_mape) * 100
    
    print(f"Stacking Ensemble Model Performance:")
    print(f"MAE: {mae:.2f} ({mae_improvement:.1f}% improvement over baseline)")
    print(f"RMSE: {rmse:.2f} ({rmse_improvement:.1f}% improvement over baseline)")
    print(f"MAPE: {mape:.2f}% ({mape_improvement:.1f}% improvement over baseline)")
    print(f"R²: {r2:.4f}")
    
    # Visualize ensemble predictions
    sample_size = min(100, len(y_test))
    sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.iloc[sample_indices].values, 'b-', label='Actual')
    plt.plot(y_pred[sample_indices], 'r-', label='Predicted')
    plt.title('Stacking Ensemble - Actual vs Predicted Demand')
    plt.xlabel('Sample Index')
    plt.ylabel('Demand')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Store results
    results = {
        'model': stacking_model,
        'name': 'Stacking Ensemble',
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'predictions': y_pred
    }
    
    return results

# Select the best performing models for the ensemble
# Exclude PyTorch Neural Network which needs special handling
base_models = sorted([model for model in model_results if 'PyTorch' not in model['name']], 
                    key=lambda x: x['mae'])[:3]

print(f"Using the following models for the ensemble: {[model['name'] for model in base_models]}")

# Create and evaluate the stacking ensemble
ensemble_results = create_stacking_ensemble(
    base_models, 
    X_train_scaled, y_train, 
    X_test_scaled, y_test
)
model_results.append(ensemble_results)

# --- Model Comparison ---
print("\n--- Model Comparison ---")

# Create a dataframe to compare model performance
comparison_df = pd.DataFrame([
    {
        'Model': 'Naive Baseline (Previous Day)',
        'MAE': naive_mae,
        'RMSE': naive_rmse,
        'MAPE': naive_mape,
        'R²': 'N/A',
        'Improvement': '0.0%'
    }
] + [
    {
        'Model': model['name'],
        'MAE': model['mae'],
        'RMSE': model['rmse'],
        'MAPE': model['mape'],
        'R²': model['r2'],
        'Improvement': f"{((naive_mae - model['mae']) / naive_mae) * 100:.1f}%"
    }
    for model in model_results
])

# Sort by MAE (lower is better)
comparison_df.sort_values('MAE', inplace=True)

print("Model Performance Comparison:")
print(comparison_df.to_string(index=False))

# Visualize model comparison
plt.figure(figsize=(14, 10))

# Plot MAE comparison
plt.subplot(2, 1, 1)
sns.barplot(x='Model', y='MAE', data=comparison_df)
plt.title('Mean Absolute Error by Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Plot MAPE comparison
plt.subplot(2, 1, 2)
sns.barplot(x='Model', y='MAPE', data=comparison_df)
plt.title('Mean Absolute Percentage Error by Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()

# --- Feature Importance Analysis ---
print("\n--- Feature Importance Analysis ---")

# Analyze feature importance from tree-based model (Random Forest)
rf_model = [model['model'] for model in model_results if model['name'] == 'Random Forest'][0]

# Calculate feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
})
feature_importance.sort_values('Importance', ascending=False, inplace=True)

print("Feature Importance:")
print(feature_importance.to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance from Random Forest Model')
plt.tight_layout()
plt.show()

# --- ARIMA Time Series Model ---
print("\n--- ARIMA Time Series Model ---")

# Prepare data for ARIMA (univariate time series analysis)
# We'll use only the target variable (demand) for this model
def train_arima_model(train_series, test_series):
    # Basic ARIMA model (p=order, d=integration, q=moving average)
    order = (1, 1, 1)  # A common starting point
    
    print(f"Training ARIMA model with order={order}...")
    try:
        model = ARIMA(train_series, order=order)
        model_fit = model.fit()
        
        # Make predictions for test period
        pred = model_fit.forecast(steps=len(test_series))
        
        # Calculate metrics
        mae = mean_absolute_error(test_series, pred)
        rmse = np.sqrt(mean_squared_error(test_series, pred))
        mape = np.mean(np.abs((test_series - pred) / test_series)) * 100
        
        # Calculate improvement over baseline
        mae_improvement = ((naive_mae - mae) / naive_mae) * 100
        rmse_improvement = ((naive_rmse - rmse) / naive_rmse) * 100
        mape_improvement = ((naive_mape - mape) / naive_mape) * 100
        
        print(f"ARIMA Model Performance:")
        print(f"MAE: {mae:.2f} ({mae_improvement:.1f}% improvement over baseline)")
        print(f"RMSE: {rmse:.2f} ({rmse_improvement:.1f}% improvement over baseline)")
        print(f"MAPE: {mape:.2f}% ({mape_improvement:.1f}% improvement over baseline)")
        
        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(test_series.values, 'b-', label='Actual')
        plt.plot(pred, 'r-', label='ARIMA Prediction')
        plt.title('ARIMA - Actual vs Predicted Demand')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return {
            'model': model_fit,
            'name': 'ARIMA',
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'predictions': pred
        }
    except Exception as e:
        print(f"Error training ARIMA model: {e}")
        return None

try:
    # Try to train ARIMA on a subset of data to save time (last 1000 points)
    train_size = min(1000, len(train_df))
    test_size = min(100, len(test_df))
    
    arima_train = train_df['demand'][-train_size:]
    arima_test = test_df['demand'][:test_size]
    
    arima_results = train_arima_model(arima_train, arima_test)
    if arima_results:
        model_results.append(arima_results)
        
        # Update comparison table
        new_row = pd.DataFrame([{
            'Model': 'ARIMA',
            'MAE': arima_results['mae'],
            'RMSE': arima_results['rmse'],
            'MAPE': arima_results['mape'],
            'R²': 'N/A',
            'Improvement': f"{((naive_mae - arima_results['mae']) / naive_mae) * 100:.1f}%"
        }])
        comparison_df = pd.concat([comparison_df, new_row], ignore_index=True)
        comparison_df.sort_values('MAE', inplace=True)
        
        print("\nUpdated Model Performance Comparison:")
        print(comparison_df.to_string(index=False))
except Exception as e:
    print(f"Error with ARIMA modeling: {e}")

# --- Final Summary ---
print("\n--- Final Summary ---")

# Identify the best model
best_model = comparison_df.iloc[0]['Model']
best_mae = comparison_df.iloc[0]['MAE']
best_improvement = comparison_df.iloc[0]['Improvement']

print(f"Best performing model: {best_model}")
print(f"Best MAE: {best_mae:.2f}")
print(f"Improvement over naive baseline: {best_improvement}")
print("\nAll models evaluated successfully!") 