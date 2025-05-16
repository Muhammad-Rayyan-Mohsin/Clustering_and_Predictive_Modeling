"""
Flask Application for Energy Demand Clustering and Forecasting

Purpose:
Provides a web interface to visualize energy demand clusters and forecast future demand
based on historical data. The application uses machine learning models for both
clustering (K-Means with PCA) and forecasting (various regressors and time series models).

Endpoints:
  GET /
    - Serves the main single-page application (index.html).

  POST /api/cluster
    - Description: Performs clustering on the energy demand data.
    - Request Body (JSON):
      {
        "city": "string",  // Currently not used for filtering dataset_cleaned.csv
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD",
        "k": "integer" // Number of clusters for K-Means
      }
    - Response (JSON):
      {
        "clusters": [
          { "x": "number", "y": "number", "label": "number" }, ...
        ],
        "pca_components": [
          { "PC1": "number", "PC2": "number", "label": "number" }, ...
        ],
        "silhouette_score": "number",
        "cluster_centers": [ { "feature1": "value", ... }, ... ],
        "error": "string" // Optional, if an error occurred
      }

  POST /api/forecast
    - Description: Forecasts energy demand using selected models.
    - Request Body (JSON):
      {
        "city": "string", // Currently not used for filtering dataset_cleaned.csv
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD",
        "lookback": "integer", // Lookback window for feature engineering
        "model": "string" // Model name: "Linear", "RandomForest", "XGBoost", "ARIMA"
      }
    - Response (JSON):
      {
        "dates": ["ISO date string", ...],
        "actual": ["number", ...],
        "predicted": ["number", ...],
        "metrics": { "mae": "number", "rmse": "number", "r2": "number", "mape": "number"},
        "normalized_metrics": {
            "mae": "number",
            "rmse": "number",
            "r2": "number",
            "mape": "number",
            "overall_score": "number"
        },
        "performance": {
            "score": "number",
            "grade": "string"
        },
        "model_used": "string",
        "features_used": ["string", ...]
      }
"""
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os

# --- Configuration ---
DATA_FILE = 'processed_data.csv'
# For clustering, using a subset of features as in 1.py for potentially better silhouette.
# Modify if other features are desired.
DEFAULT_CLUSTER_FEATURES = ['precipIntensity', 'precipProbability', 'temperature', 'apparentTemperature']
# For forecasting, features will be selected based on 2.py logic (including generated lags)
FORECAST_HORIZON = 24 # Default, can be overridden or adapted

app = Flask(__name__)
warnings.filterwarnings('ignore')

# --- Helper Functions ---
def load_and_preprocess_data(start_date_str=None, end_date_str=None, for_clustering=False, city_filter=None):
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Error: {DATA_FILE} not found.")
    
    df = pd.read_csv(DATA_FILE)

    # Apply city filter if provided and it's not the default "all cities" value
    if city_filter and city_filter != 'default_city' and 'city' in df.columns:
        df = df[df['city'].str.lower() == city_filter.lower()]
        if df.empty:
            raise ValueError(f"No data found for city: {city_filter}. Check if the city name is correct and exists in the dataset.")
    elif city_filter and city_filter != 'default_city' and 'city' not in df.columns:
        print(f"Warning: 'city' column not found in the dataset. Cannot filter by city: {city_filter}")

    # Handle 'temperature' vs 'apparentTemperature' for clustering
    if for_clustering:
        # Ensure apparentTemperature is used if 'temperature' is not preferred or available
        if 'apparentTemperature' in df.columns and 'temperature' in DEFAULT_CLUSTER_FEATURES:
             if 'temperature' in df.columns and 'apparentTemperature' not in DEFAULT_CLUSTER_FEATURES:
                pass # Keep temperature
             else: # prefer apparentTemperature or temperature is not available
                if 'temperature' in DEFAULT_CLUSTER_FEATURES:
                    DEFAULT_CLUSTER_FEATURES.remove('temperature')
                if 'apparentTemperature' not in DEFAULT_CLUSTER_FEATURES:
                    DEFAULT_CLUSTER_FEATURES.append('apparentTemperature')
        elif 'temperature' not in df.columns and 'apparentTemperature' in df.columns:
            if 'apparentTemperature' not in DEFAULT_CLUSTER_FEATURES:
                 DEFAULT_CLUSTER_FEATURES.append('apparentTemperature')
            if 'temperature' in DEFAULT_CLUSTER_FEATURES:
                DEFAULT_CLUSTER_FEATURES.remove('temperature')


    time_col_found = None
    if 'time' in df.columns:
        time_col_found = 'time'
    elif 'local_time' in df.columns: # As per 2.py
        time_col_found = 'local_time'

    if time_col_found:
        try:
            if isinstance(df[time_col_found].iloc[0], str) or df[time_col_found].dtype == 'object':
                df[time_col_found] = pd.to_datetime(df[time_col_found])
            else:
                df[time_col_found] = pd.to_datetime(df[time_col_found], unit='s')
            df.set_index(time_col_found, inplace=True)
            df.sort_index(inplace=True)
        except Exception as e:
            print(f"Warning: Could not parse or set time index: {e}")
            # Proceed without time index if parsing fails

    if start_date_str and end_date_str and df.index.is_monotonic_increasing and isinstance(df.index, pd.DatetimeIndex):
        try:
            start_date = pd.to_datetime(start_date_str)
            end_date = pd.to_datetime(end_date_str)
            df = df[(df.index >= start_date) & (df.index <= end_date)]
        except Exception as e:
            print(f"Warning: Could not filter by date: {e}")

    # Impute missing values (simple mean imputation)
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)
    
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            
    return df

def create_lag_features(df, target_col, lag_hours):
    lag_df = df.copy()
    for lag in lag_hours:
        lag_df[f'{target_col}_lag{lag}'] = lag_df[target_col].shift(lag)
    return lag_df

def create_rolling_features(df, target_col, windows, statistics):
    roll_df = df.copy()
    for window in windows:
        for stat in statistics:
            if stat == 'mean':
                roll_df[f'{target_col}_roll{window}_{stat}'] = roll_df[target_col].rolling(window=window, min_periods=1).mean()
            elif stat == 'std':
                roll_df[f'{target_col}_roll{window}_{stat}'] = roll_df[target_col].rolling(window=window, min_periods=1).std().fillna(0)
    return roll_df

def create_cyclical_features(df, col_name, period):
    df_copy = df.copy()
    df_copy[f'{col_name}_sin'] = np.sin(2 * np.pi * df_copy[col_name]/period)
    df_copy[f'{col_name}_cos'] = np.cos(2 * np.pi * df_copy[col_name]/period)
    return df_copy

# --- API Endpoints ---
@app.route('/')
def index():
    cities = ['default_city'] # Default option
    try:
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            if 'city' in df.columns:
                unique_cities = df['city'].dropna().unique().tolist()
                unique_cities.sort()
                cities.extend(unique_cities)
            else:
                print("Warning: 'city' column not found in CSV, dropdown will only have default.")
        else:
            print(f"Warning: {DATA_FILE} not found, city dropdown will only have default.")
    except Exception as e:
        print(f"Error reading cities from CSV for dropdown: {e}")
    # Ensure 'default_city' is the first option, and a more user-friendly label can be handled in template or JS if needed
    # For simplicity, template will list it as is. Or we can map it.
    return render_template('index.html', cities=cities)

@app.route('/api/cluster', methods=['POST'])
def api_cluster():
    try:
        data = request.get_json()
        city = data.get('city') # Get city from request
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        k = int(data.get('k', 3))

        df_full = load_and_preprocess_data(start_date, end_date, for_clustering=True, city_filter=city) # Pass city_filter
        
        # Select features for clustering
        # Prioritize apparentTemperature if available and temperature is also listed
        current_features = []
        if 'apparentTemperature' in df_full.columns and 'temperature' in DEFAULT_CLUSTER_FEATURES:
            current_features = [f for f in DEFAULT_CLUSTER_FEATURES if f != 'temperature']
            if 'apparentTemperature' not in current_features:
                 current_features.append('apparentTemperature')
        else:
            current_features = [f for f in DEFAULT_CLUSTER_FEATURES if f in df_full.columns]
            if not current_features: # Fallback if default features are not present
                 current_features = [col for col in ['precipIntensity', 'precipProbability', 'temperature'] if col in df_full.columns]


        if not current_features:
            return jsonify({"error": "No suitable features found for clustering in the dataset."}), 400
        
        df_features = df_full[current_features].copy()
        
        # Impute again just for the selected feature subset if any NaNs persist
        for col in current_features:
            if df_features[col].isnull().sum() > 0:
                df_features[col].fillna(df_features[col].mean(), inplace=True)
        
        if df_features.empty or df_features.isnull().values.any():
             return jsonify({"error": "Not enough data or features after preprocessing for clustering."}), 400


        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_features)

        # PCA for visualization (2 components)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # K-Means Clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        silhouette = -1 # Default if not enough samples
        if len(np.unique(cluster_labels)) > 1 and X_scaled.shape[0] > k : # Min samples for silhouette
             silhouette = silhouette_score(X_scaled, cluster_labels)


        # Cluster centers (inverse transform to original scale)
        centers_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_df = pd.DataFrame(centers_original_scale, columns=current_features)


        pca_results = [{"PC1": float(X_pca[i, 0]), "PC2": float(X_pca[i, 1]), "label": int(cluster_labels[i])} for i in range(len(X_pca))]
        
        # For scatter plot, use PCA components directly.
        # The request asks for { x: number, y: number, label: number }
        # We can use the PCA components for x and y.
        cluster_plot_data = [{"x": float(X_pca[i, 0]), "y": float(X_pca[i, 1]), "label": int(cluster_labels[i])} for i in range(len(X_pca))]

        return jsonify({
            "clusters": cluster_plot_data, # This is essentially same as pca_components with label
            "pca_components": pca_results, # For potential other uses or clarity
            "silhouette_score": float(silhouette),
            "cluster_centers": centers_df.to_dict(orient='records'),
            "features_used": current_features
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except ValueError as e:
        return jsonify({"error": f"Input error: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/api/forecast', methods=['POST'])
def api_forecast():
    try:
        data = request.get_json()
        city = data.get('city') # Get city from request
        start_date_str = data.get('start_date')
        end_date_str = data.get('end_date')
        lookback = int(data.get('lookback', 24)) # Default lookback for feature engineering
        model_name = data.get('model', 'Linear')

        df = load_and_preprocess_data(start_date_str, end_date_str, city_filter=city) # Pass city_filter

        if 'demand' not in df.columns:
            return jsonify({"error": "'demand' column not found in dataset."}), 400

        # Feature Engineering (simplified from 2.py)
        if 'hour' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
        if 'day_of_week' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
        if 'month' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df['month'] = df.index.month
        
        # Lag features
        df = create_lag_features(df, 'demand', lag_hours=[1, lookback, lookback*7]) # Example lags

        # Rolling features
        df = create_rolling_features(df, 'demand', windows=[6, lookback], statistics=['mean', 'std'])

        # Cyclical features
        if 'hour' in df.columns:
            df = create_cyclical_features(df, 'hour', 24)
        
        df.dropna(inplace=True)
        if df.empty:
            return jsonify({"error": "Not enough data after feature engineering and NaN removal."}), 400

        # Define features - this list might need adjustment based on available data after processing
        potential_features = [
            'precipIntensity', 'precipProbability', 'temperature', 'apparentTemperature',
            'humidity', 'windSpeed', # common weather features
            'hour', 'day_of_week', 'month',
            f'demand_lag1', f'demand_lag{lookback}', f'demand_lag{lookback*7}',
            f'demand_roll{lookback}_mean', 'hour_sin', 'hour_cos'
        ]
        features = [f for f in potential_features if f in df.columns]
        
        if not features:
            return jsonify({"error": "No features available for modeling after preprocessing."}), 400

        target = 'demand'
        
        # Split data - for forecasting, it's better to split chronologically
        # Using last ~20% or a fixed number of steps for testing.
        # For simplicity, using a split similar to 2.py if data is sufficient.
        split_ratio = 0.8
        if len(df) < 50 : # Minimal data for split
             return jsonify({"error": "Dataset too small for training and testing."}), 400

        split_idx = int(len(df) * split_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        if train_df.empty or test_df.empty:
            return jsonify({"error": "Train or test dataset is empty after split."}), 400

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = None
        if model_name == 'Linear':
            model = LinearRegression()
        elif model_name == 'RandomForest':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
        elif model_name == 'XGBoost':
            model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=5, learning_rate=0.1)
        elif model_name == 'ARIMA':
            # ARIMA needs univariate series and is handled differently
            try:
                # Use 'demand' from train_df for training ARIMA
                arima_train_series = train_df[target]
                
                # A common ARIMA order, can be tuned
                # Using a simple (1,1,1) order as a robust default from 2.py
                # More complex order selection (e.g., auto_arima) is possible but adds complexity
                arima_model = ARIMA(arima_train_series, order=(1, 1, 1))
                arima_model_fit = arima_model.fit()
                
                # Forecast for the length of the test set
                y_pred = arima_model_fit.forecast(steps=len(test_df))
                # Ensure y_pred is a numpy array
                y_pred = np.array(y_pred)

            except Exception as e:
                return jsonify({"error": f"ARIMA model error: {str(e)}"}), 500
        else:
            return jsonify({"error": "Invalid model name specified."}), 400

        if model_name != 'ARIMA':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        # MAPE can have issues with zero actual values, handle with care
        y_test_safe = y_test.replace(0, np.nan).dropna() # Avoid division by zero
        y_pred_safe = pd.Series(y_pred, index=y_test_safe.index).reindex(y_test_safe.index).dropna()
        mape = np.mean(np.abs((y_test_safe - y_pred_safe) / y_test_safe)) * 100 if not y_test_safe.empty else -1

        # Normalize metrics to 0-1 scale (1 is best performance)
        # For error metrics (MAE, RMSE, MAPE), we normalize so that 0 is best (low error)
        # Create reference values for normalization
        mean_y = y_test.mean()
        std_y = y_test.std()

        # Use a heuristic to normalize MAE - expect lower MAE to be around 10% of mean
        mae_norm = 1.0 - min(1.0, mae / (mean_y * 0.3))
        
        # Use similar scaling for RMSE - expect lower RMSE to be around 15% of mean
        rmse_norm = 1.0 - min(1.0, rmse / (mean_y * 0.4))
        
        # R2 is already between 0 and 1 (or negative), normalize to 0-1
        r2_norm = (r2 + 1) / 2 if r2 < 0 else r2
        
        # MAPE is in percentage, normalize to 0-1 (0% error would be ideal)
        mape_norm = 1.0 - min(1.0, mape / 100) if mape != -1 else 0.5

        # Calculate overall performance score (weighted average of normalized metrics)
        performance_score = (mae_norm * 0.3 + rmse_norm * 0.3 + r2_norm * 0.3 + mape_norm * 0.1)
        
        # Grade performance based on the score
        if performance_score >= 0.9:
            performance_grade = "Excellent"
        elif performance_score >= 0.75:
            performance_grade = "Good"
        elif performance_score >= 0.6:
            performance_grade = "Satisfactory"
        elif performance_score >= 0.4:
            performance_grade = "Fair"
        else:
            performance_grade = "Poor"

        # Dates for the forecast plot
        # Ensure test_df.index is DatetimeIndex and convert to ISO format
        if isinstance(test_df.index, pd.DatetimeIndex):
            dates = test_df.index.strftime('%Y-%m-%dT%H:%M:%SZ').tolist()
        else: # Fallback if index is not datetime
            dates = list(range(len(y_test)))

        return jsonify({
            "dates": dates,
            "actual": y_test.tolist(),
            "predicted": y_pred.tolist(),
            "metrics": {
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
                "mape": float(mape) if mape != -1 else "N/A"
            },
            "normalized_metrics": {
                "mae": float(mae_norm),
                "rmse": float(rmse_norm),
                "r2": float(r2_norm),
                "mape": float(mape_norm) if mape != -1 else 0.5,
                "overall_score": float(performance_score)
            },
            "performance": {
                "score": float(performance_score),
                "grade": performance_grade
            },
            "model_used": model_name,
            "features_used": features if model_name != 'ARIMA' else ["demand (univariate)"]
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except ValueError as e:
        return jsonify({"error": f"Input error: {str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"Forecast API error: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True) 