import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import datetime  

folder_path = r"C:\Users\krish\OneDrive\Desktop\sales forecasting"  
files_to_load = ["train.csv"]  
dataframes = []

for file_name in files_to_load:
    file_path = os.path.join(folder_path, file_name)
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower()  
        print(f"Loaded: {file_name} with shape {df.shape}")
        dataframes.append(df)
    except Exception as e:
        print(f"Failed to load {file_name}: {e}")

if not dataframes:
    raise ValueError("No valid data found. Check your CSV files.")
data = pd.concat(dataframes, ignore_index=True)
print(f"Combined DataFrame shape: {data.shape}")

required_columns = ['date', 'sales']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Missing required columns: {required_columns}. Check your dataset.")

data['sales'] = pd.to_numeric(data['sales'], errors='coerce')
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data.dropna(subset=['sales', 'date'], inplace=True)
data.fillna(method="ffill", inplace=True)
data.drop_duplicates(inplace=True)

data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['weekday'] = data['date'].dt.weekday

data['sales_lag_1'] = data['sales'].shift(1)
data['sales_lag_7'] = data['sales'].shift(7)
data['sales_lag_30'] = data['sales'].shift(30)

data['sales_roll_mean_7'] = data['sales'].rolling(window=7).mean()
data['sales_roll_sum_7'] = data['sales'].rolling(window=7).sum()

data.dropna(inplace=True)

categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

target_column = "sales"
feature_columns = [col for col in data.columns if col != target_column and col != 'date']
X = data[feature_columns]
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
accuracy_percentage = r2 * 100

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Accuracy (R² as %): {accuracy_percentage:.2f}%")

plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test.values, label="Actual", color="blue")
plt.plot(range(len(y_pred)), y_pred, label="Predicted", color="red")

forecast_date = "2025-01-01"  
plt.title(f"Actual vs Predicted Sales (Accuracy: {accuracy_percentage:.2f}%) - Forecast Date: {forecast_date}")
plt.xlabel("Samples")
plt.ylabel("Sales")
plt.legend()

output_path = os.path.join(folder_path, f"sales_forecasting_results_{forecast_date}.png")
plt.savefig(output_path)
plt.show()

print(f"Graph saved as: {output_path}")

future_features = pd.DataFrame({
    'year': [2025],
    'month': [1],
    'day': [1],
    'weekday': [2], 
    'sales_lag_1': [data['sales'].iloc[-1]],
    'sales_lag_7': [data['sales'].iloc[-7]],
    'sales_lag_30': [data['sales'].iloc[-30]],
    'sales_roll_mean_7': [data['sales'].iloc[-7:].mean()],
    'sales_roll_sum_7': [data['sales'].iloc[-7:].sum()]
})

for col in X_train.columns:
    if col not in future_features:
        future_features[col] = 0
future_features = future_features[X_train.columns]
future_features_scaled = scaler.transform(future_features)
future_sales_pred = model.predict(future_features_scaled)
print(f"Forecasted Sales for {forecast_date}: {future_sales_pred[0]:.2f}")
