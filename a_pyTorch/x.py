import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Create DataFrame with 2024 dates
data = {
    'Date': ['16-Jul-2024', '21-Jul-2024', '24-Jul-2024', '31-Jul-2024', '3-Aug-2024', 
             '10-Aug-2024', '18-Aug-2024', '21-Aug-2024', '29-Aug-2024', '4-Sep-2024', '12-Sep-2024'],
    'Spend': [100, 120, 85, 125, 110, 95, 67, 121, 102, 98, 84]
}
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
df.set_index('Date', inplace=True)

# Resample to daily frequency and interpolate missing values
df_daily = df.resample('D').interpolate()

# Plot original data
plt.figure(figsize=(12, 6))
plt.plot(df_daily.index, df_daily['Spend'], label='Original Data')
plt.title('Transaction Spend Time Series')
plt.xlabel('Date')
plt.ylabel('Spend')
plt.legend()
plt.show()

# Perform Augmented Dickey-Fuller test
result = adfuller(df_daily['Spend'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Plot ACF and PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plot_acf(df_daily['Spend'].dropna(), ax=ax1)
plot_pacf(df_daily['Spend'].dropna(), ax=ax2)
plt.show()

# Fit ARIMA model
model = ARIMA(df_daily['Spend'], order=(1,1,1))  # You might need to adjust these parameters
results = model.fit()

# Forecast
forecast_steps = 30
forecast = results.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df_daily.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot original data and forecast
plt.figure(figsize=(12, 6))
plt.plot(df_daily.index, df_daily['Spend'], label='Original Data')
plt.plot(forecast_index, forecast_mean, color='red', label='Forecast')
plt.fill_between(forecast_index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], 
                 color='pink', alpha=0.3, label='95% Confidence Interval')
plt.title('Transaction Spend Forecast (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Spend')
plt.legend()
plt.show()

# Print forecast
print("\nForecast for the next 30 days:")
print(forecast_mean)

# Print model summary
print("\nModel Summary:")
print(results.summary())

# Calculate and print additional metrics
print("\nAdditional Metrics:")
print(f"Mean Spend: ${df_daily['Spend'].mean():.2f}")
print(f"Median Spend: ${df_daily['Spend'].median():.2f}")
print(f"Standard Deviation: ${df_daily['Spend'].std():.2f}")
print(f"Minimum Spend: ${df_daily['Spend'].min():.2f}")
print(f"Maximum Spend: ${df_daily['Spend'].max():.2f}")