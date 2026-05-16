# Ex.No: 6               HOLT WINTERS METHOD
### Date: 16-05-2026



### AIM:
To run a program on HOLT WINTERS METHOD
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

data = pd.read_csv('/content/DailyDelhiClimateTest.csv', parse_dates=['date'],index_col='date')
data.head()

data.index = pd.to_datetime(data.index, format='%d-%m-%Y')
data_monthly = data.resample('MS').sum()
data_monthly.head()

scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data['humidity'].values.reshape(-1, 1)).flatten())
scaled_data.plot() 
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data['humidity'], model="additive")
decomposition.plot()
plt.show()

scaled_data=scaled_data+1
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=7).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))
ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')
np.sqrt(mean_squared_error(test_data, test_predictions_add))
np.sqrt(scaled_data.var()),scaled_data.mean()

final_model = ExponentialSmoothing(data['humidity'], trend='add', seasonal='mul', seasonal_periods=7).fit()
final_predictions = final_model.forecast(steps=30)
ax=data['humidity'].plot()
final_predictions.plot(ax=ax)
ax.legend(["Actual Humidity", "Forecasted Humidity"])
ax.set_xlabel('Date')
ax.set_ylabel('Humidity')
ax.set_title('Daily Humidity Prediction')
```

### OUTPUT:


TEST_PREDICTION
<img width="547" height="435" alt="image" src="https://github.com/user-attachments/assets/88ed3ac1-2e39-4f72-aebc-0debc9496a89" />




FINAL_PREDICTION
<img width="562" height="471" alt="image" src="https://github.com/user-attachments/assets/34211cd9-4678-483c-a8c8-e1c7306283a5" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
