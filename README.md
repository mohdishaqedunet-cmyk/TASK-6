# TASK-6

This project demonstrates a complete machine learning workflow using the California Housing dataset to predict house prices with Linear Regression. The objective is to understand data preprocessing, model training, evaluation, and interpretation of results in a structured and reproducible way.

# california_housing_data_loading.py
from sklearn.datasets import fetch_california_housing

import pandas as pd

#Load dataset

housing = fetch_california_housing()

#Create DataFrame for features

df = pd.DataFrame(housing.data, columns=housing.feature_names)

#Add target variable (house prices)

df['MedHouseValue'] = housing.target

df.head()

# california_housing_data_inspection.py
#View first few rows

df.head()

#Dataset information

df.info()

#Statistical summary

df.describe()

# feature_target_separation.py
#Features

X = df.drop('MedHouseValue', axis=1)

#Target

y = df['MedHouseValue']

#Confirm y is numeric and continuous

print(y.dtype)

# train_test_split_data.py
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, 
    
    test_size=0.2, 
    
    random_state=42
    
)

# train_linear_regression.py
from sklearn.linear_model import LinearRegression

#Create model

model = LinearRegression()

#Fit model

model.fit(X_train, y_train)

# model_predictions_comparison.py
#Make predictions

y_pred = model.predict(X_test)

#Comparison table

comparison = pd.DataFrame({

    'Actual': y_test.values[:10],
    
    'Predicted': y_pred[:10]
    
})

comparison

# model_evaluation_metrics.py
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np

#MAE

mae = mean_absolute_error(y_test, y_pred)

#RMSE

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)

print("RMSE:", rmse)

# predicted_vs_actual_plot.py
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.plot([y_test.min(), y_test.max()], 

         [y_test.min(), y_test.max()], 
         
         color='red', linestyle='--')
         
plt.xlabel('Actual House Prices')

plt.ylabel('Predicted House Prices')

plt.title('Actual vs Predicted House Prices')

plt.show()

# feature_importance_analysis.py
coefficients = pd.DataFrame({

    'Feature': X.columns,
    
    'Coefficient': model.coef_
    
}).sort_values(by='Coefficient', ascending=False)

coefficients


