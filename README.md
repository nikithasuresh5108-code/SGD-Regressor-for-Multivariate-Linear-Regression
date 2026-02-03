# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:nagalkshmi.s 
RegisterNumber:  25003017
*/
```
```
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


data = {
    'Size': [750, 800, 850, 900, 1200, 1500, 1700, 2000, 2200, 2500],
    'Rooms': [2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
    'Location': [1, 2, 1, 3, 2, 3, 1, 2, 3, 2],
    'Price': [50, 55, 60, 70, 100, 130, 150, 200, 220, 250],
    'Occupants': [3, 3, 4, 4, 5, 6, 6, 7, 8, 9]
}

df = pd.DataFrame(data)


X = df[['Size', 'Rooms', 'Location']]
y = df[['Price', 'Occupants']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


price_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
occupants_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)


price_model.fit(X_train_scaled, y_train['Price'])
occupants_model.fit(X_train_scaled, y_train['Occupants'])


y_pred_price = price_model.predict(X_test_scaled)
y_pred_occupants = occupants_model.predict(X_test_scaled)


y_pred = np.column_stack((y_pred_price, y_pred_occupants))


print("Price Prediction - MSE:", mean_squared_error(y_test['Price'], y_pred_price))
print("Price Prediction - R2 Score:", r2_score(y_test['Price'], y_pred_price))

print("Occupants Prediction - MSE:", mean_squared_error(y_test['Occupants'], y_pred_occupants))
print("Occupants Prediction - R2 Score:", r2_score(y_test['Occupants'], y_pred_occupants))

print("\nActual values:\n", y_test.values)
print("\nPredicted values:\n", y_pred)

```

## Output:
<img width="974" height="538" alt="Screenshot 2026-02-03 085626" src="https://github.com/user-attachments/assets/b3f9a442-755c-4acb-85a3-3e22a87cdd9b" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
