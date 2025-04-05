import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv('data/regression/train.csv')
test_data = pd.read_csv('data/regression/test.csv')


def preprocess_data(train_data, test_data):
    combined_data = pd.concat([train_data, test_data], axis=0)

    combined_data = combined_data.drop(
        ['id', 'brand', 'model', 'car_class', 'grbx_type_ratios'], axis=1)

    combined_data = pd.get_dummies(combined_data, columns=[
                                   'fuel_type', 'range', 'hybrid'], drop_first=True)

    combined_data = combined_data.fillna(combined_data.median())

    train_data_processed = combined_data.iloc[:len(train_data), :]
    test_data_processed = combined_data.iloc[len(train_data):, :]

    if 'co2' in test_data_processed.columns:
        test_data_processed = test_data_processed.drop('co2', axis=1)

    return train_data_processed, test_data_processed


train_data, test_data = preprocess_data(train_data, test_data)

X = train_data.drop('co2', axis=1)
y = train_data['co2']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_data = scaler.transform(test_data)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

val_predictions = regressor.predict(X_val)
mae = mean_absolute_error(y_val, val_predictions)
print(f'Random Forest Regressor MAE: {mae}')

test_predictions = regressor.predict(test_data)

submission = pd.DataFrame({
    'id': pd.read_csv('test.csv')['id'],
    'co2': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")
