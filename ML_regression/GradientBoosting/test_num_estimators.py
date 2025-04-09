import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the dataset
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data",
                          "regression", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "..", "data",
                         "regression", "test.csv")
# Chargement des données
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

# Prétraitement


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


# Préparation des données
train_data, test_data = preprocess_data(train_data, test_data)
X = train_data.drop('co2', axis=1)
y = train_data['co2']

test_size = 0.1

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=test_size, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Test de différents n_estimators
results = []

# l = list(range(20000, 30000, 1000))
l = [100000]
print(l)

for n in l:
    print(f"test {n}")
    model = GradientBoostingRegressor(
        n_estimators=n,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.9,
        max_features='sqrt',
        random_state=42
    )
    model.fit(X_train, y_train)

    if (test_size > 0):

        val_predictions = model.predict(X_val)
        mae = mean_absolute_error(y_val, val_predictions)
        results.append((n, mae))
        print(f'n_estimators: {n} → MAE: {mae:.5f}')

# Afficher le meilleur résultat
best_n, best_mae = min(results, key=lambda x: x[1])
print(f'\nMeilleur n_estimators: {best_n} avec MAE: {best_mae:.4f}')
