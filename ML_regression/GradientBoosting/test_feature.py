import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

# Chargement des données
train_data = pd.read_csv('../data/regression/train.csv')
test_data = pd.read_csv('../data/regression/test.csv')

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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# MAE de base (avec toutes les features)
model = GradientBoostingRegressor(
    n_estimators=5000,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    subsample=0.9,
    max_features='sqrt',
    random_state=42
)
model.fit(X_train_scaled, y_train)
baseline_preds = model.predict(X_val_scaled)
baseline_mae = mean_absolute_error(y_val, baseline_preds)
print(f'✅ MAE de base avec toutes les features: {baseline_mae:.4f}\n')

# 🔍 Impact de la suppression de chaque feature
results = []
for col in X.columns:
    print(f"🧪 Test sans la colonne: {col}")

    X_mod = X.drop(columns=[col])

    X_train_mod, X_val_mod, _, _ = train_test_split(
        X_mod, y, test_size=test_size, random_state=42)

    scaler_mod = StandardScaler()
    X_train_mod_scaled = scaler_mod.fit_transform(X_train_mod)
    X_val_mod_scaled = scaler_mod.transform(X_val_mod)

    model_mod = GradientBoostingRegressor(
        n_estimators=5000,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.9,
        max_features='sqrt',
        random_state=42
    )
    model_mod.fit(X_train_mod_scaled, y_train)
    val_preds_mod = model_mod.predict(X_val_mod_scaled)
    mae_mod = mean_absolute_error(y_val, val_preds_mod)

    print(f"→ MAE sans {col}: {mae_mod:.4f} (diff: {mae_mod - baseline_mae:+.4f})\n")
    results.append((col, mae_mod, mae_mod - baseline_mae))

# 🔚 Résumé
results_sorted = sorted(results, key=lambda x: abs(x[2]), reverse=True)
print("\n📊 Impact des colonnes (trié par impact absolu sur MAE):")
for col, mae, diff in results_sorted:
    print(f"{col:<25} → MAE: {mae:.4f} | Différence: {diff:+.4f}")
