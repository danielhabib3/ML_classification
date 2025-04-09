import pandas as pd
import numpy as np
import datetime
import os
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

test_size = 0.0

other_column_to_drop = ['hc', 'weight_min',
                        'ptcl', 'max_power', 'weight_max', 'co']

# Prétraitement


def preprocess_data(train_data, test_data):
    combined_data = pd.concat([train_data, test_data], axis=0)

    columns_to_drop = ['id', 'brand', 'model', 'car_class',
                       'grbx_type_ratios'] + other_column_to_drop

    combined_data = combined_data.drop(columns_to_drop, axis=1)

    combined_data = pd.get_dummies(combined_data, columns=[
        'fuel_type', 'range', 'hybrid'], drop_first=True)

    combined_data = combined_data.fillna(combined_data.median())

    train_data_processed = combined_data.iloc[:len(train_data), :]
    test_data_processed = combined_data.iloc[len(train_data):, :]

    if 'co2' in test_data_processed.columns:
        test_data_processed = test_data_processed.drop('co2', axis=1)

    return train_data_processed, test_data_processed


# Prétraitement
train_data, test_data = preprocess_data(train_data, test_data)

# Séparation features / target
X = train_data.drop('co2', axis=1)
y = train_data['co2']

# Vérification si test_size > 0 avant de faire un split
if test_size > 0:
    # Split train / validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42)
else:
    X_train = X
    y_train = y
    X_val = None
    y_val = None
    print("No validation set created (test_size=0)")

# Standardisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
test_data_scaled = scaler.transform(test_data)

if X_val is not None and y_val is not None:
    X_val = scaler.transform(X_val)

# Gradient Boosting
model = GradientBoostingRegressor(
    n_estimators=500000,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    subsample=0.9,
    max_features='sqrt',
    random_state=42
)
model.fit(X_train, y_train)

# Évaluation si test_size > 0
if test_size > 0:
    val_predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_predictions)
    print(f'Gradient Boosting Regressor MAE: {mae:.4f}')
else:
    mae = "N/A"  # Pas de MAE disponible si test_size == 0
    print("No MAE to report as test_size is 0")

# Prédiction sur les données test
test_predictions = model.predict(test_data_scaled)

# Génération du fichier de soumission
submission = pd.DataFrame({
    'id': pd.read_csv(TEST_PATH)['id'],
    'co2': test_predictions
})

# Nom fichier submit
date_str = datetime.datetime.now().strftime("%Y%m%d")
mae_str = f"{mae:.2f}" if isinstance(mae, float) else "N_A"
n_estimators = model.n_estimators
cols_used = "_".join(X.columns[:10])  # max 10 colonnes dans le nom
cols_used = cols_used.replace(' ', '').replace('/', '').replace('\\', '')

# Dossier de sortie
os.makedirs('submit', exist_ok=True)

# Construction du chemin final
filename = f"submission_{date_str}_est{n_estimators}_MAE{mae_str}.csv"
path_submit = os.path.join("submit", filename)

# Sauvegarde
submission.to_csv(path_submit, index=False)
print(f"Submission file created: {path_submit}")
