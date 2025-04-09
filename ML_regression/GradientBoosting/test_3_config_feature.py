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

# Fonction de prétraitement avec une option pour enlever des colonnes


def preprocess_data(train_data, test_data, cols_to_remove=[]):
    combined_data = pd.concat([train_data, test_data], axis=0)

    # Suppression des colonnes spécifiées dans cols_to_remove
    cols_to_remove += ['id', 'brand', 'model', 'car_class', 'grbx_type_ratios']
    combined_data = combined_data.drop(cols_to_remove, axis=1)

    # Conversion des variables catégorielles en variables dummy
    combined_data = pd.get_dummies(combined_data, columns=[
                                   'fuel_type', 'range', 'hybrid'], drop_first=True)

    # Remplir les valeurs manquantes par la médiane
    combined_data = combined_data.fillna(combined_data.median())

    # Re-séparation des données en train et test
    train_data_processed = combined_data.iloc[:len(train_data), :]
    test_data_processed = combined_data.iloc[len(train_data):, :]

    # Supprimer la target 'co2' du jeu de test
    if 'co2' in test_data_processed.columns:
        test_data_processed = test_data_processed.drop('co2', axis=1)

    return train_data_processed, test_data_processed

# Fonction pour tester différentes configurations de colonnes à enlever


def test_model(cols_to_remove):
    print(f"Testing with columns removed: {cols_to_remove}")

    # Prétraitement des données
    train_data_processed, test_data_processed = preprocess_data(
        train_data, test_data, cols_to_remove)

    # Séparation des features et de la target
    X = train_data_processed.drop('co2', axis=1)
    y = train_data_processed['co2']

    # Split train / validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42)

    # Standardisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    test_data_scaled = scaler.transform(test_data_processed)

    # Entraînement Gradient Boosting
    model = GradientBoostingRegressor(
        n_estimators=21000,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.9,
        max_features='sqrt',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Évaluation
    val_predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_predictions)
    print(f'Gradient Boosting Regressor MAE: {mae:.4f}')

    # Prédiction sur les données test
    test_predictions = model.predict(test_data_scaled)

    # Génération du fichier de soumission
    submission = pd.DataFrame({
        'id': pd.read_csv(TEST_PATH)['id'],
        'co2': test_predictions
    })

    # Création du nom de fichier avec date, n_estimators, MAE et colonnes
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    mae_str = f"{mae:.2f}".replace('.', '_')
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


# Configurations de colonnes à tester
columns_to_remove_config1 = ['hc', 'weight_min',
                             'ptcl', 'max_power', 'weight_max', 'co']
columns_to_remove_config2 = ['ptcl', 'weight_min', 'hc']
columns_to_remove_config3 = ['ptcl', 'hc']


# Tester les différentes configurations
test_model(columns_to_remove_config1)
test_model(columns_to_remove_config2)
test_model(columns_to_remove_config3)
