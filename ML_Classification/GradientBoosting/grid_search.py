import csv
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

# Chemins vers les fichiers
path_classification_data = "../data/classification/"
path_train_csv = f"{path_classification_data}/train.csv"
path_test_csv = f"{path_classification_data}/test.csv"

# Lecture CSV
with open(path_train_csv) as my_csv_file:
    header_and_data = list(csv.reader(my_csv_file, delimiter=','))

# Conversion en DataFrame
df = pd.DataFrame(header_and_data[1:], columns=header_and_data[0])

# Convertir les colonnes en numériques si nécessaire
df = df.apply(pd.to_numeric, errors='ignore')

# Séparer les données
train_size = int(len(df) * 0.9)
df_train = df.iloc[:train_size].copy()
df_test = df.iloc[train_size:].copy()

# Préparation des features et labels
X_train = df_train.drop(columns=['id', 'bc_price_evo'])
y_train = df_train['bc_price_evo']

X_test = df_test.drop(columns=['id', 'bc_price_evo'])
y_test = df_test['bc_price_evo']

# Définition des hyperparamètres à tester
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4, 5, 7]
}

# Grid Search avec validation croisée
grid_search = GridSearchCV(
    GradientBoostingClassifier(n_estimators=500, random_state=0),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# Entraînement
grid_search.fit(X_train, y_train)

# Meilleurs paramètres
print("Meilleurs paramètres trouvés :", grid_search.best_params_)

# Prédictions avec le meilleur modèle
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Création d'un DataFrame pour les résultats
df_predictions = pd.DataFrame({
    'id': df_test['id'],
    'bc_price_evo': y_pred
})

# Précision
accuracy = (df_predictions['bc_price_evo'].values == y_test.values).mean() * 100
print(f"Précision du modèle (Gradient Boosting optimisé) : {accuracy:.2f}%")

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))
