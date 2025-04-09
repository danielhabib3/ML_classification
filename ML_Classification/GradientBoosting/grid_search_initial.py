import csv
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the dataset
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data",
                          "classification", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "..", "data",
                         "classification", "test.csv")
# Lecture CSV
with open(TRAIN_PATH) as my_csv_file:
    header_and_data = list(csv.reader(my_csv_file, delimiter=','))

# Conversion en DataFrame
df = pd.DataFrame(header_and_data[1:], columns=header_and_data[0])
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

# Grid Search
grid_search = GridSearchCV(
    GradientBoostingClassifier(n_estimators=500, random_state=0), # On fixe n_estimators car on sait grâce à nos anciens tests que si c'est trop élévé, le grid_search va être trop long et d'ailleurs, o nsait déjà que plus on l'augmente mieux c'est
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# Entraînement
grid_search.fit(X_train, y_train)

print("Meilleurs paramètres trouvés :", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Création d'un DataFrame pour les résultats
df_predictions = pd.DataFrame({
    'id': df_test['id'],
    'bc_price_evo': y_pred
})

<<<<<<< HEAD:ML_Classification/GradientBoosting/grid_search.py
# Précision
accuracy = (df_predictions['bc_price_evo'].values ==
            y_test.values).mean() * 100
=======
accuracy = (df_predictions['bc_price_evo'].values == y_test.values).mean() * 100
>>>>>>> 1a1e2532ca6f48553a033f7093d2429f1e6a952b:ML_Classification/GradientBoosting/grid_search_initial.py
print(f"Précision du modèle (Gradient Boosting optimisé) : {accuracy:.2f}%")

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))
