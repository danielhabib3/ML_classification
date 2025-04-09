import csv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the dataset
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data",
                          "classification", "train.csv")


with open(TRAIN_PATH) as my_csv_file:
    header_and_data = list(csv.reader(my_csv_file, delimiter=','))

# Conversion en DataFrame
df = pd.DataFrame(header_and_data[1:], columns=header_and_data[0])

# Séparer les 90% premiers données pour l'entraînement
train_size = int(len(df) * 0.9)
df_train = df.iloc[:train_size].copy()

df_test = df.iloc[train_size:].copy()

# Nettoyage des données inutile car on voit déjà sur Kaggle qu'il n'y a pas de données manquantes, valeurs abberantes ...

print(df_train)

# Decision Tree Algorithms
# Define the parameter grid to test
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': [None, 'sqrt', 'log2'],
    'random_state': [0]
}

# Initialize the DecisionTreeClassifier
clf = DecisionTreeClassifier()

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(
    estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(df_train.drop(
    columns=['id', 'bc_price_evo']), df_train['bc_price_evo'])

# Get the best estimator
clf = grid_search.best_estimator_

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)
clf.fit(df_train.drop(
    columns=['id', 'bc_price_evo']), df_train['bc_price_evo'])

# Prédictions sur les données de test
y_pred = clf.predict(df_test.drop(columns=['id', 'bc_price_evo']))

# Création d'une DataFrame pour les prédictions
df_predictions = pd.DataFrame({
    'id': df_test['id'],
    'bc_price_evo': y_pred
})

# Comparer les prédictions avec les valeurs réelles
accuracy = (df_predictions['bc_price_evo'].values ==
            df_test['bc_price_evo'].values).mean() * 100

# Afficher le pourcentage de précision
print(f"Précision du modèle : {accuracy:.2f}%")
