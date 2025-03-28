import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import itertools
import time
import csv

# Chemins des fichiers
path_classification_data = "data/classification/"
path_train_csv = f"{path_classification_data}/train.csv"
path_test_csv = f"{path_classification_data}/test.csv"

# Chargement des données
with open(path_test_csv) as my_csv_file:
    test_data = list(csv.reader(my_csv_file, delimiter=','))

with open(path_train_csv) as my_csv_file:
    train_data = list(csv.reader(my_csv_file, delimiter=','))

# Conversion en DataFrame
df_test = pd.DataFrame(test_data[1:], columns=test_data[0])
df_train = pd.DataFrame(train_data[1:], columns=train_data[0])

# Séparation des features (X) et de la cible (y)
X = df_train[['date', 'hour', 'bc_price', 'bc_demand', 'ab_price', 'ab_demand', 'transfer']]
y = df_train['bc_price_evo']

# Conversion des types (si nécessaire)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna()

# Variables pour stocker les résultats
best_accuracy = 0
best_params = {}

# Grille d'hyperparamètres
max_depths = [5, 10, 20, 50, 100, 200, None]
min_samples_splits = [2, 5, 10, 20, 50, 100]
min_samples_leaves = [1, 5, 10, 20, 50]
criterions = ["gini", "entropy"]
max_features_options = [None, "sqrt", "log2"]
splitters = ["best", "random"]
random_states = [42, 0, 1, 42, 123]  # Différents random_state à tester

# Compteur de tests
total_tests = len(max_depths) * len(min_samples_splits) * len(min_samples_leaves) * len(criterions) * len(max_features_options) * len(splitters) * len(random_states)
test_count = 0

print(f"🔍 Début des tests... ({total_tests} combinaisons possibles)\n")

# Boucles imbriquées pour tester toutes les combinaisons
start_time = time.time()
for depth, split, leaf, criterion, max_feat, splitter, rand_state in itertools.product(
    max_depths, min_samples_splits, min_samples_leaves, criterions, max_features_options, splitters, random_states):

    # Variable pour stocker les précisions pour chaque random_state
    accuracies = []

    # Test avec plusieurs random_state
    for rs in random_states:
        # Division en ensemble d'entraînement et de test avec random_state actuel
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rs)

        # Création du modèle avec les hyperparamètres actuels
        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=depth,
            min_samples_split=split,
            min_samples_leaf=leaf,
            max_features=max_feat,
            splitter=splitter,
            random_state=rs
        )

        # Entraînement du modèle
        clf.fit(X_train, y_train)

        # Prédictions
        y_pred = clf.predict(X_test)

        # Évaluation de la précision
        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    # Calcul de la moyenne de la précision
    avg_accuracy = np.mean(accuracies)
    test_count += 1

    # Affichage des résultats intermédiaires tous les 500 tests
    if test_count % 500 == 0:
        print(f"Test {test_count}/{total_tests} -> depth={depth}, split={split}, leaf={leaf}, criterion={criterion}, max_feat={max_feat}, splitter={splitter}, random_state={rand_state} | Avg Accuracy: {avg_accuracy:.4f}")

    # Stockage des meilleurs hyperparamètres
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_params = {
            "max_depth": depth,
            "min_samples_split": split,
            "min_samples_leaf": leaf,
            "criterion": criterion,
            "max_features": max_feat,
            "splitter": splitter,
            "random_state": rand_state
        }

# Fin du test
end_time = time.time()
elapsed_time = end_time - start_time

# Affichage des meilleurs paramètres trouvés
print("\n✅ Meilleurs hyperparamètres :")
print(best_params)
print(f"🎯 Précision moyenne maximale obtenue : {best_accuracy:.4f}")
print(f"⏱️ Temps total d'exécution : {elapsed_time:.2f} secondes")