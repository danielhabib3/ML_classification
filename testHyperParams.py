import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import itertools
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

# Grille d'hyperparamètres à tester
max_depths = [5, 10, 20, 50]
min_samples_splits = [2, 5, 10, 20]
min_samples_leaves = [1, 5, 10]
criterions = ["gini", "entropy"]

# Liste des random_state à tester
random_states = [0, 10, 42, 100, 200]

# Dictionnaire pour stocker les résultats moyens de chaque combinaison
accuracies_dict = {}

# Boucles imbriquées pour tester différentes combinaisons de hyperparamètres et random_state
for depth, split, leaf, criterion in itertools.product(max_depths, min_samples_splits, min_samples_leaves, criterions):
    print(f"\n🔍 Test pour combinaison: depth={depth}, split={split}, leaf={leaf}, criterion={criterion}...")

    accuracies_per_combination = []

    for rs in random_states:
        # Division en ensemble d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rs)

        # Création du modèle avec les hyperparamètres actuels
        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=depth,
            min_samples_split=split,
            min_samples_leaf=leaf,
            random_state=rs
        )

        # Entraînement du modèle
        clf.fit(X_train, y_train)

        # Prédictions
        y_pred = clf.predict(X_test)

        # Évaluation de la précision
        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracies_per_combination.append(accuracy)

    # Calcul de la moyenne de la précision pour cette combinaison de paramètres
    mean_accuracy = np.mean(accuracies_per_combination)

    # Stockage des résultats dans le dictionnaire
    accuracies_dict[(depth, split, leaf, criterion)] = mean_accuracy

    # Affichage des résultats pour chaque combinaison
    print(f"    Moyenne précision pour cette combinaison: {mean_accuracy:.4f}")

# Trouver la meilleure combinaison d'hyperparamètres basée sur la précision moyenne
best_combination = max(accuracies_dict, key=accuracies_dict.get)
best_accuracy = accuracies_dict[best_combination]

# Affichage des meilleurs paramètres trouvés
print("\n✅ Meilleurs hyperparamètres :")
print(f"   max_depth={best_combination[0]}, min_samples_split={best_combination[1]}, "
      f"min_samples_leaf={best_combination[2]}, criterion={best_combination[3]}")
print(f"🎯 Précision moyenne maximale obtenue : {best_accuracy:.4f}")
