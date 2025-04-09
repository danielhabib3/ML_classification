import csv
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the dataset
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data",
                          "classification", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "..", "data",
                         "classification", "test.csv")

# Lecture du train.csv
with open(TRAIN_PATH) as my_csv_file:
    header_and_data = list(csv.reader(my_csv_file, delimiter=','))

df_train_full = pd.DataFrame(header_and_data[1:], columns=header_and_data[0])

# Lecture du test.csv
with open(TEST_PATH) as my_csv_test_file:
    header_and_data_test = list(csv.reader(my_csv_test_file, delimiter=','))

df_test = pd.DataFrame(
    header_and_data_test[1:], columns=header_and_data_test[0])

# Conversion en types numériques si besoin
df_train_full = df_train_full.apply(pd.to_numeric, errors='ignore')
df_test = df_test.apply(pd.to_numeric, errors='ignore')

# Séparer X/y pour train.csv
X = df_train_full.drop(columns=['id', 'bc_price_evo'])
y = df_train_full['bc_price_evo']

# Paramètres
test_size = 0.1  # Active la validation
n_estimators = 1
learning_rate = 0.1
max_depth = 7

# Fonction pour entraîner et évaluer un modèle


def train_and_evaluate(X_data, y_data):
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, test_size=test_size, random_state=42)
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=0
    )
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    return round(accuracy * 100, 2)


# Accuracy de base avec toutes les colonnes
base_accuracy = train_and_evaluate(X, y)
print(f"[BASELINE] Accuracy avec toutes les colonnes : {base_accuracy}%\n")

# Tester l'impact de chaque colonne
results = []

for col in X.columns:
    X_reduced = X.drop(columns=[col])
    acc = train_and_evaluate(X_reduced, y)
    delta = round(acc - base_accuracy, 2)
    results.append((col, acc, delta))
    print(f"Sans '{col}': Accuracy = {acc}% (Δ = {delta}%)")

# Résumé
print("\n--- Résumé ---")
results_sorted = sorted(results, key=lambda x: x[2])  # Tri par impact
for col, acc, delta in results_sorted:
    print(f"{col:30} -> Accuracy: {acc}% | Δ = {delta}%")
