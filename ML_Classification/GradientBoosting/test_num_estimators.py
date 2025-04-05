import csv
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Chemins
path_classification_data = "../data/classification/"
path_train_csv = f"{path_classification_data}/train.csv"
path_test_csv = f"{path_classification_data}/test.csv"

# Lecture du train.csv
with open(path_train_csv) as my_csv_file:
    header_and_data = list(csv.reader(my_csv_file, delimiter=','))

df_train_full = pd.DataFrame(header_and_data[1:], columns=header_and_data[0])

# Lecture du test.csv
with open(path_test_csv) as my_csv_test_file:
    header_and_data_test = list(csv.reader(my_csv_test_file, delimiter=','))

df_test = pd.DataFrame(header_and_data_test[1:], columns=header_and_data_test[0])

# Conversion en types numériques si besoin
df_train_full = df_train_full.apply(pd.to_numeric, errors='ignore')
df_test = df_test.apply(pd.to_numeric, errors='ignore')

# Séparer X/y pour train.csv
X = df_train_full.drop(columns=['id', 'bc_price_evo'])
y = df_train_full['bc_price_evo']

# Paramètres
test_size = 0.1  # Active la validation
learning_rate = 0.1
max_depth = 7

# Fonction pour entraîner et évaluer un modèle
def train_and_evaluate(X_data, y_data, n_estimators_val):
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=test_size, random_state=42)
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators_val,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=0
    )
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    return round(accuracy * 100, 2)

# Liste des n_estimators à tester
n_estimators_list = [10, 50, 100, 200, 500, 1000, 2000, 5000, 6000, 7000, 10000, 12500, 15000, 20000]

# Tester différentes valeurs de n_estimators
print("--- Test des n_estimators ---")
results = []

for n in n_estimators_list:
    acc = train_and_evaluate(X, y, n)
    results.append((n, acc))
    print(f"n_estimators = {n:5d} -> Accuracy = {acc}%")

# Résumé trié par n_estimators croissants
print("\n--- Résumé ---")
for n, acc in results:
    print(f"{n:5d} arbres -> Accuracy: {acc}%")
