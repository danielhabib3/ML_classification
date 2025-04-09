import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the dataset
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data",
                          "classification", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "..", "data",
                         "classification", "test.csv")

# Lecture des fichiers CSV
with open(TEST_PATH) as my_csv_file:
    test_data = list(csv.reader(my_csv_file, delimiter=','))

with open(TRAIN_PATH) as my_csv_file:
    train_data = list(csv.reader(my_csv_file, delimiter=','))

# Conversion en DataFrame
df_test = pd.DataFrame(test_data[1:], columns=test_data[0])
df_train = pd.DataFrame(train_data[1:], columns=train_data[0])

# Séparation des features (X) et de la cible (y) pour l'entraînement
X = df_train[['date', 'hour', 'bc_price',
              'bc_demand', 'ab_price', 'ab_demand', 'transfer']]
y = df_train['bc_price_evo']

# Division en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# Création et entraînement du modèle d'arbre de décision
clf = DecisionTreeClassifier(
    criterion="entropy", max_depth=20, random_state=42)
clf.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = clf.predict(X_test)

# Évaluation du modèle
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Précision du modèle : {accuracy:.2f}')
