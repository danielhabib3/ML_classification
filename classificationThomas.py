import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # Importation correcte
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import csv


path_classification_data = "data/classification/"
path_train_csv = f"{path_classification_data}/train.csv"
path_test_csv = f"{path_classification_data}/test.csv"

with open(path_test_csv) as my_csv_file:
  test_data = list(csv.reader(my_csv_file, delimiter=','))

with open(path_train_csv) as my_csv_file:
  train_data = list(csv.reader(my_csv_file, delimiter=','))

# Conversion en DataFrame
df_test = pd.DataFrame(test_data[1:], columns=test_data[0])
df_train= pd.DataFrame(train_data[1:], columns=train_data[0])

# Nettoyage des données inutiles car on voit déjà sur Kaggle qu'il n'y a pas de données manquantes, valeurs abberantes ...

##print(df_train)

## Decision Tree Algorithms

# On train sur données de tests

# Séparation des features (X) et de la cible (y)
X= df_train[['date','hour','bc_price','bc_demand','ab_price','ab_demand','transfer']]
y = df_train['bc_price_evo']

# # Division en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Création du modèle d'arbre de décision
clf = DecisionTreeClassifier(criterion="entropy", max_depth=20, random_state=42)

# Entraînement du modèle
clf.fit(X_train, y_train)

# Prédictions
y_pred = clf.predict(X_test)

# Évaluation
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Précision du modèle : {accuracy:.2f}')


