import csv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

path_classification_data = "data/classification/"
path_train_csv = f"{path_classification_data}/train.csv"
path_test_csv = f"{path_classification_data}/test.csv"

with open(path_train_csv) as my_csv_file:
  header_and_data = list(csv.reader(my_csv_file, delimiter=','))

# Conversion en DataFrame
df = pd.DataFrame(header_and_data[1:], columns=header_and_data[0])

# Séparer les 90% premiers données pour l'entraînement
train_size = int(len(df) * 0.7)
df_train = df.iloc[:train_size].copy()

df_test = df.iloc[train_size:].copy()





# Nettoyage des données inutiles car on voit déjà sur Kaggle qu'il n'y a pas de données manquantes, valeurs abberantes ...

print(df_train)


#Méthode de classification random forest
clf = RandomForestClassifier(
  n_estimators=100,  # Number of trees in the forest
  criterion='gini',  # or 'entropy' for information gain
  max_depth=None,    # maximum depth of the tree
  min_samples_split=2,  # minimum samples required to split an internal node
  min_samples_leaf=1,   # minimum samples required to be at a leaf node
  min_weight_fraction_leaf=0.0,  # minimum weighted fraction of the sum total of weights at a leaf node
  max_features='auto',    # number of features to consider when looking for the best split
  random_state=0,       # seed for reproducibility
  max_leaf_nodes=None,  # maximum number of leaf nodes
  min_impurity_decrease=0.0,  # minimum impurity decrease required for a split
)
clf.fit(df_train.drop(columns=['id', 'bc_price_evo']), df_train['bc_price_evo'])

# Prédictions sur les données de test
y_pred = clf.predict(df_test.drop(columns=['id', 'bc_price_evo']))

# Création d'une DataFrame pour les prédictions
df_predictions = pd.DataFrame({
    'id': df_test['id'],
    'bc_price_evo': y_pred
})

# Comparer les prédictions avec les valeurs réelles
accuracy = (df_predictions['bc_price_evo'].values == df_test['bc_price_evo'].values).mean() * 100

# Afficher le pourcentage de précision
print(f"Précision du modèle : {accuracy:.2f}%")


