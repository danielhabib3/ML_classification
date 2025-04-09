import csv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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
train_size = int(len(df) * 0.7)
df_train = df.iloc[:train_size].copy()

df_test = df.iloc[train_size:].copy()

# Nettoyage des données inutile car on voit déjà sur Kaggle qu'il n'y a pas de données manquantes, valeurs abberantes ...

print(df_train)

# Decision Tree Algorithms
clf = DecisionTreeClassifier(
    criterion='gini',  # or 'entropy' for information gain
    splitter='best',   # or 'random' for random feature splits
    max_depth=None,    # maximum depth of the tree
    min_samples_split=2,  # minimum samples required to split an internal node
    min_samples_leaf=1,   # minimum samples required to be at a leaf node
    # minimum weighted fraction of the sum total of weights at a leaf node
    min_weight_fraction_leaf=0.0,
    max_features=None,    # number of features to consider when looking for the best split
    random_state=0,       # seed for reproducibility
    max_leaf_nodes=None,  # maximum number of leaf nodes
    min_impurity_decrease=0.0,  # minimum impurity decrease required for a split
    ccp_alpha=0.0         # complexity parameter for Minimal Cost-Complexity Pruning
)
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
