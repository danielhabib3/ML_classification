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

# Séparer X/y pour train.csv
drop_add = ['transfer', 'bc_demand']
to_drop =  ['id', 'bc_price_evo'] + drop_add
X = df_train_full.drop(columns=to_drop)
y = df_train_full['bc_price_evo']

# Paramètres
test_size = 0.0  # Change à 0.1 pour activer la validation
n_estimators = 4
learning_rate = 0.1
max_depth = 7

# Gestion du split en fonction de test_size
if test_size > 0:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
else:
    X_train, y_train = X, y
    X_val, y_val = None, None

# Entraînement du modèle
clf = GradientBoostingClassifier(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    random_state=0
)
clf.fit(X_train, y_train)

# Accuracy (si validation activée)
if test_size > 0:
    y_val_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    accuracy_percent = round(accuracy * 100, 2)
    print(f"Accuracy sur validation set : {accuracy_percent}%")
else:
    accuracy_percent = "NA"
    print("Aucune validation interne (test_size = 0)")

# Prédiction sur test.csv
to_drop_test = list(set(to_drop) - {'bc_price_evo'})  # Utiliser set() pour faire la différence
X_test = df_test.drop(columns = to_drop_test)
y_test_pred = clf.predict(X_test)

# Création de la DataFrame de sortie
df_predictions = pd.DataFrame({
    'id': df_test['id'],
    'bc_price_evo': y_test_pred
})

# Nom du fichier
filename = (
    f"predictions_gb_n{n_estimators}_lr{learning_rate}_d{max_depth}_acc{accuracy_percent}.csv"
)
output_csv_path = os.path.join("submit/", filename)

# Sauvegarde
df_predictions.to_csv(output_csv_path, index=False)
print("Fichier enregistré :", output_csv_path)
