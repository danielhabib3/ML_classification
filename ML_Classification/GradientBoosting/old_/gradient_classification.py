import csv
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Chemins vers les fichiers
path_classification_data = "../data/classification/"
path_train_csv = f"{path_classification_data}/train.csv"
path_test_csv = f"{path_classification_data}/test.csv"

# Lecture CSV
with open(path_train_csv) as my_csv_file:
    header_and_data = list(csv.reader(my_csv_file, delimiter=','))

# Conversion en DataFrame
df = pd.DataFrame(header_and_data[1:], columns=header_and_data[0])

# Encodage des colonnes si nécessaire (au cas où les valeurs sont des strings)
# Ici on suppose que tout est déjà en format numérique ou utilisable

# Séparer les données
train_size = int(len(df) * 0.9)
df_train = df.iloc[:train_size].copy()
df_test = df.iloc[train_size:].copy()

#print(df_train)

# Préparation des features et labels
X_train = df_train.drop(columns=['id', 'bc_price_evo'])
y_train = df_train['bc_price_evo']

X_test = df_test.drop(columns=['id', 'bc_price_evo'])
y_test = df_test['bc_price_evo']

# Gradient Boosting Classifier
clf = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=3,
    random_state=0
)

# Entraînement
clf.fit(X_train, y_train)

# Prédictions
y_pred = clf.predict(X_test)

# Création d'un DataFrame pour les résultats
df_predictions = pd.DataFrame({
    'id': df_test['id'],
    'bc_price_evo': y_pred
})

# Calcul de la précision
accuracy = (df_predictions['bc_price_evo'].values == y_test.values).mean() * 100
print(f"Précision du modèle (Gradient Boosting) : {accuracy:.2f}%")

# Optionnel : Rapport complet
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

