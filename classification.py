import csv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

path_classification_data = "data/classification/"
path_train_csv = f"{path_classification_data}/train.csv"
path_test_csv = f"{path_classification_data}/test.csv"

with open(path_train_csv) as my_csv_train_file:
  header_and_data_train = list(csv.reader(my_csv_train_file, delimiter=','))

# Conversion en DataFrame
df_train = pd.DataFrame(header_and_data_train[1:], columns=header_and_data_train[0])


with open(path_test_csv) as my_csv_test_file:
  header_and_data_test = list(csv.reader(my_csv_test_file, delimiter=','))

# Conversion en DataFrame
df_test = pd.DataFrame(header_and_data_test[1:], columns=header_and_data_test[0])



# Nettoyage des données inutiles car on voit déjà sur Kaggle qu'il n'y a pas de données manquantes, valeurs abberantes ...

print(df_train)

## Decision Tree Algorithms
clf = DecisionTreeClassifier(random_state=0)
clf.fit(df_train.drop(columns=['id', 'bc_price_evo']), df_train['bc_price_evo'])


# Prédictions sur les données de test
y_pred = clf.predict(df_test.drop(columns=['id']))

# Création d'une DataFrame pour les prédictions
df_predictions = pd.DataFrame({
    'id': df_test['id'],
    'bc_price_evo': y_pred
})

# Création d'un fichier CSV à partir de la DataFrame
output_csv_path = f"{path_classification_data}/predictions.csv"
df_predictions.to_csv(output_csv_path, index=False)

print(df_predictions)