import csv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


def load_data_train(path_classification_data, ratio_train_test=0.9):
  with open(f"{path_classification_data}/train.csv") as my_csv_file:
    header_and_data = list(csv.reader(my_csv_file, delimiter=','))

  # Conversion en DataFrame
  df = pd.DataFrame(header_and_data[1:], columns=header_and_data[0])

  column_to_delete = ['id']

  column_numeric = ['hc', 'max_power', 'weight_min', 'weight_max', 'urb_cons', 'exturb_cons', 'overall_cons', 'co', 'nox', 'hcnox', 'ptcl', 'co2']


  # Supprimer les colonnes spécifiées dans column_to_delete
  df = df.drop(columns=column_to_delete)

  # Supprimer les lignes où 'nox' est une chaîne vide ''
  # Quand y a pas nox y a pas hc et hcnox
  df = df[df['nox'] != '']
  df = df[df['ptcl'] != '']
  df = df[df['exturb_cons'] != '']
  df = df[df['urb_cons'] != '']

  # Remplacer les valeurs manquantes de 'hcnox' par la somme de 'hc' et 'nox'
  df['hcnox'] = df.apply(
    lambda row: float(row['hc']) + float(row['nox']) if row['hcnox'] == '' else float(row['hcnox']),
    axis=1
  )

  # Remplacer les valeurs manquantes de 'hc' par la différence entre 'hcnox' et 'nox'
  df['hc'] = df.apply(
    lambda row: float(row['hcnox']) - float(row['nox']) if row['hc'] == '' else float(row['hc']),
    axis=1
  )

  # Identifier les colonnes non numériques
  non_numeric_columns = [col for col in df.columns if col not in column_numeric and col not in column_to_delete]

  # Appliquer LabelEncoder sur les colonnes non numériques
  label_encoders = {}
  for col in non_numeric_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

  # Séparer les 90% premiers données pour l'entraînement
  train_size = int(len(df) * ratio_train_test)
  df_train = df.iloc[:train_size].copy()

  df_test = df.iloc[train_size:].copy()

  # Nettoyage des données inutiles car on voit déjà sur Kaggle qu'il n'y a pas de données manquantes, valeurs abberantes ...

  return df_train, df_test

def load_data_test(path_classification_data):
  with open(f"{path_classification_data}/test.csv") as my_csv_file:
    header_and_data = list(csv.reader(my_csv_file, delimiter=','))

  # Conversion en DataFrame
  df = pd.DataFrame(header_and_data[1:], columns=header_and_data[0])

  column_to_delete = []

  column_numeric = ['hc', 'max_power', 'weight_min', 'weight_max', 'urb_cons', 'exturb_cons', 'overall_cons', 'co', 'nox', 'hcnox', 'ptcl', 'co2']


  # Supprimer les lignes où 'nox' est une chaîne vide ''
  # Quand y a pas nox y a pas hc et hcnox
  df = df[df['nox'] != '']
  df = df[df['ptcl'] != '']
  df = df[df['exturb_cons'] != '']
  df = df[df['urb_cons'] != '']

  # Remplacer les valeurs manquantes de 'hcnox' par la somme de 'hc' et 'nox'
  df['hcnox'] = df.apply(
    lambda row: float(row['hc']) + float(row['nox']) if row['hcnox'] == '' else float(row['hcnox']),
    axis=1
  )

  # Remplacer les valeurs manquantes de 'hc' par la différence entre 'hcnox' et 'nox'
  df['hc'] = df.apply(
    lambda row: float(row['hcnox']) - float(row['nox']) if row['hc'] == '' else float(row['hc']),
    axis=1
  )

  # Identifier les colonnes non numériques
  non_numeric_columns = [col for col in df.columns if col not in column_numeric and col not in column_to_delete]

  # Appliquer LabelEncoder sur les colonnes non numériques
  label_encoders = {}
  for col in non_numeric_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

  # # Séparer les 90% premiers données pour l'entraînement
  # train_size = int(len(df) * 0.7)
  # df_train = df.iloc[:train_size].copy()

  # df_test = df.iloc[train_size:].copy()

  # Nettoyage des données inutiles car on voit déjà sur Kaggle qu'il n'y a pas de données manquantes, valeurs abberantes ...

  return df