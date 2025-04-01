import pandas as pd
import numpy as np

def load_data(train_path, test_path=None):
    df_train = pd.read_csv(train_path)
    df_train.drop(columns=['id'], errors='ignore', inplace=True)
    df_train['bc_price_evo'] = df_train['bc_price_evo'].map({'DOWN': 0, 'UP': 1}) # On convertit les données catégorielles en valeurs numériques

    features = df_train.columns[:-1]  # On prend tout sauf la cible
    X = df_train[features].astype(np.float32) # On force les types
    y = df_train['bc_price_evo'].astype(np.int32)  # On force les types

    if test_path:
        df_test = pd.read_csv(test_path)
        ids = df_test.iloc[:, 0]  # Stocker les ID
        X_test = df_test.iloc[:, 1:].astype(np.float32)
        return X, y, X_test, ids
    return X, y
