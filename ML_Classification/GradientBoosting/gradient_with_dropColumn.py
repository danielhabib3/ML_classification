import csv
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the dataset
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data",
                          "classification", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "..", "data",
                         "classification", "test.csv")


# Lecture du train.csv
with open(TRAIN_PATH) as my_csv_file:
    header_and_data = list(csv.reader(my_csv_file, delimiter=','))
df_train_full = pd.DataFrame(header_and_data[1:], columns=header_and_data[0])

# Lecture du test.csv
with open(TEST_PATH) as my_csv_test_file:
    header_and_data_test = list(csv.reader(my_csv_test_file, delimiter=','))
df_test = pd.DataFrame(
    header_and_data_test[1:], columns=header_and_data_test[0])

# Cast (important si les données sont en string à cause du CSV)
df_train_full = df_train_full.astype(float, errors='ignore')
df_test = df_test.astype(float, errors='ignore')

# Combos à tester spécifiquement
combos_to_test = [
    ['transfer', 'bc_demand'],
    ['ab_demand', 'bc_demand'],
    ['transfer', 'ab_demand', 'bc_demand'],
    ['ab_demand', 'hour', 'bc_demand'],
    ['ab_demand', 'hour']
]

# Paramètres du modèle
test_size = 0.1
n_estimators = 5000
learning_rate = 0.1
max_depth = 7

best_accuracy = 0
best_combo = None

# Tester les 5 combinaisons spécifiques
for combo in combos_to_test:
    print(combo)

    # Colonnes à supprimer
    columns_to_drop = ['id', 'bc_price_evo'] + combo
    try:
        X = df_train_full.drop(columns=columns_to_drop)
        y = df_train_full['bc_price_evo']

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42)

        # Entraînement
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=0
        )
        clf.fit(X_train, y_train)

        # Évaluation
        y_val_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        accuracy_percent = round(accuracy * 100, 2)

        print(f"Combo supprimé : {combo} --> Accuracy = {accuracy_percent}%")

        # Mise à jour du meilleur score
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_combo = combo

    except Exception as e:
        print(f"Erreur avec combo {combo} : {e}")

print("\n===============================")
print(f"Meilleure combinaison : {best_combo}")
print(f"Meilleure accuracy : {round(best_accuracy * 100, 2)}%")
