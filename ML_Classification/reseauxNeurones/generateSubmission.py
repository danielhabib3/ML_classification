import torch
import pandas as pd
import argparse
from models import TransformerModel
from data_processing import load_data
from sklearn.metrics import accuracy_score  # Importer pour calculer la précision

TEST_CSV = "../../data/classification/test.csv"
SUBMISSION_CSV = "predictions.csv"

# Parser les arguments de la ligne de commande
parser = argparse.ArgumentParser(description='Script de prédiction avec un modèle Transformer')
parser.add_argument('model_path', type=str, help='Chemin vers le fichier du modèle sauvegardé')
args = parser.parse_args()

# On utilise le GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle
model = TransformerModel(input_dim=7).to(device)
model.load_state_dict(torch.load(args.model_path))
model.eval()

# Charger les données
X_train, y_train, X_test, ids = load_data("../../data/classification/train.csv", TEST_CSV)

# Conversion des données d'entraînement en tenseurs
X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1).to(device)

# Faire des prédictions sur le jeu d'entraînement
with torch.no_grad():
    y_train_pred = model(X_train_tensor)
    y_train_pred = (y_train_pred.cpu().numpy() > 0.5).astype(int)

# Calculer la précision
accuracy = accuracy_score(y_train, y_train_pred)
print(f"Précision sur l'ensemble d'entraînement : {accuracy * 100:.2f}%")

# Charger les données de test et faire des prédictions
X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred = (y_pred.cpu().numpy() > 0.5).astype(int)

# Générer le fichier CSV de soumission
df_output = pd.DataFrame({'id': ids, 'bc_price_evo': ['UP' if pred == 1 else 'DOWN' for pred in y_pred.flatten()]})
df_output.to_csv(SUBMISSION_CSV, index=False)
print(f"Fichier de soumission généré : {SUBMISSION_CSV}")