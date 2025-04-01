import torch
import pandas as pd
from train_model import TransformerModel
from data_processing import load_data

TEST_CSV = "../../data/classification/test.csv"
MODEL_PATH = "saved_models/saved_model.pth" 
SUBMISSION_CSV = "predictions.csv"

# On utilise le GPU !
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle déjà calculé
model = TransformerModel(input_dim=7).to(device) 
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Charger les données de test
_, _, X_test, ids = load_data("../../data/classification/train.csv", TEST_CSV) # Ici le chemin en 1er param est utile que parce que data_processing prend forcément jeu de train
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# Faire les prédictions
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred = (y_pred.cpu().numpy() > 0.5).astype(int)

# Générer le fichier CSV
df_output = pd.DataFrame({'id': ids, 'bc_price_evo': ['UP' if pred == 1 else 'DOWN' for pred in y_pred.flatten()]})
df_output.to_csv(SUBMISSION_CSV, index=False)
print(f"Fichier de soumission généré : {SUBMISSION_CSV}")
