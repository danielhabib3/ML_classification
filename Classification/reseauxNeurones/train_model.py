import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from data_processing import load_data
import models 
from sklearn.metrics import accuracy_score  # Importer pour calculer la précision

TRAIN_CSV = "../../data/classification/train.csv"
MODEL_PATH = "saved_models/saved_model.pth"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

# On utilise le GPU pour aller + vite !
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de : {device}")

# Argument ligne de commandes pour choisir modèle
parser = argparse.ArgumentParser(description="Entraînement d'un modèle de classification.")
parser.add_argument("--model_name", type=str, default="TransformerModel", help="Nom du modèle à utiliser (doit être défini dans models.py)")
args = parser.parse_args()

# Vérifier si le modèle existe dans models.py
if not hasattr(models, args.model_name):
    raise ValueError(f"Modèle '{args.model_name}' non trouvé dans models.py")

# Charger le modèle
ModelClass = getattr(models, args.model_name)

if __name__ == "__main__":
    # Chargement des données
    X, y = load_data(TRAIN_CSV)

    # Conversion en tenseurs PyTorch
    X_train_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)  
    y_train_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device) 

    # Création du DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialise le modèle
    model = ModelClass(X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Boucle de Training
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        all_preds = []  # Liste pour stocker toutes les prédictions
        all_labels = []  # Liste pour stocker toutes les étiquettes réelles

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Stocker les prédictions et les labels pour calculer la précision plus tard
            all_preds.append(outputs.cpu().detach().numpy())
            all_labels.append(batch_y.cpu().detach().numpy())

        # Convertir les prédictions et labels en arrays numpy
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Calcul de la précision
        all_preds = (all_preds > 0.5).astype(int)  # Convertir en 0 ou 1
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy*100:.2f}%")

    # Sauvegarde du modèle
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modèle sauvegardé sous {MODEL_PATH}")
