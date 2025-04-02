import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from data_processing import load_data
import models 
from sklearn.metrics import accuracy_score
import signal
import sys
from datetime import datetime

# Paramètres
TRAIN_CSV = "../../data/classification/train.csv"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 500
WEIGHT_DECAY = 1e-4


# Fixer la seed pour garantir la reproductibilité
SEED = 42  # Choisir une seed fixe
torch.manual_seed(SEED)
np.random.seed(SEED)

# Dispositif (CUDA si disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de : {device}")

# Parsing des arguments
parser = argparse.ArgumentParser(description="Entraînement d'un modèle de classification.")
parser.add_argument("--model_name", type=str, default="TransformerModel", help="Nom du modèle à utiliser (doit être défini dans models.py)")
parser.add_argument("--model_file", type=str, help="Chemin du fichier modèle à charger (optionnel)")
parser.add_argument("--split_eval", action="store_true", help="Séparer le dataset en 90% train et 10% évaluation")
args = parser.parse_args()

# Vérification du modèle
if not hasattr(models, args.model_name):
    raise ValueError(f"Modèle '{args.model_name}' non trouvé dans models.py")

ModelClass = getattr(models, args.model_name)

# Définir le chemin pour sauvegarder le modèle
current_date = datetime.now().strftime("%m-%d_%H-%M")
MODEL_PATH = f"saved_models/saved_{args.model_name}_{current_date}_epochs{EPOCHS}.pth"

# POur enregistrer si on arrête pendant training
def save_and_exit(sig, frame):
    print("\nInterruption détectée ! Sauvegarde du modèle en cours...")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modèle sauvegardé sous {MODEL_PATH}")
    sys.exit(0)

signal.signal(signal.SIGINT, save_and_exit)

if __name__ == "__main__":
    # Chargement des données
    X, y = load_data(TRAIN_CSV)
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)

    if args.split_eval:
        total_size = len(X_tensor)
        train_size = int(0.9 * total_size)
        eval_size = total_size - train_size

        # Séparer les données de manière reproductible avec la seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        train_dataset, eval_dataset = random_split(TensorDataset(X_tensor, y_tensor), [train_size, eval_size])
    else:
        train_dataset = TensorDataset(X_tensor, y_tensor)
        eval_dataset = None

    # Chargement des données dans des DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE) if eval_dataset else None

    # Initialisation du modèle
    model = ModelClass(input_dim=X.shape[1]).to(device)
    DROPOUT = getattr(model, 'dropout', 'unknown')  # Défaut à 'unknown' si non défini
    MODEL_PATH = f"saved_models/saved_{args.model_name}_{current_date}_epochs{EPOCHS}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}_dropout{DROPOUT}.pth"

    # Chargement du modèle pré-entraîné si spécifié
    if args.model_file:
        MODEL_PATH = args.model_file
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Le fichier spécifié '{MODEL_PATH}' n'existe pas.")
        print(f"Chargement du modèle depuis {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Aucun modèle pré-entraîné spécifié. Début de l'entraînement à partir de zéro.")

    # Définition de la fonction de perte et de l'optimiseur
    criterion = nn.BCELoss()
    #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Variables pour suivre les historiques
    loss_history = []
    train_acc_history = []
    eval_acc_history = []

    # Boucle d'entraînement
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_preds.append(outputs.cpu().detach().numpy())
            all_labels.append(batch_y.cpu().detach().numpy())

        all_preds = (np.concatenate(all_preds, axis=0) > 0.5).astype(int)
        all_labels = np.concatenate(all_labels, axis=0)
        train_accuracy = accuracy_score(all_labels, all_preds)

        loss_history.append(total_loss / len(train_loader))
        train_acc_history.append(train_accuracy)

        if eval_loader:
            model.eval()
            eval_preds, eval_labels = [], []
            with torch.no_grad():
                for batch_X, batch_y in eval_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    eval_preds.append(outputs.cpu().numpy())
                    eval_labels.append(batch_y.cpu().numpy())
            eval_preds = (np.concatenate(eval_preds, axis=0) > 0.5).astype(int)
            eval_labels = np.concatenate(eval_labels, axis=0)
            eval_accuracy = accuracy_score(eval_labels, eval_preds)
            eval_acc_history.append(eval_accuracy)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_accuracy*100:.2f}%, Eval Acc: {eval_accuracy*100:.2f}%")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_accuracy*100:.2f}%")

    # Sauvegarde du modèle après entraînement
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modèle sauvegardé sous {MODEL_PATH}")

    # Visualisation des courbes d'entraînement
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss')
    plt.plot(train_acc_history, label='Train Accuracy')
    if eval_loader:
        plt.plot(eval_acc_history, label='Eval Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.title('Training Progress')
    plt.show()
