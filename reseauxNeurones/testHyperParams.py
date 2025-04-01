import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from data_processing import load_data
import models

# Paramètres
dataset_path = "../../data/classification/train.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 50

# Parser pour récupérer le nom du modèle
parser = argparse.ArgumentParser(description="Test de différents hyperparamètres.")
parser.add_argument("model_name", type=str, help="Nom du modèle à utiliser (défini dans models.py)")
parser.add_argument("--learning_rates", nargs='+', type=float, default=[0.001, 0.01, 0.1], help="Liste des taux d'apprentissage à tester")
parser.add_argument("--activations", nargs='+', type=str, default=["ReLU", "Sigmoid", "Tanh"], help="Liste des fonctions d'activation à tester")
args = parser.parse_args()

if not hasattr(models, args.model_name):
    raise ValueError(f"Modèle '{args.model_name}' non trouvé dans models.py")
ModelClass = getattr(models, args.model_name)

# Chargement des données
X, y = load_data(dataset_path)
X_train_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Test des taux d'apprentissage
print("\nTest des taux d'apprentissage:")
for lr in args.learning_rates:
    model = ModelClass(X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"LR {lr} - Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# Test des fonctions d'activation
print("\nTest des fonctions d'activation:")
activation_functions = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh()
}

for act_name in args.activations:
    if act_name not in activation_functions:
        print(f"Fonction d'activation inconnue : {act_name}, ignorée.")
        continue
    model = ModelClass(X.shape[1], activation=activation_functions[act_name]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Activation {act_name} - Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")
