import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=128, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.LayerNorm(hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x.unsqueeze(1))
        x = self.fc(x.squeeze(1))
        return self.sigmoid(x)

class SimplifiedTransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads=2, hidden_dim=64, dropout=0.3):
        super(SimplifiedTransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)  # Une seule couche
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x.unsqueeze(1))
        x = self.fc(x.squeeze(1))
        return self.sigmoid(x)
    

class TransformerModelDropOut(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=128, num_layers=2, dropout=0.3):
        super(TransformerModelDropOut, self).__init__()  # Correction ici !
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.LayerNorm(hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x.unsqueeze(1))
        x = self.fc(x.squeeze(1))
        return self.sigmoid(x)