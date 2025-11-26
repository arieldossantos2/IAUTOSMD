# smt_app/models.py
from flask_login import UserMixin
import torch
import torch.nn as nn

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

# Copiado de train_model.py (ou app.py original) para referência
class MultiTaskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # Output: 64x8x8
            nn.Flatten(),
            nn.Linear(64*8*8, 512), nn.ReLU(),
            nn.Dropout(0.5) 
        )
        self.classifier = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x):
        features = self.backbone(x)
        prob = self.classifier(features).squeeze()
        return prob