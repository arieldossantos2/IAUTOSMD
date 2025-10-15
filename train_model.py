import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import base64
import mysql.connector
from sklearn.model_selection import train_test_split
import os

# --- Conexão com DB ---
def get_db_connection():
    return mysql.connector.connect(host="localhost", user="flaskuser", password="123456", database="smt_inspection_new")

# --- Helper de Conversão ---
def base64_to_tensor(b64):
    if not b64: return torch.zeros(3, 64, 64)
    img_bytes = base64.b64decode(b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None: return torch.zeros(3, 64, 64)
    img = cv2.resize(img, (64, 64))
    return torch.tensor(img).permute(2, 0, 1).float() / 255.0

# --- Dataset para Múltiplas Tarefas ---
class InspectionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prod_img = base64_to_tensor(s['produced_base64'])
        label_class = torch.tensor(1.0 if s['label'] == 'GOOD' else 0.0, dtype=torch.float32)
        
        # NOTA: Para treinar rotação e deslocamento, você precisaria ter esses
        # dados anotados no seu banco de dados. Por enquanto, usamos placeholders.
        label_rot = torch.tensor(0.0, dtype=torch.float32) # Placeholder
        label_disp = torch.tensor([0.0, 0.0], dtype=torch.float32) # Placeholder
        
        return prod_img, {'class': label_class, 'rot': label_rot, 'disp': label_disp}

# --- Modelo MultiTaskCNN (Novo) ---
class MultiTaskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Base convolucional compartilhada
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # Output: 64x8x8
            nn.Flatten(),
            nn.Linear(64*8*8, 512), nn.ReLU()
        )
        # Cabeça para classificação (OK/FAIL)
        self.classifier = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        # Cabeça para regressão da rotação (prevê um valor, ex: ângulo)
        self.regressor_rot = nn.Linear(512, 1)
        # Cabeça para regressão do deslocamento (prevê dx, dy)
        self.regressor_disp = nn.Linear(512, 2)

    def forward(self, x):
        features = self.backbone(x)
        prob = self.classifier(features).squeeze()
        rot = self.regressor_rot(features).squeeze()
        disp = self.regressor_disp(features).squeeze()
        return prob, rot, disp

# --- Lógica de Treinamento ---
if __name__ == '__main__':
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT produced_base64, label FROM training_samples")
    samples = cursor.fetchall()
    cursor.close()
    conn.close()

    if len(samples) < 10: # Aumenta o mínimo para um treinamento mais significativo
        print(f"Amostras insuficientes ({len(samples)}). Pelo menos 10 são necessárias. Pulando treinamento.")
    else:
        print(f"Encontradas {len(samples)} amostras. Iniciando treinamento...")
        
        train_data, val_data = train_test_split(samples, test_size=0.2, random_state=42)
        train_loader = DataLoader(InspectionDataset(train_data), batch_size=16, shuffle=True)
        val_loader = DataLoader(InspectionDataset(val_data), batch_size=16)

        model = MultiTaskCNN()
        # Funções de perda para cada tarefa
        criterion_class = nn.BCELoss() # Para OK/FAIL
        criterion_reg = nn.MSELoss()   # Para rotação/deslocamento
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10): # Mais épocas para um modelo mais complexo
            model.train()
            total_loss = 0
            for imgs, labels in train_loader:
                pred_class, pred_rot, pred_disp = model(imgs)
                
                # Calcula a perda para cada tarefa
                loss_class = criterion_class(pred_class, labels['class'])
                
                # NOTA: O treinamento de regressão só funcionará quando você tiver
                # dados reais de rotação/deslocamento. Por enquanto, a perda será 0.
                loss_rot = criterion_reg(pred_rot, labels['rot'])
                loss_disp = criterion_reg(pred_disp, labels['disp'])
                
                # Combina as perdas (pode-se adicionar pesos aqui, ex: 1.0 * loss_class)
                loss = loss_class + loss_rot + loss_disp
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

        torch.save(model.state_dict(), 'trained_model.pt')
        print("✅ Modelo salvo em trained_model.pt")