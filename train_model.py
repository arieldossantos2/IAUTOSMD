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

# --- SUGESTÃO 3: Helper de Conversão ATUALIZADO (lê caminho ou imagem) ---
def path_to_tensor(data, is_path=True):
    """Converte um caminho de arquivo de imagem ou um array de imagem em um tensor."""
    try:
        if is_path:
            if not data or not os.path.exists(data): 
                print(f"Aviso: Caminho da imagem não encontrado: {data}")
                return torch.zeros(3, 64, 64)
            img = cv2.imread(data)
        else:
            img = data # Assume que 'data' já é um array numpy (para inferência)

        if img is None: 
            return torch.zeros(3, 64, 64)
            
        img = cv2.resize(img, (64, 64))
        return torch.tensor(img).permute(2, 0, 1).float() / 255.0
    except Exception as e:
        print(f"Erro ao processar imagem: {e}")
        return torch.zeros(3, 64, 64)

# --- Dataset para Classificação (lê caminhos) ---
class InspectionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # --- SUGESTÃO 3: Usa path_to_tensor ---
        prod_img = path_to_tensor(s['produced_path'], is_path=True)
        label_class = torch.tensor(1.0 if s['label'] == 'GOOD' else 0.0, dtype=torch.float32)
        
        return prod_img, label_class

# --- Modelo CNN (Simplificado para Classificação) ---
class MultiTaskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Base convolucional compartilhada
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # Output: 64x8x8
            nn.Flatten(),
            nn.Linear(64*8*8, 512), nn.ReLU(),
            nn.Dropout(0.5) 
        )
        # Cabeça para classificação (OK/FAIL)
        self.classifier = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x):
        features = self.backbone(x)
        prob = self.classifier(features).squeeze()
        return prob

# --- Lógica de Treinamento ---
if __name__ == '__main__':
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # --- SUGESTÃO 3: Busca 'produced_path' em vez de 'produced_base64' ---
        cursor.execute("""
            SELECT produced_path, label FROM training_samples
            WHERE created_at > (NOW() - INTERVAL 30 DAY) 
              AND produced_path IS NOT NULL
            ORDER BY id DESC
            LIMIT 5000 
        """)
        samples = cursor.fetchall()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"Erro de DB ao buscar amostras: {err}")
        samples = []

    if len(samples) < 20: 
        print(f"Amostras insuficientes ({len(samples)}). Pelo menos 20 são necessárias. Pulando treinamento.")
    else:
        print(f"Encontradas {len(samples)} amostras. Iniciando treinamento...")
        
        # Filtra amostras onde o caminho é inválido
        valid_samples = [s for s in samples if s['produced_path'] and os.path.exists(s['produced_path'])]
        invalid_count = len(samples) - len(valid_samples)
        if invalid_count > 0:
            print(f"Aviso: {invalid_count} amostras foram puladas por terem caminhos de imagem inválidos.")

        if len(valid_samples) < 20:
             print(f"Amostras válidas insuficientes ({len(valid_samples)}). Pulando treinamento.")
        else:
            try:
                train_data, val_data = train_test_split(valid_samples, test_size=0.2, random_state=42, stratify=[s['label'] for s in valid_samples])
            except ValueError:
                 print("Não foi possível dividir os dados para validação (provavelmente poucas amostras de uma classe). Usando todas para treinar.")
                 train_data, val_data = valid_samples, [] # Treina com tudo

            train_loader = DataLoader(InspectionDataset(train_data), batch_size=16, shuffle=True)
            
            model = MultiTaskCNN()
            
            # Carrega o modelo existente para continuar o treinamento, se existir
            model_path = 'trained_model.pt'
            if os.path.exists(model_path):
                try:
                    model.load_state_dict(torch.load(model_path))
                    print("Carregando modelo existente para continuar o treinamento.")
                except Exception as e:
                    print(f"Não foi possível carregar modelo existente ({e}). Iniciando do zero.")

            criterion_class = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0005) # Taxa de aprendizado menor para fine-tuning

            best_val_loss = float('inf')
            epochs_no_improve = 0
            
            for epoch in range(20): 
                model.train()
                total_loss = 0
                for imgs, labels in train_loader:
                    if imgs.nelement() == 0: continue # Pula batchs vazios
                    pred_class = model(imgs)
                    loss = criterion_class(pred_class, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0

                # --- Etapa de Validação (se houver dados de validação) ---
                if val_data:
                    model.eval()
                    total_val_loss = 0
                    correct_val = 0
                    val_loader = DataLoader(InspectionDataset(val_data), batch_size=16)
                    
                    with torch.no_grad():
                        for imgs, labels in val_loader:
                            if imgs.nelement() == 0: continue
                            pred_class = model(imgs)
                            total_val_loss += criterion_class(pred_class, labels).item()
                            predicted = (pred_class > 0.5).float()
                            correct_val += (predicted == labels).sum().item()
                    
                    avg_val_loss = total_val_loss / len(val_loader)
                    val_accuracy = (correct_val / len(val_data)) * 100
                    
                    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(model.state_dict(), model_path)
                        print(f"✅ Novo melhor modelo salvo em {model_path} (Val Loss: {best_val_loss:.4f})")
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= 5:
                            print("Parada antecipada (Early Stopping).")
                            break
                else:
                    # Se não houver dados de validação, apenas salva o modelo final
                    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} (Sem dados de validação)")
                    torch.save(model.state_dict(), model_path)

            if not val_data:
                print(f"✅ Modelo salvo em {model_path} (treinamento sem validação concluído).")
            
            print("Treinamento concluído.")