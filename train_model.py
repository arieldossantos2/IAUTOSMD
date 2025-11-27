import os
import sqlite3
import json
import cv2
import numpy as np
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================================================
# 1) Configuração básica
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mesmo caminho usado em init_sqlite.py (instance/smt_inspection_new.db)
DB_PATH = os.path.join(BASE_DIR, "instance", "smt_inspection_new.db")

# Candidatos para pasta static (onde ficam as imagens salvas via save_image_to_disk)
STATIC_CANDIDATES = [
    os.path.join(BASE_DIR, "smt_app", "static"),
    os.path.join(BASE_DIR, "static"),
]


def get_db_connection():
    """Abre conexão com o SQLite usado pelo app."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def resolve_image_path(path_from_db: str) -> str:
    """
    Converte o que está salvo no banco (relativo ou absoluto)
    em um caminho absoluto existente no disco.

    - Se já for absoluto e existir, retorna direto.
    - Se for relativo, tenta dentro das pastas static candidatas.
    - Se não achar, retorna None.
    """
    if not path_from_db:
        return None

    # Se já é absoluto e existe
    if os.path.isabs(path_from_db) and os.path.exists(path_from_db):
        return path_from_db

    # Testa como relativo a cada static
    for root in STATIC_CANDIDATES:
        candidate = os.path.join(root, path_from_db)
        if os.path.exists(candidate):
            return candidate

    # Última tentativa: relativo ao projeto
    candidate = os.path.join(BASE_DIR, path_from_db)
    if os.path.exists(candidate):
        return candidate

    return None


# ============================================================
# 2) Helper para transformar imagem em tensor
#    (usado também pelo vision.predict_with_model)
# ============================================================

def path_to_tensor(data, is_path=True, size=(64, 64)) -> torch.Tensor:
    """
    Converte:
      - um caminho de arquivo de imagem (is_path=True) OU
      - um array numpy BGR (is_path=False)

    em um tensor normalizado [0,1], shape (3, H, W).
    Em caso de erro, retorna um tensor zero para não quebrar o treino/inferência.
    """
    try:
        if is_path:
            img_path = data
            if not img_path or not os.path.exists(img_path):
                print(f"[path_to_tensor] Caminho de imagem não encontrado: {img_path}")
                return torch.zeros(3, size[1], size[0])
            img = cv2.imread(img_path)
        else:
            img = data

        if img is None:
            print("[path_to_tensor] Imagem None")
            return torch.zeros(3, size[1], size[0])

        img = cv2.resize(img, size)
        # BGR -> RGB (opcional, mas em geral melhora)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        return tensor
    except Exception as e:
        print(f"[path_to_tensor] Erro ao processar imagem: {e}")
        return torch.zeros(3, size[1], size[0])


# ============================================================
# 3) Dataset para REDE SIAMESA
#    - cada amostra = (golden_tensor, produced_tensor, label)
# ============================================================

class SiameseDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        """
        samples: lista de dicts com:
          {
            'golden_path': str,
            'produced_path': str,
            'label': float (1.0 = GOOD, 0.0 = FAIL),
            'is_polarized': int (0/1),
            'sample_type': str,
          }
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        g_tensor = path_to_tensor(s["golden_path"], is_path=True)
        p_tensor = path_to_tensor(s["produced_path"], is_path=True)

        y = torch.tensor(float(s["label"]), dtype=torch.float32)

        return g_tensor, p_tensor, y


# ============================================================
# 4) Arquitetura da Rede Siamesa
#    - backbone CNN compartilha pesos para Golden e Produced
#    - head recebe |f1 - f2| e gera prob de "iguais" (GOOD)
# ============================================================

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()

        # Mesma base convolucional do seu MultiTaskCNN (64x64 -> 64 x 8 x 8) :contentReference[oaicite:5]{index=5}
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU()
        )

        # Head siamesa: recebe |f1 - f2|
        self.siamese_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        """Extrai o embedding de UMA imagem."""
        feat = self.backbone(x)
        emb = self.embedding(feat)
        return emb

    def forward(self, x1, x2):
        """
        Forward siamesa padrão:
          x1 = golden
          x2 = produced
        Retorna probabilidade de serem "iguais" (GOOD).
        """
        f1 = self.forward_once(x1)
        f2 = self.forward_once(x2)

        diff = torch.abs(f1 - f2)
        prob = self.siamese_head(diff).squeeze()
        return prob


# ============================================================
# 5) Carregar amostras do banco de dados
# ============================================================

def load_siamese_samples_from_db(max_samples: int = 5000) -> List[Dict]:
    """
    Lê training_samples + components no SQLite e resolve caminhos das imagens.
    Dá um leve oversampling em componentes polarizados para reforçar polaridade.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Pegamos também info de componente (is_polarized, polarity_box)
    query = """
        SELECT
            ts.id,
            ts.golden_path,
            ts.produced_path,
            ts.label,
            ts.sample_type,
            c.is_polarized,
            c.polarity_box
        FROM training_samples ts
        JOIN components c ON c.id = ts.component_id
        WHERE ts.golden_path IS NOT NULL
          AND ts.produced_path IS NOT NULL
          AND ts.label IN ('GOOD', 'FAIL')
        ORDER BY ts.id DESC
        LIMIT ?
    """
    cur.execute(query, (max_samples,))
    rows = cur.fetchall()
    conn.close()

    samples = []
    missing = 0
    label_map = {"GOOD": 1.0, "FAIL": 0.0}

    for row in rows:
        g_rel = row["golden_path"]
        p_rel = row["produced_path"]
        label_str = row["label"]
        is_polarized = int(row["is_polarized"] or 0)
        sample_type = row["sample_type"] or "BODY"

        g_abs = resolve_image_path(g_rel)
        p_abs = resolve_image_path(p_rel)

        if not g_abs or not p_abs:
            missing += 1
            continue

        if label_str not in label_map:
            continue

        y = label_map[label_str]

        base_sample = {
            "golden_path": g_abs,
            "produced_path": p_abs,
            "label": y,
            "is_polarized": is_polarized,
            "sample_type": sample_type,
        }

        # Amostra original
        samples.append(base_sample)

        # Oversampling leve em componentes polarizados (aprende melhor polaridade)
        if is_polarized == 1:
            samples.append(base_sample.copy())

    print(f"[load_siamese_samples_from_db] Amostras carregadas: {len(samples)} (descartadas por caminho inválido: {missing})")
    return samples


# ============================================================
# 6) Loop de treinamento da rede siamesa
# ============================================================

def train_siamese(
    max_samples: int = 5000,
    batch_size: int = 16,
    num_epochs: int = 20,
    lr: float = 5e-4,
    min_samples_required: int = 30,
    model_path: str = "siamese_model.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_siamese] Usando device: {device}")

    samples = load_siamese_samples_from_db(max_samples=max_samples)

    if len(samples) < min_samples_required:
        print(f"[train_siamese] Amostras insuficientes ({len(samples)}). "
              f"São necessárias pelo menos {min_samples_required}. Abortando treinamento.")
        return

    # Checa o balanceamento das classes
    num_good = sum(1 for s in samples if s["label"] == 1.0)
    num_fail = sum(1 for s in samples if s["label"] == 0.0)
    print(f"[train_siamese] GOOD: {num_good}, FAIL: {num_fail}")

    # Se só tiver uma classe, treino não faz sentido
    if num_good == 0 or num_fail == 0:
        print("[train_siamese] Só existe uma classe em training_samples (apenas GOOD ou apenas FAIL). "
              "Treinamento da siamesa não será realizado.")
        return

    # Split train/val (se der erro de stratify, cai para split simples)
    labels_list = [s["label"] for s in samples]
    try:
        train_data, val_data = train_test_split(
            samples,
            test_size=0.2,
            random_state=42,
            stratify=labels_list,
        )
    except ValueError as e:
        print(f"[train_siamese] Stratify falhou ({e}). Fazendo split simples.")
        split_idx = int(0.8 * len(samples))
        train_data = samples[:split_idx]
        val_data = samples[split_idx:]

    print(f"[train_siamese] Train size: {len(train_data)}, Val size: {len(val_data)}")

    train_loader = DataLoader(SiameseDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SiameseDataset(val_data), batch_size=batch_size, shuffle=False)

    model = SiameseNetwork().to(device)

    # Se já existir um modelo treinado, tentamos continuar o treino (fine-tuning)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"[train_siamese] Modelo existente carregado de {model_path} (fine-tuning).")
        except Exception as e:
            print(f"[train_siamese] Não foi possível carregar modelo existente ({e}). Treinando do zero.")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = 5

    for epoch in range(1, num_epochs + 1):
        # ---------------------------
        # Fase de Treino
        # ---------------------------
        model.train()
        total_train_loss = 0.0

        for g_imgs, p_imgs, labels in train_loader:
            g_imgs = g_imgs.to(device)
            p_imgs = p_imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            probs = model(g_imgs, p_imgs)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # ---------------------------
        # Fase de Validação
        # ---------------------------
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for g_imgs, p_imgs, labels in val_loader:
                g_imgs = g_imgs.to(device)
                p_imgs = p_imgs.to(device)
                labels = labels.to(device)

                probs = model(g_imgs, p_imgs)
                loss = criterion(probs, labels)
                total_val_loss += loss.item()

                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_acc = 100.0 * correct / max(1, total)

        print(f"[Epoch {epoch}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")

        # Early stopping + melhor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"[train_siamese] ✅ Novo melhor modelo salvo em {model_path} "
                  f"(Val Loss: {best_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("[train_siamese] Early stopping ativado.")
                break

    print("[train_siamese] Treinamento concluído.")


# ============================================================
# 7) Ponto de entrada do script
#     (chamado pelo routes_inspect.save_feedback via subprocess)
# ============================================================

if __name__ == "__main__":
    print("--------------------------------------------------")
    print("[train_model] Iniciando treinamento da rede siamesa...")
    print(f"[train_model] DB: {DB_PATH}")
    train_siamese(
        max_samples=5000,
        batch_size=16,
        num_epochs=20,
        lr=5e-4,
        min_samples_required=30,
        model_path="siamese_model.pt",
    )
    print("[train_model] Fim do treinamento.")
    print("--------------------------------------------------")
