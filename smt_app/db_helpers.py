import sqlite3
import cv2
import base64
import numpy as np
import os
import uuid
import traceback
from flask import current_app  # Usado para acessar app.config


def get_db_connection():
    """Abre uma conexão com o banco SQLite configurado na aplicação.

    - Usa app.config['SQLITE_DB_PATH'] se existir.
    - Define row_factory=sqlite3.Row para permitir acesso aos campos por nome.
    """
    db_path = current_app.config.get(
        'SQLITE_DB_PATH',
        os.path.join(current_app.instance_path, 'smt_inspection_new.db')
    )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def cv2_to_base64(image):
    if image is None:
        return None
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_cv2_img(base64_string):
    if not base64_string:
        return None
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None


def save_image_to_disk(image, folder_key, filename_prefix):
    """Salva uma imagem no disco em uma pasta configurada em IMAGE_FOLDERS.

    - folder_key: chave do dicionário IMAGE_FOLDERS (packages/results/training/uploads)
    - filename_prefix: prefixo "humano" do arquivo (ex: nome do produto)
    """
    try:
        # Pasta absoluta onde a imagem deve ser salva
        folder_path = current_app.config['IMAGE_FOLDERS'].get(folder_key)
        if not folder_path:
            raise ValueError(f"Chave de pasta de imagem inválida: {folder_key}")

        # Garante que temos uma imagem válida
        if image is None:
            raise ValueError("Imagem recebida é None em save_image_to_disk.")

        # Monta nome de arquivo seguro
        safe_prefix = "".join(
            c for c in filename_prefix if c.isalnum() or c in ('_', '-')
        ).strip()
        filename = f"{safe_prefix}_{uuid.uuid4().hex[:8]}.png"

        # Caminho absoluto final no disco
        absolute_path = os.path.join(folder_path, filename)

        # Garante que a pasta existe
        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

        # Salva a imagem de fato (AGORA usando a imagem correta)
        success = cv2.imwrite(absolute_path, image)
        if not success:
            raise IOError(f"Falha ao salvar a imagem em {absolute_path}")

        # Caminho relativo à pasta static para salvar no banco
        # Ex: "images/uploads/arquivo.png"
        relative_path_for_db = os.path.relpath(
            absolute_path, current_app.static_folder
        ).replace("\\", "/")

        print(f"Imagem salva em: {absolute_path} (DB: {relative_path_for_db})")
        return relative_path_for_db

    except Exception:
        traceback.print_exc()
        return None
