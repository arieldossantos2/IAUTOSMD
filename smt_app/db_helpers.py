# smt_app/db_helpers.py
import mysql.connector
import cv2
import base64
import numpy as np
import os
import uuid
import traceback
from flask import current_app # Usado para acessar app.config

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="flaskuser",
        password="123456",
        database="smt_inspection_new"
    )

def cv2_to_base64(image):
    if image is None: return None
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_cv2_img(base64_string):
    if not base64_string: return None
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
    try:
        folder_path = current_app.config['IMAGE_FOLDERS'].get(folder_key)
        if not folder_path:
            raise ValueError(f"Chave de pasta de imagem inválida: {folder_key}")
        
        safe_prefix = "".join(c for c in filename_prefix if c.isalnum() or c in ('_', '-')).strip()
        filename = f"{safe_prefix}_{uuid.uuid4().hex[:8]}.png"
        
        # Usa app.config['IMAGE_STORAGE'] que é relativo a 'static'
        # e os caminhos salvos no DB não incluem 'static/'
        # A pasta de armazenamento real é baseada no caminho estático do app
        relative_path_for_db = os.path.join(current_app.config['IMAGE_STORAGE'], folder_key, filename).replace('static/', '')
        absolute_path = os.path.join(current_app.static_folder, relative_path_for_db)
        
        os.makedirs(os.path.dirname(absolute_path), exist_ok=True) 
        
        success = cv2.imwrite(absolute_path, image)
        if not success:
            raise IOError(f"Falha ao salvar a imagem em {absolute_path}")
            
        print(f"Imagem salva em: {relative_path_for_db}")
        # Retorna o caminho relativo a 'static/' para o DB
        return relative_path_for_db
    except Exception as e:
        traceback.print_exc()
        return None