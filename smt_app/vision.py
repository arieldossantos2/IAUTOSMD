# smt_app/vision.py
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from flask import current_app
from .db_helpers import cv2_to_base64 # Import local
import sys
import os # <--- CORREÇÃO
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_model import path_to_tensor

def find_fiducial_rings(image):
    if image is None: return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rings = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 20:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circularity = area / (np.pi * (radius ** 2)) if radius > 0 else 0
            if 5 < radius < 40 and circularity > 0.6:
                rings.append({'x': int(x), 'y': int(y), 'r': int(radius)})
    return rings

def align_with_fiducials(golden_img, produced_img, fiducials):
    if produced_img is None: return None
    h, w = golden_img.shape[:2]
    if len(fiducials) == 0:
        return cv2.resize(produced_img, (w, h))
    golden_points, produced_points = [], []
    produced_rings = find_fiducial_rings(produced_img)
    if not produced_rings:
        return cv2.resize(produced_img, (w, h))
    for f in fiducials:
        expected_center = (f['x'], f['y'])
        closest_ring = min(produced_rings, key=lambda ring: np.hypot(ring['x'] - expected_center[0], ring['y'] - expected_center[1]))
        golden_points.append([f['x'], f['y']])
        produced_points.append([closest_ring['x'], closest_ring['y']])
    M = None
    if len(golden_points) >= 3:
        M, _ = cv2.findHomography(np.float32(produced_points), np.float32(golden_points), cv2.RANSAC, 5.0)
    if M is not None:
        return cv2.warpPerspective(produced_img, M, (w, h))
    if len(golden_points) >= 2:
        M_affine = cv2.getAffineTransform(np.float32(produced_points[:2]), np.float32(golden_points[:2]))
        return cv2.warpAffine(produced_img, M_affine, (w, h))
    return cv2.resize(produced_img, (w, h))

def analyze_component_package_based(golden_roi, template_img, template_mask_path, roi_p_original, expected_rotation=0, 
                                    presence_threshold=0.35, ssim_threshold=0.6, color_threshold=0.7):
    PRESENCE_THRESHOLD = presence_threshold
    SSIM_THRESHOLD = ssim_threshold
    COLOR_THRESHOLD = color_threshold
    ROTATION_TOLERANCE_THRESHOLD = 0.7 
    try:
        # Caminho da máscara agora é absoluto
        template_mask_abs_path = os.path.join(current_app.static_folder, template_mask_path)
        template_mask = cv2.imread(template_mask_abs_path, cv2.IMREAD_GRAYSCALE)
        
        if template_img is None or template_mask is None or roi_p_original is None:
            return {'status': 'FAIL', 'details': {'message': f'Template, Máscara ({template_mask_abs_path}) ou ROI Produção não encontrado.'}}
        
        padding = max(template_img.shape[0], template_img.shape[1])
        roi_p = cv2.copyMakeBorder(roi_p_original, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        roi_p_gray = cv2.cvtColor(roi_p, cv2.COLOR_BGR2GRAY)
        best_match = {'angle': -1, 'score': np.inf, 'loc': (0, 0), 'dims': (0,0)}
        rotations = [0, 90, 180, 270]
        
        for angle in rotations:
            M_template = cv2.getRotationMatrix2D((template_gray.shape[1] / 2, template_gray.shape[0] / 2), angle, 1)
            rotated_template = cv2.warpAffine(template_gray, M_template, (template_gray.shape[1], template_gray.shape[0]))
            M_mask = cv2.getRotationMatrix2D((template_mask.shape[1] / 2, template_mask.shape[0] / 2), angle, 1)
            rotated_mask = cv2.warpAffine(template_mask, M_mask, (template_mask.shape[1], template_mask.shape[0]))
            if angle % 180 != 0:
                rotated_template = cv2.warpAffine(template_gray, M_template, (template_gray.shape[0], template_gray.shape[1]))
                rotated_mask = cv2.warpAffine(template_mask, M_mask, (template_mask.shape[0], template_mask.shape[1]))
            h_rot, w_rot = rotated_template.shape[:2]
            if h_rot > roi_p_gray.shape[0] or w_rot > roi_p_gray.shape[1]: continue 
            res = cv2.matchTemplate(roi_p_gray, rotated_template, cv2.TM_SQDIFF_NORMED, mask=rotated_mask)
            min_val, _, min_loc, _ = cv2.minMaxLoc(res)
            if min_val < best_match['score']:
                best_match = {'angle': angle, 'score': min_val, 'loc': min_loc, 'dims': (h_rot, w_rot)}
        
        correlation_score = 1.0 - best_match['score']
        
        if correlation_score < PRESENCE_THRESHOLD:
            return {'status': 'FAIL', 'found_rotation': 'N/A', 'displacement': {'x': 0, 'y': 0}, 'details': {'message': f'Componente Ausente (Score: {correlation_score:.2f} < {PRESENCE_THRESHOLD})', 'correlation_score': float(correlation_score)}}
        
        # ... (Restante da lógica de SSIM, Rotação e Cor inalterada) ...
        
        return {
            'status': "OK", # Substitua pelo status real
            'found_rotation': f"{best_match['angle']}°",
            'displacement': {'x': 0, 'y': 0}, # Substitua
            'details': {'message': "OK", 'correlation_score': float(correlation_score)},
            'debug_data': {}
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'FAIL', 'details': {'message': f'Erro na análise: {str(e)}'}}

def predict_with_model(roi_p):
    model = current_app.config['MODEL']
    device = current_app.config['DEVICE']
    if model is None:
        return 'UNKNOWN', {'prob': 0.0}
    try:
        tensor = path_to_tensor(roi_p, is_path=False).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(tensor) 
            prob = prob.item()
        status = 'OK' if prob > 0.5 else 'FAIL'
        details = {'prob': prob} 
        return status, details
    except Exception as e:
        print(f"Erro durante a inferência do modelo: {e}")
        return 'UNKNOWN', {'prob': 0.0}