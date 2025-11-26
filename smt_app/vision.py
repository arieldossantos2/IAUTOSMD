import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from flask import current_app
from .db_helpers import cv2_to_base64 # Import local
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_model import path_to_tensor
import math

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
        closest_ring = min(
            produced_rings,
            key=lambda ring: np.hypot(
                ring['x'] - expected_center[0],
                ring['y'] - expected_center[1]
            )
        )
        golden_points.append([f['x'], f['y']])
        produced_points.append([closest_ring['x'], closest_ring['y']])
    M = None
    if len(golden_points) >= 3:
        M, _ = cv2.findHomography(
            np.float32(produced_points),
            np.float32(golden_points),
            cv2.RANSAC, 5.0
        )
    if M is not None:
        return cv2.warpPerspective(produced_img, M, (w, h))
    if len(golden_points) >= 2:
        M_affine = cv2.getAffineTransform(
            np.float32(produced_points[:2]),
            np.float32(golden_points[:2])
        )
        return cv2.warpAffine(produced_img, M_affine, (w, h))
    return cv2.resize(produced_img, (w, h))

def _rotate_template_and_mask_for_inspection(template_img, template_mask, rotation):
    """Gira a imagem do template e sua máscara pelo ângulo (0, 90, 180 ou 270 graus)."""
    rotation = int(rotation) % 360

    if rotation == 90:
        template_img = cv2.rotate(template_img, cv2.ROTATE_90_CLOCKWISE)
        template_mask = cv2.rotate(template_mask, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        template_img = cv2.rotate(template_img, cv2.ROTATE_180)
        template_mask = cv2.rotate(template_mask, cv2.ROTATE_180)
    elif rotation == 270:
        template_img = cv2.rotate(template_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        template_mask = cv2.rotate(template_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return template_img, template_mask


def analyze_component_package_based(
    golden_roi,
    template_img,
    template_mask_path,
    roi_p_original,
    expected_rotation=0,
    presence_threshold=0.35,
    ssim_threshold=0.6,
    color_threshold=0.7,
):
    """
    Template do pacote está em 0° (definido no cadastro).
    Aqui giramos template+máscara pela rotação do componente (expected_rotation)
    e fazemos matchTemplate mascarado dentro da ROI de produção.
    """
    PRESENCE_THRESHOLD = presence_threshold

    def safe_float(x, default=0.0):
        try:
            x = float(x)
            if math.isfinite(x):
                return x
            return default
        except Exception:
            return default

    try:
        template_mask_abs_path = os.path.join(current_app.static_folder, template_mask_path)
        template_mask = cv2.imread(template_mask_abs_path, cv2.IMREAD_GRAYSCALE)

        if template_img is None or template_mask is None or roi_p_original is None:
            return {
                'status': 'FAIL',
                'found_rotation': f'{int(expected_rotation) % 360}°',
                'displacement': {'x': 0, 'y': 0},
                'details': {
                    'message': f'Template, máscara ({template_mask_abs_path}) ou ROI Produção não encontrado.',
                    'correlation_score': 0.0,
                    'ssim': None,
                    'color_similar': None,
                    'color_score': None,
                }
            }

        # gira template+mask pela rotação do componente
        template_img_rot, template_mask_rot = _rotate_template_and_mask_for_inspection(
            template_img.copy(), template_mask.copy(), expected_rotation
        )

        roi_gray = cv2.cvtColor(roi_p_original, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_img_rot, cv2.COLOR_BGR2GRAY)

        h_t, w_t = template_gray.shape[:2]
        h_r, w_r = roi_gray.shape[:2]

        if h_t > h_r or w_t > w_r:
            return {
                'status': 'FAIL',
                'found_rotation': f'{int(expected_rotation) % 360}°',
                'displacement': {'x': 0, 'y': 0},
                'details': {
                    'message': (
                        f'Template maior que ROI na inspeção '
                        f'(ROI: {roi_gray.shape}, Template: {template_gray.shape})'
                    ),
                    'correlation_score': 0.0,
                    'ssim': None,
                    'color_similar': None,
                    'color_score': None,
                }
            }

        res = cv2.matchTemplate(
            roi_gray,
            template_gray,
            cv2.TM_SQDIFF_NORMED,
            mask=template_mask_rot
        )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if not math.isfinite(min_val):
            min_val = 1.0

        correlation_score = safe_float(1.0 - min_val, 0.0)
        correlation_score = max(0.0, min(1.0, correlation_score))

        if correlation_score < PRESENCE_THRESHOLD:
            return {
                'status': 'FAIL',
                'found_rotation': f'{int(expected_rotation) % 360}°',
                'displacement': {'x': 0, 'y': 0},
                'details': {
                    'message': f'Componente Ausente (Score: {correlation_score:.2f} < {PRESENCE_THRESHOLD})',
                    'correlation_score': correlation_score,
                    'ssim': None,
                    'color_similar': None,
                    'color_score': None,
                },
                'debug_data': {
                    'min_val': safe_float(min_val, 1.0),
                    'template_shape': (h_t, w_t),
                    'roi_shape': (h_r, w_r),
                }
            }

        return {
            'status': 'OK',
            'found_rotation': f'{int(expected_rotation) % 360}°',
            'displacement': {'x': 0, 'y': 0},
            'details': {
                'message': 'OK',
                'correlation_score': correlation_score,
                'ssim': None,
                'color_similar': None,
                'color_score': None,
            },
            'debug_data': {
                'min_val': safe_float(min_val, 1.0),
                'template_shape': (h_t, w_t),
                'roi_shape': (h_r, w_r),
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'status': 'FAIL',
            'found_rotation': f'{int(expected_rotation) % 360}°',
            'displacement': {'x': 0, 'y': 0},
            'details': {
                'message': f'Erro na análise: {str(e)}',
                'correlation_score': 0.0,
                'ssim': None,
                'color_similar': None,
                'color_score': None,
            }
        }


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
        status = 'OK' if prob >= 0.9 else 'FAIL'
        details = {'prob': prob}
        return status, details
    except Exception as e:
        print(f"Erro durante a inferência do modelo: {e}")
        return 'UNKNOWN', {'prob': 0.0}
