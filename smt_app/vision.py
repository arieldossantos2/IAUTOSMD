import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from flask import current_app
from .db_helpers import cv2_to_base64 # Import local
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_model import SiameseNetwork, path_to_tensor, BASE_DIR 
import math
import json

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SIAMESE_MODEL = None
_SIAMESE_PATH = "siamese_model.pt"


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
    roi_g,
    template_img,
    template_mask,
    roi_p,
    expected_rotation,
    presence_threshold=0.35,
    ssim_threshold=0.60,
    color_threshold=0.15,
    polarity_box=None,
):
    """
    Faz matching baseado em template (pacote) e retorna um dicionário com:
      - status: "OK" ou "FAIL"
      - found_rotation
      - displacement
      - details: metrics (correlation_score, ssim, color_score, polarity_ssim, polarity_corr, etc.)
      - debug_data: infos extras pra debug

    OBS: ainda usamos principalmente 'presence_threshold' pra decisão.
    As demais métricas são pensadas como 'deltas' pra IA futura.
    """
    try:
        if roi_p is None or template_img is None or template_mask is None:
            return {
                'status': 'FAIL',
                'found_rotation': f'{int(expected_rotation) % 360}°',
                'displacement': {'x': 0, 'y': 0},
                'details': {
                    'message': 'Template ou máscara inválidos (None).',
                    'correlation_score': 0.0,
                    'ssim': None,
                    'color_similar': None,
                    'color_score': None,
                    'polarity_ssim': None,
                    'polarity_corr': None,
                },
                'debug_data': {}
            }

        PRESENCE_THRESHOLD = presence_threshold if presence_threshold is not None else 0.35

        # Garante tipo float32
        roi_p = roi_p.copy()
        template_img = template_img.copy()

        # Rotaciona template + máscara pra alinhar com a rotação esperada
        def safe_float(v, default):
            try:
                if v is None:
                    return default
                return float(v)
            except Exception:
                return default

        def _rotate_template_and_mask_for_inspection(template_img, template_mask, expected_rotation):
            # mesma implementação que você já tem
            # (deixa como está; só estou chamando aqui)
            angle = expected_rotation % 360
            if angle == 0:
                return template_img, template_mask
            (h, w) = template_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_template = cv2.warpAffine(template_img, M, (w, h))
            rotated_mask = cv2.warpAffine(template_mask, M, (w, h))
            return rotated_template, rotated_mask

        template_img_rot, template_mask_rot = _rotate_template_and_mask_for_inspection(
            template_img,
            template_mask,
            expected_rotation
        )

        h_t, w_t = template_img_rot.shape[:2]
        h_r, w_r = roi_p.shape[:2]

        if h_r < h_t or w_r < w_t:
            return {
                'status': 'FAIL',
                'found_rotation': f'{int(expected_rotation) % 360}°',
                'displacement': {'x': 0, 'y': 0},
                'details': {
                    'message': f'ROI menor que template (roi={w_r}x{h_r}, tpl={w_t}x{h_t})',
                    'correlation_score': 0.0,
                    'ssim': None,
                    'color_similar': None,
                    'color_score': None,
                    'polarity_ssim': None,
                    'polarity_corr': None,
                },
                'debug_data': {
                    'template_shape': (h_t, w_t),
                    'roi_shape': (h_r, w_r),
                }
            }

        # Matching SQDIFF_NORMED (0 = igual, 1 = muito diferente)
        res = cv2.matchTemplate(
            roi_p,
            template_img_rot,
            cv2.TM_SQDIFF_NORMED,
            mask=template_mask_rot
        )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if not math.isfinite(min_val):
            min_val = 1.0

        correlation_score = safe_float(1.0 - min_val, 0.0)
        correlation_score = max(0.0, min(1.0, correlation_score))

        # Inicializa métricas extras
        ssim_score = None
        color_score = None
        color_similar = None
        mean_abs_diff = None
        polarity_ssim = None
        polarity_corr = None

        # Calcula patch alinhado ao melhor match
        x_best, y_best = min_loc
        x_end = min(x_best + w_t, w_r)
        y_end = min(y_best + h_t, h_r)

        patch_roi = roi_p[y_best:y_end, x_best:x_end]
        patch_tpl = template_img_rot[0:(y_end - y_best), 0:(x_end - x_best)]
        target_b64 = cv2_to_base64(patch_tpl)
        found_b64 = cv2_to_base64(patch_roi)


        if patch_roi.shape[:2] == patch_tpl.shape[:2] and patch_roi.size > 0:
            roi_gray = cv2.cvtColor(patch_roi, cv2.COLOR_BGR2GRAY)
            tpl_gray = cv2.cvtColor(patch_tpl, cv2.COLOR_BGR2GRAY)

            try:
                ssim_score = float(ssim(tpl_gray, roi_gray))
            except Exception:
                ssim_score = None

            # Diferença média de intensidade normalizada [0,1]
            try:
                diff = np.abs(tpl_gray.astype('float32') - roi_gray.astype('float32')) / 255.0
                mean_abs_diff = float(np.mean(diff))
            except Exception:
                mean_abs_diff = None

            # Similaridade de cor média [0,1]
            try:
                diff_color = np.abs(
                    patch_tpl.astype('float32') - patch_roi.astype('float32')
                ) / 255.0
                color_score = float(1.0 - np.mean(diff_color))
                color_score = max(0.0, min(1.0, color_score))
                color_similar = color_score >= color_threshold
            except Exception:
                color_score = None
                color_similar = None

        # Métricas de polaridade (sub-ROI)
        if polarity_box and patch_roi is not None and patch_tpl is not None:
            try:
                if isinstance(polarity_box, str):
                    pol = json.loads(polarity_box)
                else:
                    pol = polarity_box
                px = int(pol.get('x', 0))
                py = int(pol.get('y', 0))
                pw = int(pol.get('width', 0))
                ph = int(pol.get('height', 0))

                if pw > 0 and ph > 0:
                    # coords relativos ao patch (corpo)
                    px_end = min(px + pw, patch_roi.shape[1])
                    py_end = min(py + ph, patch_roi.shape[0])

                    pol_roi = patch_roi[py:py_end, px:px_end]
                    pol_tpl = patch_tpl[py:py_end, px:px_end]

                    if pol_roi.shape[:2] == pol_tpl.shape[:2] and pol_roi.size > 0:
                        pol_roi_gray = cv2.cvtColor(pol_roi, cv2.COLOR_BGR2GRAY)
                        pol_tpl_gray = cv2.cvtColor(pol_tpl, cv2.COLOR_BGR2GRAY)

                        try:
                            polarity_ssim = float(ssim(pol_tpl_gray, pol_roi_gray))
                        except Exception:
                            polarity_ssim = None

                        try:
                            # SQDIFF_NORMED no patch de polaridade
                            res_pol = cv2.matchTemplate(
                                pol_roi_gray,
                                pol_tpl_gray,
                                cv2.TM_SQDIFF_NORMED
                            )
                            min_val_pol, _, _, _ = cv2.minMaxLoc(res_pol)
                            polarity_corr = safe_float(1.0 - min_val_pol, 0.0)
                            polarity_corr = max(0.0, min(1.0, polarity_corr))
                        except Exception:
                            polarity_corr = None
            except Exception:
                traceback.print_exc()

        # Debug image simples: desenha o best match
        debug_img = roi_p.copy()
        cv2.rectangle(debug_img, (x_best, y_best), (x_best + w_t, y_best + h_t), (0, 255, 0), 2)
        debug_b64 = cv2_to_base64(debug_img)

        if correlation_score < PRESENCE_THRESHOLD:
            return {
                'status': 'FAIL',
                'found_rotation': f'{int(expected_rotation) % 360}°',
                'displacement': {'x': 0, 'y': 0},
                'details': {
                    'message': f'Componente Ausente (Score: {correlation_score:.2f} < {PRESENCE_THRESHOLD})',
                    'correlation_score': correlation_score,
                    'ssim': ssim_score,
                    'mean_abs_diff': mean_abs_diff,
                    'color_similar': color_similar,
                    'color_score': color_score,
                    'polarity_ssim': polarity_ssim,
                    'polarity_corr': polarity_corr,
                },
            'debug_data': {
                    'min_val': safe_float(min_val, 1.0),
                    'template_shape': (h_t, w_t),
                    'roi_shape': (h_r, w_r),
                    'best_match_top_left': (int(x_best), int(y_best)),
                    'debug_img_b64': debug_b64,
                }
            }

        return {
            'status': 'OK',
            'found_rotation': f'{int(expected_rotation) % 360}°',
            'displacement': {'x': 0, 'y': 0},
            'details': {
                'message': 'OK',
                'correlation_score': correlation_score,
                'ssim': ssim_score,
                'mean_abs_diff': mean_abs_diff,
                'color_similar': color_similar,
                'color_score': color_score,
                'polarity_ssim': polarity_ssim,
                'polarity_corr': polarity_corr,
            },
            'debug_data': {
                'min_val': safe_float(min_val, 1.0),
                'template_shape': (h_t, w_t),
                'roi_shape': (h_r, w_r),
                'best_match_top_left': (int(x_best), int(y_best)),
                'debug_img_b64': debug_b64,
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
                'mean_abs_diff': None,
                'color_similar': None,
                'color_score': None,
                'polarity_ssim': None,
                'polarity_corr': None,
            },
            'debug_data': {}
        }


def _load_siamese_model():
    global _siamese_model
    if _siamese_model is not None:
        return _siamese_model

    model = SiameseNetwork().to(_device)
    model_path = os.path.join(BASE_DIR, "siamese_model.pt")

    if not os.path.exists(model_path):
        print(f"[predict_with_model] Modelo {model_path} não encontrado. Usando prob 0.5.")
        _siamese_model = None
        return None

    try:
        state = torch.load(model_path, map_location=_device)
        model.load_state_dict(state)
        model.eval()
        _siamese_model = model
        print(f"[predict_with_model] Modelo siamesa carregado de {model_path}")
    except Exception as e:
        print(f"[predict_with_model] Erro ao carregar modelo siamesa: {e}")
        _siamese_model = None

    return _siamese_model

def _get_siamese_model():
    global _SIAMESE_MODEL
    if _SIAMESE_MODEL is None:
        model = SiameseNetwork().to(_DEVICE)
        if os.path.exists(_SIAMESE_PATH):
            try:
                state_dict = torch.load(_SIAMESE_PATH, map_location=_DEVICE)
                model.load_state_dict(state_dict)
                print(f"[vision] Rede siamesa carregada de '{_SIAMESE_PATH}'.")
            except Exception as e:
                print(f"[vision] Erro ao carregar '{_SIAMESE_PATH}': {e}. Usando pesos aleatórios.")
        else:
            print(f"[vision] Arquivo '{_SIAMESE_PATH}' não encontrado. Usando pesos aleatórios.")
        model.eval()
        _SIAMESE_MODEL = model
    return _SIAMESE_MODEL


def predict_with_model(roi_p_bgr, roi_g_bgr=None):
    """
    Usa a rede siamesa para comparar ROI Golden x ROI Produzida.

    - roi_g_bgr: np.ndarray BGR da golden (obrigatório para inferência correta)
    - roi_p_bgr: np.ndarray BGR da produzida
    """
    model = _get_siamese_model()

    # Se não mandarem ROI golden, fallback tosco: usa a própria produzida
    if roi_g_bgr is None:
        roi_g_bgr = roi_p_bgr

    # Converte para tensores (3, H, W)
    g_tensor = path_to_tensor(roi_g_bgr, is_path=False)
    p_tensor = path_to_tensor(roi_p_bgr, is_path=False)

    g_tensor = g_tensor.unsqueeze(0).to(_DEVICE)  # (1, 3, H, W)
    p_tensor = p_tensor.unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        prob_good = model(g_tensor, p_tensor).item()

    prob_good = float(prob_good)
    threshold = 0.5   # pode ajustar depois (ex: 0.6 / 0.7)

    ai_status = "OK" if prob_good >= threshold else "FAIL"

    ai_details = {
        "prob": prob_good,
        "threshold": threshold,
        "source": "siamese_model.pt",
    }

    return ai_status, ai_details
