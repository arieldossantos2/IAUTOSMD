# app_new.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
import numpy as np
import os
import mysql.connector
import json
import base64
from skimage.metrics import structural_similarity as ssim
from werkzeug.security import generate_password_hash, check_password_hash
import io
import traceback
import uuid
import torch
import sys
import subprocess
from train_model import MultiTaskCNN, path_to_tensor 

# --- Configuração, Modelo, Login, Helpers (Sem Alterações) ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DEBUG_FOLDER'] = 'static/debug'
app.config['SECRET_KEY'] = 'uma_chave_se_muito_segura'
app.config['IMAGE_STORAGE'] = 'static/images'
app.config['IMAGE_FOLDERS'] = {
    'packages': os.path.join(app.config['IMAGE_STORAGE'], 'packages'),
    'results': os.path.join(app.config['IMAGE_STORAGE'], 'results'),
    'training': os.path.join(app.config['IMAGE_STORAGE'], 'training')
}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DEBUG_FOLDER'], exist_ok=True)
for folder in app.config['IMAGE_FOLDERS'].values():
    os.makedirs(folder, exist_ok=True)

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_trained_model():
    global model
    model_path = 'trained_model.pt'
    model = MultiTaskCNN() 
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Modelo treinado '{model_path}' carregado com sucesso.")
        except Exception as e:
            print(f"⚠️ Erro ao carregar o modelo: {e}. Usando modelo não treinado.")
    else:
        print(f"Aviso: Arquivo de modelo '{model_path}' não encontrado. Usando modelo não treinado.")
    model.to(device)
    model.eval()
load_trained_model()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username
@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, username FROM users WHERE id = %s", (user_id,))
    user_data = cursor.fetchone()
    cursor.close()
    conn.close()
    if user_data:
        return User(user_data['id'], user_data['username'])
    return None

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
        folder_path = app.config['IMAGE_FOLDERS'].get(folder_key)
        if not folder_path:
            raise ValueError(f"Chave de pasta de imagem inválida: {folder_key}")
        safe_prefix = "".join(c for c in filename_prefix if c.isalnum() or c in ('_', '-')).strip()
        filename = f"{safe_prefix}_{uuid.uuid4().hex[:8]}.png"
        relative_path = os.path.join(folder_path, filename)
        os.makedirs(folder_path, exist_ok=True) 
        success = cv2.imwrite(relative_path, image)
        if not success:
            raise IOError(f"Falha ao salvar a imagem em {relative_path}")
        print(f"Imagem salva em: {relative_path}")
        return relative_path 
    except Exception as e:
        traceback.print_exc()
        return None

# --- Funções de Visão Computacional (find_fiducial_rings, align_with_fiducials, analyze_component_package_based) (Sem Alterações) ---
def find_fiducial_rings(image):
    # ... (código inalterado)
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
    # ... (código inalterado)
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
    # ... (código inalterado)
    PRESENCE_THRESHOLD = presence_threshold
    SSIM_THRESHOLD = ssim_threshold
    COLOR_THRESHOLD = color_threshold
    ROTATION_TOLERANCE_THRESHOLD = 0.7 
    try:
        template_mask = cv2.imread(template_mask_path, cv2.IMREAD_GRAYSCALE)
        if template_img is None or template_mask is None or roi_p_original is None:
            return {'status': 'FAIL', 'details': {'message': 'Template, Máscara ou ROI Produção não encontrado.'}}
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
        found_rotation = best_match['angle']
        h_found, w_found = best_match['dims']
        x_found, y_found = best_match['loc']
        if found_rotation != expected_rotation:
            M_expected = cv2.getRotationMatrix2D((template_gray.shape[1] / 2, template_gray.shape[0] / 2), expected_rotation, 1)
            expected_template = cv2.warpAffine(template_gray, M_expected, (template_gray.shape[1], template_gray.shape[0]))
            M_mask_expected = cv2.getRotationMatrix2D((template_mask.shape[1] / 2, template_mask.shape[0] / 2), expected_rotation, 1)
            expected_mask = cv2.warpAffine(template_mask, M_mask_expected, (template_mask.shape[1], template_mask.shape[0]))
            if expected_rotation % 180 != 0:
                expected_template = cv2.warpAffine(template_gray, M_expected, (template_gray.shape[0], template_gray.shape[1]))
                expected_mask = cv2.warpAffine(template_mask, M_mask_expected, (template_mask.shape[0], template_mask.shape[1]))
            if expected_template.shape[0] <= roi_p_gray.shape[0] and expected_template.shape[1] <= roi_p_gray.shape[1]:
                res_expected = cv2.matchTemplate(roi_p_gray, expected_template, cv2.TM_SQDIFF_NORMED, mask=expected_mask)
                expected_score, _, _, _ = cv2.minMaxLoc(res_expected)
                if best_match['score'] < (expected_score * (1.0 - (1.0 - ROTATION_TOLERANCE_THRESHOLD))):
                     return {'status': 'FAIL', 'found_rotation': f"{found_rotation}°", 'displacement': {'x': x_found - padding, 'y': y_found - padding}, 'details': {'message': f"Rotação Incorreta (Esperado: {expected_rotation}°, Encontrado: {found_rotation}°)", 'correlation_score': float(correlation_score)}}
            else:
                 print(f"Aviso: Template esperado ({expected_template.shape}) maior que ROI ({roi_p_gray.shape}). Pulando verificação de rotação.")
        found_comp_p = roi_p[y_found:y_found + h_found, x_found:x_found + w_found]
        M_color = cv2.getRotationMatrix2D((template_img.shape[1] / 2, template_img.shape[0] / 2), found_rotation, 1.0)
        target_for_ssim = cv2.warpAffine(template_img, M_color, (template_img.shape[1], template_img.shape[0]))
        if found_rotation % 180 != 0:
             target_for_ssim = cv2.warpAffine(template_img, M_color, (template_img.shape[0], template_img.shape[1]))
        if found_comp_p.shape != target_for_ssim.shape:
            found_comp_p = cv2.resize(found_comp_p, (target_for_ssim.shape[1], target_for_ssim.shape[0]))
        ssim_value = 0.0
        try:
            min_side = min(target_for_ssim.shape[0], target_for_ssim.shape[1])
            win_size = min(7, min_side)
            if win_size % 2 == 0: win_size -= 1
            if win_size >= 3:
                ssim_value, _ = ssim(target_for_ssim, found_comp_p, full=True, multichannel=True, win_size=win_size, data_range=255, channel_axis=2)
            else: 
                ssim_value = 1.0 
        except ValueError:
            try:
                target_gray_ssim = cv2.cvtColor(target_for_ssim, cv2.COLOR_BGR2GRAY)
                found_gray_ssim = cv2.cvtColor(found_comp_p, cv2.COLOR_BGR2GRAY)
                ssim_value, _ = ssim(target_gray_ssim, found_gray_ssim, full=True, data_range=255)
            except Exception as e_ssim:
                print(f"Erro SSIM: {e_ssim}")
                ssim_value = 0.0
        status = "OK" if ssim_value > SSIM_THRESHOLD else "FAIL"
        message = "OK" if status == "OK" else f"Baixa Similaridade (SSIM: {ssim_value:.2f} < {SSIM_THRESHOLD})"
        details = {'correlation_score': float(correlation_score), 'ssim': float(ssim_value)}
        if status == "OK":
            try:
                target_hsv = cv2.cvtColor(target_for_ssim, cv2.COLOR_BGR2HSV)
                found_hsv = cv2.cvtColor(found_comp_p, cv2.COLOR_BGR2HSV)
                hist_g = cv2.calcHist([target_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist_g, hist_g, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_p = cv2.calcHist([found_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist_p, hist_p, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                color_similarity = cv2.compareHist(hist_g, hist_p, cv2.HISTCMP_CORREL)
                details['color_similarity'] = float(color_similarity)
                if color_similarity < COLOR_THRESHOLD:
                    status = "FAIL"
                    message = f"Falha na Cor (Similaridade: {color_similarity:.2f} < {COLOR_THRESHOLD})"
            except Exception as e_color:
                print(f"Erro na análise de cor: {e_color}")
                details['color_similarity'] = 0.0
        debug_img = roi_p.copy()
        cv2.rectangle(debug_img, (x_found, y_found), (x_found + w_found, y_found + h_found), (0, 255, 0), 2)
        return {
            'status': status,
            'found_rotation': f"{found_rotation}°",
            'displacement': {'x': x_found - padding, 'y': y_found - padding},
            'details': {'message': message, **details},
            'debug_data': {
                'debug_img_b64': cv2_to_base64(debug_img),
                'target_ssim_b64': cv2_to_base64(target_for_ssim), 
                'found_comp_b64': cv2_to_base64(found_comp_p)    
            }
        }
    except Exception as e:
        traceback.print_exc()
        return {'status': 'FAIL', 'details': {'message': f'Erro na análise: {str(e)}'}}

def predict_with_model(roi_p):
    # ... (código inalterado)
    global model
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

# --- Rotas da Aplicação (home, login, register, logout, find_fiducials_route) (Sem Alterações) ---
@app.route('/')
@login_required
def home():
    # ... (código inalterado)
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM products")
    products = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('index.html', products=products)
@app.route('/login', methods=['GET', 'POST'])
def login():
    # ... (código inalterado)
    if current_user.is_authenticated: return redirect(url_for('home'))
    if request.method == 'POST':
        username, password = request.form.get('username'), request.form.get('password')
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, username, password_hash FROM users WHERE username = %s", (username,))
        user_data = cursor.fetchone()
        cursor.close()
        conn.close()
        if user_data and check_password_hash(user_data['password_hash'], password):
            user = User(user_data['id'], user_data['username'])
            login_user(user)
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Usuário ou senha inválidos.")
    return render_template('login.html')
@app.route('/register', methods=['GET', 'POST'])
def register():
    # ... (código inalterado)
    if current_user.is_authenticated: return redirect(url_for('home'))
    if request.method == 'POST':
        username, password = request.form.get('username'), request.form.get('password')
        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed_password))
            conn.commit()
            return redirect(url_for('login'))
        except mysql.connector.Error as err:
            error = "Nome de usuário já existe." if err.errno == 1062 else f"Erro: {err}"
            return render_template('register.html', error=error)
        finally:
            cursor.close()
            conn.close()
    return render_template('register.html')
@app.route('/logout')
@login_required
def logout():
    # ... (código inalterado)
    logout_user()
    return redirect(url_for('login'))
@app.route('/find_fiducials', methods=['POST'])
@login_required
def find_fiducials_route():
    # ... (código inalterado)
    try:
        data = request.get_json()
        base64_image = data['image_data'].split(',')[1]
        img_bytes = base64.b64decode(base64_image)
        image_cv = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_cv is None: return jsonify({'error': 'Falha ao decodificar a imagem.'}), 400
        detected_rings = find_fiducial_rings(image_cv)
        debug_image = image_cv.copy()
        if detected_rings:
            for ring in detected_rings:
                cv2.circle(debug_image, (ring['x'], ring['y']), ring['r'], (0, 255, 0), 2)
                cv2.circle(debug_image, (ring['x'], ring['y']), 2, (0, 0, 255), 3)
        return jsonify({'circles': detected_rings, 'debug_image': cv2_to_base64(debug_image)})
    except Exception as e:
        return jsonify({'error': f'Erro ao encontrar fiduciais: {str(e)}'}), 500

# --- suggest_body, preview_template (Sem Alterações) ---
@app.route('/suggest_body', methods=['POST'])
@login_required
def suggest_body():
    # ... (código inalterado)
    try:
        data = request.get_json()
        roi_b64 = data.get('component_roi_b64')
        roi_img = base64_to_cv2_img(roi_b64)
        if roi_img is None:
            return jsonify({'error': 'Imagem ROI inválida.'}), 400
        (h, w) = roi_img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return jsonify({'x': int(w*0.1), 'y': int(h*0.1), 'width': int(w*0.8), 'height': int(h*0.8)})
        min_dist = float('inf')
        best_contour = None
        for c in contours:
            if cv2.contourArea(c) < 5: continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            mX = int(M["m10"] / M["m00"])
            mY = int(M["m01"] / M["m00"])
            dist = ((mX - cX)**2) + ((mY - cY)**2)
            if dist < min_dist:
                min_dist = dist
                best_contour = c
        if best_contour is None:
             return jsonify({'x': int(w*0.1), 'y': int(h*0.1), 'width': int(w*0.8), 'height': int(h*0.8)})
        x, y, w, h = cv2.boundingRect(best_contour)
        return jsonify({'x': x, 'y': y, 'width': w, 'height': h})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
@app.route('/preview_template', methods=['POST'])
@login_required
def preview_template():
    # ... (código inalterado)
    try:
        data = request.get_json()
        roi_g_b64 = data.get('component_roi_b64')
        body_rect = data.get('body_rect')
        roi_g = base64_to_cv2_img(roi_g_b64)
        if roi_g is None:
            return jsonify({'error': 'Imagem ROI inválida.'}), 400
        if not body_rect:
             return jsonify({'error': 'Definição do corpo (body_rect) ausente.'}), 400
        x, y, w, h = body_rect['x'], body_rect['y'], body_rect['width'], body_rect['height']
        body_template_img = roi_g[y:y+h, x:x+w]
        if body_template_img.size == 0:
            return jsonify({'error': 'Região de corpo (verde) inválida.'}), 400
        gray_template = cv2.cvtColor(body_template_img, cv2.COLOR_BGR2GRAY)
        blurred_template = cv2.GaussianBlur(gray_template, (5, 5), 0)
        _, body_mask = cv2.threshold(blurred_template, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        temp_mask_path = os.path.join(app.config['DEBUG_FOLDER'], f"temp_mask_{uuid.uuid4().hex}.png")
        cv2.imwrite(temp_mask_path, body_mask)
        cv_analysis = analyze_component_package_based(
            golden_roi=roi_g, 
            template_img=body_template_img, 
            template_mask_path=temp_mask_path,
            roi_p_original=roi_g, 
            expected_rotation=0
        )
        os.remove(temp_mask_path)
        return jsonify({
            'success': True,
            'status': cv_analysis.get('status'),
            'message': cv_analysis.get('details', {}).get('message'),
            'debug_img_b64': cv_analysis.get('debug_data', {}).get('debug_img_b64')
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# --- Rota /find_body_in_roi (Sem Alterações) ---
@app.route('/find_body_in_roi', methods=['POST'])
@login_required
def find_body_in_roi():
    # ... (código inalterado)
    conn = None
    try:
        data = request.get_json()
        roi_b64 = data.get('component_roi_b64')
        package_name = data.get('package_name')
        if not roi_b64 or not package_name:
            return jsonify({'error': 'ROI ou nome do pacote ausente.'}), 400
        roi_img = base64_to_cv2_img(roi_b64)
        if roi_img is None:
            return jsonify({'error': 'Imagem ROI inválida.'}), 400
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT body_matrix, body_mask FROM packages WHERE name = %s", (package_name,))
        pkg_data = cursor.fetchone()
        cursor.close()
        conn.close()
        if not pkg_data or not pkg_data['body_matrix'] or not pkg_data['body_mask']:
            return jsonify({'error': 'Template ou máscara não encontrados para este pacote.'}), 404
        template_img = cv2.imread(pkg_data['body_matrix'])
        template_mask = cv2.imread(pkg_data['body_mask'], cv2.IMREAD_GRAYSCALE)
        if template_img is None or template_mask is None:
            return jsonify({'error': 'Falha ao ler arquivos de template/máscara do disco.'}), 500
        roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        if template_gray.shape[0] > roi_gray.shape[0] or template_gray.shape[1] > roi_gray.shape[1]:
             return jsonify({'error': 'A área desenhada é menor que o template. Desenhe uma caixa maior ao redor do componente.'}), 400
        res = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_SQDIFF_NORMED, mask=template_mask)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        (h_body, w_body) = template_gray.shape[:2]
        (x_body, y_body) = min_loc
        return jsonify({
            'body_rect': {
                'x': x_body,
                'y': y_body,
                'width': w_body,
                'height': h_body
            },
            'template_b64': cv2_to_base64(template_img)
        })
    except Exception as e:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- NOVA ROTA (CORREÇÃO): /find_body_in_roi_with_template ---
@app.route('/find_body_in_roi_with_template', methods=['POST'])
@login_required
def find_body_in_roi_with_template():
    """Recebe uma ROI bruta e um template (como ROI+Rect), e encontra o corpo.
       Usado quando o template ainda está no JS e não no DB."""
    try:
        data = request.get_json()
        roi_b64 = data.get('component_roi_b64')
        template_roi_b64 = data.get('template_roi_b64')
        template_body_rect = data.get('template_body_rect')

        if not all([roi_b64, template_roi_b64, template_body_rect]):
            return jsonify({'error': 'Dados de template ou ROI ausentes.'}), 400

        # 1. Decodifica a ROI da busca (imagem maior)
        roi_img = base64_to_cv2_img(roi_b64)
        if roi_img is None:
            return jsonify({'error': 'Imagem ROI (busca) inválida.'}), 400

        # 2. Cria o template e a máscara a partir dos dados do "primeiro componente"
        template_roi_img = base64_to_cv2_img(template_roi_b64)
        if template_roi_img is None:
            return jsonify({'error': 'Imagem ROI (template) inválida.'}), 400

        rect = template_body_rect
        x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
        template_img = template_roi_img[y:y+h, x:x+w]
        if template_img.size == 0:
            return jsonify({'error': 'Rect do corpo (template) inválido.'}), 400

        # Cria a máscara (lógica de add_product)
        gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        blurred_template = cv2.GaussianBlur(gray_template, (5, 5), 0)
        _, template_mask = cv2.threshold(blurred_template, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. Executa o match (lógica de find_body_in_roi)
        roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        if gray_template.shape[0] > roi_gray.shape[0] or gray_template.shape[1] > roi_gray.shape[1]:
             return jsonify({'error': 'A área desenhada é menor que o template. Desenhe uma caixa maior ao redor do componente.'}), 400

        res = cv2.matchTemplate(roi_gray, gray_template, cv2.TM_SQDIFF_NORMED, mask=template_mask)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        (h_body, w_body) = gray_template.shape[:2]
        (x_body, y_body) = min_loc

        return jsonify({
            'body_rect': {
                'x': x_body,
                'y': y_body,
                'width': w_body,
                'height': h_body
            },
            'template_b64': cv2_to_base64(template_img) # Retorna o template que foi usado
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
# --- FIM DA NOVA ROTA ---


# --- add_product (Sem Alterações) ---
@app.route('/add_product', methods=['GET', 'POST'])
@login_required
def add_product():
    # ... (código inalterado)
    if request.method == 'POST':
        conn, cursor = None, None
        try:
            name = request.form.get('name')
            golden_file = request.files.get('golden')
            fiducials_json = request.form.get('fiducials')
            components_json = request.form.get('components')
            if not all([name, golden_file, fiducials_json, components_json]):
                return jsonify({'error': 'Dados incompletos.'}), 400
            golden_path = os.path.join(app.config['UPLOAD_FOLDER'], 'golden_' + str(uuid.uuid4()) + '_' + golden_file.filename)
            golden_file.save(golden_path)
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True, buffered=True)
            cursor.execute("INSERT INTO products (name, golden_image, fiducials) VALUES (%s, %s, %s)", (name, golden_path, fiducials_json))
            product_id = cursor.lastrowid
            components = json.loads(components_json)
            package_info = {} 
            for comp in components:
                package_name = comp['package']
                if package_name not in package_info:
                    cursor.execute("SELECT id, body_matrix, body_mask FROM packages WHERE name = %s", (package_name,))
                    pkg_data = cursor.fetchone()
                    if not pkg_data:
                        cursor.execute("INSERT INTO packages (name, presence_threshold, ssim_threshold) VALUES (%s, %s, %s)", 
                                       (package_name, 0.35, 0.6))
                        package_id = cursor.lastrowid
                        package_info[package_name] = {'id': package_id, 'has_matrix': False}
                    else:
                        package_info[package_name] = {'id': pkg_data['id'], 'has_matrix': bool(pkg_data['body_matrix'])}
                package_id = package_info[package_name]['id']
                if not package_info[package_name]['has_matrix']:
                    if 'component_roi_b64' not in comp or 'final_body_rect' not in comp:
                        conn.rollback() 
                        return jsonify({'error': f"Componente '{comp['name']}' é o primeiro do pacote '{package_name}', mas os dados de definição do corpo (ROI e Rect) não foram enviados pelo frontend. Por favor, defina o corpo."}), 400
                    roi_g_img = base64_to_cv2_img(comp['component_roi_b64'])
                    rect = comp['final_body_rect']
                    if roi_g_img is None:
                         conn.rollback()
                         return jsonify({'error': f"Falha ao decodificar a ROI para '{comp['name']}'."}), 400
                    x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
                    body_template_img = roi_g_img[y:y+h, x:x+w]
                    if body_template_img.size == 0:
                         conn.rollback()
                         return jsonify({'error': f"Região de corpo (body_rect) inválida para '{comp['name']}'."}), 400
                    body_matrix_path = save_image_to_disk(body_template_img, 'packages', f"pkg_{package_id}_{package_name}_template")
                    gray_template = cv2.cvtColor(body_template_img, cv2.COLOR_BGR2GRAY)
                    blurred_template = cv2.GaussianBlur(gray_template, (5, 5), 0)
                    _, body_mask_img = cv2.threshold(blurred_template, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    body_mask_path = save_image_to_disk(body_mask_img, 'packages', f"pkg_{package_id}_{package_name}_mask")
                    if not body_matrix_path or not body_mask_path:
                        conn.rollback()
                        return jsonify({'error': f"Falha ao salvar arquivos de template/máscara para o pacote '{package_name}'."}), 500
                    cursor.execute("UPDATE packages SET body_matrix = %s, body_mask = %s WHERE id = %s", 
                                   (body_matrix_path, body_mask_path, package_id))
                    package_info[package_name]['has_matrix'] = True
                inspection_mask_base64 = None
                if 'inspection_regions' in comp and comp['inspection_regions']:
                    mask = np.zeros((comp['height'], comp['width']), dtype=np.uint8)
                    for region in comp['inspection_regions']:
                        rx, ry, rw, rh = region['x'], region['y'], region['width'], region['height']
                        rx1, ry1 = max(0, rx), max(0, ry)
                        rx2, ry2 = min(comp['width'], rx + rw), min(comp['height'], ry + rh)
                        if rx2 > rx1 and ry2 > ry1:
                            cv2.rectangle(mask, (rx1, ry1), (rx2, ry2), 255, -1)
                    _, buffer = cv2.imencode('.png', mask)
                    inspection_mask_base64 = base64.b64encode(buffer).decode('utf-8')
                cursor.execute(
                    """INSERT INTO components (product_id, name, x, y, width, height, package_id, rotation, inspection_mask)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (product_id, comp['name'], comp['x'], comp['y'], comp['width'], comp['height'],
                     package_id, comp.get('rotation', 0), inspection_mask_base64)
                )
            conn.commit()
            return jsonify({'success': True, 'product_id': product_id})
        except mysql.connector.Error as db_err:
            if conn: conn.rollback()
            traceback.print_exc()
            return jsonify({'error': f'Erro de Banco de Dados: {db_err}'}), 500
        except Exception as e:
            if conn: conn.rollback()
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
        finally:
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()
    
    # GET request
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name, presence_threshold, ssim_threshold, body_matrix FROM packages ORDER BY name")
    packages = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('add_product.html', packages=packages)

# --- inspect_board, feedback, main (Sem Alterações) ---
@app.route('/inspect', methods=['GET','POST'])
@login_required
def inspect_board():
    # ... (código inalterado)
    if request.method == 'POST':
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        produced_file = request.files.get('produced')
        product_id = request.form.get('product_id')
        if not produced_file or not product_id:
             return jsonify({'error': 'Produto ou imagem faltando.'}), 400
        produced_full = cv2.imdecode(np.frombuffer(produced_file.read(), np.uint8), cv2.IMREAD_COLOR)
        cursor.execute("SELECT golden_image, fiducials FROM products WHERE id=%s",(product_id,))
        product = cursor.fetchone()
        if not product:
            return jsonify({'error': 'Produto não encontrado.'}), 404
        golden_full = cv2.imread(product['golden_image'])
        if golden_full is None:
             return jsonify({'error': 'Imagem Golden não encontrada no servidor.'}), 500
        try:
            fiducials_data = json.loads(product.get('fiducials', '[]'))
            produced_aligned = align_with_fiducials(golden_full, produced_full, fiducials_data)
        except Exception as e:
            print(f"Erro no alinhamento: {e}. Usando fallback.")
            h, w = golden_full.shape[:2]
            produced_aligned = cv2.resize(produced_full, (w, h))
        if produced_aligned is None:
             return jsonify({'error': 'Falha ao alinhar ou redimensionar imagem de produção.'}), 500
        result_image = produced_aligned.copy()
        cursor.execute("""
            SELECT c.*, p.name as package_name, p.body_matrix, p.body_mask,
                   p.presence_threshold, p.ssim_threshold
            FROM components c 
            JOIN packages p ON c.package_id = p.id
            WHERE c.product_id=%s
        """, (product_id,))
        comps = cursor.fetchall()
        detailed_components_frontend = []
        total_ok, total_fail = 0, 0
        cursor.execute("INSERT INTO inspections (product_id, result, timestamp) VALUES (%s, 'IN_PROGRESS', NOW())", (product_id,))
        inspection_id = cursor.lastrowid
        for comp in comps:
            roi_g = golden_full[comp['y']:comp['y']+comp['height'], comp['x']:comp['x']+comp['width']]
            roi_p = produced_aligned[comp['y']:comp['y']+comp['height'], comp['x']:comp['x']+comp['width']]
            template_img = cv2.imread(comp['body_matrix'])
            template_mask_path = comp.get('body_mask') 
            if template_img is None or template_mask_path is None:
                print(f"ERRO: Não foi possível ler o template ou a máscara para o componente {comp['name']}")
                cv_status = 'FAIL'
                ai_status = 'UNKNOWN'
                cv_analysis = {'status': 'FAIL', 'details': {'message': 'Arquivo de template/máscara do pacote não encontrado.'}}
                ai_details = {'prob': 0.0}
            else:
                cv_analysis = analyze_component_package_based(
                    roi_g, 
                    template_img,
                    template_mask_path,
                    roi_p, 
                    comp['rotation'],
                    presence_threshold=comp.get('presence_threshold', 0.35),
                    ssim_threshold=comp.get('ssim_threshold', 0.6)
                )
                cv_status = cv_analysis.get('status', 'FAIL')
                ai_status, ai_details = predict_with_model(roi_p)
            final_status = "OK" if cv_status == "OK" and (ai_status == "OK" or ai_status == "UNKNOWN") else "FAIL"
            if final_status == "OK": total_ok += 1
            else: total_fail += 1
            x, y, w, h = comp['x'], comp['y'], comp['width'], comp['height']
            color = (0, 255, 0) if final_status == "OK" else (0, 0, 255)
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(result_image, f"{comp['name']}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            roi_g_path = save_image_to_disk(roi_g, 'results', f"insp_{inspection_id}_comp_{comp['id']}_g")
            roi_p_path = save_image_to_disk(roi_p, 'results', f"insp_{inspection_id}_comp_{comp['id']}_p")
            cursor.execute("""
                INSERT INTO inspection_results 
                (inspection_id, component_id, cv_status, ai_status, ai_status_prob, cv_details, final_status, golden_roi_image, produced_roi_image)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                inspection_id, comp['id'], cv_status, ai_status, ai_details.get('prob'), 
                json.dumps(cv_analysis.get('details')), final_status, 
                roi_g_path, roi_p_path
            ))
            debug_filename = None
            if 'debug_data' in cv_analysis and 'debug_img_b64' in cv_analysis['debug_data']:
                debug_img = base64_to_cv2_img(cv_analysis['debug_data']['debug_img_b64'])
                if debug_img is not None:
                    debug_filename = f"debug_{inspection_id}_{comp['name']}_{uuid.uuid4().hex[:4]}.png"
                    debug_path = os.path.join(app.config['DEBUG_FOLDER'], debug_filename)
                    cv2.imwrite(debug_path, debug_img)
            comp_data_for_frontend = {
                'name': comp['name'], 'component_id': comp['id'], 'package': comp['package_name'],
                'golden_image': cv2_to_base64(roi_g), 
                'produced_image': cv2_to_base64(roi_p),
                'cv_status': cv_status, 
                'ai_status': ai_status, 
                'final_status': final_status,
                'cv_details': cv_analysis.get('details', {}), 
                'ai_details': ai_details,
                'found_rotation': cv_analysis.get('found_rotation', 'N/A'),
                'displacement': cv_analysis.get('displacement', {'x':0, 'y':0}),
                'debug_filename': debug_filename, 
                'debug_data': cv_analysis.get('debug_data', {}) 
            }
            detailed_components_frontend.append(comp_data_for_frontend)
        overall_status = "OK" if total_fail == 0 else "FAIL"
        cursor.execute("UPDATE inspections SET result = %s WHERE id = %s", (overall_status, inspection_id))
        conn.commit()
        result_filename = f"inspection_result_{uuid.uuid4().hex}.png"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_image)
        cursor.close()
        conn.close()
        return jsonify({
            'total_ok': total_ok, 'total_fail': total_fail,
            'detailed_components': detailed_components_frontend,
            'result_filename': result_filename, 'product_id': product_id
        })
    # GET request
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM products ORDER BY name")
    products = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('inspect.html', products=products)

@app.route('/feedback', methods=['POST'])
@login_required
def save_feedback():
    # ... (código inalterado)
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        feedbacks = data.get('feedbacks', {})
        if not feedbacks:
            return jsonify({'success': False, 'error': 'Nenhum feedback recebido.'}), 400
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id FROM inspections WHERE product_id = %s ORDER BY timestamp DESC LIMIT 1", (product_id,))
        last_inspection = cursor.fetchone()
        if not last_inspection:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'error': 'Nenhuma inspeção encontrada para este produto.'}), 404
        inspection_id = last_inspection['id']
        samples_added_count = 0
        for component_name, feedback in feedbacks.items():
            cursor.execute("""
                INSERT INTO inspection_feedback (component_name, feedback, timestamp, user_id, inspection_id)
                VALUES (%s, %s, NOW(), %s, %s)
            """, (component_name, feedback, current_user.id, inspection_id))
            cursor.execute("SELECT id FROM components WHERE name = %s AND product_id = %s", (component_name, product_id))
            comp = cursor.fetchone()
            if not comp: continue
            cursor.execute("""
                SELECT golden_roi_image, produced_roi_image 
                FROM inspection_results 
                WHERE inspection_id = %s AND component_id = %s
            """, (inspection_id, comp['id']))
            roi_paths = cursor.fetchone()
            if roi_paths and roi_paths['produced_roi_image']:
                cursor.execute("""
                    INSERT INTO training_samples (product_id, component_id, golden_path, produced_path, label)
                    VALUES (%s, %s, %s, %s, %s)
                """, (product_id, comp['id'], roi_paths['golden_roi_image'], roi_paths['produced_roi_image'], feedback))
                samples_added_count += 1
            else:
                 print(f"Não foi possível encontrar os caminhos das ROIs para {component_name} na inspeção {inspection_id}")
        conn.commit()
        if samples_added_count > 0:
            print(f"Disparando o script de treinamento com {samples_added_count} novas amostras...")
            subprocess.Popen([sys.executable, "train_model.py"])
        return jsonify({'success': True, 'message': f'Feedback salvo! {samples_added_count} amostras adicionadas ao retreinamento.'})
    except Exception as e:
        traceback.print_exc()
        if conn: conn.rollback()
        return jsonify({'success': False, 'error': f'Erro ao salvar feedback: {str(e)}'}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

if __name__ == '__main__':
    load_trained_model()
    app.run(debug=True)