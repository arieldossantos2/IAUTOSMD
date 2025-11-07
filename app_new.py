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
from train_model import MultiTaskCNN, path_to_tensor # ATUALIZADO de base64_to_tensor

# --- Configuração Global ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DEBUG_FOLDER'] = 'static/debug'
app.config['SECRET_KEY'] = 'uma_chave_se_muito_segura'

# --- SUGESTÃO 3: Novos diretórios para armazenamento de imagens ---
app.config['IMAGE_STORAGE'] = 'static/images'
app.config['IMAGE_FOLDERS'] = {
    'packages': os.path.join(app.config['IMAGE_STORAGE'], 'packages'),
    'results': os.path.join(app.config['IMAGE_STORAGE'], 'results'),
    'training': os.path.join(app.config['IMAGE_STORAGE'], 'training') # Embora o app não grave aqui, é bom ter
}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DEBUG_FOLDER'], exist_ok=True)
for folder in app.config['IMAGE_FOLDERS'].values():
    os.makedirs(folder, exist_ok=True)
# --- FIM SUGESTÃO 3 ---


# --- Modelo de IA ---
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model():
    """Carrega o modelo MultiTaskCNN treinado ou inicializa um novo."""
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

# --- Login Manager ---
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

# --- Conexão com Banco de Dados e Helpers ---
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="flaskuser",
        password="123456", # Assumindo que você corrigiu isso
        database="smt_inspection_new"
    )

def cv2_to_base64(image):
    if image is None: return None
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_cv2_img(base64_string):
    if not base64_string: return None
    try:
        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

# --- SUGESTÃO 3: Nova função helper para salvar imagens ---
def save_image_to_disk(image, folder_key, filename_prefix):
    """Salva uma imagem (numpy array) no disco e retorna o caminho relativo."""
    try:
        folder_path = app.config['IMAGE_FOLDERS'].get(folder_key)
        if not folder_path:
            raise ValueError(f"Chave de pasta de imagem inválida: {folder_key}")
        
        filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.png"
        relative_path = os.path.join(folder_path, filename)
        
        # Garante que o diretório existe
        os.makedirs(folder_path, exist_ok=True) 
        
        # Salva a imagem
        success = cv2.imwrite(relative_path, image)
        if not success:
            raise IOError(f"Falha ao salvar a imagem em {relative_path}")
            
        print(f"Imagem salva em: {relative_path}")
        return relative_path # Retorna o caminho (ex: 'static/images/packages/pkg_1_...png')
    except Exception as e:
        traceback.print_exc()
        return None
# --- FIM SUGESTÃO 3 ---


# --- Funções de Visão Computacional ---

def find_fiducial_rings(image):
    # (código permanece o mesmo)
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

# --- SUGESTÃO 2: Fallback de Alinhamento Aprimorado ---
def align_with_fiducials(golden_img, produced_img, fiducials):
    if produced_img is None: return None
    
    h, w = golden_img.shape[:2]

    if len(fiducials) == 0:
        print("Aviso: Nenhum fiducial definido. Usando resize como fallback.")
        return cv2.resize(produced_img, (w, h))

    golden_points, produced_points = [], []
    produced_rings = find_fiducial_rings(produced_img)
    
    if not produced_rings:
        print("Aviso: Nenhum anel fiducial detectado. Usando resize como fallback.")
        return cv2.resize(produced_img, (w, h))

    # Associa os anéis detectados com os fiduciais esperados
    for f in fiducials:
        expected_center = (f['x'], f['y'])
        closest_ring = min(produced_rings, key=lambda ring: np.hypot(ring['x'] - expected_center[0], ring['y'] - expected_center[1]))
        golden_points.append([f['x'], f['y']])
        produced_points.append([closest_ring['x'], closest_ring['y']])

    M = None
    if len(golden_points) >= 3:
        M, _ = cv2.findHomography(np.float32(produced_points), np.float32(golden_points), cv2.RANSAC, 5.0)
    
    if M is not None:
        print("✅ Alinhamento com Homografia (3+ pontos) bem-sucedido.")
        return cv2.warpPerspective(produced_img, M, (w, h))
    
    if len(golden_points) >= 2:
        print("Aviso: Homografia falhou ou < 3 pontos. Tentando Transformação Affine (2 pontos).")
        # Usa os dois primeiros pontos para uma transformação Affine
        M_affine = cv2.getAffineTransform(np.float32(produced_points[:2]), np.float32(golden_points[:2]))
        return cv2.warpAffine(produced_img, M_affine, (w, h))

    print("Aviso: Menos de 2 fiduciais associados. Usando resize como fallback.")
    return cv2.resize(produced_img, (w, h))
# --- FIM SUGESTÃO 2 ---


# --- SUGESTÃO 1 & 5: Limiares por Pacote e Análise de Cor ---
def analyze_component_package_based(golden_roi, template_img, roi_p_original, expected_rotation=0, 
                                    presence_threshold=0.35, ssim_threshold=0.6, color_threshold=0.7):
    
    # Valores padrão (globais)
    PRESENCE_THRESHOLD = presence_threshold
    SSIM_THRESHOLD = ssim_threshold
    COLOR_THRESHOLD = color_threshold # Novo limiar para cor
    ROTATION_TOLERANCE_THRESHOLD = 0.7 

    try:
        if template_img is None or golden_roi is None or roi_p_original is None:
            return {'status': 'FAIL', 'details': {'message': 'Template, ROI Golden ou ROI Produção não encontrado.'}}

        padding = max(template_img.shape[0], template_img.shape[1])
        roi_p = cv2.copyMakeBorder(roi_p_original, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        roi_p_gray = cv2.cvtColor(roi_p, cv2.COLOR_BGR2GRAY)
        
        template_gray = cv2.equalizeHist(template_gray)
        roi_p_gray = cv2.equalizeHist(roi_p_gray)

        best_match = {'angle': -1, 'score': -np.inf, 'loc': (0, 0), 'dims': (0,0)}
        rotations = [0, 90, 180, 270]

        for angle in rotations:
            M = cv2.getRotationMatrix2D((template_gray.shape[1] / 2, template_gray.shape[0] / 2), angle, 1)
            rotated_template = cv2.warpAffine(template_gray, M, (template_gray.shape[1], template_gray.shape[0]))
            
            if angle % 180 != 0:
                rotated_template = cv2.warpAffine(template_gray, M, (template_gray.shape[0], template_gray.shape[1]))
            
            h_rot, w_rot = rotated_template.shape[:2]
            
            # Previne erro se o template for maior que a ROI (devido ao padding)
            if h_rot > roi_p_gray.shape[0] or w_rot > roi_p_gray.shape[1]:
                continue 

            res = cv2.matchTemplate(roi_p_gray, rotated_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_match['score']:
                best_match = {'angle': angle, 'score': max_val, 'loc': max_loc, 'dims': (h_rot, w_rot)}

        if best_match['score'] < PRESENCE_THRESHOLD:
            return {'status': 'FAIL', 'found_rotation': 'N/A', 'displacement': {'x': 0, 'y': 0}, 'details': {'message': f'Componente Ausente (Score: {best_match["score"]:.2f} < {PRESENCE_THRESHOLD})', 'correlation_score': float(best_match['score'])}}

        found_rotation = best_match['angle']
        h_found, w_found = best_match['dims']
        x_found, y_found = best_match['loc']

        # --- Verificação de Rotação ---
        if found_rotation != expected_rotation:
            M_expected = cv2.getRotationMatrix2D((template_gray.shape[1] / 2, template_gray.shape[0] / 2), expected_rotation, 1)
            expected_template = cv2.warpAffine(template_gray, M_expected, (template_gray.shape[1], template_gray.shape[0]))
            if expected_rotation % 180 != 0:
                expected_template = cv2.warpAffine(template_gray, M_expected, (template_gray.shape[0], template_gray.shape[1]))

            if expected_template.shape[0] <= roi_p_gray.shape[0] and expected_template.shape[1] <= roi_p_gray.shape[1]:
                res_expected = cv2.matchTemplate(roi_p_gray, expected_template, cv2.TM_CCOEFF_NORMED)
                _, expected_score, _, _ = cv2.minMaxLoc(res_expected)
                
                if expected_score < best_match['score'] * ROTATION_TOLERANCE_THRESHOLD:
                    return {'status': 'FAIL', 'found_rotation': f"{found_rotation}°", 'displacement': {'x': x_found - padding, 'y': y_found - padding}, 'details': {'message': f"Rotação Incorreta (Esperado: {expected_rotation}°, Encontrado: {found_rotation}°)", 'correlation_score': float(best_match['score'])}}
            else:
                 print(f"Aviso: Template esperado ({expected_template.shape}) maior que ROI ({roi_p_gray.shape}). Pulando verificação de rotação.")


        
        # --- LÓGICA DE COMPARAÇÃO SSIM ---
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
        
        details = {'correlation_score': float(best_match['score']), 'ssim': float(ssim_value)}

        # --- SUGESTÃO 5: Análise de Cor ---
        if status == "OK":
            try:
                # Converte para HSV
                target_hsv = cv2.cvtColor(target_for_ssim, cv2.COLOR_BGR2HSV)
                found_hsv = cv2.cvtColor(found_comp_p, cv2.COLOR_BGR2HSV)
                
                # Calcula histograma de Hue (Matiz) e Saturation (Saturação)
                hist_g = cv2.calcHist([target_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist_g, hist_g, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                
                hist_p = cv2.calcHist([found_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist_p, hist_p, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                
                # Compara os histogramas
                color_similarity = cv2.compareHist(hist_g, hist_p, cv2.HISTCMP_CORREL)
                details['color_similarity'] = float(color_similarity)

                if color_similarity < COLOR_THRESHOLD:
                    status = "FAIL"
                    message = f"Falha na Cor (Similaridade: {color_similarity:.2f} < {COLOR_THRESHOLD})"
            except Exception as e_color:
                print(f"Erro na análise de cor: {e_color}")
                details['color_similarity'] = 0.0
        # --- FIM SUGESTÃO 5 ---

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
    """Realiza a inferência usando o modelo PyTorch carregado."""
    global model
    if model is None:
        print("Modelo de IA não carregado, pulando inferência.")
        return 'UNKNOWN', {'prob': 0.0}
        
    try:
        # A imagem de entrada para a IA é a ROI inteira (caixa azul)
        # Usa o helper do train_model (que foi atualizado) para pré-processar
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

# --- Rotas da Aplicação ---

@app.route('/')
@login_required
def home():
    # (código permanece o mesmo)
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM products")
    products = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('index.html', products=products)

@app.route('/login', methods=['GET', 'POST'])
def login():
    # (código permanece o mesmo)
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
    # (código permanece o mesmo)
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
    # (código permanece o mesmo)
    logout_user()
    return redirect(url_for('login'))

@app.route('/find_fiducials', methods=['POST'])
@login_required
def find_fiducials_route():
    # (código permanece o mesmo)
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

@app.route('/preview_template', methods=['POST'])
@login_required
def preview_template():
    try:
        data = request.get_json()
        roi_g_b64 = data.get('component_roi_b64').split(',')[1]
        regions = data.get('inspection_regions')
        
        roi_g = base64_to_cv2_img(roi_g_b64)
        if roi_g is None:
            return jsonify({'error': 'Imagem ROI inválida.'}), 400

        # 1. Extrai o template (corpo) da ROI, assim como em /add_product
        min_x = min(r['x'] for r in regions)
        min_y = min(r['y'] for r in regions)
        max_x = max(r['x'] + r['width'] for r in regions)
        max_y = max(r['y'] + r['height'] for r in regions)
        
        body_template_img = roi_g[min_y:max_y, min_x:max_x]
        
        if body_template_img.size == 0:
            return jsonify({'error': 'Região de corpo (verde) inválida.'}), 400

        # 2. Testa o template contra a própria ROI
        #    Usamos a ROI original (roi_g) como a imagem a ser inspecionada.
        cv_analysis = analyze_component_package_based(
            golden_roi=roi_g, 
            template_img=body_template_img, 
            roi_p_original=roi_g, 
            expected_rotation=0 # Teste de 0 grau
        )
        
        # 3. Retorna o resultado
        return jsonify({
            'success': True,
            'status': cv_analysis.get('status'),
            'message': cv_analysis.get('details', {}).get('message'),
            'debug_img_b64': cv_analysis.get('debug_data', {}).get('debug_img_b64')
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/add_product', methods=['GET', 'POST'])
@login_required
def add_product():
    if request.method == 'POST':
        # --- SUGESTÃO 3: Lógica de /add_product ATUALIZADA ---
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
            golden_full = cv2.imread(golden_path)
            if golden_full is None:
                return jsonify({'error': 'Falha ao ler a imagem golden salva.'}), 400

            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True, buffered=True)

            cursor.execute("INSERT INTO products (name, golden_image, fiducials) VALUES (%s, %s, %s)", (name, golden_path, fiducials_json))
            product_id = cursor.lastrowid

            components = json.loads(components_json)
            package_info = {} 

            for comp in components:
                package_name = comp['package']
                if package_name not in package_info:
                    cursor.execute("SELECT id, body_matrix FROM packages WHERE name = %s", (package_name,))
                    pkg_data = cursor.fetchone()
                    if not pkg_data:
                        # --- SUGESTÃO 1: Insere com limiares padrão ---
                        cursor.execute("INSERT INTO packages (name, presence_threshold, ssim_threshold) VALUES (%s, %s, %s)", 
                                       (package_name, 0.35, 0.6)) # Valores padrão
                        package_id = cursor.lastrowid
                        package_info[package_name] = {'id': package_id, 'has_matrix': False}
                    else:
                        package_info[package_name] = {'id': pkg_data['id'], 'has_matrix': bool(pkg_data['body_matrix'])}

                package_id = package_info[package_name]['id']

                if not package_info[package_name]['has_matrix']:
                    print(f"Criando novo template (body_matrix) para o pacote: {package_name}")
                    
                    if 'inspection_regions' not in comp or not comp['inspection_regions']:
                        conn.rollback() 
                        return jsonify({'error': f"Componente '{comp['name']}' é o primeiro do pacote '{package_name}', mas não tem uma 'Região de Corpo' (caixa verde) definida. Por favor, defina a região do corpo."}), 400
                    
                    regions = comp['inspection_regions']
                    
                    min_x = min(r['x'] for r in regions)
                    min_y = min(r['y'] for r in regions)
                    max_x = max(r['x'] + r['width'] for r in regions)
                    max_y = max(r['y'] + r['height'] for r in regions)

                    x, y, w, h = comp['x'], comp['y'], comp['width'], comp['height']
                    roi_g = golden_full[y:y+h, x:x+w]
                    
                    body_template_img = roi_g[min_y:max_y, min_x:max_x]
                    
                    if body_template_img.size == 0:
                         conn.rollback()
                         return jsonify({'error': f"Região de corpo (verde) inválida para '{comp['name']}'."}), 400

                    # --- SUGESTÃO 3: Salva o template como arquivo ---
                    body_matrix_path = save_image_to_disk(body_template_img, 'packages', f"pkg_{package_id}_{package_name}")
                    if not body_matrix_path:
                        conn.rollback()
                        return jsonify({'error': f"Falha ao salvar a imagem do template para o pacote '{package_name}'."}), 500
                    
                    cursor.execute("UPDATE packages SET body_matrix = %s WHERE id = %s", (body_matrix_path, package_id))
                    package_info[package_name]['has_matrix'] = True
                
                # (Lógica da máscara de inspeção permanece a mesma)
                inspection_mask_base64 = None # Ainda podemos salvar isso como base64, é pequeno
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
    # --- SUGESTÃO 1: Passa os limiares para o template ---
    cursor.execute("SELECT id, name, presence_threshold, ssim_threshold FROM packages ORDER BY name")
    packages = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('add_product.html', packages=packages) # Agora 'packages' contém os limiares

@app.route('/inspect', methods=['GET','POST'])
@login_required
def inspect_board():
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
            # --- SUGESTÃO 2: Usa a nova função de alinhamento ---
            produced_aligned = align_with_fiducials(golden_full, produced_full, fiducials_data)
        except Exception as e:
            print(f"Erro no alinhamento: {e}. Usando fallback.")
            h, w = golden_full.shape[:2]
            produced_aligned = cv2.resize(produced_full, (w, h))
        
        if produced_aligned is None:
             return jsonify({'error': 'Falha ao alinhar ou redimensionar imagem de produção.'}), 500

        result_image = produced_aligned.copy()

        # --- SUGESTÃO 1: Busca os limiares do pacote ---
        cursor.execute("""
            SELECT c.*, p.name as package_name, p.body_matrix,
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
            
            # --- SUGESTÃO 3: Carrega o template (corpo) do arquivo ---
            template_img = cv2.imread(comp['body_matrix'])
            if template_img is None:
                print(f"ERRO: Não foi possível ler o template do pacote em {comp['body_matrix']} para o componente {comp['name']}")
                cv_status = 'FAIL'
                ai_status = 'UNKNOWN'
                cv_analysis = {'status': 'FAIL', 'details': {'message': 'Arquivo de template do pacote não encontrado.'}}
                ai_details = {'prob': 0.0}
            else:
                # --- SUGESTÃO 1: Passa os limiares do pacote para a função de análise ---
                cv_analysis = analyze_component_package_based(
                    roi_g, 
                    template_img, 
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

            # --- SUGESTÃO 3: Salva ROIs como arquivos e armazena caminhos ---
            roi_g_path = save_image_to_disk(roi_g, 'results', f"insp_{inspection_id}_comp_{comp['id']}_g")
            roi_p_path = save_image_to_disk(roi_p, 'results', f"insp_{inspection_id}_comp_{comp['id']}_p")

            cursor.execute("""
                INSERT INTO inspection_results 
                (inspection_id, component_id, cv_status, ai_status, ai_status_prob, cv_details, final_status, golden_roi_image, produced_roi_image)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                inspection_id, comp['id'], cv_status, ai_status, ai_details.get('prob'), 
                json.dumps(cv_analysis.get('details')), final_status, 
                roi_g_path, roi_p_path # Salva os CAMINHOS no DB
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
                # Converte para Base64 APENAS para enviar ao frontend
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
    # --- SUGESTÃO 3: Rota de feedback ATUALIZADA ---
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

            # Busca os CAMINHOS das ROIs
            cursor.execute("""
                SELECT golden_roi_image, produced_roi_image 
                FROM inspection_results 
                WHERE inspection_id = %s AND component_id = %s
            """, (inspection_id, comp['id']))
            roi_paths = cursor.fetchone()
            
            if roi_paths and roi_paths['produced_roi_image']:
                # Insere os CAMINHOS na tabela de treinamento
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
            # Não é necessário recarregar o modelo aqui, o subprocesso irá salvar o novo .pt
            # O load_trained_model() no início da próxima execução do app irá carregá-lo.
            # Se quiser recarga "ao vivo", seria necessário um mecanismo mais complexo (ex: Gunicorn reload).

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