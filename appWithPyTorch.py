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
from train_model import MultiTaskCNN, base64_to_tensor

# --- Configuração Global ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DEBUG_FOLDER'] = 'static/debug'  # NOVA LINHA
app.config['SECRET_KEY'] = 'uma_chave_secreta_muito_segura'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DEBUG_FOLDER'], exist_ok=True) # NOVA LINHA

# --- Modelo de IA ---
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model():
    """Carrega o modelo MultiTaskCNN treinado ou inicializa um novo."""
    global model
    model_path = 'trained_model.pt'
    model = MultiTaskCNN() # Usa o novo modelo
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
    # (código de load_user permanece o mesmo)
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
    # (código de get_db_connection permanece o mesmo)
    return mysql.connector.connect(
        host="localhost",
        user="flaskuser",
        password="123456",
        database="smt_inspection_new"
    )

def cv2_to_base64(image):
    # (código de cv2_to_base64 permanece o mesmo)
    if image is None: return None
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_cv2_img(base64_string):
    # (código de base64_to_cv2_img permanece o mesmo)
    if not base64_string: return None
    try:
        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

# --- Funções de Visão Computacional (Refatoradas e Novas) ---

def find_fiducial_rings(image):
    # (código de find_fiducial_rings permanece o mesmo)
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
    # (código de align_with_fiducials permanece o mesmo)
    if produced_img is None: return None
    if len(fiducials) < 3:
        print("Aviso: Menos de 3 fiduciais. Usando resize como fallback.")
        h, w = golden_img.shape[:2]
        return cv2.resize(produced_img, (w, h))
    
    golden_points, produced_points = [], []
    produced_rings = find_fiducial_rings(produced_img)
    if not produced_rings:
        print("Aviso: Nenhum anel fiducial detectado. Usando resize como fallback.")
        h, w = golden_img.shape[:2]
        return cv2.resize(produced_img, (w, h))

    # Associa os anéis detectados com os fiduciais esperados
    for f in fiducials:
        expected_center = (f['x'], f['y'])
        closest_ring = min(produced_rings, key=lambda ring: np.hypot(ring['x'] - expected_center[0], ring['y'] - expected_center[1]))
        golden_points.append([f['x'], f['y']])
        produced_points.append([closest_ring['x'], closest_ring['y']])

    if len(golden_points) < 3:
        print("Aviso: Falha ao associar pontos. Usando resize como fallback.")
        h, w = golden_img.shape[:2]
        return cv2.resize(produced_img, (w, h))
    
    M, _ = cv2.findHomography(np.float32(produced_points), np.float32(golden_points), cv2.RANSAC, 5.0)
    
    if M is not None:
        h, w = golden_img.shape[:2]
        print("✅ Alinhamento com fiduciais bem-sucedido.")
        return cv2.warpPerspective(produced_img, M, (w, h))
    
    print("Aviso: Falha na homografia. Usando resize como fallback.")
    h, w = golden_img.shape[:2]
    return cv2.resize(produced_img, (w, h))

def analyze_component_package_based(golden_roi, template_img, roi_p_original, expected_rotation=0):
    PRESENCE_THRESHOLD = 0.35
    SSIM_THRESHOLD = 0.6
    ROTATION_TOLERANCE_THRESHOLD = 0.7

    try:
        if template_img is None or golden_roi is None:
            return {'status': 'FAIL', 'details': {'message': 'Template ou ROI Golden não encontrado.'}}

        padding = 10
        roi_p = cv2.copyMakeBorder(roi_p_original, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        roi_p_gray = cv2.cvtColor(roi_p, cv2.COLOR_BGR2GRAY)
        roi_p_gray = cv2.equalizeHist(roi_p_gray)

        best_match = {'angle': -1, 'score': -1, 'loc': (0, 0)}
        rotations = [0, 90, 180, 270]
        for angle in rotations:
            M = cv2.getRotationMatrix2D((template_gray.shape[1] / 2, template_gray.shape[0] / 2), angle, 1)
            rotated_template_base = cv2.warpAffine(template_gray, M, (template_gray.shape[1], template_gray.shape[0]))
            rotated_template = cv2.equalizeHist(rotated_template_base)
            res = cv2.matchTemplate(roi_p_gray, rotated_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_match['score']:
                best_match = {'angle': angle, 'score': max_val, 'loc': max_loc}

        if best_match['score'] < PRESENCE_THRESHOLD:
            return {'status': 'FAIL', 'found_rotation': 'N/A', 'displacement': {'x': 0, 'y': 0}, 'details': {'message': 'Componente Ausente', 'correlation_score': float(best_match['score'])}}

        found_rotation = best_match['angle']

        if found_rotation != expected_rotation:
            M_expected = cv2.getRotationMatrix2D((template_gray.shape[1] / 2, template_gray.shape[0] / 2), expected_rotation, 1)
            expected_template_base = cv2.warpAffine(template_gray, M_expected, (template_gray.shape[1], template_gray.shape[0]))
            expected_template = cv2.equalizeHist(expected_template_base)
            res_expected = cv2.matchTemplate(roi_p_gray, expected_template, cv2.TM_CCOEFF_NORMED)
            _, expected_score, _, _ = cv2.minMaxLoc(res_expected)
            if expected_score < best_match['score'] * ROTATION_TOLERANCE_THRESHOLD:
                return {'status': 'FAIL', 'found_rotation': f"{found_rotation}°", 'displacement': {'x': 0, 'y': 0}, 'details': {'message': f"Rotação Incorreta (Esperado: {expected_rotation}°, Encontrado: {found_rotation}°)", 'correlation_score': float(best_match['score'])}}

        h_orig, w_orig = golden_roi.shape[:2]
        x_found, y_found = best_match['loc']
        
        # --- LÓGICA DE COMPARAÇÃO SSIM REFINADA ---
        # 1. Recorta a ROI encontrada na imagem de produção
        found_comp_p = roi_p[y_found:y_found + h_orig, x_found:x_found + w_orig]

        # 2. Cria o "alvo" para o SSIM rotacionando a ROI Golden para a orientação encontrada
        M_color = cv2.getRotationMatrix2D((w_orig / 2, h_orig / 2), found_rotation, 1.0)
        target_for_ssim = cv2.warpAffine(golden_roi, M_color, (w_orig, h_orig))

        # 3. Garante que as imagens tenham o mesmo tamanho exato para o SSIM
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
        except Exception:
            ssim_value, _ = ssim(cv2.cvtColor(target_for_ssim, cv2.COLOR_BGR2GRAY), cv2.cvtColor(found_comp_p, cv2.COLOR_BGR2GRAY), full=True, data_range=255)

        status_ssim = "OK" if ssim_value > SSIM_THRESHOLD else "FAIL"
        message = "OK" if status_ssim == "OK" else "Baixa Similaridade (SSIM)"
        
        # Prepara a imagem de depuração
        debug_img = roi_p.copy()
        # Retângulo verde onde o componente foi encontrado
        cv2.rectangle(debug_img, (x_found, y_found), (x_found + w_orig, y_found + h_orig), (0, 255, 0), 2)
        
        return {
            'status': status_ssim,
            'found_rotation': f"{found_rotation}°",
            'displacement': {'x': x_found - padding, 'y': y_found - padding},
            'details': {'message': message, 'correlation_score': float(best_match['score']), 'ssim': float(ssim_value)},
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
        return 'UNKNOWN', {'prob': 0.0, 'rot': 0.0, 'dx': 0.0, 'dy': 0.0}
        
    try:
        tensor = base64_to_tensor(cv2_to_base64(roi_p)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prob, rot, disp = model(tensor)
            prob, rot, disp = prob.item(), rot.item(), disp.cpu().numpy().flatten()
            
        status = 'OK' if prob > 0.5 else 'FAIL'
        details = {'prob': prob, 'rot': rot, 'dx': float(disp[0]), 'dy': float(disp[1])}
        return status, details
    except Exception as e:
        print("Erro durante a inferência do modelo:", e)
        return 'UNKNOWN', {'prob': 0.0, 'rot': 0.0, 'dx': 0.0, 'dy': 0.0}

# --- Rotas da Aplicação ---

@app.route('/')
# (código da rota home permanece o mesmo)
@login_required
def home():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM products")
    products = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('index.html', products=products)

@app.route('/login', methods=['GET', 'POST'])
# (código da rota login permanece o mesmo)
def login():
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
# (código da rota register permanece o mesmo)
def register():
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
# (código da rota logout permanece o mesmo)
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/find_fiducials', methods=['POST'])
# (código da rota find_fiducials permanece o mesmo)
@login_required
def find_fiducials_route():
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

@app.route('/add_product', methods=['GET', 'POST'])
@login_required
def add_product():
    if request.method == 'POST':
        # (código principal permanece, mas com a lógica de body_matrix ajustada)
        conn, cursor = None, None
        try:
            name = request.form.get('name')
            golden_file = request.files.get('golden')
            fiducials_json = request.form.get('fiducials')
            components_json = request.form.get('components')

            golden_path = os.path.join(app.config['UPLOAD_FOLDER'], 'golden_' + str(uuid.uuid4()) + '_' + golden_file.filename)
            golden_file.save(golden_path)
            golden_full = cv2.imread(golden_path)

            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True, buffered=True)

            cursor.execute("INSERT INTO products (name, golden_image, fiducials) VALUES (%s, %s, %s)", (name, golden_path, fiducials_json))
            product_id = cursor.lastrowid

            components = json.loads(components_json)
            
            # Mapeia package_name para seu ID e se já tem body_matrix
            package_info = {}

            for comp in components:
                package_name = comp['package']
                if package_name not in package_info:
                    cursor.execute("SELECT id, body_matrix FROM packages WHERE name = %s", (package_name,))
                    pkg_data = cursor.fetchone()
                    if not pkg_data:
                        cursor.execute("INSERT INTO packages (name) VALUES (%s)", (package_name,))
                        package_id = cursor.lastrowid
                        package_info[package_name] = {'id': package_id, 'has_matrix': False}
                    else:
                        package_info[package_name] = {'id': pkg_data['id'], 'has_matrix': bool(pkg_data['body_matrix'])}

                package_id = package_info[package_name]['id']

                # NOVO: Se for um novo package, cria a body_matrix
                if not package_info[package_name]['has_matrix']:
                    x, y, w, h = comp['x'], comp['y'], comp['width'], comp['height']
                    roi_g = golden_full[y:y+h, x:x+w]
                    template_small = cv2.resize(roi_g, (64, 64), interpolation=cv2.INTER_AREA)
                    body_matrix_b64 = cv2_to_base64(template_small)
                    cursor.execute("UPDATE packages SET body_matrix = %s WHERE id = %s", (body_matrix_b64, package_id))
                    package_info[package_name]['has_matrix'] = True
                
                inspection_mask_base64 = None
                # (o restante da lógica de máscara permanece o mesmo)
                if 'inspection_regions' in comp and comp['inspection_regions']:
                    mask = np.zeros((comp['height'], comp['width']), dtype=np.uint8)
                    for region in comp['inspection_regions']:
                        cv2.rectangle(mask, (region['x'], region['y']), (region['x'] + region['width'], region['y'] + region['height']), 255, -1)
                    inspection_mask_base64 = cv2_to_base64(mask)


                cursor.execute(
                    """INSERT INTO components (product_id, name, x, y, width, height, package_id, rotation, inspection_mask)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (product_id, comp['name'], comp['x'], comp['y'], comp['width'], comp['height'],
                     package_id, comp.get('rotation', 0), inspection_mask_base64)
                )

            conn.commit()
            return jsonify({'success': True, 'product_id': product_id})

        except Exception as e:
            if conn: conn.rollback()
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
        finally:
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()
    
    # GET request (código permanece o mesmo)
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM packages ORDER BY name")
    packages = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('add_product.html', packages=packages)

@app.route('/inspect', methods=['GET','POST'])
@login_required
def inspect_board():
    if request.method == 'POST':
        # (A parte inicial da função permanece a mesma, até o loop 'for')
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        produced_file = request.files.get('produced')
        product_id = request.form.get('product_id')
        produced_full = cv2.imdecode(np.frombuffer(produced_file.read(), np.uint8), cv2.IMREAD_COLOR)

        cursor.execute("SELECT golden_image, fiducials FROM products WHERE id=%s",(product_id,))
        product = cursor.fetchone()
        golden_full = cv2.imread(product['golden_image'])

        try:
            fiducials_data = json.loads(product.get('fiducials', '[]'))
            produced_aligned = align_with_fiducials(golden_full, produced_full, fiducials_data)
        except Exception as e:
            print(f"Erro no alinhamento: {e}. Usando fallback.")
            h, w = golden_full.shape[:2]
            produced_aligned = cv2.resize(produced_full, (w, h))

        result_image = produced_aligned.copy()

        cursor.execute("""
            SELECT c.*, p.name as package_name, p.body_matrix
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
            
            template_img = base64_to_cv2_img(comp['body_matrix'])

            cv_analysis = analyze_component_package_based(roi_g, template_img, roi_p, comp['rotation'])
            cv_status = cv_analysis.get('status', 'FAIL')

            ai_status, ai_details = predict_with_model(roi_p)
            
            final_status = "OK" if cv_status == "OK" and ai_status == "OK" else "FAIL"
            
            if final_status == "OK": total_ok += 1
            else: total_fail += 1

            x, y, w, h = comp['x'], comp['y'], comp['width'], comp['height']
            color = (0, 255, 0) if final_status == "OK" else (0, 0, 255)
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(result_image, f"{comp['name']}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cursor.execute("""
                INSERT INTO inspection_results 
                (inspection_id, component_id, cv_status, ai_status, ai_status_prob, cv_details, final_status, golden_roi_image, produced_roi_image)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                inspection_id, comp['id'], cv_status, ai_status, ai_details.get('prob'), 
                json.dumps(cv_analysis.get('details')), final_status, 
                cv2_to_base64(roi_g), cv2_to_base64(roi_p)
            ))
            
            # --- LÓGICA DE DEPURAÇÃO VISUAL ---
            debug_filename = None
            if 'debug_data' in cv_analysis:
                debug_img = base64_to_cv2_img(cv_analysis['debug_data']['debug_img_b64'])
                if debug_img is not None:
                    debug_filename = f"debug_{inspection_id}_{comp['name']}.png"
                    debug_path = os.path.join(app.config['DEBUG_FOLDER'], debug_filename)
                    cv2.imwrite(debug_path, debug_img)
            
            comp_data_for_frontend = {
                'name': comp['name'], 'component_id': comp['id'], 'package': comp['package_name'],
                'golden_image': cv2_to_base64(roi_g), 'produced_image': cv2_to_base64(roi_p),
                'cv_status': cv_status, 'ai_status': ai_status, 'final_status': final_status,
                'cv_details': cv_analysis.get('details', {}), 'ai_details': ai_details,
                'found_rotation': cv_analysis.get('found_rotation', 'N/A'),
                'displacement': cv_analysis.get('displacement', {'x':0, 'y':0}),
                'debug_filename': debug_filename # Adiciona o nome do arquivo para o frontend
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
    # (código refatorado para usar as novas colunas e acionar o treinamento)
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        feedbacks = data.get('feedbacks', {})

        if not feedbacks:
            return jsonify({'success': False, 'error': 'Nenhum feedback recebido.'}), 400

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Pega a última inspeção para o produto
        cursor.execute("SELECT id FROM inspections WHERE product_id = %s ORDER BY timestamp DESC LIMIT 1", (product_id,))
        last_inspection = cursor.fetchone()
        if not last_inspection:
            return jsonify({'success': False, 'error': 'Nenhuma inspeção encontrada para este produto.'}), 404
        
        inspection_id = last_inspection['id']

        for component_name, feedback in feedbacks.items():
            # Salva o feedback do usuário
            cursor.execute("""
                INSERT INTO inspection_feedback (component_name, feedback, timestamp, user_id, inspection_id)
                VALUES (%s, %s, NOW(), %s, %s)
            """, (component_name, feedback, current_user.id, inspection_id))
            
            # Encontra o component_id a partir do nome
            cursor.execute("SELECT id FROM components WHERE name = %s AND product_id = %s", (component_name, product_id))
            comp = cursor.fetchone()
            if not comp: continue

            # Busca as ROIs da inspeção correspondente para criar uma amostra de treinamento
            cursor.execute("""
                SELECT golden_roi_image, produced_roi_image 
                FROM inspection_results 
                WHERE inspection_id = %s AND component_id = %s
            """, (inspection_id, comp['id']))
            roi_images = cursor.fetchone()
            
            if roi_images:
                cursor.execute("""
                    INSERT INTO training_samples (product_id, component_id, golden_base64, produced_base64, label)
                    VALUES (%s, %s, %s, %s, %s)
                """, (product_id, comp['id'], roi_images['golden_roi_image'], roi_images['produced_roi_image'], feedback))

        conn.commit()
        cursor.close()
        conn.close()

        # Dispara o script de treinamento em um processo separado
        print("Disparando o script de treinamento em segundo plano...")
        subprocess.Popen([sys.executable, "train_model.py"])

        return jsonify({'success': True, 'message': 'Feedback salvo e retreinamento iniciado!'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Erro ao salvar feedback: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)