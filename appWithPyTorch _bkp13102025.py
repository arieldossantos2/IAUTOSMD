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
import os # Adicionado import de 'os' para verificação de arquivo
import sys # NOVO: Importa o módulo sys para obter o caminho do executável Python

# Importa a classe do modelo e a função de conversão da base de treinamento
from train_model import SimpleCNN, base64_to_tensor # Importa SimpleCNN e base64_to_tensor

# Variáveis globais para o modelo
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model():
    """Carrega o modelo treinado ou inicializa um modelo não treinado se o arquivo não existir."""
    global model
    model_path = 'trained_model.pt'
    
    # 1. Inicializa o modelo (necessário mesmo que não esteja treinado)
    model = SimpleCNN()
    
    if os.path.exists(model_path):
        try:
            # Tenta carregar o estado do modelo
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print(f"✅ Modelo treinado '{model_path}' carregado com sucesso.")
        except Exception as e:
            # Caso o arquivo exista, mas esteja corrompido ou o SimpleCNN tenha mudado.
            print(f"Erro ao carregar o estado do modelo treinado: {e}. Usando modelo inicializado/não treinado.")
            model.to(device)
            model.eval()
    else:
        # Se o arquivo não existir (primeira execução sem dados)
        print(f"Aviso: Arquivo de modelo '{model_path}' não encontrado. Iniciando com um modelo não treinado.")
        model.to(device)
        model.eval()

# Tenta carregar o modelo na inicialização
load_trained_model()


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ROI_FOLDER'] = 'static/roi_images'
app.config['SECRET_KEY'] = 'uma_chave_secreta_muito_segura'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ROI_FOLDER'], exist_ok=True)

# -------------------------------
# Login
# -------------------------------
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

# -------------------------------
# DB
# -------------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="flaskuser",
        password="123456",
        database="smt_inspection_new"
    )

# -------------------------------
# Helpers
# -------------------------------
def cv2_to_base64(image):
    if image is None:
        return None
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_cv2_mask(base64_string):
    if not base64_string:
        return None
    try:
        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return mask
    except Exception as e:
        print(f"Error decoding base64 mask: {e}")
        return None

def base64_to_cv2_img(base64_string):
    if not base64_string:
        return None
    try:
        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

def rotate_image(image, angle):
    if angle == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

# -------------------------------
# Funções de Visão Computacional
# -------------------------------

def find_and_crop_board(image):
    if image is None: return None, None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None

    placa_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(placa_contour)
    margin = 30
    x_marg, y_marg = max(0, x - margin), max(0, y - margin)
    w_marg = min(image.shape[1] - x_marg, w + 2 * margin)
    h_marg = min(image.shape[0] - y_marg, h + 2 * margin)
    cropped_board = image[y_marg:y_marg + h_marg, x_marg:x_marg + w_marg]
    return cropped_board, (x_marg, y_marg, w_marg, h_marg)

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
    if len(fiducials) < 3:
        print("Aviso: Menos de 3 fiduciais definidos. Redimensionando a imagem como fallback.")
        h, w = golden_img.shape[:2]
        return cv2.resize(produced_img, (w, h))
    
    golden_points, produced_points = [], []
    
    # Encontra os anéis fiduciais na imagem de produção inteira UMA VEZ
    produced_rings = find_fiducial_rings(produced_img)
    if not produced_rings:
        print("Aviso: Nenhum anel fiducial detectado na imagem de produção. Fallback para resize.")
        h, w = golden_img.shape[:2]
        return cv2.resize(produced_img, (w, h))

    # Associa os anéis detectados com os fiduciais esperados
    for f in fiducials:
        # Encontra o anel detectado mais próximo da posição esperada do fiducial golden
        expected_center = (f['x'], f['y'])
        closest_ring = min(produced_rings, key=lambda ring: np.hypot(ring['x'] - expected_center[0], ring['y'] - expected_center[1]))
        
        # Adiciona os pontos correspondentes para o cálculo da homografia
        golden_points.append([f['x'], f['y']])
        produced_points.append([closest_ring['x'], closest_ring['y']])

    if len(golden_points) < 3:
        print("Aviso: Falha ao associar pontos suficientes. Fallback para resize.")
        h, w = golden_img.shape[:2]
        return cv2.resize(produced_img, (w, h))
    
    # Calcula a matriz de homografia para alinhar a imagem
    M, _ = cv2.findHomography(np.float32(produced_points), np.float32(golden_points), cv2.RANSAC, 5.0)
    
    if M is not None:
        h, w = golden_img.shape[:2]
        print("✅ Alinhamento com fiduciais bem-sucedido.")
        return cv2.warpPerspective(produced_img, M, (w, h))
    
    print("Aviso: Falha ao calcular a matriz de homografia. Fallback para resize.")
    h, w = golden_img.shape[:2]
    return cv2.resize(produced_img, (w, h))

def analyze_component_revised(roi_g, roi_p, inspection_mask_base64=None):
    """
    Analisa um componente usando Template Matching e SSIM, com tratamento para ROIs pequenas.
    """
    PRESENCE_THRESHOLD = 0.6
    SSIM_THRESHOLD = 0.7

    try:
        # --- VERIFICAÇÃO INICIAL DE TAMANHO ---
        if roi_g.shape[0] < 3 or roi_g.shape[1] < 3:
            return { 'status': "FAIL", 'rotation': 'N/A', 'displacement': {'x': 0, 'y': 0}, 'details': { 'mode': 'Pré-verificação', 'message': f'ROI Golden muito pequena ({roi_g.shape[1]}x{roi_g.shape[0]}) para análise.' } }

        h, w = roi_g.shape[:2]
        roi_p_resized = cv2.resize(roi_p, (w, h))

        # --- 1. Detecção de Presença e Deslocamento ---
        roi_g_gray = cv2.cvtColor(roi_g, cv2.COLOR_BGR2GRAY)
        roi_p_gray = cv2.cvtColor(roi_p_resized, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(roi_p_gray, roi_g_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < PRESENCE_THRESHOLD:
            return { 'status': "FAIL", 'rotation': 'N/A', 'displacement': {'x': 0, 'y': 0}, 'details': { 'mode': 'Template Matching', 'ssim': 0.0, 'correlation_score': float(max_val), 'message': 'Componente Ausente' } }

        dx, dy = max_loc[0], max_loc[1]
        
        # --- 2. Análise de Qualidade com SSIM ---
        if dx + w > roi_p_resized.shape[1] or dy + h > roi_p_resized.shape[0]:
            return { 'status': 'FAIL', 'rotation': 'N/A', 'displacement': {'x': dx, 'y': dy}, 'details': { 'mode': 'Template Matching', 'message': 'Componente fora dos limites' } }
        
        found_comp_p = roi_p_resized[dy:dy+h, dx:dx+w]

        # --- NOVO: Lógica Robusta para SSIM com ROIs pequenas ---
        ssim_value = 0.0
        mode = "SSIM Color"
        try:
            # Encontra o menor lado da imagem
            min_side = min(roi_g.shape[0], roi_g.shape[1])
            
            # O win_size deve ser ímpar e menor que o menor lado
            win_size = min(7, min_side)
            if win_size % 2 == 0:
                win_size -= 1

            if win_size >= 3:
                # Tenta o SSIM colorido
                ssim_value, _ = ssim(roi_g, found_comp_p, full=True, multichannel=True, win_size=win_size, data_range=255, channel_axis=2)
            else:
                # Se a imagem for muito pequena para SSIM, consideramos OK se a presença foi confirmada
                ssim_value = 1.0 
                mode = "Presença Confirmada (ROI pequena demais para SSIM)"

        except ValueError:
            # Fallback para escala de cinza se o colorido falhar por alguma razão
            try:
                mode = "SSIM Grayscale (Fallback)"
                min_side = min(roi_g_gray.shape[0], roi_g_gray.shape[1])
                win_size = min(7, min_side)
                if win_size % 2 == 0:
                    win_size -= 1
                
                if win_size >= 3:
                    ssim_value, _ = ssim(cv2.cvtColor(roi_g, cv2.COLOR_BGR2GRAY), cv2.cvtColor(found_comp_p, cv2.COLOR_BGR2GRAY), full=True, win_size=win_size, data_range=255)
                else:
                    ssim_value = 1.0
                    mode = "Presença Confirmada (ROI pequena demais para SSIM)"
            except Exception as e_gray:
                print(f"Falha no SSIM em grayscale também: {e_gray}")
                # Se tudo falhar, retorna como erro
                return { 'status': 'FAIL', 'rotation': 'N/A', 'displacement': {'x': dx, 'y': dy}, 'details': { 'mode': 'Erro de Análise', 'message': 'Falha no cálculo do SSIM.' } }

        status = "OK" if ssim_value > SSIM_THRESHOLD else "FAIL"
        message = "OK" if status == "OK" else "Defeito de Componente (Baixo SSIM)"

        return { 'status': status, 'rotation': '0°', 'displacement': {'x': dx, 'y': dy}, 'details': { 'mode': mode, 'ssim': float(ssim_value), 'correlation_score': float(max_val), 'message': message } }

    except Exception as e:
        traceback.print_exc()
        return { 'status': 'FAIL', 'rotation': 'N/A', 'displacement': {'x': 0, 'y': 0}, 'details': { 'mode': 'Erro', 'message': f'Erro na análise: {str(e)}' } }

def predict_with_model(roi_p):
    """Realiza a inferência usando o modelo PyTorch carregado."""
    global model
    if model is None:
        print("Modelo de CNN não carregado/treinado. Retornando status 'UNKNOWN'.")
        return 'UNKNOWN', 0.0
        
    try:
        img = cv2.resize(roi_p, (64, 64))
        tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(device)
        
        with torch.no_grad():
            prob = model(tensor).item()
            
        return 'OK' if prob > 0.5 else 'FAIL', prob
    except Exception as e:
        print("Erro durante a inferência do modelo:", e)
        return 'UNKNOWN', 0.0
    

def save_inspection_results(conn, cursor, product_id, total_ok, total_fail, detailed_components):
    """
    Salva o registro da inspeção principal (inspections) e os resultados 
    detalhados dos componentes (inspection_results) no banco de dados.
    """
    try:
        overall_status = "OK" if total_fail == 0 else "FAIL"
        
        cursor.execute("""
            INSERT INTO inspections (product_id, result, timestamp)
            VALUES (%s, %s, NOW())
        """, (product_id, overall_status))
        
        inspection_id = cursor.lastrowid
        
        for comp in detailed_components:
            component_db_id = comp['component_id']
            
            cursor.execute("""
                INSERT INTO inspection_results (
                    inspection_id, 
                    component_id, 
                    ai_status, 
                    golden_roi_image, 
                    produced_roi_image
                )
                VALUES (%s, %s, %s, %s, %s)
            """, (
                inspection_id, 
                component_db_id,
                comp['status'],
                comp['golden_image'],
                comp['produced_image']
            ))

        conn.commit()
        return True, None
    except Exception as e:
        conn.rollback()
        traceback.print_exc()
        return False, str(e)


# -------------------------------
# Application Routes
# -------------------------------

@app.route('/')
def home():
    if not current_user.is_authenticated: return redirect(url_for('login'))
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM products")
    products = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('index.html', products=products)

@app.route('/login', methods=['GET', 'POST'])
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
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/find_fiducials', methods=['POST'])
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

@app.route('/get_packages', methods=['GET'])
@login_required
def get_packages():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM packages ORDER BY name")
    packages = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(packages)

@app.route('/add_product', methods=['GET', 'POST'])
@login_required
def add_product():
    if request.method == 'POST':
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
            cursor = conn.cursor(buffered=True)

            cursor.execute("INSERT INTO products (name, golden_image, fiducials) VALUES (%s, %s, %s)", (name, golden_path, fiducials_json))
            product_id = cursor.lastrowid

            components = json.loads(components_json)
            for comp in components:
                package_name = comp['package']
                cursor.execute("INSERT IGNORE INTO packages (name) VALUES (%s)", (package_name,))
                conn.commit()
                cursor.execute("SELECT id FROM packages WHERE name = %s", (package_name,))
                package_id_tuple = cursor.fetchone()
                package_id = package_id_tuple[0]
                inspection_mask_base64 = None
                if 'inspection_regions' in comp and comp['inspection_regions']:
                    mask = np.zeros((comp['height'], comp['width']), dtype=np.uint8)
                    for region in comp['inspection_regions']:
                        cv2.rectangle(mask, (region['x'], region['y']),
                                      (region['x'] + region['width'], region['y'] + region['height']), 255, -1)
                    inspection_mask_base64 = cv2_to_base64(mask)

                x, y, w, h = comp['x'], comp['y'], comp['width'], comp['height']
                roi_g = golden_full[y:y+h, x:x+w]
                body_small = cv2.resize(roi_g, (32, 32), interpolation=cv2.INTER_AREA)
                _, buf = cv2.imencode('.png', body_small)
                body_matrix_b64 = base64.b64encode(buf).decode('utf-8')

                cursor.execute(
                    """INSERT INTO components (product_id, name, x, y, width, height, package_id, rotation, inspection_mask, body_matrix)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (product_id, comp['name'], comp['x'], comp['y'], comp['width'], comp['height'],
                     package_id, comp.get('rotation', 0), inspection_mask_base64, body_matrix_b64)
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
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        produced_file = request.files.get('produced')
        product_id = request.form.get('product_id')
        
        produced_full = cv2.imdecode(np.frombuffer(produced_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # CORRIGIDO: Busca também os fiducials para o alinhamento
        cursor.execute("SELECT id, golden_image, fiducials FROM products WHERE id=%s",(product_id,))
        product = cursor.fetchone()
        golden_full = cv2.imread(product['golden_image'])

        # NOVO: Alinhamento da imagem produzida usando os fiduciais
        try:
            fiducials_data = json.loads(product.get('fiducials', '[]'))
            if not fiducials_data:
                print("Aviso: Nenhum fiducial encontrado para este produto. O alinhamento pode falhar.")
            
            produced_aligned = align_with_fiducials(golden_full, produced_full, fiducials_data)
            
            if produced_aligned is None:
                print("Erro crítico: O alinhamento da imagem falhou. Usando a imagem original como fallback.")
                produced_aligned = produced_full.copy()
        except Exception as e:
            print(f"Erro durante o alinhamento: {e}. Usando a imagem original como fallback.")
            produced_aligned = produced_full.copy()

        # A imagem de resultado agora é uma cópia da imagem ALINHADA
        result_image = produced_aligned.copy()

        cursor.execute("""
            SELECT c.*, p.name as package_name 
            FROM components c 
            JOIN packages p ON c.package_id = p.id
            WHERE c.product_id=%s
        """, (product_id,))
        comps = cursor.fetchall()

        detailed_components_frontend = []
        total_ok, total_fail = 0, 0

        for comp in comps:
            roi_g = golden_full[comp['y']:comp['y']+comp['height'], comp['x']:comp['x']+comp['width']]
            # CORRIGIDO: Usa a imagem alinhada para recortar a ROI de produção
            roi_p = produced_aligned[comp['y']:comp['y']+comp['height'], comp['x']:comp['x']+comp['width']]

            # CORRIGIDO: Usa a nova função de análise
            analysis = analyze_component_revised(roi_g, roi_p, comp.get('inspection_mask'))
            
            if analysis['status'] == "FAIL" and model:
                cnn_status, cnn_prob = predict_with_model(roi_p)
                # O CNN só substitui se ele disser que está OK.
                if cnn_status == 'OK':
                    analysis['status'] = 'OK'
                    analysis['details']['cnn_probability'] = float(cnn_prob)
                    analysis['details']['mode'] = 'CNN Override'
            else:
                analysis['details']['cnn_probability'] = None
                
            if analysis['status']=="OK": total_ok+=1
            else: total_fail+=1

            x, y, w, h = comp['x'], comp['y'], comp['width'], comp['height']
            color = (0, 255, 0) if analysis['status']=="OK" else (0, 0, 255)
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(result_image, f"{comp['name']} ({analysis['status']})", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            detailed_components_frontend.append({
                'name': comp['name'],
                'component_id': comp['id'],
                'package': comp['package_name'],
                'status': analysis['status'],
                'golden_image': cv2_to_base64(roi_g),
                'produced_image': cv2_to_base64(roi_p),
                'inspection_details': [analysis['details']],
                'rotation': analysis['rotation'],
                'displacement': analysis['displacement']
            })

        result_filename = f"inspection_result_{uuid.uuid4().hex}.png"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_image)
        
        success, error = save_inspection_results(
            conn, 
            cursor, 
            product_id, 
            total_ok, 
            total_fail, 
            detailed_components_frontend
        )

        if not success:
            print(f"Erro Crítico ao salvar resultados da inspeção: {error}")

        cursor.close()
        conn.close()

        return jsonify({
            'total_ok': total_ok,
            'total_fail': total_fail,
            'detailed_components': detailed_components_frontend,
            'result_filename': result_filename,
            'product_id': product_id
        })
    
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
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        user = data.get('user')
        feedbacks = data.get('feedbacks', {})

        if not feedbacks:
            return jsonify({'success': False, 'error': 'No feedback data received.'}), 400

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT id FROM inspections
            WHERE product_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (product_id,))
        last_inspection = cursor.fetchone()
        last_inspection_id = last_inspection['id'] if last_inspection else None

        for component_name, feedback in feedbacks.items():
            cursor.execute("""
                SELECT id, name, product_id, x, y, width, height
                FROM components
                WHERE name = %s AND product_id = %s
            """, (component_name, product_id))
            comp = cursor.fetchone()

            if not comp:
                print(f"Componente {component_name} não encontrado para o produto {product_id}")
                continue 

            cursor.execute("""
                INSERT INTO inspection_feedback (component_name, feedback, timestamp, user_id, inspection_id)
                VALUES (%s, %s, NOW(), %s, %s)
            """, (comp['name'], feedback, current_user.id, last_inspection_id))
            
            if last_inspection_id:
                cursor.execute("""
                    SELECT golden_roi_image, produced_roi_image 
                    FROM inspection_results 
                    WHERE inspection_id = %s AND component_id = %s
                """, (last_inspection_id, comp['id']))
                roi_images = cursor.fetchone()
                
                if roi_images:
                    golden_b64 = roi_images['golden_roi_image']
                    produced_b64 = roi_images['produced_roi_image']

                    cursor.execute("""
                        INSERT INTO training_samples (product_id, component_id, golden_base64, produced_base64, label)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (product_id, comp['id'], golden_b64, produced_b64, feedback))

        conn.commit()
        cursor.close()
        conn.close()

        import subprocess
        subprocess.Popen([sys.executable, "train_model.py"])

        return jsonify({'success': True, 'message': 'Feedback successfully saved!'})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Erro ao salvar feedback: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)