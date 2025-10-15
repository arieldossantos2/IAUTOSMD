from flask import Flask, render_template, request, jsonify, redirect, url_for, session
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
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'uma_chave_secreta_muito_segura'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------------
# Configuração do Flask-Login
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
# Conexão com o banco de dados
# -------------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="flaskuser",
        password="123456",
        database="smt_inspection"
    )

# -------------------------------
# Funções de Visão Computacional
# -------------------------------

def find_and_crop_board(image):
    """
    Encontra e recorta a placa na imagem.
    Aprimorado para encontrar o maior contorno retangular verde.
    """
    if image is None:
        return None, None
        
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    max_area = 0
    placa_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx) == 4 and area > max_area:
                max_area = area
                placa_contour = approx

    if placa_contour is None:
        return None, None

    x, y, w, h = cv2.boundingRect(placa_contour)
    margin = 30
    x_marg = max(0, x - margin)
    y_marg = max(0, y - margin)
    w_marg = min(image.shape[1] - x_marg, w + 2 * margin)
    h_marg = min(image.shape[0] - y_marg, h + 2 * margin)

    cropped_board = image[y_marg:y_marg+h_marg, x_marg:x_marg+w_marg]
    
    return cropped_board, (x_marg, y_marg, w_marg, h_marg)

def find_fiducial_rings(image):
    """
    Detecta anéis concêntricos (fiducials) em uma imagem usando uma abordagem de cores e Hough Transform.
    """
    if image is None:
        return []
    
    # Converte para HSV para filtrar pela cor
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Limites para a cor do fiducial (amarelo/dourado)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Encontra o contorno do anel
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rings = []
    for contour in contours:
        # Aproxima o contorno como um círculo
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Filtra por contornos que parecem círculos
        if radius > 5 and radius < 30 and cv2.contourArea(contour) / (np.pi * radius**2) > 0.6:
            rings.append({'x': center[0], 'y': center[1], 'r': radius})

    return rings

def align_with_fiducials(golden_img, produced_img, fiducials):
    """
    Alinha a imagem de produção à imagem golden usando os centros dos fiduciais.
    """
    if produced_img is None:
        print("Imagem de produção é nula. Impossível alinhar.")
        return None

    if len(fiducials) < 4:
        print("Menos de 4 fiduciais fornecidos. Usando redimensionamento simples.")
        h, w = golden_img.shape[:2]
        return cv2.resize(produced_img, (w, h))

    golden_points = []
    produced_points = []
    
    for f in fiducials:
        roi_p = produced_img[int(f['y'] - f['r']):int(f['y'] + f['r']), int(f['x'] - f['r']):int(f['x'] + f['r'])]
        detected_rings = find_fiducial_rings(roi_p)
        
        if detected_rings:
            center_x = int(f['x'] - f['r'] + detected_rings[0]['x'])
            center_y = int(f['y'] - f['r'] + detected_rings[0]['y'])
            golden_points.append([f['x'], f['y']])
            produced_points.append([center_x, center_y])
        else:
            print(f"Anel fiducial em ({f['x']},{f['y']}) não encontrado na imagem de produção. Pulando.")

    if len(golden_points) < 4:
        print("Poucos pontos fiduciais válidos. Usando redimensionamento simples.")
        h, w = golden_img.shape[:2]
        return cv2.resize(produced_img, (w, h))

    golden_pts = np.float32(golden_points)
    produced_pts = np.float32(produced_points)

    M, mask = cv2.findHomography(produced_pts, golden_pts, cv2.RANSAC, 5.0)

    if M is not None:
        h, w = golden_img.shape[:2]
        return cv2.warpPerspective(produced_img, M, (w, h))
    
    return produced_img

def analyze_component(roi_g, roi_p):
    """
    Compara duas imagens de componentes usando SSIM e histograma de cores.
    Retorna um dicionário com os resultados.
    Ajustado para ser mais tolerante com pequenas distorções de alinhamento.
    """
    try:
        roi_g_gray = cv2.cvtColor(roi_g, cv2.COLOR_BGR2GRAY)
        roi_p_gray = cv2.cvtColor(roi_p, cv2.COLOR_BGR2GRAY)
        ssim_value, _ = ssim(roi_g_gray, roi_p_gray, full=True)
    except ValueError:
        ssim_value = 0.0

    h_g = cv2.calcHist([roi_g], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    h_p = cv2.calcHist([roi_p], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_comp = cv2.compareHist(h_g, h_p, cv2.HISTCMP_CORREL)

    print(f"SSIM: {ssim_value:.2f}, Hist_corr: {hist_comp:.2f}")

    status = "OK"
    if ssim_value < 0.8:
        status = "FAIL"
    if hist_comp < 0.85:
        status = "FAIL"
    
    return {
        'ssim': float(ssim_value),
        'hist_corr': float(hist_comp),
        'status': status
    }

# Função auxiliar para converter imagem OpenCV para Base64
def cv2_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

# -------------------------------
# ROTAS DA APLICAÇÃO
# -------------------------------

@app.route('/')
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM products")
    products = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return render_template('index.html', products=products)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
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
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        hashed_password = generate_password_hash(password)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed_password))
            conn.commit()
            return redirect(url_for('login'))
        except mysql.connector.Error as err:
            if err.errno == 1062:
                return render_template('register.html', error="Nome de usuário já existe.")
            else:
                return render_template('register.html', error=f"Erro: {err}")
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
        nparr = np.frombuffer(img_bytes, np.uint8)
        image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_cv is None:
            return jsonify({'error': 'Falha ao decodificar a imagem.'}), 400
        
        detected_rings = find_fiducial_rings(image_cv)

        # Cria uma imagem de depuração com os anéis detectados
        debug_image = image_cv.copy()
        if detected_rings:
            for ring in detected_rings:
                center = (ring['x'], ring['y'])
                radius = ring['r']
                cv2.circle(debug_image, center, radius, (0, 255, 0), 2)
                cv2.circle(debug_image, center, 2, (0, 0, 255), 3)

        debug_image_base64 = cv2_to_base64(debug_image)

        return jsonify({'circles': detected_rings, 'debug_image': debug_image_base64})
    
    except Exception as e:
        print(f"Erro ao encontrar fiduciais: {e}")
        return jsonify({'error': f'Erro ao encontrar fiduciais: {str(e)}'}), 500

@app.route('/add_product', methods=['GET', 'POST'])
@login_required
def add_product():
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            golden_file = request.files.get('golden')
            fiducials_json = request.form.get('fiducials')
            components_json = request.form.get('components')

            if not name or not golden_file or not fiducials_json or not components_json:
                return jsonify({'error': 'Dados do formulário ausentes'}), 400

            fiducials = json.loads(fiducials_json)
            components = json.loads(components_json)
            
            golden_path = os.path.join(app.config['UPLOAD_FOLDER'], 'golden_' + golden_file.filename)
            golden_file.save(golden_path)

            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO products (name, golden_image, fiducials) VALUES (%s, %s, %s)",
                (name, golden_path, fiducials_json)
            )
            product_id = cursor.lastrowid
            
            for comp in components:
                cursor.execute("INSERT IGNORE INTO packages (name) VALUES (%s)", (comp['package'],))
                conn.commit()
                cursor.execute("SELECT id FROM packages WHERE name = %s", (comp['package'],))
                package_id = cursor.fetchone()[0]

                cursor.execute(
                    """INSERT INTO components (product_id, name, x, y, width, height, package_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (product_id, comp['name'], comp['x'], comp['y'], comp['width'], comp['height'], package_id)
                )
            
            conn.commit()
            
            return jsonify({'success': True, 'product_id': product_id})

        except json.JSONDecodeError as e:
            return jsonify({'error': f"Formato JSON inválido: {str(e)}"}), 400
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
            print(f"Erro ao salvar no banco de dados: {str(e)}")
            return jsonify({'error': f"Falha ao salvar no banco de dados: {str(e)}"}), 500
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    return render_template('add_product.html')

@app.route('/inspect', methods=['GET', 'POST'])
@login_required
def inspect_board():
    if request.method == 'POST':
        try:
            produced_file = request.files.get('produced')
            product_id = request.form.get('product_id')

            if not produced_file or not product_id:
                return jsonify({'error': 'Arquivo ou ID do produto ausente'}), 400
            
            produced_path = os.path.join(app.config['UPLOAD_FOLDER'], 'produced_' + produced_file.filename)
            produced_file.save(produced_path)

            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute("SELECT * FROM products WHERE id = %s", (product_id,))
            product = cursor.fetchone()
            if not product:
                cursor.close()
                conn.close()
                return jsonify({'error': 'Produto não encontrado'}), 404

            golden_fiducials = json.loads(product["fiducials"])
            
            cursor.execute("SELECT c.*, p.name as package_name FROM components c LEFT JOIN packages p ON c.package_id = p.id WHERE c.product_id = %s", (product_id,))
            golden_components = cursor.fetchall()

            cursor.close()
            conn.close()

            if not os.path.exists(product["golden_image"]):
                return jsonify({'error': f"Arquivo golden não encontrado: {product['golden_image']}"}), 500
            
            golden_cropped = cv2.imread(product["golden_image"])
            produced = cv2.imread(produced_path)
            
            if produced is None:
                return jsonify({'error': 'Falha ao carregar a imagem produzida. Verifique o formato do arquivo.'}), 500

            if golden_cropped is None:
                return jsonify({'error': 'Falha ao carregar a imagem golden. Verifique o formato do arquivo.'}), 500
            
            produced_cropped, _ = find_and_crop_board(produced)

            if produced_cropped is None:
                return jsonify({'error': 'Falha ao encontrar e recortar a placa produzida'}), 500
                
            produced_aligned = align_with_fiducials(golden_cropped, produced_cropped, golden_fiducials)
            
            if produced_aligned is None:
                return jsonify({'error': 'Falha ao alinhar a imagem produzida'}), 500

            produced_aligned = cv2.resize(produced_aligned, (golden_cropped.shape[1], golden_cropped.shape[0]))

            result_image = produced_aligned.copy()
            inspection_results = {'total_ok': 0, 'total_fail': 0, 'components': []}
            detailed_components = []

            for comp in golden_components:
                x, y, w, h = comp['x'], comp['y'], comp['width'], comp['height']
                
                roi_g = golden_cropped[y:y+h, x:x+w]
                roi_p = produced_aligned[y:y+h, x:x+w]
                
                if roi_g.size == 0 or roi_p.size == 0:
                    analysis = {'status': 'FAIL', 'details': 'ROI inválida'}
                else:
                    analysis = analyze_component(roi_g, roi_p)
                    
                if analysis['status'] == 'OK':
                    inspection_results['total_ok'] += 1
                    color = (0, 255, 0)
                else:
                    inspection_results['total_fail'] += 1
                    color = (0, 0, 255)
                    
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
                cv2.putText(result_image, comp['name'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                inspection_results['components'].append({
                    'name': comp['name'],
                    'package': comp['package_name'],
                    'status': analysis['status'],
                    'details': analysis
                })

                detailed_components.append({
                    'name': comp['name'],
                    'status': analysis['status'],
                    'golden_image': cv2_to_base64(roi_g),
                    'produced_image': cv2_to_base64(roi_p)
                })

            result_filename = f'inspection_result_{product_id}_{os.path.basename(produced_file.filename)}'
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], result_filename), result_image)
            
            inspection_results['result_filename'] = result_filename
            inspection_results['detailed_components'] = detailed_components
            
            conn_save = get_db_connection()
            cursor_save = conn_save.cursor()
            cursor_save.execute(
                """INSERT INTO inspection_results (product_id, produced_image, overall_status, details)
                VALUES (%s, %s, %s, %s)""",
                (product_id, produced_path, 'OK' if inspection_results['total_fail'] == 0 else 'FAIL', json.dumps(inspection_results['components']))
            )
            conn_save.commit()
            cursor_save.close()
            conn_save.close()
            
            return jsonify(inspection_results)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    else:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name FROM products")
        products = cursor.fetchall()
        cursor.close()
        conn.close()
        return render_template('inspect.html', products=products)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
