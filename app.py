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
        h, w = golden_img.shape[:2]
        return cv2.resize(produced_img, (w, h))
    golden_points, produced_points = [], []
    for f in fiducials:
        roi_size = int(f['r'] * 3)
        roi_x_start = max(0, int(f['x'] - roi_size))
        roi_y_start = max(0, int(f['y'] - roi_size))
        roi_x_end = min(produced_img.shape[1], int(f['x'] + roi_size))
        roi_y_end = min(produced_img.shape[0], int(f['y'] + roi_size))
        roi_p = produced_img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        if roi_p.size == 0: continue
        detected_rings = find_fiducial_rings(roi_p)
        if detected_rings:
            expected_center_in_roi = (f['x'] - roi_x_start, f['y'] - roi_y_start)
            closest_ring = min(detected_rings, key=lambda ring: np.hypot(ring['x'] - expected_center_in_roi[0], ring['y'] - expected_center_in_roi[1]))
            golden_points.append([f['x'], f['y']])
            produced_points.append([roi_x_start + closest_ring['x'], roi_y_start + closest_ring['y']])
    if len(golden_points) < 3:
        h, w = golden_img.shape[:2]
        return cv2.resize(produced_img, (w, h))
    M, _ = cv2.findHomography(np.float32(produced_points), np.float32(golden_points), cv2.RANSAC, 5.0)
    if M is not None:
        h, w = golden_img.shape[:2]
        return cv2.warpPerspective(produced_img, M, (w, h))
    h, w = golden_img.shape[:2]
    return cv2.resize(produced_img, (w, h))

def rotate_image(image, angle):
    if angle == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def analyze_component(roi_g, roi_p, inspection_mask_base64=None, body_matrix_b64=None):
    try:
        if body_matrix_b64:
            ref_img = base64_to_cv2_img(body_matrix_b64)
            body_small_p = cv2.resize(roi_p, (32, 32), interpolation=cv2.INTER_AREA)
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            prod_gray = cv2.cvtColor(body_small_p, cv2.COLOR_BGR2GRAY)
            score, _ = ssim(ref_gray, prod_gray, full=True)
            status = "OK" if score > 0.80 else "FAIL"
            return {
                'status': status,
                'rotation': '0°',
                'displacement': {'x': 0, 'y': 0},
                'details': {
                    'mode': 'Body Matrix',
                    'ssim': float(score),
                    'body_matrix_ref': body_matrix_b64,
                    'body_matrix_prod': cv2_to_base64(body_small_p)
                }
            }
    except Exception as e:
        print(f"Erro análise body_matrix: {e}")

    inspection_mask = base64_to_cv2_mask(inspection_mask_base64)
    if inspection_mask is None:
        try:
            h, w = roi_g.shape[:2]
            roi_p_resized = cv2.resize(roi_p, (w, h))
            roi_g_gray = cv2.cvtColor(roi_g, cv2.COLOR_BGR2GRAY)
            roi_p_gray = cv2.cvtColor(roi_p_resized, cv2.COLOR_BGR2GRAY)
            ssim_value, _ = ssim(roi_g_gray, roi_p_gray, full=True)
        except:
            ssim_value = 0.0
        status = "OK" if ssim_value > 0.80 else "FAIL"
        return {
            'status': status,
            'rotation': '0°',
            'displacement': {'x': 0, 'y': 0},
            'details': {'mode': 'ROI Fallback', 'ssim': float(ssim_value)}
        }
    else:
        masked_g = cv2.bitwise_and(roi_g, roi_g, mask=inspection_mask)
        masked_p = cv2.bitwise_and(roi_p, roi_p, mask=inspection_mask)
        try:
            ssim_value, _ = ssim(
                cv2.cvtColor(masked_g, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(masked_p, cv2.COLOR_BGR2GRAY),
                full=True
            )
        except:
            ssim_value = 0.0
        status = "OK" if ssim_value > 0.80 else "FAIL"
        return {
            'status': status,
            'rotation': '0°',
            'displacement': {'x': 0, 'y': 0},
            'details': {'mode': 'Mask Fallback', 'ssim': float(ssim_value)}
        }

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
            cursor = conn.cursor()
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

                # gerar body_matrix
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

        cursor.execute("SELECT * FROM products WHERE id=%s",(product_id,))
        product = cursor.fetchone()
        golden_full = cv2.imread(product['golden_image'])

        cursor.execute("SELECT * FROM components WHERE product_id=%s",(product_id,))
        comps = cursor.fetchall()

        result_image = produced_full.copy()
        detailed_components_frontend = []
        total_ok,total_fail = 0,0

        for comp in comps:
            roi_g = golden_full[comp['y']:comp['y']+comp['height'], comp['x']:comp['x']+comp['width']]
            roi_p = produced_full[comp['y']:comp['y']+comp['height'], comp['x']:comp['x']+comp['width']]

            analysis = analyze_component(roi_g, roi_p, comp.get('inspection_mask'), comp.get('body_matrix'))

            if analysis['status']=="OK": total_ok+=1
            else: total_fail+=1

            # desenha no overlay
            x, y, w, h = comp['x'], comp['y'], comp['width'], comp['height']
            color = (0, 255, 0) if analysis['status']=="OK" else (0, 0, 255)
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(result_image, comp['name'], (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            detailed_components_frontend.append({
                'name': comp['name'],
                'package': comp['package_id'],
                'status': analysis['status'],
                'golden_image': cv2_to_base64(roi_g),
                'produced_image': cv2_to_base64(roi_p),
                'inspection_details': [analysis['details']],
                'rotation': analysis['rotation'],
                'displacement': analysis['displacement']
            })

        # salva imagem resultante
        result_filename = f"inspection_result_{uuid.uuid4().hex}.png"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_image)

        return jsonify({
            'total_ok': total_ok,
            'total_fail': total_fail,
            'detailed_components': detailed_components_frontend,
            'result_filename': result_filename
        })

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM products ORDER BY name")
    products = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('inspect.html', products=products)

if __name__ == '__main__':
    app.run(debug=True)