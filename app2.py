from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import cv2
import numpy as np
import os
import mysql.connector
import json
import base64
from skimage.metrics import structural_similarity as ssim
import bcrypt
import re
import datetime

# Inicialização do Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = os.urandom(24)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------------
# Configuração do Banco de Dados
# -------------------------------
DB_CONFIG = {
    'host': "localhost",
    'user': "flaskuser",
    'password': "123456",
    'database': "smt_inspection_new"
}

def get_db_connection():
    """Cria e retorna uma conexão com o banco de dados."""
    return mysql.connector.connect(**DB_CONFIG)

# -------------------------------
# Funções de Visão Computacional
# -------------------------------

def find_and_crop_board(image):
    """
    Encontra e recorta a placa na imagem de entrada.
    Esta versão é mais robusta, encontrando o maior contorno retangular.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detecção de bordas usando Canny
    edged = cv2.Canny(blurred, 50, 150)
    
    # Encontra contornos na imagem
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Inicializa variáveis para o maior contorno quadrilátero
    max_area = 0
    placa_contour = None
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        area = cv2.contourArea(contour)
        
        # Filtra por contornos que são quadriláteros e têm área considerável
        if len(approx) == 4 and area > 1000 and area > max_area:
            max_area = area
            placa_contour = approx

    if placa_contour is None:
        return None, None, None

    # Encontra o retângulo delimitador
    x, y, w, h = cv2.boundingRect(placa_contour)
    
    # Extrai a região da placa com uma pequena margem
    margin = 30
    x_marg = max(0, x - margin)
    y_marg = max(0, y - margin)
    w_marg = min(image.shape[1] - x_marg, w + 2 * margin)
    h_marg = min(image.shape[0] - y_marg, h + 2 * margin)
    
    cropped_board = image[y_marg:y_marg + h_marg, x_marg:x_marg + w_marg]
    
    # Retorna a placa recortada e as coordenadas originais
    return cropped_board, (x, y, w, h), (x_marg, y_marg, w_marg, h_marg)


def align_with_fiducials(golden_cropped, produced_cropped, fiducials_g):
    """
    Alinha a imagem produzida com a imagem golden usando pontos fiduciais.
    A função espera que fiducials_g seja uma lista de coordenadas [[x1, y1], [x2, y2], ...].
    """
    sift = cv2.SIFT_create()
    
    golden_gray = cv2.cvtColor(golden_cropped, cv2.COLOR_BGR2GRAY)
    produced_gray = cv2.cvtColor(produced_cropped, cv2.COLOR_BGR2GRAY)
    
    # Encontra os pontos-chave e descritores para as imagens
    kp_g, des_g = sift.detectAndCompute(golden_gray, None)
    kp_p, des_p = sift.detectAndCompute(produced_gray, None)

    if des_g is None or des_p is None:
        return cv2.resize(produced_cropped, (golden_cropped.shape[1], golden_cropped.shape[0]))

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_g, des_p, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Pega as coordenadas dos pontos correspondentes
    src_pts = np.float32([kp_g[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_p[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Encontra a matriz de homografia usando RANSAC
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    # Usa a matriz de homografia para alinhar a imagem
    if M is not None:
        aligned_image = cv2.warpPerspective(produced_cropped, M, (golden_cropped.shape[1], golden_cropped.shape[0]))
        return aligned_image
    
    # Se falhar, retorna a imagem redimensionada
    return cv2.resize(produced_cropped, (golden_cropped.shape[1], golden_cropped.shape[0]))


def analyze_component(roi_g, roi_p):
    """
    Analisa a similaridade entre duas ROIs (Regiões de Interesse).
    Inclui SSIM, correlação de histograma, e diferença de contornos.
    """
    # Verifica se as imagens são válidas
    if roi_g.size == 0 or roi_p.size == 0 or roi_g.shape != roi_p.shape:
        return { 'status': 'FAIL', 'details': 'Regiões de interesse inválidas ou de tamanhos diferentes.' }

    # Converte para escala de cinza
    roi_g_gray = cv2.cvtColor(roi_g, cv2.COLOR_BGR2GRAY)
    roi_p_gray = cv2.cvtColor(roi_p, cv2.COLOR_BGR2GRAY)

    # 1. Análise de similaridade estrutural (SSIM)
    try:
        ssim_value, _ = ssim(roi_g_gray, roi_p_gray, full=True)
    except ValueError:
        ssim_value = 0.0

    # 2. Análise de correlação de histograma de cores
    h_g = cv2.calcHist([roi_g], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    h_p = cv2.calcHist([roi_p], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_comp = cv2.compareHist(h_g, h_p, cv2.HISTCMP_CORREL)

    # 3. Análise de diferença de contornos
    _, thresh_g = cv2.threshold(roi_g_gray, 128, 255, cv2.THRESH_BINARY)
    _, thresh_p = cv2.threshold(roi_p_gray, 128, 255, cv2.THRESH_BINARY)
    contours_g, _ = cv2.findContours(thresh_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_p, _ = cv2.findContours(thresh_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_diff = abs(len(contours_g) - len(contours_p))

    # 4. Análise de paleta de cores para futuro treinamento de IA (vago, por isso simplificado)
    # Extrai os valores médios de cor
    avg_color_g = np.mean(roi_g, axis=(0, 1)).tolist()
    avg_color_p = np.mean(roi_p, axis=(0, 1)).tolist()

    status = "OK"
    if ssim_value < 0.75:
        status = "FAIL"
    if hist_comp < 0.8:
        status = "FAIL"
    if contour_diff > 2:
        status = "FAIL"
    
    return {
        'status': status,
        'ssim_value': float(ssim_value),
        'hist_correlation': float(hist_comp),
        'contour_diff': int(contour_diff),
        'extra_data_json': {
            'golden_avg_color': avg_color_g,
            'produced_avg_color': avg_color_p
        }
    }

def get_db_stats():
    """Retorna estatísticas do banco de dados para a página inicial."""
    conn = get_db_connection()
    cursor = conn.cursor()
    stats = {}
    cursor.execute("SELECT COUNT(*) FROM users")
    stats['users'] = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM products")
    stats['products'] = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM results")
    stats['inspections'] = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return stats

# -------------------------------
# ROTAS DO FLASK
# -------------------------------

@app.before_request
def check_authentication():
    """Verifica se o usuário está logado antes de processar a requisição, exceto para rotas públicas."""
    public_routes = ['login', 'static']
    if request.endpoint not in public_routes and 'logged_in' not in session:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Rota para o login de usuário."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT password_hash FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        
        if result and bcrypt.checkpw(password.encode('utf-8'), result[0].encode('utf-8')):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('index.html', page='login', error="Usuário ou senha inválidos.")
    return render_template('index.html', page='login')

@app.route('/logout')
def logout():
    """Rota para o logout de usuário."""
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/')
def dashboard():
    """Rota para o dashboard principal."""
    stats = get_db_stats()
    return render_template('index.html', page='dashboard', stats=stats)

@app.route('/products')
def products_page():
    """Rota para a página de produtos, listando todos os produtos cadastrados."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name, golden_image_path FROM products ORDER BY created_at DESC")
    products = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('index.html', page='products', products=products)

@app.route('/products/add')
def add_product_page():
    """Rota para a página de adicionar um novo produto."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM packages")
    packages = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('index.html', page='add_product', packages=packages)

@app.route('/products/inspect/<int:product_id>')
def inspect_product_page(product_id):
    """Rota para a página de inspeção de um produto específico."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM products WHERE id = %s", (product_id,))
    product = cursor.fetchone()
    cursor.close()
    conn.close()
    if not product:
        return redirect(url_for('dashboard'))
    return render_template('index.html', page='inspect_product', product=product)

@app.route('/api/find_and_crop', methods=['POST'])
def find_and_crop_api():
    """API para encontrar e recortar a placa da imagem enviada."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    img_stream = image_file.read()
    nparr = np.frombuffer(img_stream, np.uint8)
    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    cropped_board, _, coords = find_and_crop_board(image_cv)
    if cropped_board is None:
        return jsonify({'error': 'Failed to find and crop the board'}), 500

    retval, buffer = cv2.imencode('.jpg', cropped_board)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'cropped_image': base64_image,
        'golden_board_coords': coords
    })

@app.route('/api/products', methods=['POST'])
def add_product_api():
    """API para adicionar um novo produto (golden board) ao banco de dados."""
    try:
        name = request.form.get('name')
        golden_file = request.files.get('golden')
        fiducials_json = request.form.get('fiducials')
        components_json = request.form.get('components')
        golden_board_coords_json = request.form.get('golden_board_coords')

        if not name or not golden_file or not fiducials_json or not components_json or not golden_board_coords_json:
            return jsonify({'error': 'Missing form data'}), 400

        # Salva a imagem golden no servidor
        filename = f"golden_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{golden_file.filename}"
        golden_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        golden_file.save(golden_path)

        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insere o produto na tabela `products`
        cursor.execute(
            "INSERT INTO products (name, golden_image_path, fiducials_json, golden_board_coords) VALUES (%s, %s, %s, %s)",
            (name, golden_path, fiducials_json, golden_board_coords_json)
        )
        product_id = cursor.lastrowid

        # Insere os componentes na tabela `components`
        components = json.loads(components_json)
        for comp in components:
            cursor.execute(
                """INSERT INTO components (product_id, package_id, name, x, y, width, height)
                VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (product_id, comp['package_id'], comp['name'], comp['x'], comp['y'], comp['width'], comp['height'])
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

@app.route('/api/inspect', methods=['POST'])
def inspect_board_api():
    """API para inspecionar uma nova placa produzida."""
    produced_file = request.files.get('produced')
    product_id = request.form.get('product_id')

    if not produced_file or not product_id:
        return jsonify({'error': 'Missing file or product ID'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Busca os dados do produto golden no banco de dados
    cursor.execute("SELECT * FROM products WHERE id = %s", (product_id,))
    product = cursor.fetchone()
    if not product:
        cursor.close()
        conn.close()
        return jsonify({'error': 'Product not found'}), 404

    # Busca os componentes associados ao produto golden
    cursor.execute("SELECT * FROM components WHERE product_id = %s", (product_id,))
    golden_components = cursor.fetchall()
    
    # Salva a imagem produzida e a carrega com OpenCV
    produced_filename = f"produced_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{produced_file.filename}"
    produced_path = os.path.join(app.config['UPLOAD_FOLDER'], produced_filename)
    produced_file.save(produced_path)
    
    produced_img = cv2.imread(produced_path)
    golden_img = cv2.imread(product["golden_image_path"])

    # Recorta as placas
    produced_cropped, produced_coords, _ = find_and_crop_board(produced_img)
    if produced_cropped is None:
        return jsonify({'error': 'Failed to find and crop the produced board'}), 500

    # Alinha a imagem produzida com a golden
    fiducials_g = json.loads(product['fiducials_json'])
    produced_aligned = align_with_fiducials(golden_img, produced_cropped, fiducials_g)

    # Inicializa variáveis para os resultados
    inspection_results = {'total_ok': 0, 'total_fail': 0, 'components': [], 'result_filename': None}
    
    # Cria uma cópia da imagem alinhada para desenhar os retângulos
    result_image = produced_aligned.copy()

    # Prepara a imagem golden para a análise de componentes
    golden_gray = cv2.cvtColor(golden_img, cv2.COLOR_BGR2GRAY)
    produced_aligned_gray = cv2.cvtColor(produced_aligned, cv2.COLOR_BGR2GRAY)
    
    overall_status = "OK"

    # Itera sobre cada componente para inspecionar
    for comp in golden_components:
        x_g, y_g, w_g, h_g = comp['x'], comp['y'], comp['width'], comp['height']
        
        # Recorta a ROI do componente golden
        roi_g = golden_img[y_g:y_g + h_g, x_g:x_g + w_g]
        
        # Encontra a posição do componente na imagem produzida alinhada
        # Usa template matching para encontrar a melhor correspondência
        template = golden_gray[y_g:y_g + h_g, x_g:x_g + w_g]
        
        if template.size == 0 or template.shape[0] < 5 or template.shape[1] < 5:
            analysis = {'status': 'FAIL', 'details': 'Template inválido. Pulando.'}
        else:
            result_match = cv2.matchTemplate(produced_aligned_gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result_match)
            
            x_p, y_p = max_loc
            
            # Recorta a ROI do componente produzido na posição encontrada
            roi_p = produced_aligned[y_p:y_p + h_g, x_p:x_p + w_g]
            
            # Realiza a análise do componente
            analysis = analyze_component(roi_g, roi_p)
            
            # Desenha o retângulo na imagem de resultado
            if analysis['status'] == 'OK':
                inspection_results['total_ok'] += 1
                color = (0, 255, 0)
            else:
                inspection_results['total_fail'] += 1
                color = (0, 0, 255)
                overall_status = "FAIL"
            
            cv2.rectangle(result_image, (x_p, y_p), (x_p + w_g, y_p + h_g), color, 3)
            cv2.putText(result_image, f"{comp['name']}: {analysis['status']}", (x_p, y_p - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            inspection_results['components'].append({
                'name': comp['name'],
                'status': analysis['status'],
                'details': analysis
            })

    # Salva a imagem de resultado da inspeção
    result_filename = f'inspection_result_{product_id}_{os.path.basename(produced_filename)}'
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], result_filename), result_image)
    
    # Insere o resultado geral da inspeção no banco de dados
    cursor.execute(
        "INSERT INTO results (product_id, produced_image_path, overall_status) VALUES (%s, %s, %s)",
        (product_id, produced_path, overall_status)
    )
    result_id = cursor.lastrowid
    
    # Insere os resultados detalhados dos componentes
    for comp_result in inspection_results['components']:
        component_id = next(item['id'] for item in golden_components if item['name'] == comp_result['name'])
        details = comp_result['details']
        cursor.execute(
            """INSERT INTO component_results (result_id, component_id, status, ssim_value, hist_correlation, contour_diff, extra_data_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (result_id, component_id, details['status'], details['ssim_value'], details['hist_correlation'], details['contour_diff'], json.dumps(details['extra_data_json']))
        )
    conn.commit()

    inspection_results['result_filename'] = result_filename
    cursor.close()
    conn.close()
    
    return jsonify(inspection_results)

# -------------------------------
# Frontend HTML/CSS/JS (um único arquivo)
# -------------------------------

@app.route('/<page_name>')
def render_dynamic_page(page_name):
    # Rota genérica para carregar as páginas do frontend
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    if page_name == 'products':
        cursor.execute("SELECT id, name, golden_image_path FROM products ORDER BY created_at DESC")
        data = cursor.fetchall()
        return render_template('index.html', page=page_name, products=data)
    elif page_name == 'add_product':
        cursor.execute("SELECT id, name FROM packages")
        data = cursor.fetchall()
        return render_template('index.html', page=page_name, packages=data)
    elif 'inspect' in page_name:
        product_id = re.search(r'\d+', page_name).group()
        cursor.execute("SELECT id, name FROM products WHERE id = %s", (product_id,))
        data = cursor.fetchone()
        return render_template('index.html', page='inspect_product', product=data)
    else:
        stats = get_db_stats()
        return render_template('index.html', page=page_name, stats=stats)


@app.route('/inspections/results/<int:product_id>')
def inspection_results_page(product_id):
    """Rota para a página de resultados de inspeção."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT
            r.overall_status,
            r.timestamp,
            r.id,
            p.name as product_name
        FROM results r
        JOIN products p ON r.product_id = p.id
        WHERE r.product_id = %s
        ORDER BY r.timestamp DESC
    """, (product_id,))
    results = cursor.fetchall()
    
    cursor.execute("SELECT name FROM products WHERE id = %s", (product_id,))
    product_name = cursor.fetchone()['name']
    
    cursor.close()
    conn.close()
    return render_template('index.html', page='results_page', results=results, product_id=product_id, product_name=product_name)

@app.route('/inspections/details/<int:result_id>')
def inspection_details_page(result_id):
    """Rota para ver os detalhes de uma inspeção específica."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT
            p.name AS product_name,
            p.golden_image_path,
            r.produced_image_path,
            r.timestamp,
            r.overall_status
        FROM results r
        JOIN products p ON r.product_id = p.id
        WHERE r.id = %s
    """, (result_id,))
    main_result = cursor.fetchone()

    cursor.execute("""
        SELECT
            c.name AS component_name,
            c.x AS golden_x, c.y AS golden_y, c.width AS golden_w, c.height AS golden_h,
            cr.status,
            cr.ssim_value,
            cr.hist_correlation,
            cr.contour_diff,
            cr.extra_data_json
        FROM component_results cr
        JOIN components c ON cr.component_id = c.id
        WHERE cr.result_id = %s
    """, (result_id,))
    component_results = cursor.fetchall()

    cursor.close()
    conn.close()
    
    if not main_result:
        return redirect(url_for('dashboard'))

    return render_template('index.html', page='details_page', main_result=main_result, component_results=component_results)

@app.route('/api/get_product_data/<int:product_id>')
def get_product_data(product_id):
    """API para obter dados de um produto para exibição."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT golden_image_path, fiducials_json, golden_board_coords FROM products WHERE id = %s", (product_id,))
    product = cursor.fetchone()
    
    cursor.close()
    conn.close()
    if not product:
        return jsonify({'error': 'Product not found'}), 404
    
    product['fiducials_json'] = json.loads(product['fiducials_json'])
    product['golden_board_coords'] = json.loads(product['golden_board_coords'])
    
    return jsonify(product)


@app.route('/api/get_packages')
def get_packages_api():
    """API para obter a lista de pacotes para o frontend."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM packages")
    packages = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(packages)

@app.route('/api/get_components_by_product/<int:product_id>')
def get_components_by_product_api(product_id):
    """API para obter os componentes de um produto específico."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name, x, y, width, height FROM components WHERE product_id = %s", (product_id,))
    components = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(components)


if __name__ == '__main__':
    # Exemplo de hash de senha para 'admin'
    # hashed_password = bcrypt.hashpw('admin'.encode('utf-8'), bcrypt.gensalt())
    # print(hashed_password) # Use este hash para a tabela `users`
    
    app.run(debug=True, use_reloader=False)
