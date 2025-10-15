from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import mysql.connector
import json
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------------
# Conexão com o banco
# -------------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="flaskuser",
        password="123456",
        database="smt_inspection"
    )

# -------------------------------
# Função para alinhar com fiduciais
# -------------------------------
def align_with_fiducials(golden, produced, fiducials_g, fiducials_p):
    g_pts = np.float32([[p["x"], p["y"]] for p in fiducials_g])
    p_pts = np.float32([[p["x"], p["y"]] for p in fiducials_p])

    n = min(len(g_pts), len(p_pts))

    if n < 2:
        print("⚠️ Menos de 2 fiduciais, sem alinhamento")
        return produced

    if n == 2 or n == 3:
        M, _ = cv2.estimateAffinePartial2D(p_pts[:n], g_pts[:n])
        if M is not None:
            return cv2.warpAffine(produced, M, (golden.shape[1], golden.shape[0]))
    elif n >= 4:
        M = cv2.getPerspectiveTransform(p_pts[:4], g_pts[:4])
        return cv2.warpPerspective(produced, M, (golden.shape[1], golden.shape[0]))

    return produced

# -------------------------------
# Função para inspecionar
# -------------------------------
def inspect_board(product_id, produced_path):
    # 1️⃣ Buscar produto e componentes do banco
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Produto
    cursor.execute("SELECT * FROM products WHERE id = %s", (product_id,))
    product = cursor.fetchone()
    if not product:
        cursor.close()
        conn.close()
        return None

    # Componentes
    cursor.execute("SELECT * FROM components WHERE product_id = %s", (product_id,))
    components = cursor.fetchall()

    cursor.close()
    conn.close()

    # 2️⃣ Carregar imagens
    golden = cv2.imread(product["golden_image"])
    produced = cv2.imread(produced_path)

    # Redimensiona a imagem produzida para o mesmo tamanho do golden
    produced = cv2.resize(produced, (golden.shape[1], golden.shape[0]))

    result = produced.copy()

    # 3️⃣ Inspecionar cada componente
    conn = get_db_connection()
    cursor = conn.cursor()

    for comp in components:
        x, y = comp['x'], comp['y']
        w = comp.get('width', 50)
        h = comp.get('height', 50)

        roi_g = golden[y:y+h, x:x+w]
        roi_p = produced[y:y+h, x:x+w]

        roi_g_gray = cv2.cvtColor(roi_g, cv2.COLOR_BGR2GRAY)
        roi_p_gray = cv2.cvtColor(roi_p, cv2.COLOR_BGR2GRAY)

        sim, diff = ssim(roi_g_gray, roi_p_gray, full=True)
        status = "OK" if sim > 0.85 else "FAIL"
        color = (0, 255, 0) if status == "OK" else (0, 0, 255)

        # Desenhar retângulo e texto
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 3)
        cv2.putText(result, f"{comp['name']} {status}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Salvar resultado no banco
        cursor.execute("""
            INSERT INTO results (product_id, filename, component_name, similarity, status)
            VALUES (%s, %s, %s, %s, %s)
        """, (product_id, f'inspection_result_{product_id}.png', comp['name'], float(sim), status))

    conn.commit()
    cursor.close()
    conn.close()

    # 4️⃣ Salvar imagem final
    result_filename = f'inspection_result_{product_id}.png'
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], result_filename), result)

    return result_filename

# -------------------------------
# Rotas
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        produced_file = request.files.get('produced')
        product_id = request.form.get('product_id')

        if not produced_file or not product_id:
            return "Precisa enviar a imagem produzida e escolher produto"

        produced_path = os.path.join(app.config['UPLOAD_FOLDER'], 'produced_' + produced_file.filename)
        produced_file.save(produced_path)

        result_filename = inspect_board(product_id, produced_path)

        return render_template('index.html',
                               produced_filename='produced_' + produced_file.filename,
                               result_filename=result_filename)

    # Carregar lista de produtos para seleção
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM products")
    products = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('index.html', products=products)

# -------------------------------
# Rota para cadastrar novo produto (CORRIGIDA E SIMPLIFICADA)
# -------------------------------
@app.route('/add_product', methods=['GET', 'POST'])
def add_product():
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            golden_file = request.files.get('golden')
            fiducials_json = request.form.get('fiducials')
            components_json = request.form.get('components')

            if not name or not golden_file or not fiducials_json or not components_json:
                return "Faltam dados"

            # Validar JSONs
            fiducials = json.loads(fiducials_json)
            components = json.loads(components_json)
            
            if len(fiducials) < 2:
                return "É necessário marcar pelo menos 2 pontos fiduciais"
            if len(components) < 1:
                return "É necessário marcar pelo menos 1 componente"

            # Salva a imagem
            golden_path = os.path.join(app.config['UPLOAD_FOLDER'], 'golden_' + golden_file.filename)
            golden_file.save(golden_path)

            # Conectar ao banco
            conn = get_db_connection()
            cursor = conn.cursor()

            # Insert produto
            cursor.execute(
                "INSERT INTO products (name, golden_image, fiducials) VALUES (%s, %s, %s)",
                (name, golden_path, fiducials_json)
            )
            product_id = cursor.lastrowid
            print(f"Produto inserido com ID: {product_id}")

            # Insert componentes
            for comp in components:
                cursor.execute(
                    """INSERT INTO components (product_id, name, x, y, rotation, package) 
                    VALUES (%s, %s, %s, %s, %s, %s)""",
                    (product_id, comp['name'], comp['x'], comp['y'], 
                     comp['rotation'], comp['type'])
                )
                print(f"Componente inserido: {comp}")

            # Commit final
            conn.commit()
            print("Commit realizado com sucesso!")
            
            return "Produto cadastrado com sucesso!"

        except json.JSONDecodeError as e:
            return f"Formato inválido de JSON: {str(e)}"
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
            print(f"Erro no banco de dados: {str(e)}")
            return f"Erro ao salvar no banco: {str(e)}"
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    return render_template('add_product.html')

if __name__ == '__main__':
    print("Iniciando Flask...")
    app.run(debug=True, use_reloader=False)