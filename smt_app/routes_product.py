# smt_app/routes_product.py
from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required
import mysql.connector
import json
import traceback
import cv2
import numpy as np
import os
from .db_helpers import get_db_connection, base64_to_cv2_img, save_image_to_disk, cv2_to_base64
from .vision import find_fiducial_rings, align_with_fiducials, analyze_component_package_based, predict_with_model
import base64


bp = Blueprint('product', __name__)

@bp.route('/find_fiducials', methods=['POST'])
@login_required
def find_fiducials_route():
    try:
        data = request.get_json()
        base64_image = data['image_data'].split(',')[1]
        img_bytes = base64.b64decode(base64_image)
        image_cv = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_cv is None: return jsonify({'error': 'Falha ao decodificar a imagem.'}), 400
        
        # Esta função agora é importada de .vision
        detected_rings = find_fiducial_rings(image_cv) 
        
        debug_image = image_cv.copy()
        if detected_rings:
            for ring in detected_rings:
                cv2.circle(debug_image, (ring['x'], ring['y']), ring['r'], (0, 255, 0), 2)
                cv2.circle(debug_image, (ring['x'], ring['y']), 2, (0, 0, 255), 3)
        return jsonify({'circles': detected_rings, 'debug_image': cv2_to_base64(debug_image)})
    except Exception as e:
        traceback.print_exc() # Adiciona isso para ver o erro real no console
        return jsonify({'error': f'Erro ao encontrar fiduciais: {str(e)}'}), 500
# --- FIM DA ROTA MOVIDA ---

@bp.route('/add_product', methods=['GET', 'POST'])
@login_required
def add_product():
    if request.method == 'POST':
        conn, cursor = None, None
        try:
            name = request.form.get('name')
            golden_file = request.files.get('golden')
            fiducials_json = request.form.get('fiducials')
            components_json = request.form.get('components')

            if not all([name, golden_file, fiducials_json, components_json]):
                return jsonify({'error': 'Dados incompletos.'}), 400

            golden_path_db = save_image_to_disk(
                cv2.imdecode(np.frombuffer(golden_file.read(), np.uint8), cv2.IMREAD_COLOR),
                'uploads',
                f"golden_{str(uuid.uuid4())[:8]}"
            )
            if not golden_path_db:
                 return jsonify({'error': 'Falha ao salvar a imagem golden.'}), 500

            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True, buffered=True)

            cursor.execute("INSERT INTO products (name, golden_image, fiducials) VALUES (%s, %s, %s)", (name, golden_path_db, fiducials_json))
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

                # Se o pacote NÃO TIVER um template/máscara no DB...
                if not package_info[package_name]['has_matrix']:
                    print(f"Criando novo template e máscara multi-região para: {package_name}")
                    
                    # ATUALIZADO: Espera 'final_body_rects' (lista)
                    if 'component_roi_b64' not in comp or 'final_body_rects' not in comp:
                        conn.rollback() 
                        return jsonify({'error': f"Componente '{comp['name']}' é o primeiro do pacote '{package_name}', mas os dados de definição do corpo (ROI e Rects) não foram enviados."}), 400
                    
                    roi_g_img = base64_to_cv2_img(comp['component_roi_b64'])
                    rects = comp['final_body_rects']
                    
                    if roi_g_img is None:
                         conn.rollback()
                         return jsonify({'error': f"Falha ao decodificar a ROI para '{comp['name']}'."}), 400
                    if not rects:
                         conn.rollback()
                         return jsonify({'error': f"Nenhuma região de corpo definida para '{comp['name']}'."}), 400

                    # 1. Encontra o Bounding Box de todas as regiões
                    min_x = min(r['x'] for r in rects)
                    min_y = min(r['y'] for r in rects)
                    max_x = max(r['x'] + r['width'] for r in rects)
                    max_y = max(r['y'] + r['height'] for r in rects)
                    
                    bb_x, bb_y = min_x, min_y
                    bb_w, bb_h = max_x - min_x, max_y - min_y

                    # 2. Cria a IMAGEM DO TEMPLATE (o recorte do bounding box)
                    body_template_img = roi_g_img[bb_y:bb_y+bb_h, bb_x:bb_x+bb_w]
                    
                    if body_template_img.size == 0:
                         conn.rollback()
                         return jsonify({'error': f"Região de corpo (Bounding Box) inválida para '{comp['name']}'."}), 400

                    body_matrix_path = save_image_to_disk(body_template_img, 'packages', f"pkg_{package_id}_{package_name}_template")
                    
                    # 3. Cria a IMAGEM DA MÁSCARA (multi-região)
                    body_mask_img = np.zeros((bb_h, bb_w), dtype=np.uint8)
                    
                    for r in rects:
                        # Ajusta as coordenadas do rect para serem relativas ao bounding box
                        rel_x = r['x'] - bb_x
                        rel_y = r['y'] - bb_y
                        cv2.rectangle(body_mask_img, 
                                      (rel_x, rel_y), 
                                      (rel_x + r['width'], rel_y + r['height']), 
                                      255, -1)

                    body_mask_path = save_image_to_disk(body_mask_img, 'packages', f"pkg_{package_id}_{package_name}_mask")

                    if not body_matrix_path or not body_mask_path:
                        conn.rollback()
                        return jsonify({'error': f"Falha ao salvar arquivos de template/máscara para o pacote '{package_name}'."}), 500
                    
                    cursor.execute("UPDATE packages SET body_matrix = %s, body_mask = %s WHERE id = %s", 
                                   (body_matrix_path, body_mask_path, package_id))
                    package_info[package_name]['has_matrix'] = True
                
                # Salva o componente
                cursor.execute(
                    """INSERT INTO components (product_id, name, x, y, width, height, package_id, rotation)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (product_id, comp['name'], comp['x'], comp['y'], comp['width'], comp['height'],
                     package_id, comp.get('rotation', 0))
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


@bp.route('/suggest_body', methods=['POST'])
@login_required
def suggest_body():
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
            return jsonify([{'x': int(w*0.1), 'y': int(h*0.1), 'width': int(w*0.8), 'height': int(h*0.8)}])

        # Retorna TODOS os contornos centrais (ou apenas o mais central)
        # Vamos manter o mais central por enquanto, o JS implementa multi-região
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
             return jsonify([{'x': int(w*0.1), 'y': int(h*0.1), 'width': int(w*0.8), 'height': int(h*0.8)}])

        x, y, w, h = cv2.boundingRect(best_contour)
        # Retorna como uma LISTA para ser compatível com a nova lógica
        return jsonify([{'x': x, 'y': y, 'width': w, 'height': h}])

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@bp.route('/find_body_in_roi', methods=['POST'])
@login_required
def find_body_in_roi():
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
        
        template_img = cv2.imread(os.path.join(current_app.static_folder, pkg_data['body_matrix']))
        template_mask = cv2.imread(os.path.join(current_app.static_folder, pkg_data['body_mask']), cv2.IMREAD_GRAYSCALE)

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
            'body_rect': { 'x': x_body, 'y': y_body, 'width': w_body, 'height': h_body },
            'template_b64': cv2_to_base64(template_img)
        })
    except Exception as e:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500



@bp.route('/find_body_in_roi_with_template', methods=['POST'])
@login_required
def find_body_in_roi_with_template():
    try:
        data = request.get_json()
        roi_b64 = data.get('component_roi_b64')
        template_roi_b64 = data.get('template_roi_b64')
        template_body_rects = data.get('template_body_rects')
        rotation = int(data.get('rotation', 0))  # mantém sua rotação

        if not all([roi_b64, template_roi_b64, template_body_rects]):
            return jsonify({'error': 'Dados de template ou ROI ausentes.'}), 400

        # Decodifica imagens
        roi_img = base64_to_cv2_img(roi_b64)
        if roi_img is None:
            return jsonify({'error': 'Imagem ROI (busca) inválida.'}), 400

        template_roi_img = base64_to_cv2_img(template_roi_b64)
        if template_roi_img is None:
            return jsonify({'error': 'Imagem ROI (template) inválida.'}), 400

        th, tw = template_roi_img.shape[:2]

        # ----- NOVO: normaliza rects para inteiros e faz clamp dentro da ROI do template
        norm_rects = []
        for r in template_body_rects:
            rx = int(round(r.get('x', 0)))
            ry = int(round(r.get('y', 0)))
            rw = int(round(r.get('width', 0)))
            rh = int(round(r.get('height', 0)))

            # garante largura/altura positivas
            if rw < 0:
                rx += rw
                rw = -rw
            if rh < 0:
                ry += rh
                rh = -rh

            # clamp nos limites da imagem template
            rx = max(0, min(rx, tw - 1))
            ry = max(0, min(ry, th - 1))
            # ajusta w/h para não ultrapassar a borda
            rw = max(1, min(rw, tw - rx))
            rh = max(1, min(rh, th - ry))

            norm_rects.append({'x': rx, 'y': ry, 'width': rw, 'height': rh})

        if not norm_rects:
            return jsonify({'error': 'Nenhuma região de corpo válida recebida.'}), 400

        # Bounding box do conjunto de regiões (agora com inteiros)
        min_x = min(r['x'] for r in norm_rects)
        min_y = min(r['y'] for r in norm_rects)
        max_x = max(r['x'] + r['width'] for r in norm_rects)
        max_y = max(r['y'] + r['height'] for r in norm_rects)

        bb_x = int(min_x)
        bb_y = int(min_y)
        bb_w = int(max(1, max_x - min_x))
        bb_h = int(max(1, max_y - min_y))

        # Clamp final da BB ao tamanho do template ROI
        if bb_x >= tw or bb_y >= th:
            return jsonify({'error': 'Bounding box do corpo fora da ROI do template.'}), 400
        bb_w = min(bb_w, tw - bb_x)
        bb_h = min(bb_h, th - bb_y)

        # Recorte do template e criação da máscara
        template_img = template_roi_img[bb_y:bb_y + bb_h, bb_x:bb_x + bb_w]
        if template_img is None or template_img.size == 0:
            return jsonify({'error': 'Rects do corpo (template) inválidos.'}), 400

        template_mask = np.zeros((bb_h, bb_w), dtype=np.uint8)
        for r in norm_rects:
            rel_x = int(r['x'] - bb_x)
            rel_y = int(r['y'] - bb_y)
            rel_w = int(r['width'])
            rel_h = int(r['height'])
            # clamp por segurança dentro da máscara
            if rel_x < 0 or rel_y < 0:
                continue
            x2 = min(rel_x + rel_w, bb_w)
            y2 = min(rel_y + rel_h, bb_h)
            if x2 > rel_x and y2 > rel_y:
                cv2.rectangle(template_mask, (rel_x, rel_y), (x2, y2), 255, -1)

        # --- rotação preservada (usa o seu helper interno) ---
        template_img, template_mask = _rotate_template_and_mask(template_img, template_mask, rotation)

        # Match no espaço em tons de cinza, como no seu fluxo
        roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

        if template_gray.shape[0] > roi_gray.shape[0] or template_gray.shape[1] > roi_gray.shape[1]:
            return jsonify({'error': f'A área desenhada é menor que o template. (ROI: {roi_gray.shape}, Template: {template_gray.shape})'}), 400

        res = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_SQDIFF_NORMED, mask=template_mask)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        (h_body, w_body) = template_gray.shape[:2]
        (x_body, y_body) = min_loc

        return jsonify({
            'body_rect': {'x': x_body, 'y': y_body, 'width': w_body, 'height': h_body},
            'template_b64': cv2_to_base64(template_img)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def _rotate_template_and_mask(template_img, template_mask, rotation):
    """Gira a imagem do template e sua máscara pelo ângulo (90, 180 ou 270 graus)."""
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
