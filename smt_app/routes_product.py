from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
import sqlite3
import json
import traceback
import cv2
import numpy as np
import os
from .db_helpers import get_db_connection, base64_to_cv2_img, save_image_to_disk, cv2_to_base64
from .vision import find_fiducial_rings, align_with_fiducials, analyze_component_package_based, predict_with_model
import base64
import uuid

bp = Blueprint('product', __name__)


@bp.route('/find_fiducials', methods=['POST'])
@login_required
def find_fiducials_route():
    try:
        data = request.get_json()
        base64_image = data['image_data'].split(',')[1]
        img_bytes = base64.b64decode(base64_image)
        image_cv = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_cv is None:
            return jsonify({'error': 'Falha ao decodificar a imagem.'}), 400

        # Esta função agora é importada de .vision
        detected_rings = find_fiducial_rings(image_cv)

        debug_image = image_cv.copy()
        if detected_rings:
            for ring in detected_rings:
                cv2.circle(debug_image, (ring['x'], ring['y']), ring['r'], (0, 255, 0), 2)
                cv2.circle(debug_image, (ring['x'], ring['y']), 2, (0, 0, 255), 3)
        return jsonify({'circles': detected_rings, 'debug_image': cv2_to_base64(debug_image)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Erro ao encontrar fiduciais: {str(e)}'}), 500
# --- FIM DA ROTA MOVIDA ---


@bp.route('/add_product', methods=['GET', 'POST'])
@login_required
def add_product():
    if request.method == 'GET':
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, presence_threshold, ssim_threshold,
                   body_matrix, template_roi_width, template_roi_height
            FROM packages
            ORDER BY name
        """)
        packages = cursor.fetchall()
        cursor.close()
        conn.close()

        return render_template('add_product.html', packages=packages)

    # POST
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        name = request.form.get('name')
        golden_file = request.files.get('golden')
        fiducials_json = request.form.get('fiducials')
        components_json = request.form.get('components')
        package_templates_json = request.form.get('package_templates') or '{}'

        if not name or not golden_file or not components_json:
            return jsonify({'error': 'Nome, imagem Golden ou componentes faltando.'}), 400

        try:
            fiducials = json.loads(fiducials_json) if fiducials_json else []
        except Exception:
            fiducials = []

        try:
            components = json.loads(components_json)
        except Exception as e:
            print("[add_product] Erro ao decodificar components_json:", e)
            return jsonify({'error': 'JSON de componentes inválido.'}), 400

        try:
            package_templates = json.loads(package_templates_json)
        except Exception as e:
            print("[add_product] Erro ao decodificar package_templates_json:", e)
            package_templates = {}

        # Salva a Golden
        golden_bytes = golden_file.read()
        golden_cv = cv2.imdecode(np.frombuffer(golden_bytes, np.uint8), cv2.IMREAD_COLOR)
        if golden_cv is None:
            return jsonify({'error': 'Falha ao ler imagem Golden.'}), 400

        golden_rel_path = save_image_to_disk(
            golden_cv,
            'uploads',
            f"golden_{uuid.uuid4().hex[:8]}"
        )

        cursor.execute(
            """
            INSERT INTO products (name, golden_image, fiducials, created_by)
            VALUES (?, ?, ?, ?)
            """,
            (name, golden_rel_path, json.dumps(fiducials), current_user.id)
        )
        product_id = cursor.lastrowid

        # --- Pacotes e componentes ---
        package_info = {}  # cache: package_name -> {id, body_matrix, body_mask, template_roi_width, template_roi_height}

        for comp in components:
            comp_name = comp.get('name')
            x = int(comp.get('x', 0))
            y = int(comp.get('y', 0))
            width = int(comp.get('width', 0))
            height = int(comp.get('height', 0))
            rotation = int(comp.get('rotation', 0))
            package_name = comp.get('package') or 'DEFAULT'
            is_polarized = 1 if comp.get('is_polarized') else 0
            polarity_box = comp.get('polarity_rect')
            if polarity_box is not None and not isinstance(polarity_box, str):
                polarity_box = json.dumps(polarity_box)

            if not comp_name or width <= 0 or height <= 0:
                print(f"[add_product] Componente ignorado por dados inválidos: {comp}")
                continue

            # --- Busca ou cria o pacote ---
            if package_name not in package_info:
                cursor.execute(
                    """
                    SELECT id, body_matrix, body_mask, template_roi_width, template_roi_height
                    FROM packages
                    WHERE name = ?
                    """,
                    (package_name,)
                )
                pkg_data = cursor.fetchone()

                body_matrix_rel = None
                body_mask_rel = None
                t_w = None
                t_h = None

                tmpl = None
                if isinstance(package_templates, dict):
                    tmpl = package_templates.get(package_name)

                if not pkg_data:
                    # Pacote novo: tenta criar template, se o front mandou
                    if tmpl:
                        body_matrix_rel, body_mask_rel, t_w, t_h = build_package_template_from_frontend(
                            tmpl, package_name
                        )

                    cursor.execute(
                        """
                        INSERT INTO packages (name, body_matrix, body_mask, template_roi_width, template_roi_height)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (package_name, body_matrix_rel, body_mask_rel, t_w, t_h)
                    )
                    pkg_id = cursor.lastrowid
                    package_info[package_name] = {
                        'id': pkg_id,
                        'body_matrix': body_matrix_rel,
                        'body_mask': body_mask_rel,
                        'template_roi_width': t_w,
                        'template_roi_height': t_h,
                    }
                else:
                    # Pacote já existe
                    body_matrix_rel = pkg_data['body_matrix']
                    body_mask_rel = pkg_data['body_mask']
                    t_w = pkg_data['template_roi_width']
                    t_h = pkg_data['template_roi_height']

                    # Se não tinha template ainda mas o front mandou agora, atualiza
                    if tmpl and (body_matrix_rel is None or body_mask_rel is None):
                        new_body_matrix_rel, new_body_mask_rel, new_w, new_h = build_package_template_from_frontend(
                            tmpl, package_name
                        )
                        if new_body_matrix_rel and new_body_mask_rel:
                            body_matrix_rel = new_body_matrix_rel
                            body_mask_rel = new_body_mask_rel
                            t_w = new_w
                            t_h = new_h
                            cursor.execute(
                                """
                                UPDATE packages
                                SET body_matrix = ?, body_mask = ?, template_roi_width = ?, template_roi_height = ?
                                WHERE id = ?
                                """,
                                (body_matrix_rel, body_mask_rel, t_w, t_h, pkg_data['id'])
                            )

                    package_info[package_name] = {
                        'id': pkg_data['id'],
                        'body_matrix': body_matrix_rel,
                        'body_mask': body_mask_rel,
                        'template_roi_width': t_w,
                        'template_roi_height': t_h,
                    }

            pkg_id = package_info[package_name]['id']

            # --- Insere componente ---
            cursor.execute(
                """
                INSERT INTO components
                (product_id, name, x, y, width, height, package_id, rotation, is_polarized, polarity_box, inspection_mask)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product_id,
                    comp_name,
                    x, y, width, height,
                    pkg_id,
                    rotation,
                    is_polarized,
                    polarity_box,
                    None  # inspection_mask ainda não usamos
                )
            )

        conn.commit()
        return jsonify({'message': 'Produto cadastrado com sucesso.', 'product_id': product_id})

    except Exception as e:
        conn.rollback()
        traceback.print_exc()
        return jsonify({'error': f'Erro ao cadastrar produto: {str(e)}'}), 500

    finally:
        cursor.close()
        conn.close()

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
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return jsonify([{
                'x': int(w * 0.1), 'y': int(h * 0.1),
                'width': int(w * 0.8), 'height': int(h * 0.8)
            }])

        min_dist = float('inf')
        best_contour = None
        for c in contours:
            if cv2.contourArea(c) < 5:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            mX = int(M["m10"] / M["m00"])
            mY = int(M["m01"] / M["m00"])
            dist = ((mX - cX) ** 2) + ((mY - cY) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_contour = c

        if best_contour is None:
            return jsonify([{
                'x': int(w * 0.1), 'y': int(h * 0.1),
                'width': int(w * 0.8), 'height': int(h * 0.8)
            }])

        x, y, w2, h2 = cv2.boundingRect(best_contour)
        return jsonify([{'x': x, 'y': y, 'width': w2, 'height': h2}])

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@bp.route('/find_body_in_roi', methods=['POST'])
@login_required
def find_body_in_roi():
    conn = None
    cursor = None
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
        cursor = conn.cursor()
        cursor.execute(
            "SELECT body_matrix, body_mask FROM packages WHERE name = ?",
            (package_name,)
        )
        pkg_data = cursor.fetchone()
        cursor.close()
        conn.close()
        if not pkg_data or not pkg_data['body_matrix'] or not pkg_data['body_mask']:
            return jsonify({
                'error': 'Template ou máscara não encontrados para este pacote.'
            }), 404

        template_img = cv2.imread(
            os.path.join(current_app.static_folder, pkg_data['body_matrix'])
        )
        template_mask = cv2.imread(
            os.path.join(current_app.static_folder, pkg_data['body_mask']),
            cv2.IMREAD_GRAYSCALE
        )

        if template_img is None or template_mask is None:
            return jsonify({
                'error': 'Falha ao ler arquivos de template/máscara do disco.'
            }), 500

        roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        if (template_gray.shape[0] > roi_gray.shape[0] or
                template_gray.shape[1] > roi_gray.shape[1]):
            return jsonify({
                'error': 'A área desenhada é menor que o template. '
                         'Desenhe uma caixa maior ao redor do componente.'
            }), 400
        res = cv2.matchTemplate(
            roi_gray, template_gray, cv2.TM_SQDIFF_NORMED, mask=template_mask
        )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        (h_body, w_body) = template_gray.shape[:2]
        (x_body, y_body) = min_loc
        return jsonify({
            'body_rect': {
                'x': x_body, 'y': y_body,
                'width': w_body, 'height': h_body
            },
            'template_b64': cv2_to_base64(template_img)
        })
    except Exception as e:
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

        # ----- normaliza rects -----
        norm_rects = []
        for r in template_body_rects:
            rx = int(round(r.get('x', 0)))
            ry = int(round(r.get('y', 0)))
            rw = int(round(r.get('width', 0)))
            rh = int(round(r.get('height', 0)))

            if rw < 0:
                rx += rw
                rw = -rw
            if rh < 0:
                ry += rh
                rh = -rh

            rx = max(0, min(rx, tw - 1))
            ry = max(0, min(ry, th - 1))
            rw = max(1, min(rw, tw - rx))
            rh = max(1, min(rh, th - ry))

            norm_rects.append({'x': rx, 'y': ry, 'width': rw, 'height': rh})

        if not norm_rects:
            return jsonify({'error': 'Nenhuma região de corpo válida recebida.'}), 400

        min_x = min(r['x'] for r in norm_rects)
        min_y = min(r['y'] for r in norm_rects)
        max_x = max(r['x'] + r['width'] for r in norm_rects)
        max_y = max(r['y'] + r['height'] for r in norm_rects)

        bb_x = int(min_x)
        bb_y = int(min_y)
        bb_w = int(max(1, max_x - min_x))
        bb_h = int(max(1, max_y - min_y))

        if bb_x >= tw or bb_y >= th:
            return jsonify({
                'error': 'Bounding box do corpo fora da ROI do template.'
            }), 400
        bb_w = min(bb_w, tw - bb_x)
        bb_h = min(bb_h, th - bb_y)

        template_img = template_roi_img[bb_y:bb_y + bb_h, bb_x:bb_x + bb_w]
        if template_img is None or template_img.size == 0:
            return jsonify({'error': 'Rects do corpo (template) inválidos.'}), 400

        template_mask = np.zeros((bb_h, bb_w), dtype=np.uint8)
        for r in norm_rects:
            rel_x = int(r['x'] - bb_x)
            rel_y = int(r['y'] - bb_y)
            rel_w = int(r['width'])
            rel_h = int(r['height'])
            if rel_x < 0 or rel_y < 0:
                continue
            x2 = min(rel_x + rel_w, bb_w)
            y2 = min(rel_y + rel_h, bb_h)
            if x2 > rel_x and y2 > rel_y:
                cv2.rectangle(template_mask, (rel_x, rel_y), (x2, y2), 255, -1)

        template_img, template_mask = _rotate_template_and_mask(
            template_img, template_mask, rotation
        )

        roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

        if (template_gray.shape[0] > roi_gray.shape[0] or
                template_gray.shape[1] > roi_gray.shape[1]):
            return jsonify({
                'error': f'A área desenhada é menor que o template. '
                         f'(ROI: {roi_gray.shape}, Template: {template_gray.shape})'
            }), 400

        res = cv2.matchTemplate(
            roi_gray, template_gray, cv2.TM_SQDIFF_NORMED, mask=template_mask
        )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        (h_body, w_body) = template_gray.shape[:2]
        (x_body, y_body) = min_loc

        return jsonify({
            'body_rect': {
                'x': x_body, 'y': y_body,
                'width': w_body, 'height': h_body
            },
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

def build_package_template_from_frontend(tmpl, package_name):
    """
    Constrói e salva:
      - body_matrix: ROI do corpo (imagem)
      - body_mask: máscara binária (255 onde é corpo)
      - template_roi_width/height: tamanho da ROI

    tmpl vem do front, algo como:
    {
      "roi_b64": "data:image/png;base64,...",
      "body_rects": [{x, y, width, height}, ...],
      "base_rotation": 0
    }
    """
    try:
        if not tmpl:
            return None, None, None, None

        roi_b64 = tmpl.get('roi_b64')
        body_rects = tmpl.get('body_rects') or []

        if not roi_b64 or not body_rects:
            return None, None, None, None

        roi_img = base64_to_cv2_img(roi_b64)
        if roi_img is None:
            print(f"[build_package_template] Falha ao decodificar ROI para pacote {package_name}")
            return None, None, None, None

        h, w = roi_img.shape[:2]

        # Máscara do mesmo tamanho da ROI: 255 apenas onde há corpo
        mask = np.zeros((h, w), dtype=np.uint8)
        for rect in body_rects:
            x = int(rect.get('x', 0))
            y = int(rect.get('y', 0))
            rw = int(rect.get('width', 0))
            rh = int(rect.get('height', 0))
            if rw > 0 and rh > 0:
                cv2.rectangle(mask, (x, y), (x + rw, y + rh), 255, -1)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        safe_name = ''.join(c if c.isalnum() else '_' for c in str(package_name))[:40]
        rand = uuid.uuid4().hex[:6]

        body_matrix_rel = save_image_to_disk(
            roi_img,
            'uploads',
            f"pkg_{safe_name}_body_{rand}"
        )
        body_mask_rel = save_image_to_disk(
            mask_bgr,
            'uploads',
            f"pkg_{safe_name}_mask_{rand}"
        )

        print(f"[build_package_template] Template salvo para pacote '{package_name}':")
        print(f"  body_matrix={body_matrix_rel}, body_mask={body_mask_rel}, w={w}, h={h}")

        return body_matrix_rel, body_mask_rel, w, h

    except Exception:
        traceback.print_exc()
        return None, None, None, None