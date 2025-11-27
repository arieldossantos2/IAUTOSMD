from flask import Blueprint, render_template, request, jsonify, current_app, url_for
from flask_login import login_required, current_user
import sqlite3
import json
import traceback
import cv2
import numpy as np
import os
import uuid
import sys
import subprocess
import base64
from .db_helpers import get_db_connection, base64_to_cv2_img, save_image_to_disk, cv2_to_base64
from .vision import find_fiducial_rings, align_with_fiducials, analyze_component_package_based, predict_with_model

bp = Blueprint('inspect', __name__)


@bp.route('/find_fiducials', methods=['POST'])
@login_required
def find_fiducials_route():
    # (Movido do app.py original)
    try:
        data = request.get_json()
        base64_image = data['image_data'].split(',')[1]
        img_bytes = base64.b64decode(base64_image)
        image_cv = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_cv is None:
            return jsonify({'error': 'Falha ao decodificar a imagem.'}), 400
        detected_rings = find_fiducial_rings(image_cv)
        debug_image = image_cv.copy()
        if detected_rings:
            for ring in detected_rings:
                cv2.circle(debug_image, (ring['x'], ring['y']), ring['r'], (0, 255, 0), 2)
                cv2.circle(debug_image, (ring['x'], ring['y']), 2, (0, 0, 255), 3)
        return jsonify({'circles': detected_rings, 'debug_image': cv2_to_base64(debug_image)})
    except Exception as e:
        return jsonify({'error': f'Erro ao encontrar fiduciais: {str(e)}'}), 500


@bp.route('/inspect', methods=['GET', 'POST'])
@login_required
def inspect_board():
    if request.method == 'POST':
        conn = get_db_connection()
        cursor = conn.cursor()
        produced_file = request.files.get('produced')
        product_id = request.form.get('product_id')

        if not produced_file or not product_id:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Produto ou imagem faltando.'}), 400

        produced_full = cv2.imdecode(np.frombuffer(produced_file.read(), np.uint8), cv2.IMREAD_COLOR)

        cursor.execute(
            "SELECT golden_image, fiducials FROM products WHERE id = ?",
            (product_id,)
        )
        product = cursor.fetchone()
        if not product:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Produto não encontrado.'}), 404

        golden_full_path = os.path.join(current_app.static_folder, product['golden_image'])
        golden_full = cv2.imread(golden_full_path)

        if golden_full is None:
            cursor.close()
            conn.close()
            return jsonify({'error': f"Imagem Golden não encontrada no servidor: {golden_full_path}"}), 500

        # Tenta decodificar as fiduciais; se der ruim, usa lista vazia
        raw_fiducials = product['fiducials'] or '[]'
        try:
            fiducials_data = json.loads(raw_fiducials)
        except Exception as e:
            print(f"Fiduciais inválidas para produto {product_id}: {e}. Usando lista vazia.")
            fiducials_data = []

        # align_with_fiducials já sabe lidar com lista vazia (faz apenas resize)
        produced_aligned = align_with_fiducials(golden_full, produced_full, fiducials_data)

        if produced_aligned is None:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Falha ao alinhar ou redimensionar imagem de produção.'}), 500

        result_image = produced_aligned.copy()

        cursor.execute("""
            SELECT c.*, p.name AS package_name, p.body_matrix, p.body_mask,
                   p.presence_threshold, p.ssim_threshold
            FROM components c 
            JOIN packages p ON c.package_id = p.id
            WHERE c.product_id = ?
        """, (product_id,))
        comps_rows = cursor.fetchall()
        comps = [dict(row) for row in comps_rows]

        detailed_components_frontend = []
        total_ok, total_fail = 0, 0

        cursor.execute(
            "INSERT INTO inspections (product_id, result, timestamp) VALUES (?, 'IN_PROGRESS', CURRENT_TIMESTAMP)",
            (product_id,)
        )
        inspection_id = cursor.lastrowid

        for comp in comps:
            # Recorta as ROIs
            roi_g = golden_full[comp['y']:comp['y'] + comp['height'],
                                comp['x']:comp['x'] + comp['width']]
            roi_p = produced_aligned[comp['y']:comp['y'] + comp['height'],
                                     comp['x']:comp['x'] + comp['width']]

            # Caminhos do template e da máscara
            body_matrix_rel = comp.get('body_matrix')
            if body_matrix_rel:
                template_img_path = os.path.join(current_app.static_folder, body_matrix_rel)
                template_img = cv2.imread(template_img_path)
            else:
                template_img = None

            # Carrega a máscara como imagem em escala de cinza
            template_mask = None
            body_mask_rel = comp.get('body_mask')
            if body_mask_rel:
                mask_abs_path = os.path.join(current_app.static_folder, body_mask_rel)
                template_mask = cv2.imread(mask_abs_path, cv2.IMREAD_GRAYSCALE)

            # Thresholds com default
            presence_threshold = comp.get('presence_threshold', 0.35)
            ssim_threshold = comp.get('ssim_threshold', 0.60)

            # Inicializa estados
            cv_status = 'UNKNOWN'
            cv_msg = ''
            cv_analysis = {
                'status': 'UNKNOWN',
                'details': {},
                'debug_data': {},
                'found_rotation': 'N/A',
                'displacement': {'x': 0, 'y': 0},
            }
            ai_status = 'UNKNOWN'
            ai_details = {'prob': 0.0}
            final_status = 'FAIL'  # default conservador

            # =======================
            # VISÃO COMPUTACIONAL
            # =======================
            if template_img is None or template_mask is None:
                print(f"ERRO: Não foi possível ler template/máscara para {comp['name']}")
                cv_status = 'FAIL'
                cv_msg = "Arquivo de template/máscara do pacote não encontrado."
                cv_analysis['status'] = 'FAIL'
                cv_analysis['details'] = {'message': cv_msg}

                # Nesse caso, IA não é usada; final_status permanece FAIL
                ai_status = 'UNKNOWN'
                ai_details = {
                    'prob': 0.0,
                    'used_in_final': False,
                    'hard_fail_low_prob': False,
                    'decision_rule': (
                        "IA pode virar o componente para OK se prob >= 0.85, "
                        "ou virar para FAIL se prob <= 0.05. "
                        "Caso contrário, a decisão segue a visão computacional."
                    ),
                    'suggestion': "IA não foi utilizada porque o template/máscara está ausente.",
                    'influence_text': "ℹ️ IA não foi utilizada para este componente."
                }
                final_status = 'FAIL'
            else:
                cv_analysis = analyze_component_package_based(
                    roi_g,
                    template_img,
                    template_mask,
                    roi_p,
                    comp.get('rotation', 0),
                    presence_threshold=presence_threshold,
                    ssim_threshold=ssim_threshold,
                    polarity_box=comp.get('polarity_box')
                )
                cv_status = cv_analysis.get('status', 'FAIL')
                cv_msg = cv_analysis.get('details', {}).get('message', '')

                # =======================
                # IA SIAMESA
                # =======================
                ai_status, ai_details = predict_with_model(
                    roi_g,
                    roi_p,
                    template_img=template_img,
                    template_mask=template_mask,
                    polarity_box=comp.get('polarity_box'),
                    is_polarized=bool(comp.get('is_polarized', 0)),
                )

                prob = ai_details.get('prob') or 0.0

                ia_ok = (ai_status == "OK" and prob >= 0.85)
                ia_fail = (ai_status == "FAIL" and prob <= 0.05)

                # Regra final:
                # 1) IA pode virar para OK (prob >= 0.85)
                # 2) IA pode virar para FAIL (prob <= 0.05)
                # 3) Caso contrário CV decide
                if ia_ok:
                    final_status = "OK"
                elif ia_fail:
                    final_status = "FAIL"
                else:
                    final_status = cv_status

                # Completa ai_details com info de como ela foi usada
                ai_details = ai_details or {}
                ai_details.setdefault('prob', prob)
                ai_details['used_in_final'] = bool(ia_ok or ia_fail)
                ai_details['hard_fail_low_prob'] = bool(ia_fail)
                ai_details['decision_rule'] = (
                    "IA pode virar o componente para OK se prob >= 0.85, "
                    "ou virar para FAIL se prob <= 0.05. "
                    "Caso contrário, a decisão final segue a visão computacional."
                )

                if ia_ok:
                    ai_details['influence_text'] = "✅ A IA determinou que o componente está OK (prob >= 0.85)."
                elif ia_fail:
                    ai_details['influence_text'] = "⚠️ A IA determinou que o componente está FAIL (prob <= 0.05)."
                else:
                    ai_details['influence_text'] = "ℹ️ A opinião da IA NÃO mudou a decisão final; foi apenas um apoio à visão computacional."

                ai_details.setdefault(
                    'suggestion',
                    "IA sugere que o componente está OK (match com golden)." if ai_status == "OK"
                    else "IA sugere que o componente está com desvio em relação ao golden."
                )

            # Contagem geral
            if final_status == "OK":
                total_ok += 1
            else:
                total_fail += 1

            # Desenha retângulo no overlay
            x, y, w, h = comp['x'], comp['y'], comp['width'], comp['height']
            color = (0, 255, 0) if final_status == "OK" else (0, 0, 255)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result_image, f"{comp['name']}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Salva ROIs em disco
            roi_g_path = save_image_to_disk(roi_g, 'results', f"insp_{inspection_id}_comp_{comp['id']}_g")
            roi_p_path = save_image_to_disk(roi_p, 'results', f"insp_{inspection_id}_comp_{comp['id']}_p")

            # Persiste no inspection_results
            cursor.execute("""
                INSERT INTO inspection_results 
                (inspection_id, component_id, cv_status, ai_status, ai_status_prob, cv_details, final_status, golden_roi_image, produced_roi_image)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                inspection_id, comp['id'], cv_status, ai_status, ai_details.get('prob'),
                json.dumps(cv_analysis.get('details', {})), final_status,
                roi_g_path, roi_p_path
            ))

            # Debug image opcional
            debug_filename = None
            debug_data = cv_analysis.get('debug_data', {})
            if isinstance(debug_data, dict) and 'debug_img_b64' in debug_data:
                debug_img = base64_to_cv2_img(debug_data['debug_img_b64'])
                if debug_img is not None:
                    debug_filename = f"debug_{inspection_id}_{comp['name']}_{uuid.uuid4().hex[:4]}.png"
                    debug_path = os.path.join(current_app.config['DEBUG_FOLDER'], debug_filename)
                    cv2.imwrite(debug_path, debug_img)

            # Dados para o frontend
            comp_data_for_frontend = {
                'name': comp['name'],
                'component_id': comp['id'],
                'package': comp['package_name'],
                'rotation': comp['rotation'],
                'is_polarized': bool(comp.get('is_polarized', 0)),
                'golden_image': cv2_to_base64(roi_g),
                'produced_image': cv2_to_base64(roi_p),
                'cv_status': cv_status,
                'cv_msg': cv_msg,
                'ai_status': ai_status,
                'ai_details': ai_details,
                'final_status': final_status,
                'cv_details': cv_analysis.get('details', {}),
                'found_rotation': cv_analysis.get('found_rotation', 'N/A'),
                'displacement': cv_analysis.get('displacement', {'x': 0, 'y': 0}),
                'debug_filename': debug_filename,
                'debug_data': debug_data,
            }

            detailed_components_frontend.append(comp_data_for_frontend)

        overall_status = "OK" if total_fail == 0 else "FAIL"
        cursor.execute("UPDATE inspections SET result = ? WHERE id = ?", (overall_status, inspection_id))
        conn.commit()

        result_filename_db = save_image_to_disk(result_image, 'uploads', f"result_{inspection_id}")

        cursor.close()
        conn.close()

        return jsonify({
            'total_ok': total_ok, 'total_fail': total_fail,
            'detailed_components': detailed_components_frontend,
            'result_image_url': url_for('static', filename=result_filename_db),
            'product_id': product_id
        })

    # GET request
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM products ORDER BY name")
    products = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('inspect.html', products=products)


@bp.route('/feedback', methods=['POST'])
@login_required
def save_feedback():
    conn = None
    cursor = None
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        feedbacks = data.get('feedbacks', {})

        if not feedbacks:
            return jsonify({'success': False, 'error': 'Nenhum feedback recebido.'}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id FROM inspections WHERE product_id = ? ORDER BY timestamp DESC LIMIT 1",
            (product_id,)
        )
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
                VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?)
            """, (component_name, feedback, current_user.id, inspection_id))

            cursor.execute(
                "SELECT id FROM components WHERE name = ? AND product_id = ?",
                (component_name, product_id)
            )
            comp = cursor.fetchone()
            if not comp:
                continue

            cursor.execute("""
                SELECT golden_roi_image, produced_roi_image 
                FROM inspection_results 
                WHERE inspection_id = ? AND component_id = ?
            """, (inspection_id, comp['id']))
            roi_paths = cursor.fetchone()

            if roi_paths and roi_paths['produced_roi_image']:
                cursor.execute("""
                    INSERT INTO training_samples (product_id, component_id, golden_path, produced_path, label)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    product_id, comp['id'],
                    roi_paths['golden_roi_image'], roi_paths['produced_roi_image'],
                    feedback
                ))
                samples_added_count += 1
            else:
                print(f"Não foi possível encontrar os caminhos das ROIs para {component_name} na inspeção {inspection_id}")

        conn.commit()

        if samples_added_count > 0:
            print(f"Disparando o script de treinamento com {samples_added_count} novas amostras...")
            # Garante que o train_model.py seja encontrado (assumindo que está no root)
            train_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train_model.py'))
            subprocess.Popen([sys.executable, train_script_path])

        return jsonify({'success': True, 'message': f'Feedback salvo! {samples_added_count} amostras adicionadas ao retreinamento.'})

    except Exception as e:
        traceback.print_exc()
        if conn:
            conn.rollback()
        return jsonify({'success': False, 'error': f'Erro ao salvar feedback: {str(e)}'}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
