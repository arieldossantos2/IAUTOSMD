# smt_app/routes_inspect.py
from flask import Blueprint, render_template, request, jsonify, current_app, url_for
from flask_login import login_required, current_user
import mysql.connector
import json
import traceback
import cv2
import numpy as np
import os
import uuid
import sys
import subprocess
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

@bp.route('/inspect', methods=['GET','POST'])
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
            
        golden_full_path = os.path.join(current_app.static_folder, product['golden_image'])
        golden_full = cv2.imread(golden_full_path)
        
        if golden_full is None:
             return jsonify({'error': f"Imagem Golden não encontrada no servidor: {golden_full_path}"}), 500

        try:
            fiducials_data = json.loads(product.get('fiducials', '[]'))
            produced_aligned = align_with_fiducials(golden_full, produced_full, fiducials_data)
        except Exception as e:
            print(f"Erro no alinhamento: {e}. Usando fallback.")
            h, w = golden_full.shape[:2]
            produced_aligned = cv2.resize(produced_full, (w, h))
        
        if produced_aligned is None:
             return jsonify({'error': 'Falha ao alinhar ou redimensionar imagem de produção.'}), 500

        result_image = produced_aligned.copy()

        cursor.execute("""
            SELECT c.*, p.name as package_name, p.body_matrix, p.body_mask,
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
            
            template_img_path = os.path.join(current_app.static_folder, comp['body_matrix'])
            template_img = cv2.imread(template_img_path)
            template_mask_path = comp.get('body_mask') # Este já é o caminho relativo correto
            
            if template_img is None or template_mask_path is None:
                print(f"ERRO: Não foi possível ler o template ({template_img_path}) ou a máscara ({template_mask_path}) para {comp['name']}")
                cv_status = 'FAIL'
                ai_status = 'UNKNOWN'
                cv_analysis = {'status': 'FAIL', 'details': {'message': 'Arquivo de template/máscara do pacote não encontrado.'}}
                ai_details = {'prob': 0.0}
            else:
                cv_analysis = analyze_component_package_based(
                    roi_g, 
                    template_img,
                    template_mask_path, # Passa o CAMINHO da máscara
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

            roi_g_path = save_image_to_disk(roi_g, 'results', f"insp_{inspection_id}_comp_{comp['id']}_g")
            roi_p_path = save_image_to_disk(roi_p, 'results', f"insp_{inspection_id}_comp_{comp['id']}_p")

            cursor.execute("""
                INSERT INTO inspection_results 
                (inspection_id, component_id, cv_status, ai_status, ai_status_prob, cv_details, final_status, golden_roi_image, produced_roi_image)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                inspection_id, comp['id'], cv_status, ai_status, ai_details.get('prob'), 
                json.dumps(cv_analysis.get('details')), final_status, 
                roi_g_path, roi_p_path
            ))
            
            debug_filename = None
            if 'debug_data' in cv_analysis and 'debug_img_b64' in cv_analysis['debug_data']:
                debug_img = base64_to_cv2_img(cv_analysis['debug_data']['debug_img_b64'])
                if debug_img is not None:
                    debug_filename = f"debug_{inspection_id}_{comp['name']}_{uuid.uuid4().hex[:4]}.png"
                    debug_path = os.path.join(current_app.config['DEBUG_FOLDER'], debug_filename)
                    cv2.imwrite(debug_path, debug_img)
            
            comp_data_for_frontend = {
                'name': comp['name'], 'component_id': comp['id'], 'package': comp['package_name'],
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

        result_filename_db = save_image_to_disk(result_image, 'uploads', f"result_{inspection_id}")
        
        cursor.close()
        conn.close()

        return jsonify({
            'total_ok': total_ok, 'total_fail': total_fail,
            'detailed_components': detailed_components_frontend,
            # Retorna o caminho relativo para o JS
            'result_image_url': url_for('static', filename=result_filename_db), 
            'product_id': product_id
        })
    
    # GET request
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM products ORDER BY name")
    products = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('inspect.html', products=products)

@bp.route('/feedback', methods=['POST'])
@login_required
def save_feedback():
    conn = None
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

            cursor.execute("""
                SELECT golden_roi_image, produced_roi_image 
                FROM inspection_results 
                WHERE inspection_id = %s AND component_id = %s
            """, (inspection_id, comp['id']))
            roi_paths = cursor.fetchone()
            
            if roi_paths and roi_paths['produced_roi_image']:
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
            # Garante que o train_model.py seja encontrado (assumindo que está no root)
            train_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train_model.py'))
            subprocess.Popen([sys.executable, train_script_path])

        return jsonify({'success': True, 'message': f'Feedback salvo! {samples_added_count} amostras adicionadas ao retreinamento.'})
    
    except Exception as e:
        traceback.print_exc()
        if conn: conn.rollback()
        return jsonify({'success': False, 'error': f'Erro ao salvar feedback: {str(e)}'}), 500
    finally:
        if cursor: 
            cursor.close()
        if conn and conn.is_connected():
            conn.close()