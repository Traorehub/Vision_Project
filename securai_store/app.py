"""
Backend Flask pour SecurAI Store.
Expose le flux vidéo MJPEG et les API de contrôle avec multithreading.
Reconnaissance déportée sur GPU Colab via Cloudflare Tunnel.
"""
import os
import cv2
import time
import numpy as np
import threading
import base64
import requests
import logging
import queue
from dataclasses import dataclass
from flask import Flask, Response, request, jsonify, render_template
from werkzeug.utils import secure_filename

from modules.face_detector import FaceDetector
from modules.face_recognizer import FaceRecognizer
from modules.fgsm_attacker import FGSMAttacker
from modules.patch_attacker import PatchAttacker
from modules.defender import Defender
from modules.anomaly_detector import AnomalyDetector
from rights_manager import RightsManager
from paths import BASE_DIR, MODELS_DIR, ENROLLED_DIR, AUDIT_LOG

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
REMOTE_INFERENCE_URL = 'http://vision.api.near-u-api.org/infer'
COLAB_FRAME_SKIP     = 3
FACE_CROP_SIZE       = 160

app = Flask(__name__)
os.makedirs(ENROLLED_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    filename=AUDIT_LOG,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Système SecurAI démarré.")

# ─────────────────────────────────────────────
# ÉTAT GLOBAL (Thread-safe)
# ─────────────────────────────────────────────
@dataclass
class SystemState:
    identity:         str   = "Aucun"
    access_level:     str   = "DENIED"
    permissions:      dict  = None
    anomaly_detected: bool  = False
    anomaly_score:    float = 0.0
    attack_active:    bool  = False
    model_mode:       str   = "standard"
    fps:              int   = 0
    confidence:       float = 0.0
    latest_frame:     bytes = None

state             = SystemState()
state.permissions = {'entrance': False, 'stock': False, 'cashier': False, 'server': False}
state_lock        = threading.Lock()

# ─────────────────────────────────────────────
# INITIALISATION DES MODULES
# ─────────────────────────────────────────────
print("Initialisation des modules IA...")
face_detector    = FaceDetector()
rights_manager   = RightsManager()
face_recognizer  = FaceRecognizer(mode='standard')
fgsm_attacker    = FGSMAttacker(face_recognizer.model, epsilon=0.03)
defender         = Defender()
anomaly_detector = AnomalyDetector()
patch_attacker   = PatchAttacker(face_recognizer.model, epsilon=0.35, steps=40, alpha=0.02)

print("Enrôlement des visages connus...")
for filename in os.listdir(ENROLLED_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        identity_name = os.path.splitext(filename)[0].split('-')[0]
        filepath      = os.path.join(ENROLLED_DIR, filename)
        img           = cv2.imread(filepath)
        if img is not None:
            bboxes = face_detector.detect(img)
            if bboxes:
                face_crop = face_detector.crop_face(img, bboxes[0], size=FACE_CROP_SIZE)
                if face_crop is not None:
                    face_recognizer.enroll_face(identity_name, face_crop)
                    # Synchroniser aussi vers Colab
                    try:
                        _, buf = cv2.imencode('.jpg', face_crop)
                        b64    = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
                        requests.post(
                            'http://vision.api.near-u-api.org/enroll',
                            json={"name": identity_name, "image": b64},
                            timeout=5
                        )
                        print(f"[SYNC COLAB] {identity_name} enrôlé sur Colab")
                    except Exception as e:
                        print(f"[SYNC ERREUR] {identity_name} : {e}")

print(f"{len(face_recognizer.enrolled_embeddings)} identités enrôlées.")
print("Modules prêts.")

# ─────────────────────────────────────────────
# INFÉRENCE DISTANTE (Colab GPU)
# ─────────────────────────────────────────────
def send_crop_to_colab(face_crop: np.ndarray):
    """Envoie le crop 160x160 vers Colab GPU. Retourne le résultat JSON ou None."""
    try:
        _, buf = cv2.imencode('.jpg', face_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64    = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        t0     = time.time()
        resp   = requests.post(REMOTE_INFERENCE_URL, json={"image": b64}, timeout=3)
        ms     = int((time.time() - t0) * 1000)
        resp.raise_for_status()
        result = resp.json()
        name   = result.get('name', 'Inconnu')
        result['access'] = 'DENIED' if name in ['Unknown', 'Inconnu'] else 'GRANTED'
        print(f"[COLAB GPU] {name} | conf={result.get('confidence', 0):.2f} | {ms}ms")
        return result
    except Exception as e:
        print(f"[COLAB ERREUR] {e}")
        logging.error(f"Erreur Colab: {e}")
        return None

def send_fgsm_to_colab(face_crop: np.ndarray) -> np.ndarray:
    """Envoie le crop vers Colab pour calcul FGSM sur GPU."""
    try:
        _, buf = cv2.imencode('.jpg', face_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64    = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        resp   = requests.post(
            'http://vision.api.near-u-api.org/fgsm',
            json={"image": b64, "target": "Manager_Demo"},
            timeout=3
        )
        resp.raise_for_status()
        result  = resp.json()
        img_b64 = result.get('attacked_image')
        if img_b64:
            img_bytes    = base64.b64decode(img_b64.split(',')[-1])
            np_arr       = np.frombuffer(img_bytes, np.uint8)
            attacked_crop = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return attacked_crop
    except Exception as e:
        print(f"[FGSM ERREUR] {e}")
    return None

# ─────────────────────────────────────────────
# THREAD COLAB — séparé, ne bloque jamais la vidéo
# ─────────────────────────────────────────────
colab_queue = queue.Queue(maxsize=1)

def colab_worker():
    while True:
        try:
            face_crop = colab_queue.get(timeout=1)
            result    = send_crop_to_colab(face_crop)
            if result:
                with state_lock:
                    name               = result.get('name', 'Inconnu')
                    state.identity     = name
                    state.confidence   = result.get('confidence', 0.0)
                    state.access_level = result.get('access', 'DENIED')
                    state.permissions  = rights_manager.get_permissions(name)
        except queue.Empty:
            continue

threading.Thread(target=colab_worker, daemon=True).start()

# ─────────────────────────────────────────────
# THREAD VIDÉO — ultra-léger, jamais bloqué
# ─────────────────────────────────────────────
def video_processing_thread():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    start_time  = time.time()

    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.01)
            continue

        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            with state_lock:
                state.fps = int(frame_count / elapsed)
            frame_count = 0
            start_time  = time.time()

        with state_lock:
            attack_active = state.attack_active
            current_mode  = state.model_mode

        # ── Détection locale (YOLOv8n — rapide CPU) ──
        bboxes      = face_detector.detect(frame)
        is_attacked = False
        anom_score  = 0.0

        if bboxes:
            bbox      = bboxes[0]
            face_crop = face_detector.crop_face(frame, bbox, size=FACE_CROP_SIZE)

            if face_crop is not None:
                # FGSM déporté sur Colab GPU
                if attack_active and frame_count % COLAB_FRAME_SKIP == 0:
                    fgsm_crop = send_fgsm_to_colab(face_crop)
                    if fgsm_crop is not None:
                        face_crop = fgsm_crop

                # Défense locale
                if current_mode == 'hardened':
                    face_crop = defender.apply_defense(face_crop, defense_type='gaussian')

                # Détection anomalie (FFT local — ~2ms)
                is_attacked, anom_score = anomaly_detector.analyze(face_crop)

                # Envoyer vers Colab SANS ATTENDRE
                if frame_count % COLAB_FRAME_SKIP == 0:
                    try:
                        colab_queue.put_nowait(face_crop)
                    except queue.Full:
                        pass

            # Overlay avec le dernier résultat connu
            with state_lock:
                identity = state.identity
                conf     = state.confidence

            x1, y1, x2, y2 = bbox
            ui_config = rights_manager.get_ui_config(identity)
            color_hex = ui_config['color'].lstrip('#')
            color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(frame, f"{identity} ({conf:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        # ── Mise à jour état global ──
        with state_lock:
            state.anomaly_detected = is_attacked
            state.anomaly_score    = anom_score
            if is_attacked:
                logging.warning(f"ANOMALIE - score={anom_score:.2f} - {state.identity}")
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                state.latest_frame = buffer.tobytes()


threading.Thread(target=video_processing_thread, daemon=True).start()

# ─────────────────────────────────────────────
# ROUTES FLASK
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('EntranceControl.html')

@app.route('/static_analysis')
def static_analysis():
    return render_template('StaticAnalysis.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with state_lock:
                frame = state.latest_frame
            if frame is not None:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status', methods=['GET'])
def get_status():
    with state_lock:
        return jsonify({
            'identity':         state.identity,
            'access_level':     state.access_level,
            'permissions':      state.permissions,
            'anomaly_detected': state.anomaly_detected,
            'anomaly_score':    state.anomaly_score,
            'attack_active':    state.attack_active,
            'model_mode':       state.model_mode,
            'fps':              state.fps,
            'confidence':       state.confidence
        })

@app.route('/api/toggle_attack', methods=['POST'])
def toggle_attack():
    data = request.json
    if 'active' not in data:
        return jsonify({"success": False, "error": "Paramètre 'active' manquant."}), 400
    with state_lock:
        state.attack_active = data['active']
    return jsonify({"success": True, "message": f"Attaque {'activée' if data['active'] else 'désactivée'}."})

@app.route('/api/toggle_mode', methods=['POST'])
def toggle_mode():
    data = request.json
    if 'mode' not in data or data['mode'] not in ['standard', 'hardened']:
        return jsonify({"success": False, "error": "Mode invalide."}), 400
    try:
        face_recognizer.switch_mode(data['mode'])
        with state_lock:
            state.model_mode = data['mode']
        return jsonify({"success": True, "message": f"Mode → {data['mode']}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/enroll', methods=['POST'])
def enroll():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"success": False, "error": "Données manquantes"}), 400
    file  = request.files['image']
    name  = request.form['name']
    level = request.form.get('level', RightsManager.EMPLOYEE)
    if file.filename == '':
        return jsonify({"success": False, "error": "Fichier vide"}), 400
    filename  = secure_filename(f"{name}_{file.filename}")
    save_path = os.path.join(ENROLLED_DIR, filename)
    file.save(save_path)
    try:
        rights_manager.add_identity(name, level)
        return jsonify({"success": True, "message": f"{name} enrôlé comme {level}"})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/analyze_static', methods=['POST'])
def analyze_static():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Aucune image envoyée"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "Fichier vide"}), 400
    try:
        nparr = np.frombuffer(file.read(), np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "error": "Format invalide"}), 400
        bboxes = face_detector.detect(img)
        if not bboxes:
            return jsonify({"success": True, "results": {
                "identity": "Aucun visage", "access_level": "DENIED",
                "confidence": 0, "anomaly_detected": False, "anomaly_score": 0.0
            }})
        bbox      = bboxes[0]
        face_crop = face_detector.crop_face(img, bbox, size=FACE_CROP_SIZE)
        if face_crop is None:
            return jsonify({"success": False, "error": "Erreur recadrage"}), 500

        is_attacked, anom_score = anomaly_detector.analyze(face_crop)
        result   = send_crop_to_colab(face_crop)
        identity = result.get('name', 'Inconnu') if result else 'Inconnu'
        conf     = result.get('confidence', 0.0)  if result else 0.0

        access_level = rights_manager.get_access_level(identity)
        permissions  = rights_manager.get_permissions(identity)

        x1, y1, x2, y2 = bbox
        ui_config = rights_manager.get_ui_config(identity)
        color_hex = ui_config['color'].lstrip('#')
        color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 3)
        cv2.putText(img, f"{identity} ({conf:.2f})",
                    (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)

        _, buffer  = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"success": True, "results": {
            "identity":         identity,
            "confidence":       conf,
            "access_level":     access_level,
            "permissions":      permissions,
            "anomaly_detected": is_attacked,
            "anomaly_score":    anom_score,
            "image_base64":     img_base64
        }})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/generate_glasses_attack', methods=['POST'])
def generate_glasses_attack():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Paramètre 'image' manquant."}), 400
    target_name = request.form.get('target', 'Manager_Demo')
    target_emb  = face_recognizer.enrolled_embeddings.get(target_name)
    if target_emb is None:
        return jsonify({"success": False,
                        "error": f"'{target_name}' non enrôlé.",
                        "enrolled": list(face_recognizer.enrolled_embeddings.keys())}), 404
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "Fichier vide."}), 400
    try:
        nparr = np.frombuffer(file.read(), np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "error": "Format invalide."}), 400
        bboxes = face_detector.detect(img)
        if not bboxes:
            return jsonify({"success": False, "error": "Aucun visage détecté."}), 422

        bbox            = bboxes[0]
        x1, y1, x2, y2 = bbox
        face_crop       = face_detector.crop_face(img, bbox, size=FACE_CROP_SIZE)
        if face_crop is None:
            return jsonify({"success": False, "error": "Erreur recadrage."}), 500

        r_before    = send_crop_to_colab(face_crop) or {}
        id_before   = r_before.get('name', 'Inconnu')
        conf_before = r_before.get('confidence', 0.0)

        attacked_crop = patch_attacker.attack(face_crop, target_emb)

        r_after      = send_crop_to_colab(attacked_crop) or {}
        id_after     = r_after.get('name', 'Inconnu')
        conf_after   = r_after.get('confidence', 0.0)
        access_after = r_after.get('access', 'DENIED')

        is_anom, anom_score = anomaly_detector.analyze(attacked_crop)
        permissions_after   = rights_manager.get_permissions(id_after)

        result_img   = img.copy()
        img_h, img_w = img.shape[:2]
        rx1, ry1     = max(0, x1), max(0, y1)
        rx2, ry2     = min(img_w, x2), min(img_h, y2)
        result_img[ry1:ry2, rx1:rx2] = cv2.resize(attacked_crop, (rx2-rx1, ry2-ry1))

        ui_cfg  = rights_manager.get_ui_config(id_after)
        col_hex = ui_cfg['color'].lstrip('#')
        col_bgr = tuple(int(col_hex[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(result_img, (x1, y1), (x2, y2), col_bgr, 3)
        cv2.putText(result_img, f"{id_after} ({conf_after:.2f}) [PATCH]",
                    (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_bgr, 2)

        _, buf  = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buf).decode('utf-8')

        logging.info(f"Patch: {id_before}({conf_before:.2f}) → {id_after}({conf_after:.2f})")

        return jsonify({"success": True, "target": target_name,
                        "before": {"identity": id_before, "confidence": conf_before},
                        "after":  {"identity": id_after, "confidence": conf_after,
                                   "access_level": access_after, "permissions": permissions_after,
                                   "anomaly_detected": is_anom, "anomaly_score": anom_score,
                                   "image_base64": img_b64}})
    except Exception as e:
        logging.error(f"generate_glasses_attack: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)