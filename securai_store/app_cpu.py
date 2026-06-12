"""
Backend Flask — SecurAI Store (MODE CPU LOCAL)
Toute l'inférence tourne sur le CPU local.
Aucun appel réseau pendant la démo.
FPS attendu : 10-20 FPS selon le CPU.
"""
import os, cv2, time, numpy as np, threading, base64, logging, queue
from dataclasses import dataclass, field
from flask import Flask, Response, request, jsonify, render_template
from werkzeug.utils import secure_filename

from modules.face_detector    import FaceDetector
from modules.face_recognizer  import FaceRecognizer
from modules.fgsm_attacker    import FGSMAttacker
from modules.patch_attacker   import PatchAttacker
from modules.defender         import Defender
from modules.anomaly_detector import AnomalyDetector
from rights_manager import RightsManager
from paths import BASE_DIR, MODELS_DIR, ENROLLED_DIR, AUDIT_LOG, read_audit_log_lines

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FACE_CROP_SIZE  = 160
FGSM_SKIP       = 5     # calcule FGSM 1 frame sur 5 (lourd sur CPU)
DETECT_SKIP     = 2     # détecte 1 frame sur 2 (YOLO)
DEBUG           = True

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
logging.info("SecurAI CPU mode démarré.")

def dbg(msg):
    if DEBUG:
        print(msg)

# ─────────────────────────────────────────────
# ÉTAT GLOBAL
# ─────────────────────────────────────────────
@dataclass
class SystemState:
    identity:         str   = "Aucun"
    access_level:     str   = "DENIED"
    permissions:      dict  = field(default_factory=lambda: {
                                'entrance': False, 'stock': False,
                                'cashier':  False, 'server': False})
    anomaly_detected: bool  = False
    anomaly_score:    float = 0.0
    attack_active:         bool  = False
    glasses_attack_active: bool  = False
    glasses_attack_target: str   = ''
    patch_generating:      bool  = False   # True pendant le calcul PGD
    patch_ready:           bool  = False   # True quand le tenseur est dans le cache
    model_mode:       str   = "standard"
    defense_type:     str   = "gaussian"
    fps:              int   = 0
    confidence:       float = 0.0
    latest_frame:     bytes = None
    infer_ms:         float = 0.0
    fgsm_ms:          float = 0.0
    fgsm_epsilon:     float = 0.15
    perturbation_heatmap: bytes = None
    strict_block:     bool  = False

state      = SystemState()
state_lock = threading.Lock()

# ─────────────────────────────────────────────
# INITIALISATION MODULES
# ─────────────────────────────────────────────
print("\n══════════════════════════════════════")
print("  SecurAI — MODE CPU LOCAL")
print("══════════════════════════════════════")
print("[INIT] Chargement des modules IA...")

face_detector    = FaceDetector()
rights_manager   = RightsManager()
face_recognizer  = FaceRecognizer(mode='standard')
fgsm_attacker    = FGSMAttacker(face_recognizer.model, epsilon=0.15)
defender         = Defender()
anomaly_detector = AnomalyDetector()
patch_attacker   = PatchAttacker(face_recognizer.model, epsilon=0.35, steps=40, alpha=0.02)

print("[INIT] Enrôlement des visages connus...")
enrolled_count = 0
for filename in os.listdir(ENROLLED_DIR):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    identity_name = os.path.splitext(filename)[0].split('-')[0]
    img = cv2.imread(os.path.join(ENROLLED_DIR, filename))
    if img is None:
        continue
    bboxes = face_detector.detect(img, force=True)
    if not bboxes:
        print(f"  ⚠️  Aucun visage détecté dans {filename}")
        continue
    face_crop = face_detector.crop_face(img, bboxes[0], size=FACE_CROP_SIZE)
    if face_crop is None:
        continue
    face_recognizer.enroll_face(identity_name, face_crop)
    enrolled_count += 1
    print(f"  ✅ {identity_name} enrôlé")

print(f"[INIT] {enrolled_count} identité(s) enrôlée(s).")

# ─────────────────────────────────────────────
# BENCHMARK CPU AU DÉMARRAGE
# ─────────────────────────────────────────────
print("\n[BENCHMARK] Test vitesse CPU...")
dummy = np.zeros((160, 160, 3), dtype=np.uint8)

# Test inférence
times = []
for _ in range(5):
    t0 = time.time()
    face_recognizer.predict(dummy)
    times.append((time.time()-t0)*1000)
avg_infer = sum(times)/len(times)
print(f"  Inférence FaceNet : {avg_infer:.0f}ms/frame → {1000/avg_infer:.1f} FPS théorique")

# Test FGSM
times_fgsm = []
for _ in range(3):
    t0 = time.time()
    fgsm_attacker.attack(dummy, None)
    times_fgsm.append((time.time()-t0)*1000)
avg_fgsm = sum(times_fgsm)/len(times_fgsm)
print(f"  Calcul FGSM      : {avg_fgsm:.0f}ms/calcul")
print(f"  FGSM skip={FGSM_SKIP} → impact réel : {avg_fgsm/FGSM_SKIP:.0f}ms/frame")
print("══════════════════════════════════════\n")

print("[READY] Modules prêts — lancement serveur Flask...")

# ─────────────────────────────────────────────
# QUEUE FGSM — thread séparé pour ne pas bloquer la vidéo
# ─────────────────────────────────────────────
fgsm_queue       = queue.Queue(maxsize=1)
fgsm_result_lock = threading.Lock()
fgsm_last_crop   = [None]   # résultat partagé entre threads
fgsm_last_clean  = [None]   # crop propre associé (pour heatmap)

def _fgsm_worker():
    """Calcule FGSM en arrière-plan. La vidéo n'attend jamais."""
    while True:
        try:
            face_crop = fgsm_queue.get(timeout=1)
            t0        = time.time()
            with state_lock:
                epsilon = state.fgsm_epsilon
            target_emb = face_recognizer.enrolled_embeddings.get('Manager_Demo')
            attacked  = fgsm_attacker.attack(face_crop, target_emb, epsilon=epsilon)
            ms        = int((time.time()-t0)*1000)
            dbg(f"[FGSM LOCAL] {ms}ms")
            with fgsm_result_lock:
                fgsm_last_crop[0] = attacked
                fgsm_last_clean[0] = face_crop.copy()
            with state_lock:
                state.fgsm_ms = ms
                heatmap = FGSMAttacker.perturbation_heatmap(face_crop, attacked)
                if heatmap is not None:
                    ok_hm, buf_hm = cv2.imencode('.jpg', heatmap)
                    if ok_hm:
                        state.perturbation_heatmap = buf_hm.tobytes()
        except queue.Empty:
            continue
        except Exception as e:
            dbg(f"[FGSM ERREUR] {e}")

threading.Thread(target=_fgsm_worker, daemon=True, name="fgsm-worker").start()

# ─────────────────────────────────────────────
# HELPER : recoller le crop patché dans la frame complète
# ─────────────────────────────────────────────
def _overlay_patch(frame: np.ndarray, bbox: tuple, patched_face: np.ndarray) -> np.ndarray:
    """Overlay ONLY the glasses region on the frame, preventing rectangular boundaries."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1 or patched_face is None:
        return frame
    try:
        target_w = x2 - x1
        target_h = y2 - y1
        patched_resized = cv2.resize(patched_face, (target_w, target_h))
        
        # Get binary glasses mask and resize it to the bounding box size
        mask_160 = patch_attacker.create_glasses_mask(160, 160)
        mask_resized = cv2.resize(mask_160, (target_w, target_h))
        
        # Expand dims to (H, W, 1) to broadcast to 3 BGR channels
        mask_3d = np.expand_dims(mask_resized, axis=2)
        
        # Blend the glasses overlay only where mask == 1
        roi = frame[y1:y2, x1:x2]
        blended = (patched_resized * mask_3d + roi * (1.0 - mask_3d)).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended
    except Exception as e:
        print(f"Error in overlay_patch: {e}")
    return frame

# ─────────────────────────────────────────────
# THREAD VIDÉO
# ─────────────────────────────────────────────
def _video_thread():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Webcam inaccessible.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count  = 0
    fps_count    = 0
    fps_start    = time.time()
    last_bboxes  = []   # cache détection YOLO

    while True:
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame_count += 1
        fps_count   += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            with state_lock:
                state.fps = int(fps_count / elapsed)
            fps_count = 0
            fps_start = time.time()

        with state_lock:
            attack_active = state.attack_active
            current_mode  = state.model_mode
            active_defense_type = state.defense_type

        # ── Détection YOLO (1 frame sur DETECT_SKIP) ──
        if frame_count % DETECT_SKIP == 0:
            last_bboxes = face_detector.detect(frame)
        bboxes = last_bboxes

        is_attacked = False
        anom_score  = 0.0

        if bboxes:
            bbox      = bboxes[0]
            face_crop = face_detector.crop_face(frame, bbox, size=FACE_CROP_SIZE)

            if face_crop is not None:
                patch_applied = False

                # ── Patch lunettes live (applique le patch pré-calculé, <5ms) ──
                with state_lock:
                    glasses_active = state.glasses_attack_active
                    glasses_target = state.glasses_attack_target
                if glasses_active and glasses_target:
                    cached_perturbation = patch_attacker.get_cached_patch(glasses_target)
                    if cached_perturbation is not None:
                        patched = patch_attacker.apply_patch(face_crop, cached_perturbation)
                        if patched is not None:
                            face_crop = patched
                            patch_applied = True

                # ── Mode attaque : FGSM en arrière-plan ──
                if attack_active:
                    # Envoyer vers le worker FGSM sans bloquer
                    if frame_count % FGSM_SKIP == 0:
                        try:
                            fgsm_queue.put_nowait(face_crop.copy())
                        except queue.Full:
                            pass
                    # Appliquer le dernier crop attaqué calculé
                    with fgsm_result_lock:
                        if fgsm_last_crop[0] is not None:
                            face_crop = fgsm_last_crop[0]
                else:
                    with state_lock:
                        state.perturbation_heatmap = None
                    with fgsm_result_lock:
                        fgsm_last_clean[0] = None

                # ── Défense ──
                if current_mode == 'hardened':
                    face_crop = defender.apply_defense(face_crop, defense_type=active_defense_type, face_recognizer=face_recognizer)

                # ── Anomalie FFT (~2ms) ──
                is_attacked, anom_score = anomaly_detector.analyze(face_crop)

                # ── Inférence FaceNet locale ──
                t0             = time.time()
                is_denoised    = (active_defense_type in ('clean_pipeline', 'neural_lr_recover', 'neural_cnn_recover'))
                identity, conf = face_recognizer.predict(face_crop, is_denoised=is_denoised)
                infer_ms       = int((time.time()-t0)*1000)
                dbg(f"[INFER LOCAL] {identity} | conf={conf:.2f} | {infer_ms}ms")

                # Si la défense neuronale a détecté une attaque active, on marque l'anomalie
                if current_mode == 'hardened' and getattr(defender, 'attack_detected', False):
                    is_attacked = True
                    anom_score = 1.0
                    # Si la méthode sélectionnée exige le blocage strict
                    if active_defense_type in ('neural_lr_reject', 'neural_cnn_reject'):
                        identity = "Inconnu"
                        conf = 0.0
                        dbg("[DEFENSE NEURALE] ⚠️ Attaque détectée et bloquée (Stricte) ! Accès refusé.")
                    else:
                        dbg(f"[DEFENSE NEURALE] ⚠️ Attaque détectée et nettoyée. Identité restaurée : {identity}")

                with state_lock:
                    state.identity     = identity
                    state.confidence   = conf
                    state.access_level = rights_manager.get_access_level(identity)
                    state.permissions  = rights_manager.get_permissions(identity)
                    state.infer_ms     = infer_ms

                # ── Overlay ──
                x1, y1, x2, y2 = bbox
                ui_cfg    = rights_manager.get_ui_config(identity)
                hex_color = ui_cfg['color'].lstrip('#')
                color_bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
                cv2.putText(frame, f"{identity} ({conf:.2f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
                # Overlay debug
                with state_lock:
                    fps_val  = state.fps
                    fgsm_val = state.fgsm_ms
                mode_label = "ATK" if attack_active else ("HRD" if current_mode=='hardened' else "STD")
                cv2.putText(frame,
                            f"FPS:{fps_val} INFER:{infer_ms}ms FGSM:{fgsm_val:.0f}ms [{mode_label}]",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 1)

                # Recoller le crop patché (lunettes) dans la frame
                if patch_applied:
                    frame = _overlay_patch(frame, bbox, face_crop)

        with state_lock:
            state.anomaly_detected = is_attacked
            state.anomaly_score    = anom_score
            if is_attacked:
                logging.warning(f"ANOMALIE score={anom_score:.2f} identite={state.identity}")
            ok2, buf = cv2.imencode('.jpg', frame)
            if ok2:
                state.latest_frame = buf.tobytes()

threading.Thread(target=_video_thread, daemon=True, name="video-thread").start()

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
            if frame:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    with state_lock:
        return jsonify({
            'identity':         state.identity,
            'access_level':     state.access_level,
            'permissions':      state.permissions,
            'anomaly_detected': state.anomaly_detected,
            'anomaly_score':    round(state.anomaly_score, 4),
            'attack_active':    state.attack_active,
            'model_mode':       state.model_mode,
            'defense_type':     state.defense_type,
            'fps':              state.fps,
            'confidence':       round(state.confidence, 4),
            'infer_ms':         state.infer_ms,
            'fgsm_ms':          state.fgsm_ms,
            'fgsm_epsilon':     round(state.fgsm_epsilon, 3),
            'strict_block':     state.strict_block,
            'mode':             'CPU_LOCAL'
        })

@app.route('/api/audit_logs')
def get_audit_logs():
    try:
        return jsonify({"success": True, "logs": read_audit_log_lines(100)})
    except Exception as e:
        logging.error(f"audit_logs: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/debug')
def debug_info():
    with state_lock:
        return jsonify({
            'mode':             'CPU_LOCAL',
            'fps':              state.fps,
            'infer_ms':         state.infer_ms,
            'fgsm_ms':          state.fgsm_ms,
            'enrolled':         list(face_recognizer.enrolled_embeddings.keys()),
            'attack_active':    state.attack_active,
            'model_mode':       state.model_mode,
        })

@app.route('/api/toggle_attack', methods=['POST'])
def toggle_attack():
    data = request.json or {}
    if 'active' not in data:
        return jsonify({"success": False, "error": "Parametre 'active' manquant."}), 400
    with state_lock:
        state.attack_active = bool(data['active'])
        # Reset le crop FGSM quand on désactive
        if not data['active']:
            state.perturbation_heatmap = None
            with fgsm_result_lock:
                fgsm_last_crop[0] = None
                fgsm_last_clean[0] = None
    status = "activée" if data['active'] else "désactivée"
    logging.info(f"Attaque {status}")
    return jsonify({"success": True, "message": f"Attaque {status}."})


@app.route('/api/perturbation_heatmap')
def get_perturbation_heatmap():
    """Image JPEG : carte de chaleur du bruit adversarial (visible si attaque active)."""
    with state_lock:
        data = state.perturbation_heatmap
    if not data:
        return '', 204
    return Response(data, mimetype='image/jpeg')


@app.route('/api/set_fgsm_epsilon', methods=['POST'])
def set_fgsm_epsilon():
    """Règle la force de l'attaque FGSM (epsilon) depuis l'interface."""
    data = request.json or {}
    try:
        epsilon = float(data.get('epsilon', 0.15))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "epsilon invalide"}), 400
    epsilon = max(0.0, min(0.30, epsilon))
    with state_lock:
        state.fgsm_epsilon = epsilon
    fgsm_attacker.epsilon = epsilon
    with fgsm_result_lock:
        fgsm_last_crop[0] = None
        fgsm_last_clean[0] = None
    with state_lock:
        state.perturbation_heatmap = None
    logging.info(f"FGSM epsilon -> {epsilon}")
    return jsonify({"success": True, "epsilon": epsilon})

@app.route('/api/toggle_mode', methods=['POST'])
def toggle_mode():
    data = request.json or {}
    mode = data.get('mode', '')
    if mode not in ('standard', 'hardened'):
        return jsonify({"success": False, "error": "Mode invalide."}), 400
    try:
        face_recognizer.switch_mode(mode)
        with state_lock:
            state.model_mode = mode
        return jsonify({"success": True, "message": f"Mode → {mode}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/toggle_strict_block', methods=['POST'])
def toggle_strict_block():
    data = request.json or {}
    active = data.get('active', False)
    with state_lock:
        state.strict_block = active
    status = "activé" if active else "désactivé"
    logging.info(f"Rejecteur d'attaque {status}")
    return jsonify({"success": True, "message": f"Rejecteur d'attaque {status}."})

@app.route('/api/set_defense_type', methods=['POST'])
def set_defense_type():
    data = request.json or {}
    defense_type = data.get('defense_type', '')
    allowed = ('none', 'clean_pipeline', 'neural_lr_recover', 'neural_lr_reject', 'neural_cnn_recover', 'neural_cnn_reject')
    if defense_type not in allowed:
        return jsonify({"success": False, "error": f"Type de défense invalide. Choix possibles: {allowed}"}), 400
    with state_lock:
        state.defense_type = defense_type
    logging.info(f"Type de défense configuré : {defense_type}")
    return jsonify({"success": True, "message": f"Défense configurée à : {defense_type}"})

@app.route('/api/enroll', methods=['POST'])
def enroll():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"success": False, "error": "Données manquantes."}), 400
    file  = request.files['image']
    name  = request.form['name']
    level = request.form.get('level', RightsManager.EMPLOYEE)
    if not file.filename:
        return jsonify({"success": False, "error": "Fichier vide."}), 400
    save_path = os.path.join(ENROLLED_DIR, secure_filename(f"{name}_{file.filename}"))
    file.save(save_path)
    try:
        rights_manager.add_identity(name, level)
        return jsonify({"success": True, "message": f"{name} enrôlé comme {level}."})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/analyze_static', methods=['POST'])
def analyze_static():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Aucune image envoyée."}), 400
    file = request.files['image']
    if not file.filename:
        return jsonify({"success": False, "error": "Fichier vide."}), 400
    try:
        nparr = np.frombuffer(file.read(), np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "error": "Format invalide."}), 400

        bboxes = face_detector.detect(img, force=True)
        if not bboxes:
            return jsonify({"success": True, "results": {
                "identity": "Aucun visage", "access_level": "DENIED",
                "confidence": 0.0, "anomaly_detected": False, "anomaly_score": 0.0
            }})

        bbox      = bboxes[0]
        face_crop = face_detector.crop_face(img, bbox, size=FACE_CROP_SIZE)
        if face_crop is None:
            return jsonify({"success": False, "error": "Erreur recadrage."}), 500

        is_attacked, anom_score = anomaly_detector.analyze(face_crop)
        identity, conf          = face_recognizer.predict(face_crop)
        access_level            = rights_manager.get_access_level(identity)
        permissions             = rights_manager.get_permissions(identity)

        x1, y1, x2, y2 = bbox
        ui_cfg    = rights_manager.get_ui_config(identity)
        hex_color = ui_cfg['color'].lstrip('#')
        color_bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 3)
        cv2.putText(img, f"{identity} ({conf:.2f})",
                    (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)

        _, buf     = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buf).decode()

        return jsonify({"success": True, "results": {
            "identity":         identity,
            "confidence":       round(float(conf), 4),
            "access_level":     access_level,
            "permissions":      permissions,
            "anomaly_detected": is_attacked,
            "anomaly_score":    round(float(anom_score), 4),
            "image_base64":     img_base64,
        }})
    except Exception as e:
        logging.error(f"analyze_static: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/generate_glasses_attack', methods=['GET', 'POST'])
def generate_glasses_attack():
    # Support GET for quick test / documentation
    if request.method == 'GET':
        return jsonify({
            "success": True,
            "message": "Endpoint generate_glasses_attack expects a POST with 'image' and optional 'target'."
        })

    # POST handling
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Parametre 'image' manquant."}), 400
    target_name = request.form.get('target', 'Manager_Demo')
    target_emb = face_recognizer.enrolled_embeddings.get(target_name)
    if target_emb is None:
        return jsonify({
            "success": False,
            "error":   f"'{target_name}' non enrôlé.",
            "enrolled": list(face_recognizer.enrolled_embeddings.keys())
        }), 404

    file = request.files['image']
    if not file.filename:
        return jsonify({"success": False, "error": "Fichier vide."}), 400

    try:
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "error": "Format invalide."}), 400

        bboxes = face_detector.detect(img, force=True)
        if not bboxes:
            return jsonify({"success": False, "error": "Aucun visage détecté."}), 422

        bbox = bboxes[0]
        x1, y1, x2, y2 = bbox
        face_crop = face_detector.crop_face(img, bbox, size=FACE_CROP_SIZE)
        if face_crop is None:
            return jsonify({"success": False, "error": "Erreur recadrage."}), 500

        id_before, conf_before = face_recognizer.predict(face_crop)
        attacked_crop          = patch_attacker.attack(face_crop, target_emb, target_name=target_name)
        id_after, conf_after   = face_recognizer.predict(attacked_crop)
        is_anom, anom_score    = anomaly_detector.analyze(attacked_crop)
        access_after           = rights_manager.get_access_level(id_after)
        permissions_after      = rights_manager.get_permissions(id_after)

        # Activer automatiquement l'attaque glasses live pour le flux vidéo
        with state_lock:
            state.glasses_attack_active = True
            state.glasses_attack_target = target_name
            state.patch_ready           = True
            state.patch_generating      = False

        result_img   = img.copy()
        img_h, img_w = img.shape[:2]
        rx1 = max(0, x1); ry1 = max(0, y1)
        rx2 = min(img_w, x2); ry2 = min(img_h, y2)
        result_img[ry1:ry2, rx1:rx2] = cv2.resize(attacked_crop, (rx2-rx1, ry2-ry1))

        ui_cfg    = rights_manager.get_ui_config(id_after)
        hex_color = ui_cfg['color'].lstrip('#')
        col_bgr   = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(result_img, (x1, y1), (x2, y2), col_bgr, 3)
        cv2.putText(result_img, f"{id_after} ({conf_after:.2f}) [PATCH]",
                    (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_bgr, 2)

        _, buf  = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buf).decode()

        logging.info(f"Patch: {id_before}({conf_before:.2f}) -> {id_after}({conf_after:.2f})")

        return jsonify({
            "success": True,
            "target":  target_name,
            "before":  {"identity": id_before,  "confidence": round(float(conf_before), 4)},
            "after":   {
                "identity":         id_after,
                "confidence":       round(float(conf_after), 4),
                "access_level":     access_after,
                "permissions":      permissions_after,
                "anomaly_detected": is_anom,
                "anomaly_score":    round(float(anom_score), 4),
                "image_base64":     img_b64,
            }
        })
    except Exception as e:
        logging.error(f"generate_glasses_attack: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ─────────────────────────────────────────────
# ROUTES LUNETTES (feature_lunettes)
# ─────────────────────────────────────────────
@app.route('/glasses_attack')
def glasses_attack():
    return render_template('glasses_attack.html')


@app.route('/api/enrolled_list')
def enrolled_list():
    """Retourne la liste des identités enrôlées dans FaceRecognizer."""
    names = sorted(face_recognizer.enrolled_embeddings.keys())
    return jsonify({'enrolled': names, 'count': len(names)})


@app.route('/api/patch_status')
def patch_status():
    """Retourne l'état du patch pré-calculé : génération en cours, prêt, cible."""
    with state_lock:
        return jsonify({
            'generating':  state.patch_generating,
            'ready':       state.patch_ready,
            'active':      state.glasses_attack_active,
            'target':      state.glasses_attack_target,
        })


@app.route('/api/toggle_patch_live', methods=['POST'])
def toggle_patch_live():
    """
    Active ou désactive l'attaque lunettes en temps réel.
    Body JSON : { "active": true, "target": "Manager_Demo" }  → génère + active
                { "active": false }                           → désactive
    """
    data        = request.get_json(force=True, silent=True) or {}
    active      = bool(data.get('active', False))
    target_name = data.get('target', 'Manager_Demo')
    enrolled    = sorted(face_recognizer.enrolled_embeddings.keys())

    if not active:
        with state_lock:
            state.glasses_attack_active = False
            state.glasses_attack_target = ''
            state.patch_ready           = False
            state.patch_generating      = False
        logging.info("Patch live désactivé")
        return jsonify({'success': True, 'message': 'Attaque lunettes désactivée.'})

    # Vérifier que la cible est enrôlée
    target_emb = face_recognizer.enrolled_embeddings.get(target_name)
    if target_emb is None:
        return jsonify({
            'success': False,
            'error':   f"'{target_name}' non enrôlé.",
            'enrolled': enrolled
        }), 404

    # Si le patch est déjà en cache, on l'active directement
    if patch_attacker.get_cached_patch(target_name) is not None:
        with state_lock:
            state.glasses_attack_active = True
            state.glasses_attack_target = target_name
            state.patch_ready           = True
            state.patch_generating      = False
        logging.info(f"Patch live activé (cache) pour {target_name}")
        return jsonify({
            'success':    True,
            'message':    f"Patch activé depuis le cache pour {target_name}.",
            'generating': False,
            'target':     target_name,
        })

    # Sinon, lancer la génération dans un thread background
    with state_lock:
        state.glasses_attack_active = True
        state.glasses_attack_target = target_name
        state.patch_generating      = True
        state.patch_ready           = False

    def _generate():
        """Thread background — calcule le patch PGD et met à jour l'état."""
        try:
            print(f"[PATCH] Génération du patch pour '{target_name}'...")
            emb = face_recognizer.enrolled_embeddings.get(target_name)
            if emb is None:
                raise ValueError(f"Embedding introuvable pour {target_name}")
            patch_attacker.generate_patch(emb, target_name)
            with state_lock:
                state.patch_generating = False
                state.patch_ready      = True
            print(f"[PATCH] Patch prêt pour '{target_name}'.")
            logging.info(f"Patch live prêt pour {target_name}")
        except Exception as exc:
            with state_lock:
                state.patch_generating      = False
                state.glasses_attack_active = False
                state.glasses_attack_target = ''
            logging.error(f"Génération patch échouée : {exc}")
            print(f"[PATCH] ERREUR génération : {exc}")

    threading.Thread(target=_generate, daemon=True,
                     name=f"patch-gen-{target_name}").start()
    logging.info(f"Patch live : génération lancée pour {target_name}")
    return jsonify({
        'success':    True,
        'message':    f"Génération du patch en cours pour {target_name}…",
        'generating': True,
        'target':     target_name,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
