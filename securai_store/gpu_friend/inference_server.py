"""
=============================================================
  SecurAI — Serveur d'inférence GPU
  À lancer chez ton pote sur son PC avec GPU.
  
  Démarrage :
      python inference_server.py
  
  Ensuite ton pote lance le tunnel Cloudflare avec son token :
      cloudflared tunnel run --token <TON_TOKEN>
=============================================================
"""

import os
import cv2
import time
import numpy as np
import torch
import joblib
from flask import Flask, request, jsonify
from facenet_pytorch import InceptionResnetV1

# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------
app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU uniquement"
print("=" * 60)
print(f"  SecurAI Inference Server")
print(f"  Device  : {DEVICE}")
print(f"  GPU     : {GPU_NAME}")
print("=" * 60)

# ------------------------------------------------------------------
# Chargement des modèles
# ------------------------------------------------------------------

# 1. FaceNet (pour extraction des embeddings 512D)
print("[*] Chargement de FaceNet (InceptionResnetV1)...")
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
print("[OK] FaceNet chargé.")

# Échauffement (Warmup) du GPU pour éviter le temps de latence au premier appel
print("[*] Échauffement du GPU avec un tenseur fantôme...")
try:
    with torch.no_grad():
        dummy_tensor = torch.zeros((1, 3, 160, 160)).to(DEVICE)
        _ = facenet(dummy_tensor)
    print("[OK] GPU prêt et chaud.")
except Exception as e:
    print(f"[WARN] Échec du warmup GPU : {e}")

# 2. Defender — Régression Logistique (si dispo)
clf = None
if os.path.exists("models/defender.pkl"):
    try:
        clf = joblib.load("models/defender.pkl")
        print("[OK] Defender LR (defender.pkl) chargé.")
    except Exception as e:
        print(f"[WARN] Impossible de charger defender.pkl : {e}")
else:
    print("[WARN] models/defender.pkl introuvable — détection LR désactivée.")

# 3. Defender — CNN TorchScript (si dispo)
cnn_model = None
if os.path.exists("models/defender_cnn.pt"):
    try:
        cnn_model = torch.jit.load("models/defender_cnn.pt", map_location=DEVICE)
        cnn_model.eval()
        print("[OK] Defender CNN (defender_cnn.pt) chargé.")
    except Exception as e:
        print(f"[WARN] Impossible de charger defender_cnn.pt : {e}")
else:
    print("[WARN] models/defender_cnn.pt introuvable — détection CNN désactivée.")


# ------------------------------------------------------------------
# Utilitaires
# ------------------------------------------------------------------

def decode_image(raw_bytes: bytes) -> np.ndarray | None:
    """Décode une image reçue en bytes (JPEG/PNG) vers un tableau NumPy BGR."""
    np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def preprocess_for_facenet(img_bgr: np.ndarray) -> torch.Tensor:
    """Convertit un crop BGR en tenseur prêt pour FaceNet (160×160, normalisé)."""
    img_resized = cv2.resize(img_bgr, (160, 160))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = (img_rgb.astype(np.float32) - 127.5) / 128.0
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return tensor


def detect_attack(embedding: list[float]) -> tuple[bool, float]:
    """Passe l'embedding dans le classificateur Defender et retourne (attaque?, prob)."""
    prob_adv = 0.0
    attack_detected = False

    if cnn_model is not None:
        with torch.no_grad():
            emb_t = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            logits = cnn_model(emb_t)
            prob_adv = torch.sigmoid(logits).item()
            attack_detected = prob_adv > 0.5
    elif clf is not None:
        probs = clf.predict_proba([embedding])[0]
        prob_adv = float(probs[1])
        attack_detected = prob_adv > 0.5

    return attack_detected, prob_adv


# ------------------------------------------------------------------
# Routes API
# ------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Test rapide de connexion et état du serveur."""
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "gpu_name": GPU_NAME,
        "facenet_loaded": True,
        "defender_lr": clf is not None,
        "defender_cnn": cnn_model is not None,
    })


@app.route("/inference", methods=["POST"])
def inference():
    """
    Reçoit un crop de visage (JPEG/PNG en multipart).
    Retourne l'embedding FaceNet 512D + résultat de détection d'attaque.
    """
    if "image" not in request.files:
        return jsonify({"error": "Champ 'image' manquant dans la requête."}), 400

    t_start = time.perf_counter()

    raw_bytes = request.files["image"].read()
    img = decode_image(raw_bytes)

    if img is None:
        return jsonify({"error": "Image invalide ou corrompue."}), 400

    # 1. Extraction de l'embedding
    with torch.no_grad():
        tensor = preprocess_for_facenet(img)
        embedding_tensor = facenet(tensor)
        # Normalisation L2
        embedding_tensor = torch.nn.functional.normalize(embedding_tensor, p=2, dim=1)
        embedding = embedding_tensor.cpu().numpy().squeeze().tolist()

    # 2. Détection d'attaque adversariale
    attack_detected, prob_adv = detect_attack(embedding)

    latency_ms = (time.perf_counter() - t_start) * 1000

    return jsonify({
        "embedding": embedding,          # Liste de 512 floats
        "attack_detected": attack_detected,
        "prob_adv": round(prob_adv, 4),
        "device_used": str(DEVICE),
        "latency_ms": round(latency_ms, 2),
    })


# ------------------------------------------------------------------
# Point d'entrée
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("\n[*] Serveur en écoute sur http://0.0.0.0:5000")
    print("[*] En attente du tunnel Cloudflare...\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
