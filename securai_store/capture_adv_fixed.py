#!/usr/bin/env python
"""
Capture d'images adversariales (FGSM) à partir du flux webcam.

Ce script :
1️⃣  Charge toutes les images présentes dans le répertoire d’enrôlement
    (`data/enrolled`) et les **enrôle** dans un `FaceRecognizer`.
2️⃣  Vérifie que la cible demandée (par défaut « Manager_Demo ») existe.
3️⃣  Capture une frame toutes les N frames, applique FGSM sur le crop du visage
    (ou sur l’image entière si aucun visage n’est détecté) et
    enregistre le résultat dans `data/defense/adv/`.
"""

import os
import sys
import cv2
import time
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------
# Ajout du répertoire du projet au PYTHONPATH (nécessaire si le script est
# lancé hors du serveur Flask)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Import des modules du projet
from modules.face_detector    import FaceDetector
from modules.face_recognizer  import FaceRecognizer
from modules.fgsm_attacker    import FGSMAttacker
from paths                    import ENROLLED_DIR   # <- chemin correct du projet
# Convert the string path to a pathlib.Path for filesystem operations
ENROLLED_PATH = Path(ENROLLED_DIR)

# ----------------------------------------------------------------------
# Configuration
INTERVAL      = 2               # 1 image sauvegardée toutes les INTERVAL frames
TARGET_NAME   = "Manager_Demo"  # modifier si vous avez une autre identité
EPSILON       = 0.03            # magnitude de l'attaque FGSM
OUTPUT_DIR    = PROJECT_ROOT / "data" / "defense" / "adv"
FACE_SIZE     = 160             # même taille que le modèle FaceNet

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------------------------------------------------
# 1️⃣  Enrôler toutes les images du dossier `ENROLLED_DIR`
logging.info(f"Chargement des visages enrôlés depuis {ENROLLED_DIR}")

face_recognizer = FaceRecognizer(mode='standard')
face_detector   = FaceDetector()

# Parcourir le répertoire d’enrôlement
for file_path in ENROLLED_PATH.iterdir():
    if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    # Le nom de l’identité est la partie avant le premier '-'
    identity = file_path.stem.split("-")[0]
    img = cv2.imread(str(file_path))
    if img is None:
        logging.warning(f"Impossible de lire {file_path}")
        continue
    # Détecter le (premier) visage et le recadrer
    bboxes = face_detector.detect(img)
    if bboxes:
        crop = face_detector.crop_face(img, bboxes[0], size=FACE_SIZE)
    else:
        crop = cv2.resize(img, (FACE_SIZE, FACE_SIZE))
    face_recognizer.enroll_face(identity, crop)
    logging.info(f"Enrôlé : {identity} ← {file_path.name}")

# ----------------------------------------------------------------------
# 2️⃣  Vérifier que la cible demandée existe bien
target_emb = face_recognizer.enrolled_embeddings.get(TARGET_NAME)
if target_emb is None:
    raise RuntimeError(
        f"Cible '{TARGET_NAME}' non enrôlée – vérifiez les images d'enrôlement dans {ENROLLED_DIR}"
    )
logging.info(f"Cible d'attaque trouvée : {TARGET_NAME}")

# ----------------------------------------------------------------------
# 3️⃣  Initialiser l'attaquant FGSM
fgsm_attacker = FGSMAttacker(face_recognizer.model, epsilon=EPSILON)

# ----------------------------------------------------------------------
# 4️⃣  Boucle de capture webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam inaccessible.")

frame_counter = 0
logging.info("Démarrage de la capture – appuyez sur ESC pour quitter.")
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame_counter += 1
        if frame_counter % INTERVAL != 0:
            # Affichage optionnel (décommenter si vous voulez voir le flux)
            # cv2.imshow("Webcam (clean)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # ---- Détection du visage ----
        bboxes = face_detector.detect(frame)
        if bboxes:
            crop = face_detector.crop_face(frame, bboxes[0], size=FACE_SIZE)
        else:
            # Aucun visage détecté – on attaque l’image brute
            crop = cv2.resize(frame, (FACE_SIZE, FACE_SIZE))

        # ---- Attaque FGSM ----
        attacked_crop = fgsm_attacker.attack(crop, target_emb)

        # ---- Sauvegarde ----
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = OUTPUT_DIR / f"adv_{ts}.jpg"
        cv2.imwrite(str(out_path), attacked_crop)
        logging.info(f"Sauvegarde : {out_path}")

        # Affichage du résultat (optionnel)
        cv2.imshow("Adversarial (FGSM)", attacked_crop)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC pour quitter
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
