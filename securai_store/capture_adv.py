# capture_adv.py
"""Capture des images perturbées (FGSM) depuis la webcam.

Ce script lit la webcam, applique l'attaque FGSM sur chaque Nᵉ‑ème frame
et sauvegarde le résultat dans `dataset/adv/`.

Il réutilise le même modèle FaceRecognizer que l'application principale
pour obtenir l'embedding cible (par défaut `Manager_Demo`).

Usage : `python capture_adv.py`
"""
import os
import cv2
import time
from datetime import datetime

# Import du code du projet (assurez‑vous que le répertoire parent est dans le PYTHONPATH)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.face_recognizer import FaceRecognizer
from modules.fgsm_attacker import FGSMAttacker

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dataset", "adv")
FRAME_SKIP = 2          # appliquer l'attaque 1 frame sur FRAME_SKIP
TARGET_NAME = "Manager_Demo"  # cible de l'attaque (embedding du visage cible)
EPSILON = 0.03          # magnitude FGSM – ajustez selon vos besoins

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialiser les modules
recognizer = FaceRecognizer(mode='standard')
fgsm = FGSMAttacker(recognizer.model, epsilon=EPSILON)

# Récupérer l'embedding de la cible
target_emb = recognizer.enrolled_embeddings.get(TARGET_NAME)
if target_emb is None:
    raise RuntimeError(f"Cible '{TARGET_NAME}' non enrôlée – vérifiez les images d'enrôlement.")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam inaccessible.")

frame_counter = 0
print("[capture_adv] Démarrage – appuyez sur Ctrl+C pour quitter.")
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        frame_counter += 1
        # Sauvegarder/lafter FGSM chaque FRAME_SKIP‑ième frame
        if frame_counter % FRAME_SKIP == 0:
            # Appliquer FGSM sur le crop du visage (ou sur la frame entière si aucun visage détecté)
            # Ici on utilise la frame brute pour simplifier – vous pouvez intégrer le detector si besoin.
            attacked = fgsm.attack(frame, target_emb)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(OUTPUT_DIR, f"adv_{ts}.jpg")
            cv2.imwrite(filename, attacked)
            print(f"[capture_adv] Sauvegarde {filename}")
        # Affichage optionnel
        cv2.imshow('Capture adv (Esc pour quitter)', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
