# capture_clean.py
"""Capture des images propres depuis la webcam.

Ce script capture une frame toutes les N frames (défaut 2) et la sauvegarde dans
`dataset/clean/` avec un timestamp dans le nom de fichier.

Usage : `python capture_clean.py`
"""
import os
import cv2
import time
from datetime import datetime

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dataset", "clean")
FRAME_SKIP = 2  # sauvegarder 1 frame sur FRAME_SKIP

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam inaccessible.")

frame_counter = 0
print("[capture_clean] Démarrage – appuyez sur Ctrl+C pour quitter.")
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        frame_counter += 1
        if frame_counter % FRAME_SKIP == 0:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(OUTPUT_DIR, f"clean_{ts}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[capture_clean] Sauvegarde {filename}")
        # Affichage optionnel, désactivez si vous ne voulez pas de fenêtre
        cv2.imshow('Capture clean (Esc pour quitter)', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
