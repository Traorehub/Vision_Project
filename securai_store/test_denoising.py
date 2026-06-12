import os
import sys
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path

# Ajouter la racine du projet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.face_recognizer import FaceRecognizer
from modules.face_detector import FaceDetector
from modules.defender import Defender
from paths import ENROLLED_DIR

dataset_dir = r"C:/Users/MOH/Desktop/Vision_Project/securai_store"
clean_dir = os.path.join(dataset_dir, "dataset", "clean")
adv_dir = r"C:/Users/MOH/Desktop/Vision_Project/securai_store/dataset/adv"

recognizer = FaceRecognizer(mode='standard')
detector = FaceDetector()
defender = Defender()

# Enrôler les visages connus
print("Enrôlement des identités...")
ENROLLED_PATH = Path(ENROLLED_DIR)
for file_path in ENROLLED_PATH.iterdir():
    if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    identity = file_path.stem.split("-")[0]
    img = cv2.imread(str(file_path))
    if img is None:
        continue
    bboxes = detector.detect(img)
    if bboxes:
        crop = detector.crop_face(img, bboxes[0], size=160)
    else:
        crop = cv2.resize(img, (160, 160))
    recognizer.enroll_face(identity, crop)

# Charger un échantillon d'images adverses
adv_images = [os.path.join(adv_dir, f) for f in os.listdir(adv_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not adv_images:
    print("Aucune image adverse trouvée dans le dataset/adv !")
    sys.exit(1)

# Charger une image clean de référence (première image du dossier clean)
clean_images = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
first_clean = None
if clean_images:
    first_clean = cv2.imread(clean_images[0])
    if first_clean is None:
        print('[test_denoising] Warning: unable to read reference clean image')
else:
    print('[test_denoising] Warning: no clean images found in dataset/clean/')

print(f"Chargement de {len(adv_images)} images adverses pour le test...")

methods = {
    "Original (Attaqué)": lambda img: img,
    "Gaussian Blur 3x3": lambda img: cv2.GaussianBlur(img, (3, 3), 0),
    "Gaussian Blur 5x5": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
    "Gaussian Blur 7x7": lambda img: cv2.GaussianBlur(img, (7, 7), 0),
    "Median Blur 3": lambda img: cv2.medianBlur(img, 3),
    "Median Blur 5": lambda img: cv2.medianBlur(img, 5),
    "Median Blur 7": lambda img: cv2.medianBlur(img, 7),
    "Bilateral Filter": lambda img: cv2.bilateralFilter(img, 9, 75, 75),
    "Rescale 80x80 -> 160x160": lambda img: cv2.resize(cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA), (160, 160), interpolation=cv2.INTER_CUBIC),
    "Rescale 64x64 -> 160x160": lambda img: cv2.resize(cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA), (160, 160), interpolation=cv2.INTER_CUBIC),
    "Rescale 80x80 + Median 3": lambda img: cv2.medianBlur(cv2.resize(cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA), (160, 160), interpolation=cv2.INTER_CUBIC), 3),
    "Rescale 80x80 + Gaussian 3x3": lambda img: cv2.GaussianBlur(cv2.resize(cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA), (160, 160), interpolation=cv2.INTER_CUBIC), (3, 3), 0),
    "[Defender] clean_image pipeline": lambda img: defender.clean_image(img),
}

# Tester les 5 premières images
for idx, img_path in enumerate(adv_images[:5]):
    print(f"\n================ Image {idx+1}: {os.path.basename(img_path)} ================")
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    for name, method in methods.items():
        sanitized = method(img)
        is_denoised = (name != "Original (Attaqué)")
        pred_name, score = recognizer.predict(sanitized, is_denoised=is_denoised)
        print(f"[{name}] -> Identité prédite : {pred_name} (conf={score:.4f})")
