import cv2
import numpy as np
from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path='yolov8n-face.pt', frame_skip=3):
        """
        Initialise le détecteur YOLOv8 dédié aux visages.
        * **model_path** – chemin du poids « yolov8n-face.pt ». Aucun fallback :
          si le fichier est absent ou trop petit, on lève une exception explicite.
        * **frame_skip** – nombre de frames à ignorer pour alléger le CPU.
        """
        # ------------------------------------------------------------------------
        # Vérification stricte et chargement du modèle.
        # Aucun fallback vers un modèle d'objets généraliste (ex: YOLOv5-n) n'est autorisé.
        # ------------------------------------------------------------------------
        import pathlib
        
        model_file = pathlib.Path(model_path)
        loaded = False
        
        # Liste des chemins potentiels à tester
        paths_to_try = [model_file]
        if model_path == 'yolov8n-face.pt':
            base_dir = pathlib.Path(__file__).parent.parent
            # On cherche dans 'models/' ou directement à la racine de securai_store
            paths_to_try.append(base_dir / 'models' / 'yolov8n-face.pt')
            paths_to_try.append(base_dir / 'yolov8n-face.pt')
            
        for path_opt in paths_to_try:
            if path_opt.is_file() and path_opt.stat().st_size >= 1_000_000:
                try:
                    self.model = YOLO(str(path_opt))
                    loaded = True
                    print(f"[FaceDetector] Modèle chargé avec succès depuis : {path_opt.absolute()}")
                    break
                except Exception as e:
                    print(f"[FaceDetector] Tentative de chargement depuis {path_opt} échouée : {e}")
                    
        if not loaded:
            raise FileNotFoundError(
                f"Impossible de charger un modèle YOLOv8-face valide depuis les chemins testés : {[str(p.absolute()) for p in paths_to_try]}. "
                "Le fichier est peut-être absent, corrompu (KeyError: 'model'), ou incomplet.\n"
                "Veuillez exécuter 'python securai_store/download.py' pour télécharger et installer le modèle correct."
            )
            
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.last_results = []

    def detect(self, frame: np.ndarray, force: bool = False):
        """
        Détecte les visages avec un système de frame skipping pour le CPU.
        Si force=True, on court-circuite le frame skipping (utile pour l'enrôlement ou l'analyse statique).
        """
        if force:
            results = self.model(frame, verbose=False, conf=0.5, imgsz=320)
            boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    boxes.append((x1, y1, x2, y2))
            return boxes

        self.frame_count += 1
        
        # On ne traite qu'une frame sur N
        if self.frame_count % self.frame_skip == 0 or not self.last_results:
            # Augmentation de imgsz de 160 à 320 pour une meilleure précision des coordonnées
            results = self.model(frame, verbose=False, conf=0.5, imgsz=320)
            self.last_results = []
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    self.last_results.append((x1, y1, x2, y2))
                    
        return self.last_results
 
    def crop_face(self, frame, bbox, size=128, margin_ratio=0.30):
        """
        Découpe et redimensionne le visage en ajoutant une marge de 15% pour correspondre
        aux attentes géométriques de FaceNet et augmenter la précision de la similarité.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        bw = x2 - x1
        bh = y2 - y1
        
        # Ajouter la marge
        x1_m = max(0, int(x1 - bw * margin_ratio))
        y1_m = max(0, int(y1 - bh * margin_ratio))
        x2_m = min(w, int(x2 + bw * margin_ratio))
        y2_m = min(h, int(y2 + bh * margin_ratio))
        
        face = frame[y1_m:y2_m, x1_m:x2_m]
        if face.size == 0:
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                return None
            
        face_resized = cv2.resize(face, (size, size))
        return face_resized
