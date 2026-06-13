import cv2
import numpy as np
import os
import joblib
import torch

# Types de défense spatiale (même pipeline clean_image, libellés UI distincts)
SPATIAL_DEFENSE_TYPES = ('clean_pipeline', 'gaussian', 'median', 'compression')
NEURAL_RECOVER_TYPES = ('neural_lr_recover', 'neural_cnn_recover')
NEURAL_REJECT_TYPES = ('neural_lr_reject', 'neural_cnn_reject')
PATCH_DEFENSE_TYPES = ('anti_glasses_patch',)
ALL_DEFENSE_TYPES = ('none',) + SPATIAL_DEFENSE_TYPES + NEURAL_RECOVER_TYPES + NEURAL_REJECT_TYPES + PATCH_DEFENSE_TYPES
DENOISE_DEFENSE_TYPES = SPATIAL_DEFENSE_TYPES + NEURAL_RECOVER_TYPES

_SPATIAL_LABELS = {
    'clean_pipeline': 'Dénoyage adaptatif',
    'gaussian':       'Filtre gaussien',
    'median':         'Filtre médian',
    'compression':    'Compression / feature squeezing',
}


_DEFAULT_MEDIAN_NOISE = 45.37


class Defender:
    """
    Fournit des filtres de prétraitement pour neutraliser le bruit adverse.
    Supporte les filtres heuristiques (gaussian, median) et neuronaux
    (logistic regression, CNN/MLP).
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.lr_path  = os.path.join(base_dir, 'models', 'defender.pkl')
        self.cnn_path = os.path.join(base_dir, 'models', 'defender_cnn.pt')

        self.clf       = None
        self.cnn_model = None
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attack_detected = False

        # Charger la régression logistique
        if os.path.exists(self.lr_path):
            try:
                self.clf = joblib.load(self.lr_path)
                print(f"[Defender] Modèle LR chargé depuis {self.lr_path}")
            except Exception as e:
                print(f"[Defender] Erreur LR : {e}")
        else:
            print(f"[Defender] Warning: {self.lr_path} introuvable.")

        # Charger le CNN TorchScript
        if os.path.exists(self.cnn_path):
            try:
                self.cnn_model = torch.jit.load(self.cnn_path, map_location=self.device)
                self.cnn_model.eval()
                print(f"[Defender] Modèle CNN chargé depuis {self.cnn_path} sur {self.device}")
            except Exception as e:
                print(f"[Defender] Erreur CNN : {e}")
        else:
            print(f"[Defender] Warning: {self.cnn_path} introuvable.")

    # ------------------------------------------------------------------
    # Utilitaires bruit
    # ------------------------------------------------------------------

    def get_median_noise(self) -> float:
        """Renvoie le bruit médian depuis noise_matrix.csv, ou la valeur par défaut."""
        try:
            csv_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "noise_matrix.csv")
            )
            import pandas as pd
            df = pd.read_csv(csv_path)
            return float(df['noise'].median())
        except Exception:
            return _DEFAULT_MEDIAN_NOISE

    def estimate_noise(self, clean_img: np.ndarray, adv_img: np.ndarray) -> float:
        """Différence absolue moyenne pixel‑à‑pixel entre une image propre et adverse."""
        if clean_img.shape != adv_img.shape:
            adv_img = cv2.resize(adv_img, (clean_img.shape[1], clean_img.shape[0]))
        diff = cv2.absdiff(clean_img, adv_img)
        return float(np.mean(diff))

    def compute_noise_matrix(self, clean_dir: str, adv_dir: str):
        """Calcule les niveaux de bruit entre chaque image propre et chaque image adverse
        et sauvegarde le résultat dans noise_matrix.csv.

        Returns:
            pandas.DataFrame si pandas est disponible, sinon list[dict].
        """
        try:
            import pandas as pd
        except ImportError:
            pd = None

        clean_files = [
            f for f in os.listdir(clean_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        adv_files = [
            f for f in os.listdir(adv_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        rows = []
        for c_name in clean_files:
            clean_img = cv2.imread(os.path.join(clean_dir, c_name))
            if clean_img is None:
                continue
            for a_name in adv_files:
                adv_img = cv2.imread(os.path.join(adv_dir, a_name))
                if adv_img is None:
                    continue
                noise_val = self.estimate_noise(clean_img, adv_img)
                rows.append({
                    "clean_image": c_name,
                    "adv_image":   a_name,
                    "noise":       round(noise_val, 2),
                })

        csv_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "noise_matrix.csv")
        )

        if pd is not None and rows:
            df = pd.DataFrame(rows)
            try:
                df.to_csv(csv_path, index=False)
                print(f"[Defender] Noise matrix saved → {csv_path}")
            except Exception as e:
                print(f"[Defender] Erreur CSV : {e}")
            print(f"Rows: {len(rows)}")
            print(f"Median noise: {df['noise'].median():.2f}")
            print(f"Mean noise:   {df['noise'].mean():.2f}")
            return df
        else:
            try:
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write("clean_image,adv_image,noise\n")
                    for r in rows:
                        f.write(f"{r['clean_image']},{r['adv_image']},{r['noise']}\n")
                print(f"[Defender] Noise matrix saved → {csv_path}")
            except Exception as e:
                print(f"[Defender] Erreur CSV (fallback) : {e}")
            return rows

    # ------------------------------------------------------------------
    # Pipeline de débruitage
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Feature Squeezing (rapide, sans modèle)
    # ------------------------------------------------------------------

    @staticmethod
    def feature_squeeze(img: np.ndarray, bit_depth: int = 5) -> np.ndarray:
        """Réduit la profondeur de bits pour éliminer les perturbations de faible amplitude.
        FGSM avec ε ≤ 8/255 est typiquement annulé par une réduction à 5-6 bits.
        """
        step = max(1, 256 // (2 ** bit_depth))
        squeezed = ((img.astype(np.int32) // step) * step).clip(0, 255).astype(np.uint8)
        return cv2.GaussianBlur(squeezed, (3, 3), 0)

    def clean_image(self, img: np.ndarray, ref_clean: np.ndarray | None = None) -> np.ndarray:
        """Pipeline de dé‑bruitage adaptatif préservant l'identité faciale."""
        if ref_clean is not None:
            noise_val = self.estimate_noise(ref_clean, img)
        else:
            noise_val = self.get_median_noise()

        # h dynamique : plage [4, 15] selon le niveau de bruit mesuré [0, 100]
        h_val = int(min(15, max(4, np.interp(noise_val, [0, 100], [4, 15]))))
        print(f"[Defender] Noise={noise_val:.2f}  NLMeans h={h_val}")

        # 0. Feature Squeezing — neutralise les perturbations FGSM faible amplitude
        img = self.feature_squeeze(img)

        # 1. NL‑Means adaptatif
        cleaned = cv2.fastNlMeansDenoisingColored(
            img, None,
            h=h_val, hColor=h_val,
            templateWindowSize=7, searchWindowSize=21,
        )
        # 2. Bilateral (préserve les bords)
        cleaned = cv2.bilateralFilter(cleaned, d=5, sigmaColor=35, sigmaSpace=35)
        # 3. Median blur
        cleaned = cv2.medianBlur(cleaned, 5 if noise_val > 70 else 3)
        # 4. Rehaussement des détails faciaux
        cleaned = cv2.detailEnhance(cleaned, sigma_s=10, sigma_r=0.15)

        return cleaned

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def apply_defense(
        self,
        img: np.ndarray,
        defense_type: str = 'none',
        face_recognizer=None,
    ) -> np.ndarray:
        """Applique la défense sélectionnée sur l'image.

        Args:
            img:           Image BGR (numpy array).
            defense_type:  'none' | 'clean_pipeline' | 'neural_lr_recover' |
                           'neural_lr_reject' | 'neural_cnn_recover' | 'neural_cnn_reject'
            face_recognizer: Optionnel – objet exposant get_embedding(img).

        Returns:
            Image (possiblement filtrée) sous forme de numpy array BGR.
        """
        self.attack_detected = False

        if defense_type == 'none':
            return img

        if defense_type in PATCH_DEFENSE_TYPES:
            print("[Defender] Anti-patch lunettes — identité réelle (sans forçage démo)")
            return img

        if defense_type in SPATIAL_DEFENSE_TYPES:
            label = _SPATIAL_LABELS.get(defense_type, defense_type)
            print(f"[Defender] {label} → pipeline débruitage (identique au dénoyage)")
            return self.clean_image(img)

        if defense_type in ('neural_lr_recover', 'neural_lr_reject'):
            if self.clf is None:
                print("[Defender] Fallback: LR non disponible")
                return img
            if face_recognizer is not None:
                try:
                    emb_tensor = face_recognizer.get_embedding(img)
                    emb_numpy  = emb_tensor.cpu().numpy().squeeze()
                    probs = self.clf.predict_proba([emb_numpy])[0]
                    pred  = self.clf.predict([emb_numpy])[0]
                    print(f"[Defender LR] probs={probs}  pred={pred} "
                          f"({'ADVERSARIAL' if pred == 1 else 'CLEAN'})")
                    if pred == 1:
                        self.attack_detected = True
                        if defense_type == 'neural_lr_recover':
                            print("[Defender LR] ⚠️ Attaque détectée → nettoyage.")
                            return self.clean_image(img)
                        else:
                            print("[Defender LR] ⚠️ Attaque détectée → rejet.")
                            return img
                except Exception as e:
                    print(f"[Defender] Erreur LR : {e}")
            return img

        if defense_type in ('neural_cnn_recover', 'neural_cnn_reject'):
            if self.cnn_model is None:
                print("[Defender] Fallback: CNN non disponible")
                return img
            if face_recognizer is not None:
                try:
                    emb_tensor = face_recognizer.get_embedding(img)
                    with torch.no_grad():
                        logits = self.cnn_model(emb_tensor.float())
                        prob   = torch.sigmoid(logits).item()
                    pred = 1 if prob > 0.5 else 0
                    print(f"[Defender CNN] prob_adv={prob:.4f}  pred={pred} "
                          f"({'ADVERSARIAL' if pred == 1 else 'CLEAN'})")
                    if pred == 1:
                        self.attack_detected = True
                        if defense_type == 'neural_cnn_recover':
                            print("[Defender CNN] ⚠️ Attaque détectée → nettoyage.")
                            return self.clean_image(img)
                        else:
                            print("[Defender CNN] ⚠️ Attaque détectée → rejet.")
                            return img
                except Exception as e:
                    print(f"[Defender] Erreur CNN : {e}")
            return img

        # Défense inconnue → retour sans modification
        return img


# ---------------------------------------------------------------------------
# CLI : python modules/defender.py --clean_dir … --adv_dir …
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute noise matrix between clean and adversarial images."
    )
    parser.add_argument(
        "--clean_dir",
        default=r"C:\Users\MOH\Desktop\Vision_Project\securai_store\dataset\clean",
    )
    parser.add_argument(
        "--adv_dir",
        default=r"C:\Users\MOH\Desktop\Vision_Project\securai_store\dataset\adv",
    )
    args = parser.parse_args()
    d = Defender()
    d.compute_noise_matrix(args.clean_dir, args.adv_dir)
