import torch
import torch.nn.functional as F
import numpy as np
import cv2

class FGSMAttacker:
    def __init__(self, model, epsilon=0.03):
        """
        Génère des attaques FGSM sur le modèle FaceNet (InceptionResnetV1).
        L'attaque vise à manipuler l'embedding (vecteur 512D).
        """
        self.model = model
        self.epsilon = epsilon
        self.device = next(model.parameters()).device
        # FaceNet normalization
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)

    def preprocess(self, face_crop: np.ndarray):
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        # Resize to 160x160 as expected by FaceNet
        face_resized = cv2.resize(face_rgb, (160, 160))
        img_tensor = torch.from_numpy(face_resized.transpose(2, 0, 1)).float() / 255.0
        # Normalize
        img_tensor = (img_tensor - self.mean.cpu().squeeze(0)) / self.std.cpu().squeeze(0)
        return img_tensor.unsqueeze(0).to(self.device)

    def attack(self, face_crop: np.ndarray, target_embedding=None, steps=10, epsilon=None):
        """
        Applique l'attaque I-FGSM (Iterative FGSM / PGD) sur un crop BGR.
        FaceNet est très robuste, une attaque itérative est nécessaire.
        epsilon : force de la perturbation (0 = aucune, 0.25+ = forte). Si None, utilise self.epsilon.
        """
        eps = float(epsilon if epsilon is not None else self.epsilon)
        eps = max(0.0, min(0.30, eps))
        if eps <= 0.0:
            return face_crop.copy()

        alpha = max(eps / steps, 0.001)

        img_tensor_orig = self.preprocess(face_crop)
        img_tensor = img_tensor_orig.clone().detach().requires_grad_(True)
        
        for step in range(steps):
            embedding = self.model(img_tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
            
            if target_embedding is not None:
                # S'assurer que l'embedding cible est sur le bon device
                target_embedding = target_embedding.to(self.device)
                # Attaque ciblée
                loss = 1.0 - F.cosine_similarity(embedding, target_embedding)
                sign = -1.0
            else:
                # Attaque non ciblée
                loss = embedding.sum()
                sign = 1.0
                
            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                # Avancer d'un petit pas dans la direction du gradient
                adv_img = img_tensor + sign * alpha * img_tensor.grad.sign()
                
                # S'assurer qu'on ne dépasse pas la limite de perturbation maximale (epsilon)
                eta = torch.clamp(adv_img - img_tensor_orig, min=-eps, max=eps)
                img_tensor = img_tensor_orig + eta
                
            img_tensor.requires_grad = True
            
        # Denormalize
        perturbed_image = img_tensor * self.std + self.mean
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        # Retour BGR
        perturbed_np = perturbed_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        perturbed_bgr = cv2.cvtColor((perturbed_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        h, w = face_crop.shape[:2]
        return cv2.resize(perturbed_bgr, (w, h))

    @staticmethod
    def perturbation_heatmap(clean_bgr: np.ndarray, adv_bgr: np.ndarray, amplify: float = 8.0) -> np.ndarray | None:
        """
        Carte de chaleur de la différence clean vs attaqué (visualisation pédagogique).
        Amplifie le bruit invisible pour l'affichage dans l'interface.
        """
        if clean_bgr is None or adv_bgr is None:
            return None
        clean = cv2.resize(clean_bgr, (160, 160))
        adv = cv2.resize(adv_bgr, (160, 160))
        diff = np.abs(adv.astype(np.float32) - clean.astype(np.float32))
        magnitude = np.max(diff, axis=2)
        magnitude = np.clip(magnitude * amplify, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)
        return cv2.addWeighted(clean, 0.3, heat, 0.7, 0)

    def get_perturbation_visual(self, face_crop):
        """Retourne uniquement le bruit ajouté (amplifié x10) pour l'UI."""
        img_tensor = self.preprocess(face_crop).requires_grad_(True)
        
        embedding = self.model(img_tensor)
        loss = embedding.sum()
        self.model.zero_grad()
        loss.backward()
        
        noise = self.epsilon * img_tensor.grad.data.sign()
        noise = torch.clamp(noise * 10, -1, 1) 
        
        noise_np = noise.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        return ((noise_np + 1) / 2 * 255).astype(np.uint8)
