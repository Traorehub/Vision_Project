import sys
import os
import torch
import cv2
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from securai_store.modules.face_recognizer import FaceRecognizer
from securai_store.modules.defender import Defender

dataset_dir = "c:/Users/MOH/Desktop/Vision_Project/securai_store/dataset"
clean_dir = os.path.join(dataset_dir, "clean")
adv_dir = os.path.join(dataset_dir, "adv")

# 1. Charger FaceRecognizer
recognizer = FaceRecognizer(mode='standard')
defender = Defender()

print(f"Logistic Regression: {'Loaded' if defender.clf is not None else 'None'}")
print(f"CNN/MLP: {'Loaded' if defender.cnn_model is not None else 'None'}")

# 2. Charger les chemins
clean_images = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
adv_images = [os.path.join(adv_dir, f) for f in os.listdir(adv_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Total clean images: {len(clean_images)}")
print(f"Total adversarial images: {len(adv_images)}")

def evaluate_models():
    # Évaluation LR
    lr_correct_clean = 0
    lr_correct_adv = 0
    
    # Évaluation CNN
    cnn_correct_clean = 0
    cnn_correct_adv = 0
    
    print("\n--- Évaluation Clean Images ---")
    for img_path in clean_images[:50]:  # tester un échantillon de 50
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Obtenir l'embedding
        emb_tensor = recognizer.get_embedding(img)
        emb_numpy = emb_tensor.cpu().numpy().squeeze()
        
        # LR predict
        if defender.clf is not None:
            pred_lr = defender.clf.predict([emb_numpy])[0]
            if pred_lr == 0:
                lr_correct_clean += 1
                
        # CNN predict
        if defender.cnn_model is not None:
            with torch.no_grad():
                logits = defender.cnn_model(emb_tensor.float())
                prob = torch.sigmoid(logits).item()
                pred_cnn = 1 if prob > 0.5 else 0
            if pred_cnn == 0:
                cnn_correct_clean += 1
                
    n_clean_tested = min(len(clean_images), 50)
    print(f"LR Clean Accuracy: {lr_correct_clean}/{n_clean_tested} ({lr_correct_clean/n_clean_tested*100:.2f}%)" if defender.clf is not None else "LR not loaded")
    print(f"CNN Clean Accuracy: {cnn_correct_clean}/{n_clean_tested} ({cnn_correct_clean/n_clean_tested*100:.2f}%)" if defender.cnn_model is not None else "CNN not loaded")
    
    print("\n--- Évaluation Adversarial Images ---")
    for img_path in adv_images[:50]:  # tester un échantillon de 50
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Obtenir l'embedding
        emb_tensor = recognizer.get_embedding(img)
        emb_numpy = emb_tensor.cpu().numpy().squeeze()
        
        # LR predict
        if defender.clf is not None:
            pred_lr = defender.clf.predict([emb_numpy])[0]
            if pred_lr == 1:
                lr_correct_adv += 1
                
        # CNN predict
        if defender.cnn_model is not None:
            with torch.no_grad():
                logits = defender.cnn_model(emb_tensor.float())
                prob = torch.sigmoid(logits).item()
                pred_cnn = 1 if prob > 0.5 else 0
            if pred_cnn == 1:
                cnn_correct_adv += 1
                
    n_adv_tested = min(len(adv_images), 50)
    print(f"LR Adversarial Accuracy: {lr_correct_adv}/{n_adv_tested} ({lr_correct_adv/n_adv_tested*100:.2f}%)" if defender.clf is not None else "LR not loaded")
    print(f"CNN Adversarial Accuracy: {cnn_correct_adv}/{n_adv_tested} ({cnn_correct_adv/n_adv_tested*100:.2f}%)" if defender.cnn_model is not None else "CNN not loaded")

evaluate_models()
