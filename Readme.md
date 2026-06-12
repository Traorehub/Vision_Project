# SecurAI

Plateforme de contrôle d'accès biométrique avec reconnaissance faciale en temps réel, simulation d'attaques adversariales et mécanismes de défense intégrés.

SecurAI combine détection de visages, identification par embeddings et audit de sécurité dans une interface web opérationnelle. Le système permet d'observer, en conditions réelles, comment des perturbations adversariales peuvent compromettre un pipeline de reconnaissance faciale — et comment des contre-mesures (filtrage spatial, analyse spectrale, modèles durcis) peuvent les atténuer.

---

## Fonctionnalités

- **Reconnaissance faciale** — FaceNet (InceptionResnetV1, VGGFace2) + détection YOLOv8
- **Contrôle d'accès** — gestion des rôles et permissions (Manager, Employé, Refus)
- **Attaque FGSM** — usurpation ciblée sur flux webcam en temps réel
- **Attaque par patch** — optimisation PGD d'un patch occlusif (lunettes adversariales)
- **Défenses** — filtre gaussien/médian, modèle durci, détection d'anomalie FFT
- **Dashboard web** — live feed, analyse statique, export PDF de session
- **Journal d'audit** — traçabilité des événements dans `security_audit.log`

---

## Architecture

```text
Vision_Project/
├── README.md
├── requirements_fianl.txt      # PyTorch CUDA 12.1 (GPU NVIDIA)
└── securai_store/
    ├── app_cpu.py              # Inférence locale CPU
    ├── app_gpu_direct.py       # Inférence locale GPU
    ├── app.py                  # Variante inférence distante
    ├── paths.py
    ├── rights_manager.py
    ├── data/enrolled/          # Visages enrôlés (Role_Nom-N.jpg)
    ├── models/
    ├── modules/
    │   ├── face_detector.py
    │   ├── face_recognizer.py
    │   ├── fgsm_attacker.py
    │   ├── patch_attacker.py
    │   ├── defender.py
    │   └── anomaly_detector.py
    └── templates/
        ├── EntranceControl.html
        ├── StaticAnalysis.html
        └── glasses_attack.html
```

---

## Installation

### Environnement CPU

```powershell
cd Vision_Project
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r securai_store\requirements.txt
```

Pour une installation PyTorch CPU pure, utiliser les wheels depuis [pytorch.org](https://pytorch.org).

### Environnement GPU (NVIDIA + CUDA 12.1)

```powershell
cd Vision_Project
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements_fianl.txt
```

Vérification CUDA :

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Lancement

Depuis `securai_store/` :

| Commande | Description |
|----------|-------------|
| `python app_cpu.py` | Stack complète sur CPU |
| `python app_gpu_direct.py` | Stack complète sur GPU local |
| `python app.py` | Variante avec inférence distante |

Interface : **http://127.0.0.1:5000**

Au démarrage, le serveur affiche le device actif, un benchmark inférence/FGSM et le nombre d'identités enrôlées.

---

## Enrôlement

1. Photo du visage (centré, éclairage uniforme).
2. Nommer le fichier : `Role_Nom-Numero.jpg`  
   Exemples : `Manager_Demo-1.jpg`, `Employee_Alice-1.jpg`
3. Placer dans `securai_store/data/enrolled/`.
4. Redémarrer l'application.

---

## Interface

- **Entrance Control** — flux webcam, activation attaque FGSM, bascule mode standard/durci, blocage strict.
- **Analyse statique** — comparaison image originale / attaquée / défendue sur upload.
- **Patch lunettes** — génération et application live d'un patch adversarial.
- **Export PDF** — rapport de session depuis le journal d'événements.

Métriques disponibles en overlay : FPS, latence d'inférence, score d'anomalie FFT.

---

## Prérequis

| Mode | Matériel | Débit indicatif |
|------|----------|-----------------|
| CPU | Processeur multi-cœur | 5–15 FPS |
| GPU | NVIDIA, CUDA 12.x | 20–40 FPS |

Webcam requise pour le flux live. Navigateur Chromium recommandé.
