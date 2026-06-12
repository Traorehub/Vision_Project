# Guide d'entraînement du modèle de défense

## 1️⃣ Pourquoi ce guide ?
Vous avez déjà les scripts `capture_clean.py` et `capture_adv.py` qui vous permettent de créer deux dossiers :
- `data/defense/clean/` : images sans perturbation.
- `data/defense/adv/` : images avec l'attaque FGSM appliquée.
Ce guide vous montre comment **préparer** les données, **entraîner** un classificateur (logistic regression ou petite CNN) sur un GPU (Colab recommandé), **sauvegarder** le modèle et l’**intégrer** dans votre serveur Flask.

---

## 📂 Structure des dossiers attendue
```
securai_store/
│   capture_clean.py
│   capture_adv.py
│   requirements.txt
│   ...
└─ data/
   └─ defense/
       ├─ clean/   # images capturées sans attaque
       └─ adv/    # images capturées après FGSM
```
Assurez‑vous que les deux dossiers existent avant de lancer les captures.

---

## 2️⃣ Enrôler une cible pour `capture_adv.py`
Le script `capture_adv.py` cherche l’embedding d’une identité nommée **`Manager_Demo`** dans le dictionnaire `FaceRecognizer.enrolled_embeddings`. Si cette identité n'existe pas :
1. Ouvrez l’interface web (`http://localhost:5000/`), choisissez **Enrôlement** et chargez une photo de la personne que vous voulez utiliser comme cible.
2. Donnez‑lui exactement le nom **`Manager_Demo`** (casse respectée). Le serveur ajoutera l’image dans `enrolled/` et mettra à jour `enrolled_embeddings`.
3. Redémarrez le serveur Flask (ou rechargez le module) pour que le nouveau mapping soit pris en compte.

> **Alternative** : modifiez la constante `TARGET_NAME` dans `capture_adv.py` pour correspondre à une identité déjà enrôlée (ex. « John_Doe »).

---

## 3️⃣ Collecte des images
### 3.1 Capture des images propres
```bash
python capture_clean.py   # appuyez sur q pour quitter
```
Les images seront enregistrées sous `data/defense/clean/` avec un timestamp.

### 3.2 Capture des images perturbées (FGSM)
```bash
python capture_adv.py    # appuyez sur q pour quitter
```
Assurez‑vous que `TARGET_NAME` correspond à une identité enrôlée, sinon le script s’arrêtera avec l’erreur que vous avez rencontrée.

---

## 4️⃣ Entraînement sur Google Colab (GPU T4)
1. **Créer un nouveau notebook** → **Runtime → Change runtime type → GPU**.
2. **Monter votre Drive** afin d’accéder aux dossiers.
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. **Copier le projet** dans le notebook (zip ou `git clone`).
```bash
!unzip -q /content/drive/MyDrive/Vision_Project/securai_store.zip -d /content/drive/MyDrive/Vision_Project/
```
4. **Installer les dépendances** :
```python
!pip -q install opencv-python-headless scikit-learn torch torchvision tqdm
```
5. **Ajouter le chemin du projet** :
```python
import sys, os
PROJECT_ROOT = '/content/drive/MyDrive/Vision_Project/securai_store'
sys.path.append(PROJECT_ROOT)
```
6. **Importer les modules et préparer les données** :
```python
from modules.face_recognizer import FaceRecognizer
import cv2, numpy as np, os
from tqdm import tqdm

recognizer = FaceRecognizer(mode='standard')

clean_dir = os.path.join(PROJECT_ROOT, 'data', 'defense', 'clean')
adv_dir   = os.path.join(PROJECT_ROOT, 'data', 'defense', 'adv')

def embed_path(p):
    img = cv2.imread(p)
    return recognizer.model.predict(img[np.newaxis, ...]).squeeze()

X, y = [], []
for f in tqdm(os.listdir(clean_dir)):
    X.append(embed_path(os.path.join(clean_dir, f)))
    y.append(0)
for f in tqdm(os.listdir(adv_dir)):
    X.append(embed_path(os.path.join(adv_dir, f)))
    y.append(1)

X = np.stack(X)
y = np.array(y)
print('Dataset shape :', X.shape, y.shape)
```
7. **Entraîner un classificateur** (par ex. Logistic Regression) :
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=2000, n_jobs=-1)
clf.fit(Xtr, ytr)
print(classification_report(yte, clf.predict(Xte)))
print('Confusion matrix:\n', confusion_matrix(yte, clf.predict(Xte)))
```
8. **Sauvegarder le modèle** dans votre Drive :
```python
import joblib, pathlib
model_path = pathlib.Path('/content/drive/MyDrive/Vision_Project/securai_store/models/defender.pkl')
joblib.dump(clf, model_path)
print('Modèle enregistré →', model_path)
```
---

## 5️⃣ Intégrer le modèle dans le serveur Flask
Modifiez `modules/defender.py` :
```python
import joblib, os
from .face_recognizer import FaceRecognizer

class Defender:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'defender.pkl')
        self.clf = joblib.load(model_path)
        self.recognizer = FaceRecognizer(mode='standard')

    def _embedding(self, img):
        return self.recognizer.model.predict(img[np.newaxis, ...]).squeeze()

    def apply_neural_defense(self, img):
        emb = self._embedding(img)
        pred = self.clf.predict([emb])[0]
        if pred == 1:          # image suspecte → appliquer filtre léger
            return self.preprocess_gaussian(img)
        return img
```
Puis, dans votre flux vidéo (`app_cpu.py` ou `app.py`) :
```python
if current_mode == 'hardened':
    face_crop = defender.apply_neural_defense(face_crop)
```
---

## 6️⃣ Test rapide en local
```bash
# lancer le serveur
(.venv) python app_cpu.py
# Activer le mode hardened via l'API
curl -X POST http://localhost:5000/api/toggle_mode -H "Content-Type: application/json" -d '{"mode":"hardened"}'
```
Observez que les visages suspectés sont d’abord filtrés (flou gaussien) ; vous pouvez mesurer le temps d’exécution dans les logs (`defender.apply_neural_defense`).

---

## 📌 Récapitulatif des actions à faire
1. Enrôler **une cible** nommée `Manager_Demo` (ou changer `TARGET_NAME`).
2. Lancer `capture_clean.py` → remplir `data/defense/clean/`.
3. Lancer `capture_adv.py` → remplir `data/defense/adv/`.
4. Suivre les étapes du notebook Colab pour entraîner le modèle.
5. Copier le fichier `defender.pkl` dans `securai_store/models/`.
6. Mettre à jour `modules/defender.py` comme indiqué.
7. Redémarrer le serveur et activer le mode **hardened**.

---

*Ce guide est conservé sous forme de fichier markdown afin que vous puissiez le relire plus tard.*
