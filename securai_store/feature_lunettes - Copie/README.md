# Feature : Attaque patch lunettes

Ce dossier contient les fichiers principaux utilisés pour la feature d'attaque adversariale par patch lunettes dans SecurAI.

## Contenu
- `securai_store/app.py` : orchestration de la capture vidéo, de l'état global et des routes API.
- `securai_store/modules/patch_attacker.py` : génération et application du patch lunettes via PGD.
- `securai_store/modules/face_recognizer.py` : calcul des embeddings FaceNet et enrôlement des visages.
- `securai_store/modules/face_detector.py` : détection des visages avec YOLOv8-face.
- `securai_store/modules/anomaly_detector.py` : analyse d'anomalie locale par FFT.
- `securai_store/modules/defender.py` : filtre de défense locale (gaussian/median).
- `securai_store/modules/fgsm_attacker.py` : support de l’attaque FGSM distante.
- `securai_store/paths.py` : chemins partagés.
- `securai_store/rights_manager.py` : gestion simple des permissions et accès.
- `securai_store/templates/glasses_attack.html` : interface web de contrôle de l'attaque.
- `securai_store/test_app.py` : tests unitaires pour l'API patch live.

## Architecture

1. Détection de visage
   - `face_detector.py` détecte le visage dans la frame webcam.
   - le crop est redimensionné à 160x160.

2. Reconnaissance et enrôlement
   - `face_recognizer.py` calcule un embedding 512D FaceNet.
   - les visages connus sont enrôlés et stockés pour comparaison.

3. Patch lunettes live
   - `patch_attacker.py` crée un patch adversarial localisé en forme de lunettes.
   - le patch est généré par PGD pour améliorer la similarité avec une identité cible.
   - le patch est appliqué en temps réel via `apply_patch()`.

4. Application dans le flux vidéo
   - `app.py` active le patch live si `glasses_attack_active` est vrai.
   - le patch pré-calculé est récupéré du cache et appliqué au crop du visage.
   - le résultat est recollé dans la frame vidéo et affiché.

5. Défense et analyse
   - `defender.py` peut appliquer un filtre local si le mode `hardened` est activé.
   - `anomaly_detector.py` calcule un score d'anomalie local.

## Attaque

- Type : patch adversarial localisé (lunettes).
- Zone modifiée : uniquement la monture de lunettes et le pont du nez.
- Algorithme : Projected Gradient Descent (PGD) avec contrainte L-infinity.
- Objectif : faire en sorte que l'embedding du visage modifié se rapproche de celui de la cible.

### Workflow

1. Génération du patch : `generate_patch(target_embedding, target_name)`.
2. Mise en cache du patch.
3. Application live : `apply_patch(face_bgr, perturbation)` sur chaque frame.

## Pourquoi ce dossier existe

Ce dossier a été créé pour isoler la feature patch lunettes et permettre à un collègue de lire le code essentiel sans parcourir tout le projet.
