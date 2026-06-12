# Documentation de l'attaque "patch lunettes" et du mode live

## 1. Vue d'ensemble

Ce document décrit la fonctionnalité de patch adversarial en forme de lunettes dans le projet `Deepfake`, en se concentrant sur :
- le module `securai_store/modules/patch_attacker.py`
- les points d'intégration dans `securai_store/app.py`
- le comportement live de l'attaque lunettes
- les routes API associées

L'objectif est de tromper un système de reconnaissance faciale basé sur FaceNet en appliquant une perturbation localisée sur la région des lunettes.

## 1.1 Fichiers impliqués

Les fichiers principaux ayant servi à implémenter cette feature sont :
- `securai_store/modules/patch_attacker.py` : génération, cache et application du patch lunettes.
- `securai_store/app.py` : orchestration du flux vidéo, activation live, état patch et API.
- `securai_store/modules/face_recognizer.py` : calcul des embeddings FaceNet et enrôlement des identités.
- `securai_store/modules/face_detector.py` : détection de visages et recadrage de la région de la face.
- `securai_store/modules/anomaly_detector.py` : analyse d'anomalie utilisée en complément.
- `securai_store/modules/defender.py` : mode `hardened` appliquant une défense locale.
- `securai_store/templates/glasses_attack.html` : interface de contrôle de l'attaque lunettes (contrôle UI).
- `securai_store/test_app.py` : tests unitaires et cas de validation sur l'API patch/live.

## 1.2 Algorithme et technique utilisés

La feature repose sur une attaque adversariale localisée basée sur 3 concepts principaux :
- une perturbation limitée à une forme de lunettes (`glasses mask`)
- l'utilisation d'un modèle de reconnaissance faciale FaceNet pour optimiser la ressemblance vers une cible
- la méthode d'optimisation Projected Gradient Descent (PGD) pour calculer la perturbation

### Technique

La technique utilisée est un patch adversarial localisé. Au lieu d'altérer l'image entière, le système :
1. définit une zone `glasses mask` couvrant les lunettes et le pont du nez,
2. calcule une perturbation uniquement dans cette zone,
3. applique cette perturbation sur le crop du visage,
4. conserve le patch en cache pour une application rapide lors du streaming vidéo.

### Algorithme

L'algorithme est une optimisation PGD (Projected Gradient Descent) sur un patch :
- on représente le patch comme un tenseur de perturbation `perturbation` de taille `(1, 3, 160, 160)`;
- on applique le patch uniquement là où `mask == 1.0` ;
- on calcule l'image adv via `img_tensor + perturbation * mask_t` ;
- on passe cette image dans FaceNet pour obtenir un embedding 512D ;
- on évalue la perte comme `1 - cosine_similarity(embedding, target_embedding)` ;
- on met à jour la perturbation par gradient de signe : `perturbation = perturbation - alpha * sign(grad)` ;
- on projette la perturbation dans l'intervalle `[-epsilon, epsilon]`.

Ce procédé est implémenté dans :
- `PatchAttacker.generate_patch()` : génération du patch pré-calculé ;
- `PatchAttacker.attack()` : génération directe sur un crop, utilisée pour tests ou API.

### Caractéristique clé

- Patch localisé : seul le masque lunettes est modifié.
- Limitation de la norme L-infinity via `epsilon`.
- Normalisation cosinus de FaceNet pour cibler une identité spécifique.
- Cache mémoire pour réutiliser un patch déjà calculé et éviter un recalcul coûteux.

## 2. Architecture globale

### 2.1 Composants clés

- `PatchAttacker` : crée, conserve et applique le patch lunettes.
- `FaceRecognizer` : calcule et stocke les embeddings FaceNet des visages enrôlés.
- `FaceDetector` : détecte les visages et recadre la zone pour l'attaque.
- `app.py` : orchestre la capture vidéo, l'attaque live, le cache des patches et les API.
- `anomaly_detector` / `defender` : offrent des mécanismes de défense et d'analyse, mais ne sont pas au centre de l'attaque patch lunettes.

### 2.2 Différence entre attaque statique et live

- `patch_attacker.attack(...)` : calcule la perturbation adversariale en direct sur un crop, ce qui est plus lent (~2s) et généralement utilisé dans une route API d'analyse ou de démonstration.
- `patch_attacker.generate_patch(...)` + `patch_attacker.apply_patch(...)` : workflow live préféré, où le patch est pré-généré (PGD) et appliqué ensuite très rapidement (< 5ms/frame).

## 3. `PatchAttacker` — génération et application du patch

Fichier principal : `securai_store/modules/patch_attacker.py`

### 3.1 Objectif

Créer un patch en forme de lunettes qui modifie uniquement les pixels dans une région ciblée du visage afin de faire ressembler l'empreinte faciale extraite par FaceNet à une identité cible.

### 3.2 Initialisation

Constructeur :
- `model` : modèle FaceNet (InceptionResnetV1) déjà chargé et en mode `eval()`.
- `epsilon` : amplitude maximale de la perturbation (0.35 par défaut).
- `steps` : nombre d'itérations PGD pour la génération de patch (40 par défaut).
- `alpha` : pas de mise à jour du patch à chaque itération (0.02 par défaut).
- Cache interne `_patch_cache` pour stocker les perturbations par cible.

### 3.3 Masque lunettes — `create_glasses_mask(h, w)`

Ce masque binaire définit la zone attaquable :
- deux ellipses pour les verres
- un pont du nez
- des branches de lunettes à gauche et à droite

Le masque est centré sur un visage recadré en `160x160` et retourne un tableau float32 où `1.0` correspond à la zone modifiée.

### 3.4 Prétraitement / post-traitement

- `_preprocess(bgr_crop)` :
  - convertit BGR -> RGB
  - redimensionne à `160x160`
  - normalise selon FaceNet
  - retourne un tenseur `(1, 3, 160, 160)`

- `_postprocess(tensor, original_size)` :
  - inverse la normalisation
  - convertit en BGR
  - redimensionne à la taille du crop original

### 3.5 Génération du patch — `generate_patch(target_embedding, target_name)`

Processus :
1. construit une image neutre `160x160` grise
2. génère le masque de lunettes
3. initialise une perturbation nulle
4. exécute `steps` itérations PGD :
   - calcule l'image adv = image + perturbation * masque
   - normalise l'embedding produit par FaceNet
   - minimise `1 - cosine_similarity(embedding, target_embedding)`
   - met à jour la perturbation avec `alpha` et clamp à `[-epsilon, epsilon]`
5. stocke le patch dans `_patch_cache[target_name]`

Important : cette génération est conçue pour être lancée une seule fois par cible, de préférence dans un thread background.

### 3.6 Application du patch live — `apply_patch(face_bgr, perturbation)`

Processus :
1. prétraite le crop de visage
2. applique le masque lunettes
3. ajoute la perturbation pré-calculée sous le masque
4. clamp le résultat puis post-traite vers BGR

Ce code ne recalcul pas le patch à chaque frame, il effectue uniquement une addition et une normalisation, ce qui est très rapide.

### 3.7 Attaque directe — `attack(face_bgr, target_embedding)`

Fonction plus lente :
- calcule une perturbation PGD directement sur le crop fourni
- n'utilise pas de cache
- utile pour des cas où le patch n'a pas été pré-généré

Cette méthode est maintenue pour compatibilité, mais le live patch se repose sur `generate_patch` + `apply_patch`.

### 3.8 Outils visuels

- `get_glasses_overlay(size)` : retourne une image BGRA de l'emplacement des lunettes pour visualisation.
- `get_colored_mask_preview(size)` : retourne une image BGR du masque.

Ces fonctions sont utiles pour le debug ou l'interface, mais ne sont pas nécessaires à l'attaque elle-même.

## 4. Intégration dans `securai_store/app.py`

Ce fichier orchestre le mode live, le flux vidéo et les API.

### 4.1 État global

`SystemState` contient :
- `attack_active` : active l'attaque FGSM distante
- `glasses_attack_active` : active le patch lunettes live
- `glasses_attack_target` : cible actuelle du patch live
- `patch_generating` : indique qu'un patch est en cours de génération
- `patch_ready` : indique qu'un patch est prêt dans le cache
- `model_mode` : mode `standard` ou `hardened`

### 4.2 Initialisation

Au démarrage :
- `FaceDetector()` charge YOLOv8-face
- `FaceRecognizer(mode='standard')` charge FaceNet
- `PatchAttacker(face_recognizer.model, epsilon=0.35, steps=40, alpha=0.02)` instancie le module de patch
- `AnomalyDetector` et `Defender` sont aussi initialisés

### 4.3 Thread vidéo `_video_thread()`

Responsabilités principales :
- lecture de la webcam
- détection de visage
- crop du visage
- application du patch live si actif
- envoi périodique vers HF pour inference / FGSM
- analyse d'anomalie
- mise à jour d'un buffer JPEG partagé

### 4.4 Application du patch dans le flux vidéo

Lorsque `glasses_attack_active` est vrai et qu'un patch est caché pour la cible :
- on récupère `cached_perturbation = patch_attacker.get_cached_patch(glasses_attack_target)`
- si présent, on appelle `patch_attacker.apply_patch(face_crop, cached_perturbation)`
- si le résultat est valide, on redéfinit `face_crop = patched`
- on positionne `patch_applied = True`
- après les traitements, on recolle le face crop patché dans l'image full frame avec `_overlay_patch(...)`

Cette logique garantit que le patch est visible dans la sortie vidéo, et pas seulement utilisé pour la reconnaissance locale.

### 4.5 Fonction helper `_overlay_patch(...)`

`app.py` contient une fonction qui remplace la zone du visage détecté dans l'image complète par le crop patché :
- recadre les coins aux dimensions de l'image
- redimensionne le crop patché à la taille du `bbox`
- copie le crop sur la frame

Ceci permet d'afficher la version attaquée de la face dans le flux vidéo.

### 4.6 Mode `hardened`

Si `state.model_mode == 'hardened'`, le flux applique `defender.apply_defense(face_crop, defense_type='gaussian')` avant l'analyse. Cela montre que l'attaque patch est combinée à un mode de défense local.

### 4.7 Envoi vers HF et fallback

- `hf_queue` conserve un seul crop en attente
- `_hf_worker()` envoie ce crop à `send_to_hf_infer(face_crop)`
- si HF est indisponible, le système bascule en inference locale
- les routes API exposent l'état HF et l'état du patch live

## 5. API pertinentes pour l'attaque lunettes

### 5.1 `/api/patch_status`
Retourne l'état du patch live :
- `generating`
- `ready`
- `active`
- `target`

### 5.2 `/api/toggle_patch_live`
Permet :
- activer le patch live pour une cible enrôlée
- désactiver le patch live

Flux d'activation :
1. Vérifie si la cible est enregistrée dans `face_recognizer.enrolled_embeddings`
2. Si un patch existe déjà dans le cache, active directement le mode live
3. Sinon, lance un thread background `_generate()` qui calcule `patch_attacker.generate_patch(...)`
4. met à jour `state.patch_generating` et `state.patch_ready`

En cas d'échec, le patch live est désactivé et l'état revient à `inactive`.

### 5.3 `/api/generate_glasses_attack`
Route plus ancienne / complémentaire :
- si `live=true` : active uniquement le mode live sans générer le patch ici
- si `stop=true` : désactive le mode live
- sinon, prend une image uploadée, détecte un visage, effectue :
  - inference avant attaque via `send_to_hf_infer(face_crop)`
  - attaque statique `patch_attacker.attack(face_crop, target_emb)`
  - inference après attaque
  - renvoie l'image patchée encodée en base64

Cette route est utile pour tester l'effet d'un patch sur une image unique et mesurer la reconnaissance avant/après.

## 6. Comportement du cache de patch

`PatchAttacker` stocke les perturbations en mémoire sous forme de tenseurs torch dans `_patch_cache`.

- clé : `target_name` (par exemple `Manager_Demo`)
- valeur : tenseur perturbation `(1, 3, 160, 160)`

Quand `toggle_patch_live` est appelé pour une cible déjà présente dans le cache, l'attaque live démarre immédiatement sans recalcul.

## 7. Séquence d'utilisation live

1. le système démarre et enrôle les visages existants depuis `ENROLLED_DIR`
2. l'utilisateur active l'attaque lunettes live via l'API ou l'interface
3. si nécessaire, le serveur calcule un patch en thread background
4. quand le patch est prêt, `state.patch_ready = True`
5. pendant la boucle vidéo :
   - détection du visage
   - recadrage du visage
   - application du patch pré-calculé
   - collage du crop attaqué dans la frame
   - affichage vidéo mise à jour
6. le système continue à analyser et à envoyer des crops à HF pour mise à jour d'identité

## 8. Points importants et limitations

- Le patch est appliqué localement uniquement si `glasses_attack_active` et que le patch est en cache.
- La génération `generate_patch()` peut durer plusieurs secondes sur CPU.
- L'attaque live est conçue pour être un flux rapide : patch pré-calculé + application rapide.
- L'image patchée est recollée dans le flux vidéo, mais la perturbation ne modifie pas la détection du visage : seule la région est altérée.
- Si la cible n'est pas enrôlée, le patch live ne peut pas démarrer.

## 9. Extension et debug

### 9.1 Ligne de débogage

Le système logge des événements importants :
- activation/désactivation du patch
- lancement de génération
- patch prêt
- erreurs de génération
- attaques FGSM et HF

### 9.2 Ajout de nouvelles cibles

Pour ajouter une cible :
1. déposer une photo dans `ENROLLED_DIR`
2. redémarrer le service ou appeler l'enrôlement via API
3. la cible apparaît dans `face_recognizer.enrolled_embeddings`

### 9.3 Améliorations possibles

- persister le cache de patch entre redémarrages
- afficher le masque lunettes dans l'UI de `glasses_attack.html`
- séparer plus clairement l'attaque statique et le mode live
- ajouter un seuil de confiance plus fin pour la sélection de cible

## 9.4 Fichiers ayant servi à implémenter cette fonctionnalité

- `securai_store/modules/patch_attacker.py`
- `securai_store/app.py`
- `securai_store/modules/face_recognizer.py`
- `securai_store/modules/face_detector.py`
- `securai_store/modules/anomaly_detector.py`
- `securai_store/modules/defender.py`
- `securai_store/templates/glasses_attack.html`
- `securai_store/test_app.py`

## 9.5 Résumé technique

La fonctionnalité est un exemple concret d'attaque adversariale ciblée sur un système de reconnaissance faciale.

- La technique est une attaque par patch localisé, non une attaque globale sur toute l'image.
- L'algorithme est un PGD contraint par un masque et un `epsilon` sur la norme L-infinity.
- Le modèle utilisé pour l'objectif est FaceNet (InceptionResnetV1), qui produit des embeddings 512D.
- La métrique d'optimisation est la similarité cosinus entre embedding généré et embedding cible.
- Le mode live utilise un cache de perturbation pré-générée pour un rendu temps réel.

## 9.6 Ce qui est expliqué dans ce document

Ce document couvre :
- le fonctionnement du patch lunettes,
- le rôle de chaque fichier impliqué,
- l'algorithme PGD utilisé,
- la logique du flux vidéo live,
- les routes API et l'état du patch.


## 10. Fichiers principaux à connaître

- `securai_store/modules/patch_attacker.py`
- `securai_store/modules/face_recognizer.py`
- `securai_store/modules/face_detector.py`
- `securai_store/app.py`
- `securai_store/templates/glasses_attack.html`

## 11. Résumé

La fonctionnalité patch lunettes du projet combine un patch adversarial localisé et un mode live. Le patch est généré une fois puis appliqué en temps réel sur le crop de visage détecté, ce qui permet de démontrer un contournement de la reconnaissance FaceNet sans recalcul intensif à chaque frame.

Cette documentation couvre le mécanisme de génération, l'application live, les APIs utilisées et les états système impliqués dans le workflow.
