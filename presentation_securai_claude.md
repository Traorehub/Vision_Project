# Brief Claude — Générer PowerPoint SecurAI (15 slides)

## Ta mission

Produire un fichier **PowerPoint (.pptx)** professionnel pour une **soutenance académique EHTP**.  
**Maximum 15 slides.** Style **visuel ET lisible** : schémas, icônes, formes, textures — mais **assez de texte** pour que le jury comprenne sans lire le rapport.

**Équilibre texte / visuel :**
- **4 à 6 puces par slide** (phrases courtes complètes, pas des mots isolés)
- **1 phrase d’accroche** sous le titre de chaque slide (sous-titre gris/cyan)
- **Pas de paragraphes** de plus de 2 lignes
- **Pas de murs de texte** (pas de copier-coller du rapport)
- Visuels dominants à **60 %**, texte à **40 %**

Durée visée : **8–10 min** de présentation → **démo live** (slide 14) → conclusion (slide 15 **après** la démo).

---

## Charte graphique obligatoire

| Élément | Valeur |
|---------|--------|
| Fond slide | `#0B0F19` (bleu nuit) |
| Accent / titres / lignes | `#00F0FF` (cyan) |
| Texte principal | `#E8EAF0` (gris clair) |
| Succès / accès OK | `#00E676` (vert) |
| Alerte / attaque | `#FF1744` (rouge) |
| Secondaire | `#1565A8` (bleu) |
| Police titres | Montserrat Bold ou Segoe UI Bold, 28–36 pt |
| Police corps | Inter ou Calibri, 16–20 pt |

**Style visuel :** fond sombre cyber-sécurité, hexagones ou grilles cyan en filigrane (5–8 % opacité), icônes flat, 1 schéma fort par slide.

---

## Vocabulaire projet (strict)

- **Projet** = « Attaque par évasion »
- **SecurAI** = surnom du POC développé (pas le nom du projet)
- Rôle élevé = **MANAGER** (identité démo `Manager_Demo`) — jamais « admin » / « administrateur »
- Équipe : **NADAHE Mohamed · TRAORE Fanogo Mohamed · ZOGBELEMOU Francois · SAFARE Yassin**
- **EHTP** · **10 semaines** · **5 sprints Agile**
- **Ne pas mentionner** la feature liveness (abandonnée)

---

## Contenu des 15 slides

---

### Slide 1 — Page de titre

**Visuel :** Logo EHTP en haut (placeholder si besoin). Grand titre centré. Hexagone cyan filigrane. Bandeau bas avec noms.

**Titre :** Attaque par évasion

**Sous-titre :** SecurAI — Proof of Concept de contrôle d'accès biométrique

**Texte sur slide :**
- École Hassania des Travaux Publics (EHTP)
- Projet : Attaque par évasion · Solution : SecurAI
- Durée : 10 semaines · Méthode : Agile (5 sprints)
- Équipe : NADAHE Mohamed · TRAORE Fanogo Mohamed · ZOGBELEMOU Francois · SAFARE Yassin
- Soutenance suivie d'une **démonstration en direct**

---

### Slide 2 — Contexte et problématique

**Visuel :** Schéma comparatif : visage normal → FaceNet → identité correcte | visage + perturbation invisible → FaceNet → **usurpation d'identité**. Icône point d'interrogation rouge.

**Titre :** Pourquoi ce projet ?

**Sous-titre :** Les réseaux de neurones peuvent être trompés par des perturbations imperceptibles

**Texte sur slide :**
- Les systèmes biométriques (FaceNet, YOLO) sont utilisés pour sécuriser des accès physiques
- Une **attaque adversariale** ajoute un bruit calculé, invisible à l'œil, qui modifie la prédiction du modèle
- L'objectif n'est pas seulement de « brouiller » le système, mais d'**usurper une identité cible** (ex. manager)
- Question centrale : *comment attaquer, détecter et se défendre en temps réel ?*
- Réponse : POC **SecurAI** — chaîne complète attaque + défense + interface web

---

### Slide 3 — Scénarios de menace

**Visuel :** Deux cartes côte à côte. **Scénario A** (grande, mise en avant cyan/vert) : caméra + porte sécurisée. **Scénario B** (plus petite, grisée) : smartphone + banque.

**Titre :** Scénarios étudiés

**Sous-titre :** Deux cas réalistes — un retenu pour le POC

**Texte sur slide :**

**Scénario A — Contrôle d'accès biométrique (retenu ✓)**
- Caméra devant une zone sensible (serveurs, locaux admin)
- Attaque FGSM ou patch lunettes pour obtenir les droits d'un **manager**
- Impact : accès physique non autorisé à plusieurs zones

**Scénario B — Fraude KYC mobile (perspective)**
- Attaque sur flux caméra frontale d'une application bancaire
- Hors périmètre : contraintes mobile, réglementation, délai 10 semaines

---

### Slide 4 — Présentation de SecurAI

**Visuel :** Pipeline horizontal avec icônes : Webcam → YOLO (visage) → FaceNet (512D) → RBAC (4 zones). Branches rouge (FGSM, lunettes) et cyan (défenses).

**Titre :** SecurAI — Vue d'ensemble

**Sous-titre :** Un POC complet pour démontrer vulnérabilité et défense en live

**Texte sur slide :**
- **Détection** du visage (YOLOv8-face) et recadrage automatique
- **Reconnaissance** par embedding FaceNet (similarité cosinus, seuil 0,60)
- **Décision d'accès** par rôle : MANAGER, EMPLOYÉ ou REFUS — 4 zones simulées
- **Attaques** intégrées : FGSM itératif (I-FGSM) et patch adversarial « lunettes »
- **Défenses** activables : FFT, dénoyage adaptatif, classificateurs LR/CNN
- Interface web type « salle de sécurité » — flux MJPEG temps réel

---

### Slide 5 — Architecture technique

**Visuel :** Schéma 3 couches empilées + flèches. Icône verrou « threading ».

**Titre :** Architecture modulaire

**Sous-titre :** Séparation claire entre capture, IA et interface

**Texte sur slide :**

**Couche présentation (Frontend)**
- Live Feed : contrôle d'entrée et monitoring
- Analyse statique : test sur image uploadée
- Page dédiée attaque lunettes

**Couche application (Backend Flask)**
- Serveur multithreadé : vidéo + API REST en parallèle
- État partagé (`SystemState`) protégé par verrous
- Streaming MJPEG vers le navigateur

**Couche IA (Modules Python)**
- Détection, reconnaissance, attaque, défense — modules indépendants et testables

---

### Slide 6 — Stack et modes d'exécution

**Visuel :** Grille de logos/icônes technologies + badge CPU / GPU.

**Titre :** Technologies utilisées

**Sous-titre :** Stack vision + deep learning, exécutable en local

**Texte sur slide :**
- **Backend :** Python 3, Flask, threading, OpenCV
- **Vision :** YOLOv8-face (détection), FaceNet / InceptionResnetV1 (VGGFace2)
- **Deep Learning :** PyTorch, scikit-learn (régression logistique défense)
- **Frontend :** HTML/CSS glassmorphism, JavaScript (polling AJAX)
- **Mode CPU local** (`app_cpu.py`) : démo complète pour la soutenance — 10 à 20 FPS
- **Mode GPU** (RTX 4050) : accélération FaceNet + FGSM, démo plus fluide en attaque active

---

### Slide 7 — Attaque FGSM / I-FGSM

**Visuel :** 3 panneaux : visage original | heatmap perturbation (rouge) | visage attaqué. Formule en bas : `x_adv = x + ε · sign(∇L)`. Badge : I-FGSM, ε ≈ 0,15, 10 itérations.

**Titre :** Attaque par évitement — FGSM

**Sous-titre :** Perturbation invisible orientée vers l'identité cible

**Texte sur slide :**
- Basée sur **Goodfellow et al. (2014)** — gradient de la perte pour modifier l'image
- Variante **I-FGSM** (10 pas) : plus efficace qu'un FGSM mono-étape
- Perte optimisée : rapprocher l'embedding FaceNet de la cible **`Manager_Demo`**
- Le bruit est **invisible** à l'œil mais modifie la décision du classificateur
- Calcul **asynchrone** sur CPU (thread worker) pour ne pas bloquer la vidéo
- Curseur ε réglable + **carte de chaleur** du bruit dans l'interface

---

### Slide 8 — Attaque patch lunettes

**Visuel :** Visage avec zone lunettes surlignée. Flèche « PGD » vers patch. Mention **< 5 ms/frame** après cache.

**Titre :** Attaque localisée — Patch adversarial

**Sous-titre :** Deuxième vecteur : perturbation confinée à la zone des lunettes

**Texte sur slide :**
- Patch optimisé par **PGD** (Projected Gradient Descent) sur un masque « lunettes »
- Seule la région des yeux / monture est modifiée — attaque **discrète**
- Objectif : tromper FaceNet pour reconnaître une **identité choisie**
- Patch **pré-calculé** puis mis en cache → application live **< 5 ms par frame**
- Page dédiée : test sur photo uploadée ou activation sur le flux webcam
- Complète le FGSM : attaque globale vs attaque **localisée**

---

### Slide 9 — Mécanismes de défense

**Visuel :** 3 colonnes avec icônes bouclier : FFT (spectre) | Filtre (dénoyage) | Réseau (LR/CNN).

**Titre :** Défenses complémentaires

**Sous-titre :** Détecter, nettoyer ou rejeter — 6 modes sélectionnables

**Texte sur slide :**

**1. Détection FFT (analyse fréquentielle)**
- Le bruit FGSM crée une énergie anormale en **hautes fréquences**
- Alerte visuelle sur le Live Feed avant ou pendant la reconnaissance

**2. Pipeline de dénoyage (NL-Means adaptatif)**
- Tente de **restaurer** l'image et récupérer la vraie identité
- Seuil de reconnaissance assoupli après nettoyage (0,60 → 0,48)

**3. Classificateurs LR + CNN (embeddings clean vs adversarial)**
- Modes **recover** (nettoyage) ou **reject** (blocage strict → « Inconnu »)

---

### Slide 10 — Interface et contrôle d'accès

**Visuel :** Mockup écran sombre « salle de sécurité » : flux vidéo, nom, confiance, 4 voyants zones, panneau alerte.

**Titre :** Interface web et RBAC

**Sous-titre :** Impact visuel immédiat pour le jury et la démo

**Texte sur slide :**
- Design **glassmorphism** — ambiance cybersécurité / salle de contrôle
- Affichage temps réel : identité, score de confiance, **FPS**, alertes
- **4 zones simulées :** Entrée · Stock · Caisse · Serveur
- Couleurs par rôle : **MANAGER** (vert, tout autorisé) · **EMPLOYÉ** (jaune, accès partiel) · **REFUS** (rouge)
- Boutons : activer attaque FGSM, mode hardened, choix de la défense
- Journal des **20 derniers événements** côté client

---

### Slide 11 — Planning Agile (5 sprints)

**Visuel :** Timeline horizontale S1 → S5 avec icônes et ✓. Une seule slide pour tout le planning.

**Titre :** Organisation du projet

**Sous-titre :** 10 semaines · 5 sprints · livrables incrémentaux

**Texte sur slide (sous la timeline) :**

| Sprint | Période | Livrable principal |
|--------|---------|-------------------|
| **S1** | Sem. 1–2 | Reconnaissance faciale + interface Live Feed + RBAC |
| **S2** | Sem. 3–4 | Attaque FGSM async + analyse statique + classificateur LR |
| **S3** | Sem. 5–6 | Défenses FFT, dénoyage, modes hardened, CNN défense |
| **S4** | Sem. 7–8 | Patch lunettes live + accélération GPU RTX 4050 |
| **S5** | Sem. 9–10 | Stabilisation, tests, documentation, rapport, soutenance |

- Méthode **Agile** : démo fonctionnelle dès le sprint 1, enrichie à chaque itération

---

### Slide 12 — Résultats et démonstrations validées

**Visuel :** 4 cartes ou icônes ✓ avec chiffres mis en avant (pas un tableau dense).

**Titre :** Résultats obtenus

**Sous-titre :** Scénarios testés et validés sur le POC

**Texte sur slide :**
- ✓ **Accès légitime** : visage enrôlé reconnu, zones vertes/jaunes selon le rôle
- ✓ **Usurpation FGSM** : attaque active → identité basculée vers `Manager_Demo`
- ✓ **Alerte FFT** : score d'anomalie élevé, panneau rouge sur le Live Feed
- ✓ **Récupération défense** : mode hardened + dénoyage → identité réelle restaurée
- ✓ **Patch lunettes** : usurpation localisée, application **< 5 ms/frame** en live
- ✓ **Analyse statique** : diagnostic complet sur image uploadée (drag & drop)
- Performances CPU : **10–20 FPS** selon machine — suffisant pour démo pédagogique

---

### Slide 13 — Performances CPU vs GPU

**Visuel :** Graphique barres comparatif CPU / GPU pour inférence FaceNet et calcul FGSM. Logo NVIDIA discret. Mention RTX 4050.

**Titre :** Performances

**Sous-titre :** CPU = référence soutenance · GPU = confort en mode attaque

**Texte sur slide :**
- **Mode CPU** (`app_cpu.py`) : pipeline intégral, benchmark automatique au démarrage
- Inférence FaceNet seule : ~50–150 ms/frame (variable selon processeur)
- FGSM (10 itérations) : ~0,5–2 s — atténué par thread async + frame skip (1/5)
- **Mode GPU RTX 4050** : inférence FaceNet typiquement **5–20 ms/frame**
- FGSM fortement accéléré → flux plus stable quand l'attaque est activée
- Serveur distant possible (`gpu_friend/inference_server.py`) pour inférence HTTP
- **Choix soutenance :** CPU pour prouver que tout fonctionne sans matériel dédié

---

### Slide 14 — Démonstration live

**Visuel :** Liste numérotée 1→5 avec icônes + capture ou mockup Live Feed. Flèche « On passe au navigateur ».

**Titre :** Démonstration en direct

**Sous-titre :** SecurAI — Live Feed · `app_cpu.py`

**Texte sur slide :**
1. **Accès normal** — visage enrôlé, voyants zones selon le rôle
2. **Activer l'attaque FGSM** — usurpation vers le profil **Manager_Demo**
3. **Observer l'alerte FFT** — panneau anomalie + heatmap de perturbation
4. **Activer le mode défense** — dénoyage ou LR/CNN recover → identité récupérée
5. **(Si le temps le permet)** Patch lunettes sur le flux ou page dédiée

- Application lancée en local : `python app_cpu.py` (environnement virtuel)
- **→ Passage immédiat à la démo dans le navigateur**

---

### Slide 15 — Conclusion (après la démo)

**Visuel :** 3 mots-clés grands au centre. Bandeau bas avec GitHub. « Merci de votre attention ».

**Titre :** Conclusion

**Sous-titre :** SecurAI — Attaque par évasion

**Texte sur slide :**
- Les systèmes biométriques modernes sont **vulnérables** aux attaques adversariales numériques
- Des défenses **complémentaires** (FFT, dénoyage, classificateurs) permettent de **limiter l'impact**
- SecurAI **démontre** la chaîne complète : attaque → détection → défense → décision d'accès
- Perspectives : attaque physique (lunettes imprimées), extension KYC mobile, CI/CD
- Dépôt GitHub : **github.com/Traorehub/Vision_Project**

**Merci de votre attention — Questions ?**

*Afficher cette slide **après** la démonstration live.*

---

## Annexe — Script oral rapide (notes PowerPoint, optionnel)

Durée cible ~1 min par slide slides 2–13, 30 s pour slides 1, 14, 15.

**Transition slide 14 :** « Nous passons maintenant à la démonstration en conditions réelles. »

**Transition slide 15 :** « Pour conclure après ce que vous venez de voir… »

---

## Livrable attendu

- Fichier **.pptx** · **15 slides** · ratio **16:9**
- Charte couleurs respectée
- **Texte substantiel** (4–6 puces lisibles par slide) + **visuels explicites**
- Notes orateur optionnelles (masquées en présentation)

---

## Ce qu'il ne faut PAS faire

- Plus de 15 slides ou une slide par sprint détaillée (sauf slide 11 résumé)
- Paragraphes de 5+ lignes ou copier-coller du rapport
- Slides avec seulement 3 mots (trop vide)
- Mention liveness / PAD photo téléphone
- Fond blanc · terme « administrateur » (utiliser **MANAGER**)
