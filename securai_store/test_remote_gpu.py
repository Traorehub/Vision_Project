"""
=============================================================
  SecurAI — Test de connexion au serveur GPU distant
  À lancer sur TON PC local pour vérifier la connexion.

  Usage :
      python test_remote_gpu.py
  
  Ou avec une image spécifique :
      python test_remote_gpu.py --image chemin/vers/visage.jpg
=============================================================
"""

import argparse
import time
import cv2
import numpy as np
import requests

# ==============================================================
# CONFIGURATION — Modifie uniquement cette ligne si besoin
# ==============================================================
REMOTE_URL = "http://vision.api.near-u-api.org"
TIMEOUT    = 5  # secondes


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_health() -> dict | None:
    """Test 1 : vérifie que le serveur répond et affiche son état."""
    print_header("TEST 1 : Vérification de la connexion (Health Check)")
    print(f"  URL : {REMOTE_URL}/health")
    try:
        t0 = time.perf_counter()
        resp = requests.get(f"{REMOTE_URL}/health", timeout=TIMEOUT)
        ping_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code == 200:
            data = resp.json()
            print(f"\n  ✅  Serveur en ligne !")
            print(f"  Ping réseau      : {ping_ms:.1f} ms")
            print(f"  Device utilisé   : {data.get('device_used', data.get('device', '?'))}")
            print(f"  GPU              : {data.get('gpu_name', 'N/A')}")
            print(f"  FaceNet chargé   : {'✅' if data.get('facenet_loaded') else '❌'}")
            print(f"  Defender LR      : {'✅' if data.get('defender_lr') else '❌ (non chargé)'}")
            print(f"  Defender CNN     : {'✅' if data.get('defender_cnn') else '❌ (non chargé)'}")
            return data
        else:
            print(f"\n  ❌ Réponse HTTP inattendue : {resp.status_code}")
            return None
    except requests.ConnectionError:
        print(f"\n  ❌ ERREUR : Impossible de contacter {REMOTE_URL}")
        print("  → Vérifie que le serveur de ton pote est bien démarré.")
        print("  → Vérifie que Cloudflare Tunnel est actif chez lui.")
        return None
    except requests.Timeout:
        print(f"\n  ❌ TIMEOUT : Le serveur n'a pas répondu en {TIMEOUT}s.")
        return None


def generate_test_face() -> np.ndarray:
    """Génère une image de visage de test (visage synthétique basique)."""
    img = np.zeros((160, 160, 3), dtype=np.uint8)
    # Fond clair
    img[:] = (200, 190, 180)
    # Cercle (tête)
    cv2.ellipse(img, (80, 75), (60, 70), 0, 0, 360, (220, 190, 160), -1)
    # Yeux
    cv2.circle(img, (55, 65), 8, (50, 50, 50), -1)
    cv2.circle(img, (105, 65), 8, (50, 50, 50), -1)
    # Nez
    cv2.line(img, (80, 75), (75, 95), (150, 120, 100), 2)
    # Bouche
    cv2.ellipse(img, (80, 105), (20, 8), 0, 0, 180, (100, 60, 60), 2)
    return img


def test_inference(img: np.ndarray) -> None:
    """Test 2 : envoie une image au serveur et vérifie la réponse d'inférence."""
    print_header("TEST 2 : Inférence (Embedding + Détection d'attaque)")
    print(f"  URL : {REMOTE_URL}/inference")
    print(f"  Taille image envoyée : {img.shape[1]}×{img.shape[0]} px")

    _, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_bytes = img_encoded.tobytes()
    print(f"  Taille paquet envoyé : {len(img_bytes) / 1024:.1f} Ko")

    try:
        t0 = time.perf_counter()
        resp = requests.post(
            f"{REMOTE_URL}/inference",
            files={"image": ("face.jpg", img_bytes, "image/jpeg")},
            timeout=TIMEOUT
        )
        round_trip_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code == 200:
            data = resp.json()
            embedding = data.get("embedding", [])
            attack    = data.get("attack_detected", False)
            prob_adv  = data.get("prob_adv", 0.0)
            server_ms = data.get("latency_ms", "N/A")
            device    = data.get("device_used", "?")

            print(f"\n  ✅  Inférence réussie !")
            print(f"  Device serveur         : {device}")
            print(f"  Latence calcul serveur : {server_ms} ms")
            print(f"  Latence totale A/R     : {round_trip_ms:.1f} ms")
            print(f"  Embedding reçu         : {len(embedding)} dimensions")
            print(f"  Embedding extrait      : [{embedding[0]:.4f}, {embedding[1]:.4f}, ..., {embedding[-1]:.4f}]")
            print(f"  Attaque détectée       : {'⚠️  OUI' if attack else '✅  NON'} (probabilité adv = {prob_adv:.2%})")

            # Évaluation de la qualité de la connexion
            print("\n  — Évaluation de la qualité de la connexion —")
            if round_trip_ms < 100:
                print(f"  ⭐⭐⭐ Excellent  (<100ms) — Idéal pour du temps réel")
            elif round_trip_ms < 300:
                print(f"  ⭐⭐   Bon        (100-300ms) — Utilisable pour la démo")
            else:
                print(f"  ⭐     Lent       (>300ms) — Possible freeze du flux vidéo")
        else:
            print(f"\n  ❌ Erreur serveur : {resp.status_code}")
            print(f"  Réponse : {resp.text}")

    except requests.Timeout:
        print(f"\n  ❌ TIMEOUT : Le serveur n'a pas répondu en {TIMEOUT}s.")
    except Exception as e:
        print(f"\n  ❌ Erreur inattendue : {e}")


def test_throughput() -> None:
    """Test 3 : envoie 10 requêtes de suite pour mesurer le FPS réel."""
    print_header("TEST 3 : Débit (10 requêtes consécutives — simulation FPS)")

    img = generate_test_face()
    _, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_bytes = img_encoded.tobytes()

    latencies = []
    print("  Envoi en cours ", end="", flush=True)

    for i in range(10):
        try:
            t0 = time.perf_counter()
            resp = requests.post(
                f"{REMOTE_URL}/inference",
                files={"image": ("face.jpg", img_bytes, "image/jpeg")},
                timeout=TIMEOUT
            )
            elapsed = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                latencies.append(elapsed)
                print(".", end="", flush=True)
            else:
                print("X", end="", flush=True)
        except Exception:
            print("!", end="", flush=True)

    print("\n")
    if latencies:
        avg = sum(latencies) / len(latencies)
        fps = 1000.0 / avg if avg > 0 else 0
        print(f"  Latence moyenne : {avg:.1f} ms")
        print(f"  Latence min     : {min(latencies):.1f} ms")
        print(f"  Latence max     : {max(latencies):.1f} ms")
        print(f"  FPS estimé      : {fps:.1f} frames/seconde")
        if fps >= 15:
            print("  ✅  Débit suffisant pour une démo fluide (>15 FPS)")
        elif fps >= 5:
            print("  ⚠️  Débit acceptable pour une démo lente (5-15 FPS)")
        else:
            print("  ❌  Débit trop faible — risque de freeze pendant la démo")
    else:
        print("  ❌  Toutes les requêtes ont échoué.")


# ==============================================================
# Main
# ==============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test de connexion au serveur GPU distant SecurAI"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Chemin vers une image de visage réelle à tester (optionnel)"
    )
    args = parser.parse_args()

    print("\n  SecurAI — Diagnostic du serveur GPU distant")
    print(f"  Serveur ciblé : {REMOTE_URL}\n")

    # Test 1 : Connexion et état du serveur
    health_data = test_health()

    if health_data is None:
        print("\n  ⛔ Le serveur est inaccessible. Les tests suivants sont annulés.")
        print("  → Vérifications à faire :")
        print("     1. Ton pote a-t-il lancé : python inference_server.py ?")
        print("     2. Le tunnel Cloudflare est-il actif ? (cloudflared tunnel run ...)")
        print("     3. Teste manuellement : https://vision.api.near-u-api.org/health")
    else:
        # Test 2 : Inférence
        if args.image:
            test_img = cv2.imread(args.image)
            if test_img is None:
                print(f"\n  ⚠️  Image introuvable : {args.image} — utilisation d'un visage synthétique.")
                test_img = generate_test_face()
        else:
            print("\n  [Info] Aucune image fournie — utilisation d'un visage synthétique.")
            test_img = generate_test_face()

        test_inference(test_img)

        # Test 3 : Débit
        test_throughput()

    print("\n" + "=" * 60)
    print("  Diagnostic terminé.")
    print("=" * 60 + "\n")
