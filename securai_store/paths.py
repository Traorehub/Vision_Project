"""Chemins partagés — local Windows et Colab (SECURAI_BASE sur Drive)."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.environ.get("SECURAI_BASE", BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, "models")
ENROLLED_DIR = os.path.join(BASE_DIR, "data", "enrolled")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

TUNNEL_LOG = os.path.join(LOGS_DIR, "tunnel.log")
TRAIN_LOG = os.path.join(LOGS_DIR, "train.log")
AUDIT_LOG = os.path.join(BASE_DIR, "security_audit.log")


def read_audit_log_lines(max_lines: int = 100) -> list[str]:
    """Dernières lignes du journal d'audit (UTF-8, tolérant CP1252)."""
    if not os.path.exists(AUDIT_LOG):
        return []
    with open(AUDIT_LOG, "r", encoding="utf-8", errors="replace") as f:
        return [line.strip() for line in f.readlines()[-max_lines:]]

# Racine Google Drive (nom neutre pour Colab)
DRIVE_ROOT = "/content/drive/MyDrive/Vision_project"
DRIVE_SECURAI = os.path.join(DRIVE_ROOT, "securai_store")
