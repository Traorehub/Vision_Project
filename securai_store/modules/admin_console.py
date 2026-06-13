"""Console administrateur — journal temps réel + authentification."""
from __future__ import annotations

import logging
import sys
from collections import deque
from functools import wraps
from threading import Lock

from flask import jsonify, render_template, request, session

ADMIN_PASSWORD = "SecurAI_Administrator"
_CONSOLE_LOCK = Lock()
_CONSOLE_LINES: deque[str] = deque(maxlen=800)


class ConsoleBufferHandler(logging.Handler):
    """Capture les logs Python dans un buffer mémoire (affichage admin)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            with _CONSOLE_LOCK:
                _CONSOLE_LINES.append(line)
        except Exception:
            pass


def install_admin_logging(audit_log_path: str) -> None:
    """Configure logging fichier + buffer console admin."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(audit_log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)

    buffer_handler = ConsoleBufferHandler()
    buffer_handler.setFormatter(fmt)

    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(buffer_handler)


def console_log(message: str, level: str = "INFO") -> None:
    """Messages debug/terminal → buffer admin + stdout (sans doublon logging)."""
    text = str(message).strip()
    if not text:
        return
    with _CONSOLE_LOCK:
        _CONSOLE_LINES.append(text)
    print(text, file=sys.stdout, flush=True)


def get_console_lines(max_lines: int = 200) -> list[str]:
    with _CONSOLE_LOCK:
        lines = list(_CONSOLE_LINES)
    return lines[-max_lines:]


def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("admin_authenticated"):
            return jsonify({"success": False, "error": "Accès refusé — authentification requise."}), 401
        return fn(*args, **kwargs)

    return wrapper


def register_admin_routes(app, state, state_lock, mode_label: str, read_audit_fn) -> None:
    """Enregistre /admin et les APIs protégées sur une app Flask."""
    import os
    app.secret_key = os.environ.get("SECURAI_SECRET", "securai-console-demo-key")

    @app.route("/admin")
    def admin_page():
        return render_template("admin.html")

    @app.route("/api/admin/login", methods=["POST"])
    def admin_login():
        data = request.get_json(silent=True) or {}
        if data.get("password") == ADMIN_PASSWORD:
            session["admin_authenticated"] = True
            logging.info("Console admin : connexion autorité réussie")
            with _CONSOLE_LOCK:
                _CONSOLE_LINES.append("Console admin : connexion autorité réussie")
            return jsonify({"success": True})
        logging.warning("Console admin : tentative de connexion refusée")
        return jsonify({"success": False, "error": "Mot de passe incorrect"}), 401

    @app.route("/api/admin/logout", methods=["POST"])
    def admin_logout():
        session.pop("admin_authenticated", None)
        return jsonify({"success": True})

    @app.route("/api/admin/check")
    def admin_check():
        return jsonify({"authenticated": bool(session.get("admin_authenticated"))})

    @app.route("/api/admin/stream")
    @admin_required
    def admin_stream():
        with state_lock:
            metrics = {
                "identity":              state.identity,
                "access_level":          state.access_level,
                "fps":                   state.fps,
                "infer_ms":              state.infer_ms,
                "confidence":            round(state.confidence, 4),
                "attack_active":         state.attack_active,
                "fgsm_epsilon":          round(state.fgsm_epsilon, 3),
                "fgsm_ms":               state.fgsm_ms,
                "model_mode":            state.model_mode,
                "defense_type":          state.defense_type,
                "anomaly_detected":      state.anomaly_detected,
                "anomaly_score":         round(state.anomaly_score, 4),
                "glasses_attack_active": state.glasses_attack_active,
                "glasses_attack_target": state.glasses_attack_target,
                "patch_generating":      state.patch_generating,
                "patch_ready":           state.patch_ready,
                "mode":                  mode_label,
            }
        return jsonify({
            "success": True,
            "console": get_console_lines(250),
            "audit":   read_audit_fn(150),
            "metrics": metrics,
        })
