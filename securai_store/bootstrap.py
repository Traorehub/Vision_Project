"""Initialise le PYTHONPATH et les imports optionnels (console admin)."""
from __future__ import annotations

import logging
import os
import sys
from collections import deque
from functools import wraps
from threading import Lock

_STORE_DIR = os.path.dirname(os.path.abspath(__file__))
_ADMIN_PASSWORD = "SecurAI_Administrator"
_CONSOLE_LOCK = Lock()
_CONSOLE_LINES: deque[str] = deque(maxlen=800)


def setup_store_path() -> str:
    """Ajoute securai_store/ au path pour `modules`, `paths`, etc."""
    if _STORE_DIR not in sys.path:
        sys.path.insert(0, _STORE_DIR)
    return _STORE_DIR


def _get_console_lines(max_lines: int = 200) -> list[str]:
    with _CONSOLE_LOCK:
        lines = list(_CONSOLE_LINES)
    return lines[-max_lines:]


def _register_admin_routes_impl(app, state, state_lock, mode_label, read_audit_fn) -> None:
    """Routes /admin — utilisées si admin_console.py est absent."""
    from flask import jsonify, render_template, request, session

    app.secret_key = os.environ.get("SECURAI_SECRET", "securai-console-demo-key")

    def admin_required(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not session.get("admin_authenticated"):
                return jsonify({"success": False, "error": "Accès refusé — authentification requise."}), 401
            return fn(*args, **kwargs)
        return wrapper

    @app.route("/admin")
    def admin_page():
        return render_template("admin.html")

    @app.route("/api/admin/login", methods=["POST"])
    def admin_login():
        data = request.get_json(silent=True) or {}
        if data.get("password") == _ADMIN_PASSWORD:
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
            "console": _get_console_lines(250),
            "audit":   read_audit_fn(150),
            "metrics": metrics,
        })


def load_admin_console():
    """Importe la console admin ; repli intégré si le fichier est absent."""
    try:
        from modules.admin_console import (
            install_admin_logging,
            console_log,
            register_admin_routes,
        )
        return install_admin_logging, console_log, register_admin_routes
    except ImportError:
        def install_admin_logging(audit_log_path: str) -> None:
            logging.basicConfig(
                level=logging.INFO,
                filename=audit_log_path,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )

        def console_log(message: str, level: str = "INFO") -> None:
            text = str(message).strip()
            if text:
                with _CONSOLE_LOCK:
                    _CONSOLE_LINES.append(text)
                print(f"[{level}] {text}", flush=True)

        def register_admin_routes(app, state, state_lock, mode_label, read_audit_fn) -> None:
            _register_admin_routes_impl(app, state, state_lock, mode_label, read_audit_fn)

        return install_admin_logging, console_log, register_admin_routes


_SPATIAL_DEFENSE_TYPES = ('clean_pipeline', 'gaussian', 'median', 'compression')
_NEURAL_RECOVER_TYPES = ('neural_lr_recover', 'neural_cnn_recover')
_NEURAL_REJECT_TYPES = ('neural_lr_reject', 'neural_cnn_reject')
_PATCH_DEFENSE_TYPES = ('anti_glasses_patch',)
_DEFAULT_ALL_DEFENSE_TYPES = ('none',) + _SPATIAL_DEFENSE_TYPES + _NEURAL_RECOVER_TYPES + _NEURAL_REJECT_TYPES + _PATCH_DEFENSE_TYPES
_DEFAULT_DENOISE_DEFENSE_TYPES = _SPATIAL_DEFENSE_TYPES + _NEURAL_RECOVER_TYPES


def load_defender():
    """Importe Defender + constantes ; compatible avec une ancienne defender.py."""
    from modules import defender as defender_mod

    Defender = defender_mod.Defender
    all_types = getattr(defender_mod, 'ALL_DEFENSE_TYPES', None)
    denoise_types = getattr(defender_mod, 'DENOISE_DEFENSE_TYPES', None)

    if all_types is None:
        all_types = _DEFAULT_ALL_DEFENSE_TYPES
        defender_mod.ALL_DEFENSE_TYPES = all_types
    if denoise_types is None:
        denoise_types = _DEFAULT_DENOISE_DEFENSE_TYPES
        defender_mod.DENOISE_DEFENSE_TYPES = denoise_types

    if not hasattr(defender_mod, 'SPATIAL_DEFENSE_TYPES'):
        spatial = set(_SPATIAL_DEFENSE_TYPES)
        original = Defender.apply_defense

        def apply_defense(self, img, defense_type='none', face_recognizer=None):
            if defense_type in spatial:
                if hasattr(self, 'clean_image'):
                    return self.clean_image(img)
                if defense_type == 'gaussian' and hasattr(self, 'preprocess_gaussian'):
                    return self.preprocess_gaussian(img)
                if defense_type == 'median' and hasattr(self, 'preprocess_median'):
                    return self.preprocess_median(img)
                return img
            try:
                return original(self, img, defense_type, face_recognizer=face_recognizer)
            except TypeError:
                return original(self, img, defense_type)

        Defender.apply_defense = apply_defense

    return Defender, all_types, denoise_types


setup_store_path()
