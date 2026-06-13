import importlib.util
import os
import sys
import threading
import pathlib
import types

import pytest
import numpy as np

# Ensure the local securai_store directory is on sys.path so imports of top-level modules work.
APP_DIR = pathlib.Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


class DummySession:
    def __init__(self):
        self.headers = {}

    def post(self, *args, **kwargs):
        class DummyResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {'name': 'Inconnu', 'confidence': 0.0, 'access': 'DENIED'}

        return DummyResponse()


class DummyFaceDetector:
    def __init__(self, *args, **kwargs):
        self.frame_skip = 1
        self.frame_count = 0
        self.last_results = []

    def detect(self, frame):
        return [(0, 0, 10, 10)]

    def crop_face(self, frame, bbox, size=160):
        return np.zeros((size, size, 3), dtype=np.uint8)


class DummyFaceRecognizer:
    def __init__(self, mode='standard'):
        self.mode = mode
        self.model = object()
        self.enrolled_embeddings = {}

    def switch_mode(self, mode):
        self.mode = mode

    def enroll_face(self, name, face_crop):
        self.enrolled_embeddings[name] = np.zeros((160, 160, 3), dtype=np.uint8)
        return True

    def predict(self, face_crop):
        return 'Inconnu', 0.0


class DummyAttacker:
    def __init__(self, *args, **kwargs):
        pass

    def attack(self, face_crop, target_emb):
        return face_crop


class DummyDefender:
    def apply_defense(self, face_crop, defense_type='gaussian'):
        return face_crop


class DummyAnomalyDetector:
    def analyze(self, face_crop):
        return False, 0.0


@pytest.fixture
def app_module(monkeypatch):
    # Prevent background threads from starting during import.
    monkeypatch.setattr(threading.Thread, 'start', lambda self: None)

    # Stub network sessions and HF sync calls.
    import requests
    monkeypatch.setattr(requests, 'Session', DummySession)

    # Create dummy top-level module imports for heavy IA modules.
    modules = {
        'modules.face_detector': types.ModuleType('modules.face_detector'),
        'modules.face_recognizer': types.ModuleType('modules.face_recognizer'),
        'modules.fgsm_attacker': types.ModuleType('modules.fgsm_attacker'),
        'modules.patch_attacker': types.ModuleType('modules.patch_attacker'),
        'modules.defender': types.ModuleType('modules.defender'),
        'modules.anomaly_detector': types.ModuleType('modules.anomaly_detector'),
    }

    modules['modules.face_detector'].FaceDetector = DummyFaceDetector
    modules['modules.face_recognizer'].FaceRecognizer = DummyFaceRecognizer
    modules['modules.fgsm_attacker'].FGSMAttacker = DummyAttacker
    modules['modules.patch_attacker'].PatchAttacker = DummyAttacker
    modules['modules.defender'].Defender = DummyDefender
    modules['modules.anomaly_detector'].AnomalyDetector = DummyAnomalyDetector

    for module_name, module_obj in modules.items():
        sys.modules[module_name] = module_obj

    # Load app.py from its file location.
    app_path = APP_DIR / 'app.py'
    spec = importlib.util.spec_from_file_location('test_app_module', str(app_path))
    app = importlib.util.module_from_spec(spec)
    sys.modules['test_app_module'] = app
    spec.loader.exec_module(app)

    yield app


def test_api_hf_health_endpoint(app_module):
    client = app_module.app.test_client()
    response = client.get('/api/hf_health')
    data = response.get_json()

    assert response.status_code == 200
    assert data['available'] is True
    assert data['status'] == 'OK'
    assert data['max_failures'] == app_module.hf_status['max_failures']


def test_api_status_and_toggle_attack(app_module):
    client = app_module.app.test_client()
    response = client.post('/api/toggle_attack', json={'active': True})
    data = response.get_json()

    assert response.status_code == 200
    assert data['success'] is True
    assert app_module.state.attack_active is True

    status_response = client.get('/api/status')
    status_data = status_response.get_json()

    assert status_response.status_code == 200
    assert status_data['attack_active'] is True


def test_api_toggle_mode_invalid_value(app_module):
    client = app_module.app.test_client()
    response = client.post('/api/toggle_mode', json={'mode': 'invalid'})
    data = response.get_json()

    assert response.status_code == 400
    assert data['success'] is False


def test_api_generate_glasses_attack_live(app_module):
    client = app_module.app.test_client()
    app_module.face_recognizer.enrolled_embeddings['Manager_Demo'] = np.zeros((160, 160, 3), dtype=np.uint8)

    response = client.post('/api/generate_glasses_attack', data={'target': 'Manager_Demo', 'live': 'true'})
    data = response.get_json()

    assert response.status_code == 200
    assert data['success'] is True
    assert app_module.state.glasses_attack_active is True
    assert app_module.state.glasses_attack_target == 'Manager_Demo'


def test_api_generate_glasses_attack_live_stop(app_module):
    client = app_module.app.test_client()
    response = client.post('/api/generate_glasses_attack', data={'stop': 'true'})
    data = response.get_json()

    assert response.status_code == 200
    assert data['success'] is True
    assert app_module.state.glasses_attack_active is False
    assert app_module.state.glasses_attack_target == ''
