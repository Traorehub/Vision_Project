"""
Point d'entrée GPU — alias de app_gpu_direct.py
    python app_gpu.py
"""
from app_gpu_direct import app, RUN_MODE, GPU_DEVICE

if __name__ == '__main__':
    print(f"[READY] SecurAI {RUN_MODE} ({GPU_DEVICE}) — http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
