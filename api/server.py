import sys
import traceback
import threading
from pathlib import Path
import cv2
import yaml
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from deployment.detector import FatigueDetector

app = FastAPI(
    title="Fatigue Detection API",
    description="API for real-time fatigue detection using hybrid attention mechanism",
    version="1.0.0"
)

detector = None
# Lock to prevent concurrent mediapipe/torch inference (not thread-safe)
_detect_lock = threading.Lock()


@app.on_event("startup")
def startup_event():
    global detector
    try:
        config_path = Path(__file__).parent.parent / 'config' / 'train_config.yaml'
        model_path = Path(__file__).parent.parent / 'models' / 'checkpoints' / 'best_model.pth'

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}.")
            return

        detector = FatigueDetector(str(model_path), config)
        print("Model loaded successfully.")
    except Exception as e:
        traceback.print_exc()
        print(f"Error loading model: {e}")


@app.post("/api/v1/detect")
def detect(file: UploadFile = File(...)):
    """Synchronous endpoint to avoid asyncio/mediapipe conflicts"""
    global detector
    if detector is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please restart the server after training.")

    try:
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # Serialize all inference calls - mediapipe FaceLandmarker is not thread-safe
        with _detect_lock:
            result = detector.detect(frame)

        # Convert landmarks numpy array to list for JSON serialization
        if 'landmarks' in result and result['landmarks'] is not None:
            result['landmarks'] = result['landmarks'].tolist()

        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/reset")
def reset_session():
    """Reset the fatigue detector state (clear LSTM temporal buffers)"""
    global detector
    if detector is not None:
        with _detect_lock:
            detector.ear_buffer.clear()
            detector.mar_buffer.clear()
            detector.ear_consecutive_frames = 0
            detector.mar_consecutive_frames = 0
        return {"status": "success", "message": "Session state reset."}
    return {"status": "error", "message": "Detector not initialized."}


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False, workers=1)
