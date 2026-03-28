from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from services.media_processing import (
    load_image,
    read_image_metadata,
    read_video_metadata,
    sample_video_frames,
)
from services.model_inference import predict_face

app = FastAPI(title="SheShield Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenCV built-in Haar Cascade for face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def detect_faces_opencv(image_bgr) -> Dict[str, Any]:
    """
    Detect faces using OpenCV Haar Cascade.
    Returns face count and bounding boxes.
    """
    if image_bgr is None:
        return {"face_count": 0, "faces": []}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    results: List[Dict[str, Any]] = []

    for (x, y, w, h) in faces:
        results.append(
            {
                "confidence": None,
                "bbox": {
                    "x_min": int(x),
                    "y_min": int(y),
                    "x_max": int(x + w),
                    "y_max": int(y + h),
                    "width": int(w),
                    "height": int(h),
                },
            }
        )

    return {
        "face_count": len(results),
        "faces": results,
    }


def crop_first_face(image_bgr, face_result: Dict[str, Any]):
    """
    Crop first detected face from image.
    """
    if face_result["face_count"] == 0:
        return None

    bbox = face_result["faces"][0]["bbox"]

    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    x_max = bbox["x_max"]
    y_max = bbox["y_max"]

    face_crop = image_bgr[y_min:y_max, x_min:x_max]

    if face_crop is None or face_crop.size == 0:
        return None

    return face_crop


def analyze_sampled_video_frames(frame_paths: List[str]) -> Dict[str, Any]:
    """
    Detect faces and run model inference on sampled frames.
    """
    frame_results: List[Dict[str, Any]] = []
    frames_with_faces = 0
    total_faces_detected = 0
    predictions: List[Dict[str, Any]] = []

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)

        if frame is None:
            frame_results.append(
                {
                    "frame_name": Path(frame_path).name,
                    "readable": False,
                    "face_count": 0,
                    "faces": [],
                    "prediction": {
                        "label": "frame_unreadable",
                        "confidence": 0.0,
                        "reason_flags": ["frame_read_failed"],
                    },
                }
            )
            continue

        face_result = detect_faces_opencv(frame)
        face_crop = crop_first_face(frame, face_result)

        if face_result["face_count"] > 0:
            frames_with_faces += 1
            total_faces_detected += face_result["face_count"]

        prediction = predict_face(face_crop)

        if prediction["label"] in ["real", "fake"]:
            predictions.append(prediction)

        frame_results.append(
            {
                "frame_name": Path(frame_path).name,
                "readable": True,
                "face_count": face_result["face_count"],
                "faces": face_result["faces"],
                "prediction": prediction,
            }
        )

    return {
        "sampled_frames_analyzed": len(frame_paths),
        "frames_with_faces": frames_with_faces,
        "total_faces_detected": total_faces_detected,
        "frame_results": frame_results,
        "valid_predictions": predictions,
    }


def aggregate_video_prediction(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate frame-level predictions into one video-level result.
    """
    if not predictions:
        return {
            "label": "no_face_detected",
            "confidence": 0.0,
            "reason_flags": ["no_clear_face_found_in_sampled_frames"],
        }

    fake_scores = []
    real_scores = []

    for pred in predictions:
        raw_scores = pred.get("raw_scores", {})
        fake_scores.append(float(raw_scores.get("fake", 0.0)))
        real_scores.append(float(raw_scores.get("real", 0.0)))

    avg_fake = sum(fake_scores) / len(fake_scores)
    avg_real = sum(real_scores) / len(real_scores)

    if avg_fake > avg_real:
        return {
            "label": "fake",
            "confidence": round(avg_fake, 2),
            "reason_flags": ["multiple_frames_show_fake_patterns"],
            "raw_scores": {
                "real": round(avg_real, 4),
                "fake": round(avg_fake, 4),
            },
        }

    return {
        "label": "real",
        "confidence": round(avg_real, 2),
        "reason_flags": ["multiple_frames_show_natural_patterns"],
        "raw_scores": {
            "real": round(avg_real, 4),
            "fake": round(avg_fake, 4),
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Backend is running",
    }


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    suffix = Path(file.filename).suffix if file.filename else ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        try:
            metadata = read_image_metadata(temp_path)
            image = load_image(temp_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        face_result = detect_faces_opencv(image)
        face_crop = crop_first_face(image, face_result)
        prediction = predict_face(face_crop)

        return {
            "filename": file.filename,
            "type": "image",
            "metadata": metadata,
            "face_detection": {
                "face_count": face_result["face_count"],
                "faces": face_result["faces"],
                "face_crop_generated": face_crop is not None,
            },
            **prediction,
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/predict-frame")
async def predict_frame(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded frame must be an image.")

    suffix = Path(file.filename).suffix if file.filename else ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        try:
            metadata = read_image_metadata(temp_path)
            image = load_image(temp_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        face_result = detect_faces_opencv(image)
        face_crop = crop_first_face(image, face_result)
        prediction = predict_face(face_crop)

        return {
            "filename": file.filename,
            "type": "frame",
            "metadata": metadata,
            "face_detection": {
                "face_count": face_result["face_count"],
                "faces": face_result["faces"],
                "face_crop_generated": face_crop is not None,
            },
            **prediction,
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video.")

    suffix = Path(file.filename).suffix if file.filename else ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_video_path = temp_file.name

    frames_dir = tempfile.mkdtemp(prefix="sampled_frames_")

    try:
        try:
            video_metadata = read_video_metadata(temp_video_path)
            sampled_frames = sample_video_frames(
                file_path=temp_video_path,
                output_dir=frames_dir,
                sample_every_n_frames=30,
                max_frames=10,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        analysis = analyze_sampled_video_frames(sampled_frames)
        final_prediction = aggregate_video_prediction(analysis["valid_predictions"])

        return {
            "filename": file.filename,
            "type": "video",
            "metadata": video_metadata,
            "sampled_frames_count": len(sampled_frames),
            "sampled_frame_names": [Path(frame).name for frame in sampled_frames],
            "face_detection_summary": {
                "sampled_frames_analyzed": analysis["sampled_frames_analyzed"],
                "frames_with_faces": analysis["frames_with_faces"],
                "total_faces_detected": analysis["total_faces_detected"],
            },
            "frame_results": analysis["frame_results"],
            **final_prediction,
        }
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)