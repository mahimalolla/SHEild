from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from services.dummy_predictor import predict_dummy
from services.face_detection import (
    crop_first_face,
    detect_face_in_image,
    detect_faces_in_frame_files,
    save_face_crop,
)
from services.media_processing import (
    load_image,
    read_image_metadata,
    read_video_metadata,
    sample_video_frames,
)

app = FastAPI(title="SheShield Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "message": "Backend is running"}


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    suffix = Path(file.filename).suffix if file.filename else ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    temp_face_path = None

    try:
        try:
            metadata = read_image_metadata(temp_path)
            image = load_image(temp_path)
            face_result = detect_face_in_image(image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        face_crop = crop_first_face(image, face_result)

        if face_crop is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as face_file:
                temp_face_path = face_file.name
            save_face_crop(face_crop, temp_face_path)

        result = predict_dummy()

        return {
            "filename": file.filename,
            "type": "image",
            "metadata": metadata,
            "face_detection": {
                "face_count": face_result["face_count"],
                "faces": face_result["faces"],
                "face_crop_generated": face_crop is not None,
            },
            **result,
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if temp_face_path and os.path.exists(temp_face_path):
            os.remove(temp_face_path)


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
            face_result = detect_face_in_image(image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = predict_dummy()

        return {
            "filename": file.filename,
            "type": "frame",
            "metadata": metadata,
            "face_detection": face_result,
            **result,
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
            frame_face_results = detect_faces_in_frame_files(sampled_frames)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = predict_dummy()

        return {
            "filename": file.filename,
            "type": "video",
            "metadata": video_metadata,
            "sampled_frames_count": len(sampled_frames),
            "sampled_frame_names": [Path(frame).name for frame in sampled_frames],
            "face_detection_summary": {
                "sampled_frames_analyzed": frame_face_results["sampled_frames_analyzed"],
                "frames_with_faces": frame_face_results["frames_with_faces"],
                "total_faces_detected": frame_face_results["total_faces_detected"],
            },
            "frame_results": frame_face_results["frame_results"],
            **result,
        }
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)