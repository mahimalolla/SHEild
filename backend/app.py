from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from services.dummy_predictor import predict_dummy
from services.media_processing import (
    read_image_metadata,
    read_video_metadata,
    sample_video_frames,
)

app = FastAPI(title="SheShield Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later to frontend origin
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

    try:
        try:
            metadata = read_image_metadata(temp_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = predict_dummy()

        return {
            "filename": file.filename,
            "type": "image",
            "metadata": metadata,
            **result,
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
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = predict_dummy()

        return {
            "filename": file.filename,
            "type": "frame",
            "metadata": metadata,
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
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = predict_dummy()

        return {
            "filename": file.filename,
            "type": "video",
            "metadata": video_metadata,
            "sampled_frames_count": len(sampled_frames),
            "sampled_frame_names": [Path(frame).name for frame in sampled_frames],
            **result,
        }
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)