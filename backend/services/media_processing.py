from __future__ import annotations

import cv2
from pathlib import Path
from typing import Any, Dict, List


def read_image_metadata(file_path: str) -> Dict[str, Any]:
    image = cv2.imread(file_path)

    if image is None:
        raise ValueError("Could not read image file.")

    height, width, channels = image.shape

    return {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
    }


def load_image(file_path: str):
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError("Could not read image file.")
    return image


def read_video_metadata(file_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_seconds = (frame_count / fps) if fps > 0 else 0.0

    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": round(float(fps), 2),
        "frame_count": frame_count,
        "duration_seconds": round(float(duration_seconds), 2),
    }


def sample_video_frames(
    file_path: str,
    output_dir: str,
    sample_every_n_frames: int = 30,
    max_frames: int = 10,
) -> List[str]:
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file for frame sampling.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_frames: List[str] = []
    frame_index = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % sample_every_n_frames == 0:
            frame_filename = output_path / f"frame_{frame_index}.jpg"
            write_ok = cv2.imwrite(str(frame_filename), frame)

            if write_ok:
                saved_frames.append(str(frame_filename))
                saved_count += 1

            if saved_count >= max_frames:
                break

        frame_index += 1

    cap.release()
    return saved_frames