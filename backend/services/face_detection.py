from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp


mp_face_detection = mp.solutions.face_detection


def detect_face_in_image(
    image_bgr,
    min_detection_confidence: float = 0.5,
) -> Dict[str, Any]:
    """
    Detect faces in a BGR image using MediaPipe.
    Returns bounding boxes and face count.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = image_bgr.shape

    detections_output: List[Dict[str, Any]] = []

    with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=min_detection_confidence,
    ) as face_detection:
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                x_min = max(0, int(bbox.xmin * width))
                y_min = max(0, int(bbox.ymin * height))
                box_width = int(bbox.width * width)
                box_height = int(bbox.height * height)

                x_max = min(width, x_min + box_width)
                y_max = min(height, y_min + box_height)

                score = detection.score[0] if detection.score else 0.0

                detections_output.append(
                    {
                        "confidence": round(float(score), 3),
                        "bbox": {
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max,
                            "width": max(0, x_max - x_min),
                            "height": max(0, y_max - y_min),
                        },
                    }
                )

    return {
        "face_count": len(detections_output),
        "faces": detections_output,
    }


def crop_first_face(image_bgr, detections: Dict[str, Any]):
    """
    Crop the first detected face from the image.
    Returns cropped face image or None.
    """
    if detections["face_count"] == 0:
        return None

    bbox = detections["faces"][0]["bbox"]
    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    x_max = bbox["x_max"]
    y_max = bbox["y_max"]

    if x_max <= x_min or y_max <= y_min:
        return None

    face_crop = image_bgr[y_min:y_max, x_min:x_max]

    if face_crop.size == 0:
        return None

    return face_crop


def save_face_crop(face_crop, output_path: str) -> bool:
    """
    Save cropped face image to disk.
    """
    return cv2.imwrite(output_path, face_crop)


def detect_faces_in_frame_files(frame_paths: List[str]) -> Dict[str, Any]:
    """
    Detect faces across sampled video frames.
    """
    analyzed_frames: List[Dict[str, Any]] = []
    frames_with_faces = 0
    total_faces = 0

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            analyzed_frames.append(
                {
                    "frame_name": Path(frame_path).name,
                    "face_count": 0,
                    "faces": [],
                    "readable": False,
                }
            )
            continue

        detection_result = detect_face_in_image(frame)

        if detection_result["face_count"] > 0:
            frames_with_faces += 1
            total_faces += detection_result["face_count"]

        analyzed_frames.append(
            {
                "frame_name": Path(frame_path).name,
                "face_count": detection_result["face_count"],
                "faces": detection_result["faces"],
                "readable": True,
            }
        )

    return {
        "sampled_frames_analyzed": len(frame_paths),
        "frames_with_faces": frames_with_faces,
        "total_faces_detected": total_faces,
        "frame_results": analyzed_frames,
    }