from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


class HeuristicInferenceService:
    """
    Hackathon-friendly inference service.

    Instead of using an untrained neural net, this uses visual heuristics
    from the detected face crop to produce a stable and explainable score.
    """

    def __init__(self) -> None:
        pass

    def _variance_of_laplacian(self, gray: np.ndarray) -> float:
        """
        Blur metric:
        lower value -> blurrier image
        """
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _brightness_score(self, gray: np.ndarray) -> float:
        """
        Mean brightness of the face crop.
        """
        return float(np.mean(gray))

    def _contrast_score(self, gray: np.ndarray) -> float:
        """
        Standard deviation of grayscale intensities.
        """
        return float(np.std(gray))

    def _edge_density(self, gray: np.ndarray) -> float:
        """
        Ratio of edge pixels in the face crop.
        """
        edges = cv2.Canny(gray, 100, 200)
        return float(np.count_nonzero(edges) / edges.size)

    def analyze_face(self, face_image: np.ndarray) -> Dict[str, float]:
        if face_image is None or face_image.size == 0:
            raise ValueError("Face image is empty or invalid.")

        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        blur_score = self._variance_of_laplacian(gray)
        brightness = self._brightness_score(gray)
        contrast = self._contrast_score(gray)
        edge_density = self._edge_density(gray)

        return {
            "blur_score": round(blur_score, 4),
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "edge_density": round(edge_density, 4),
        }

    def predict_face(self, face_image: np.ndarray | None) -> Dict[str, Any]:
        if face_image is None:
            return {
                "label": "no_face_detected",
                "confidence": 0.0,
                "reason_flags": ["no_face"],
            }

        try:
            metrics = self.analyze_face(face_image)
        except ValueError:
            return {
                "label": "no_face_detected",
                "confidence": 0.0,
                "reason_flags": ["invalid_face_crop"],
            }

        blur_score = metrics["blur_score"]
        brightness = metrics["brightness"]
        contrast = metrics["contrast"]
        edge_density = metrics["edge_density"]

        fake_risk = 0.0
        flags = []

        # Very blurry faces often look suspicious / low quality
        if blur_score < 80:
            fake_risk += 0.3
            flags.append("high_blur_detected")
        elif blur_score < 150:
            fake_risk += 0.15

        # Overly dark or bright images can be low confidence / suspicious
        if brightness < 60 or brightness > 200:
            fake_risk += 0.2
            flags.append("abnormal_brightness")

        # Very low contrast can indicate smoothed/generated appearance
        if contrast < 30:
            fake_risk += 0.25
            flags.append("low_contrast_face")

        # Strange lack of edges/detail can look synthetic or overly smoothed
        if edge_density < 0.03:
            fake_risk += 0.25
            flags.append("low_facial_detail")

        fake_risk = min(fake_risk, 0.95)
        real_score = 1.0 - fake_risk

        if fake_risk >= 0.5:
            return {
                "label": "fake",
                "confidence": round(fake_risk, 2),
                "reason_flags": flags if flags else ["facial_artifacts_detected"],
                "raw_scores": {
                    "real": round(real_score, 4),
                    "fake": round(fake_risk, 4),
                },
                "analysis_metrics": metrics,
            }

        return {
            "label": "real",
            "confidence": round(real_score, 2),
            "reason_flags": ["natural_face_pattern"] if not flags else flags,
            "raw_scores": {
                "real": round(real_score, 4),
                "fake": round(fake_risk, 4),
            },
            "analysis_metrics": metrics,
        }


inference_service = HeuristicInferenceService()


def predict_face(face_image: np.ndarray | None) -> Dict[str, Any]:
    return inference_service.predict_face(face_image)