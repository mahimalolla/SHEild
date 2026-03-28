from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class SimpleModel(nn.Module):
    """
    Very simple baseline classifier for hackathon pipeline testing.

    Input shape after preprocessing:
    [batch_size, 3, 224, 224]

    Output:
    logits for 2 classes:
    - index 0 -> real
    - index 1 -> fake
    """

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(224 * 224 * 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.fc(x)


class ModelInferenceService:
    """
    Loads the model once and provides face prediction.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleModel().to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        self.class_names = ["real", "fake"]

    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Convert BGR face crop to model-ready tensor.
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Face image is empty or invalid.")

        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
        return tensor

    def predict_face(self, face_image: np.ndarray | None) -> Dict[str, Any]:
        """
        Predict whether the given face crop is real or fake.
        """
        if face_image is None:
            return {
                "label": "no_face_detected",
                "confidence": 0.0,
                "reason_flags": ["no_face"],
            }

        try:
            tensor = self.preprocess_face(face_image)
        except ValueError:
            return {
                "label": "no_face_detected",
                "confidence": 0.0,
                "reason_flags": ["invalid_face_crop"],
            }

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        real_prob = float(probabilities[0])
        fake_prob = float(probabilities[1])

        if fake_prob > real_prob:
            label = "fake"
            confidence = round(fake_prob, 2)
            flags = ["facial_artifacts_detected"]
        else:
            label = "real"
            confidence = round(real_prob, 2)
            flags = ["natural_face_pattern"]

        return {
            "label": label,
            "confidence": confidence,
            "reason_flags": flags,
            "raw_scores": {
                "real": round(real_prob, 4),
                "fake": round(fake_prob, 4),
            },
        }


# Global singleton so model loads once
inference_service = ModelInferenceService()


def predict_face(face_image: np.ndarray | None) -> Dict[str, Any]:
    """
    Convenience wrapper for importing directly in app.py
    """
    return inference_service.predict_face(face_image)