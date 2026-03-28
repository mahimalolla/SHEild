from typing import Dict
import random


def predict_dummy() -> Dict:
    label = random.choice(["real", "fake"])
    confidence = round(random.uniform(0.75, 0.98), 2)

    if label == "fake":
        flags = ["face_inconsistency", "visual_artifacts"]
    else:
        flags = ["natural_face_pattern"]

    return {
        "label": label,
        "confidence": confidence,
        "reason_flags": flags,
    }