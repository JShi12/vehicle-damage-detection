from __future__ import annotations

from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox


class PredictResponse(BaseModel):
    detections: list[Detection]
    image_width: int
    image_height: int
    inference_ms: float
    model_source: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str
    git_sha: str | None
    source: str = "https://github.com/JShi12/vehicle-damage-detection"
