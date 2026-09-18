"""Streamlit demo UI for the CarDD vehicle damage detector.

This page has no model of its own and never imports `cardd`, `ultralytics`, or torch - it is a
plain HTTP client of the real REST API (`src/cardd/serving/app.py`), exactly like `curl` or any
other consumer would be. Its only job is to make that API's `/predict` contract easy to try
visually (upload an image, see boxes drawn on it) instead of reading raw JSON in Swagger's
`/docs` UI.

Run the API first, then this, in two separate processes:

    uvicorn cardd.serving.app:app --reload          # terminal 1
    API_URL=http://127.0.0.1:8000 streamlit run ui/app.py   # terminal 2

Set API_URL to point elsewhere (e.g. the live Render deployment) instead of localhost.
"""
from __future__ import annotations

import io
import os

import requests
import streamlit as st
from PIL import Image, ImageDraw

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

# Render's free tier is CPU-throttled hard enough that a single /predict call has taken ~99s in
# practice (see README's Known limitations) - this timeout has to cover that, not just guard
# against a hang. Health checks get a much shorter budget since they don't run the model.
PREDICT_TIMEOUT_S = 120
HEALTH_TIMEOUT_S = 10

# One fixed colour per class so the same damage type always draws the same colour across images -
# matches CarDD's own six classes exactly (src/cardd/serving/schemas.py has no class list of its
# own; this only needs to be readable, not authoritative).
CLASS_COLORS = {
    "dent": "#e6194B",
    "scratch": "#f58231",
    "crack": "#ffe119",
    "glass shatter": "#4363d8",
    "lamp broken": "#911eb4",
    "tire flat": "#3cb44b",
}
DEFAULT_COLOR = "#42d4f4"


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Draws each detection's box + "class conf" label onto a copy of `image`. Pure function of
    the API's own response - draws exactly what /predict returned, nothing inferred client-side.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for det in detections:
        box = det["box"]
        xyxy = (box["x1"], box["y1"], box["x2"], box["y2"])
        color = CLASS_COLORS.get(det["class_name"], DEFAULT_COLOR)
        draw.rectangle(xyxy, outline=color, width=3)

        label = f"{det['class_name']} {det['confidence']:.2f}"
        text_bbox = draw.textbbox((0, 0), label)
        label_w, label_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        label_origin = (xyxy[0], max(0, xyxy[1] - label_h - 4))
        draw.rectangle(
            [label_origin, (label_origin[0] + label_w + 6, label_origin[1] + label_h + 4)],
            fill=color,
        )
        draw.text((label_origin[0] + 3, label_origin[1] + 1), label, fill="black")
    return annotated


st.set_page_config(page_title="CarDD Damage Detector", page_icon="🚗", layout="centered")

st.title("🚗 Vehicle Damage Detector")
st.caption(
    f"Demo client of the real REST API at `{API_URL}` "
    f"([Swagger docs]({API_URL}/docs)) - see the "
    "[repo README](https://github.com/JShi12/vehicle-damage-detection) for the full project. "
    "This page never touches the model directly; every result below is a live HTTP round trip."
)

with st.expander("API status"):
    try:
        health = requests.get(f"{API_URL}/health", timeout=HEALTH_TIMEOUT_S).json()
        st.json(health)
    except requests.RequestException as e:
        st.warning(
            f"Couldn't reach the API at {API_URL}: {e}. If this is the live Render deployment, "
            "it may just be waking up from an idle spin-down (free tier, ~1 min cold start)."
        )

uploaded = st.file_uploader("Upload a vehicle photo", type=["jpg", "jpeg", "png"])
conf_col, iou_col = st.columns(2)
conf = conf_col.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
iou = iou_col.slider("IoU threshold", 0.0, 1.0, 0.7, 0.05)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Detect damage", type="primary"):
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)
        with st.spinner(
            "Calling the API - can take up to ~90s on Render's free-tier CPU if it's been idle."
        ):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    files={"file": ("upload.jpg", buf, "image/jpeg")},
                    params={"conf": conf, "iou": iou},
                    timeout=PREDICT_TIMEOUT_S,
                )
                resp.raise_for_status()
            except requests.RequestException as e:
                st.error(f"Request to the API failed: {e}")
            else:
                result = resp.json()
                detections = result["detections"]
                st.success(
                    f"{len(detections)} detection(s) in {result['inference_ms']:.0f} ms "
                    f"(model: `{result['model_source']}`)"
                )
                st.image(
                    draw_detections(image, detections),
                    caption="Detections",
                    use_container_width=True,
                )
                if detections:
                    st.table(
                        [
                            {"class": d["class_name"], "confidence": round(d["confidence"], 3)}
                            for d in detections
                        ]
                    )
                with st.expander("Raw API response (JSON)"):
                    st.json(result)
