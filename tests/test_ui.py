"""Tests for the Streamlit demo UI (ui/app.py).

Streamlit's AppTest (streamlit.testing.v1) can re-run the script and inspect its rendered
elements, but has no simulated st.file_uploader interaction as of this streamlit version - so the
upload -> POST /predict -> draw boxes path isn't exercisable end-to-end without a browser. Two
things are exercised for real instead: draw_detections() directly, since it's a pure function with
no Streamlit dependency at all, and a full script run with no API reachable, which is the one path
guaranteed to happen on every real page load before a user has uploaded anything.
"""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "ui"))
from app import draw_detections  # noqa: E402


def make_detection(class_name="dent", confidence=0.9, box=(10, 10, 50, 50)):
    x1, y1, x2, y2 = box
    return {
        "class_name": class_name,
        "confidence": confidence,
        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


def test_draw_detections_returns_same_size_image_without_mutating_input():
    image = Image.new("RGB", (100, 100), color="white")
    annotated = draw_detections(image, [make_detection()])
    assert annotated.size == image.size
    assert annotated is not image
    # original untouched - draw_detections must copy, not draw in place
    assert list(image.getdata())[0] == (255, 255, 255)


def test_draw_detections_handles_empty_list():
    image = Image.new("RGB", (64, 64), color="white")
    annotated = draw_detections(image, [])
    assert annotated.size == image.size


def test_draw_detections_draws_something_for_each_detection():
    image = Image.new("RGB", (200, 200), color="white")
    detections = [
        make_detection("dent", 0.8, (10, 10, 60, 60)),
        make_detection("glass shatter", 0.6, (100, 100, 150, 150)),
    ]
    annotated = draw_detections(image, detections)
    # a box was actually drawn: pixels inside each bbox border are no longer pure white
    assert list(annotated.getdata()) != list(image.getdata())


def test_draw_detections_unknown_class_falls_back_to_default_color():
    image = Image.new("RGB", (64, 64), color="white")
    # must not KeyError on a class name outside CLASS_COLORS
    draw_detections(image, [make_detection("unlisted_class")])


def test_app_runs_without_exception_when_api_unreachable(monkeypatch):
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    # port 1 refuses the connection immediately - fails fast, no DNS/timeout wait
    monkeypatch.setenv("API_URL", "http://127.0.0.1:1")
    at = AppTest.from_file(str(Path(__file__).parent.parent / "ui" / "app.py"))
    at.run(timeout=15)

    assert not at.exception
    assert at.title[0].value == "🚗 Vehicle Damage Detector"
    assert any("Couldn't reach the API" in w.value for w in at.warning)
    del streamlit  # imported only to skip cleanly if streamlit isn't installed
