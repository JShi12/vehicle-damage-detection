FROM python:3.11-slim

# ultralytics imports opencv-python (cv2) as a hard transitive dependency, and cv2's non-headless
# build needs X11/GL shared libraries that python:3.11-slim doesn't ship - confirmed by hitting
# "ImportError: libxcb.so.1: cannot open shared object file" without this. Standard fix for
# ultralytics-on-slim-Debian, not a headless-opencv swap (that fights pip's dependency resolution
# since ultralytics itself requires plain opencv-python).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app

COPY pyproject.toml ./
COPY src/ ./src/

# CPU-only torch build - Render (and most free hosts) have no GPU, and this keeps the image much
# smaller than the default CUDA-enabled wheels ultralytics would otherwise pull in.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install --no-cache-dir ".[serve]"

# Points at the rolling "champion" release tag (its assets get replaced on each promotion, see
# promote.yml), not a fixed version - this default should never need editing again after this.
ENV CHAMPION_WEIGHTS_URL="https://github.com/JShi12/vehicle-damage-detection/releases/download/champion/best.pt" \
    PORT=8000

USER appuser
EXPOSE 8000

CMD ["sh", "-c", "uvicorn cardd.serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
