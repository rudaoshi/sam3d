# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_HOME=/workspace/models/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        build-essential wget curl git ca-certificates \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python venv ----------
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ---------- Python dependencies ----------
ENV PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" \
    PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

COPY sam3d/docker/requirements.txt /tmp/requirements.txt
COPY sam3d/docker/requirements.dev.txt /tmp/requirements.dev.txt
COPY sam3d/docker/requirements.p3d.txt /tmp/requirements.p3d.txt
COPY sam3d/docker/requirements.inference.txt /tmp/requirements.inference.txt

RUN python -m ensurepip --upgrade || true
RUN python -m pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt 
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.dev.txt 
# RUN --mount=type=cache,target=/root/.cache/pip \
RUN python -m pip install --no-build-isolation -r /tmp/requirements.p3d.txt

ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html
    
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0+PTX"
RUN --mount=type=cache,target=/root/.cache/pip \
pip install --no-build-isolation -r /tmp/requirements.inference.txt

COPY sam3d/docker/patching/hydra /tmp/hydra_patch.py
RUN python /tmp/hydra_patch.py && rm /tmp/hydra_patch.py

# ---------- sam3d wrapper (HTTP service) ----------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install 'sam3d[service]'

# ---------- Cleanup ----------
RUN rm -rf /tmp/* /tmp/.[!.]* 2>/dev/null || true \
    && find /opt/venv -type d -name __pycache__ -print0 | xargs -0 rm -rf 2>/dev/null || true \
    && find /opt/venv -type f -name "*.pyc" -delete 2>/dev/null || true

# ---------- Models ----------
COPY models/ /workspace/models/

RUN sed -i \
    's|/home/sun/Projects/embody/models/sam-3d-objects|/workspace/models/sam-3d-objects|g' \
    /workspace/models/sam-3d-objects/checkpoints/hf/pipeline.yaml

# ---------- Runtime ----------
WORKDIR /workspace
EXPOSE 8000

ENTRYPOINT ["sam3d-http", "--workspace-dir", "/workspace/models", "--host", "0.0.0.0"]
CMD ["--port", "8000"]
