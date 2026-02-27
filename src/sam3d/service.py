from __future__ import annotations

import argparse
import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from .api import SAM3D


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _decode_image(image_path: str | None, image_base64: str | None) -> str | np.ndarray:
    if image_path:
        return str(Path(image_path).expanduser())
    if image_base64:
        raw = base64.b64decode(image_base64)
        image = Image.open(BytesIO(raw)).convert("RGB")
        return np.asarray(image)
    raise ValueError("Either image_path or image_base64 must be provided.")


class BasePredictRequest(BaseModel):
    image_path: str | None = None
    image_base64: str | None = None

    @model_validator(mode="after")
    def _check_image_source(self) -> "BasePredictRequest":
        if bool(self.image_path) == bool(self.image_base64):
            raise ValueError("Provide exactly one of image_path or image_base64.")
        return self


class BodyPredictRequest(BasePredictRequest):
    bboxes: list[list[float]] | None = None
    masks: list[list[list[float]]] | None = None
    cam_int: list[list[float]] | None = None
    bbox_thr: float = 0.5
    nms_thr: float = 0.3
    use_mask: bool = False
    inference_type: str = "full"


class ObjectsPredictRequest(BasePredictRequest):
    mask: list[list[float]]
    seed: int | None = None
    pointmap: list[list[list[float]]] | None = None


class ServiceState:
    def __init__(self, workspace_dir: str, device: str, compile_objects: bool):
        self.workspace_dir = workspace_dir
        self.device = device
        self.compile_objects = compile_objects
        self.client = SAM3D.from_defaults(
            workspace_dir=workspace_dir,
            device=device,
            compile_objects=compile_objects,
        )


def create_app(
    workspace_dir: str,
    device: str = "cuda",
    compile_objects: bool = False,
):
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    state = ServiceState(
        workspace_dir=workspace_dir,
        device=device,
        compile_objects=compile_objects,
    )
    app = FastAPI(title="sam3d-http", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.sam3d_state = state

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/info")
    def info() -> dict[str, Any]:
        return _to_jsonable(state.client.info())

    @app.post("/predict/body")
    def predict_body(req: BodyPredictRequest) -> dict[str, Any]:
        try:
            image = _decode_image(req.image_path, req.image_base64)
            bboxes = np.asarray(req.bboxes, dtype=np.float32) if req.bboxes is not None else None
            masks = np.asarray(req.masks, dtype=np.float32) if req.masks is not None else None
            cam_int = np.asarray(req.cam_int, dtype=np.float32) if req.cam_int is not None else None
            out = state.client.predict_body(
                image=image,
                bboxes=bboxes,
                masks=masks,
                cam_int=cam_int,
                bbox_thr=req.bbox_thr,
                nms_thr=req.nms_thr,
                use_mask=req.use_mask,
                inference_type=req.inference_type,
            )
            return _to_jsonable(out)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/predict/objects")
    def predict_objects(req: ObjectsPredictRequest) -> dict[str, Any]:
        try:
            image = _decode_image(req.image_path, req.image_base64)
            mask = np.asarray(req.mask, dtype=np.float32)
            pointmap = np.asarray(req.pointmap, dtype=np.float32) if req.pointmap is not None else None
            out = state.client.predict_objects(
                image=image,
                mask=mask,
                seed=req.seed,
                pointmap=pointmap,
            )
            return _to_jsonable(out)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sam3d HTTP API service.")
    parser.add_argument("--workspace-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--compile-objects", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import uvicorn

    app = create_app(
        workspace_dir=args.workspace_dir,
        device=args.device,
        compile_objects=args.compile_objects,
    )
    uvicorn.run(app, host=args.host, port=args.port)
