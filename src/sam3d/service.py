from __future__ import annotations

import argparse
import base64
import logging
import tempfile
from io import BytesIO
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from .api import SAM3D

logger = logging.getLogger(__name__)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        data = {
            k: _to_jsonable(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
        if data:
            data["__type__"] = type(value).__name__
            return data
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Last-resort fallback for non-JSON-native third-party objects.
    return repr(value)


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


class OneTimeFileStore:
    def __init__(self, root_dir: Path, ttl_seconds: int = 30 * 60):
        self.root_dir = root_dir
        self.ttl_seconds = ttl_seconds
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def register(self, file_path: Path, media_type: str) -> str:
        self.cleanup_expired()
        file_id = uuid4().hex
        now = time.time()
        with self._lock:
            self._entries[file_id] = {
                "path": str(file_path),
                "media_type": media_type,
                "filename": file_path.name,
                "created_at": now,
            }
        return file_id

    def pop(self, file_id: str) -> dict[str, Any] | None:
        self.cleanup_expired()
        with self._lock:
            return self._entries.pop(file_id, None)

    def cleanup_expired(self) -> None:
        now = time.time()
        expired_entries: list[dict[str, Any]] = []
        with self._lock:
            expired_ids = [
                file_id
                for file_id, entry in self._entries.items()
                if now - float(entry["created_at"]) >= self.ttl_seconds
            ]
            for file_id in expired_ids:
                entry = self._entries.pop(file_id, None)
                if entry is not None:
                    expired_entries.append(entry)

        for entry in expired_entries:
            _cleanup_downloaded_file(entry["path"])


def _cleanup_downloaded_file(file_path: str) -> None:
    path = Path(file_path)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        logger.exception("Failed to remove downloaded file: %s", file_path)
    try:
        parent = path.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except Exception:
        logger.exception("Failed to cleanup download directory for: %s", file_path)


def create_app(
    workspace_dir: str,
    device: str = "cuda",
    compile_objects: bool = False,
):
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTask

    state = ServiceState(
        workspace_dir=workspace_dir,
        device=device,
        compile_objects=compile_objects,
    )
    file_store = OneTimeFileStore(
        Path(tempfile.gettempdir()) / "sam3d" / "http_exports",
        ttl_seconds=30 * 60,
    )
    cleanup_stop_event = threading.Event()
    cleanup_thread: threading.Thread | None = None

    def _cleanup_loop() -> None:
        # Periodically remove files that were never downloaded within TTL.
        while not cleanup_stop_event.wait(5 * 60):
            try:
                file_store.cleanup_expired()
            except Exception:
                logger.exception("Periodic cleanup failed.")

    app = FastAPI(title="sam3d-http", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.sam3d_state = state

    @app.on_event("startup")
    def _start_cleanup_worker() -> None:
        nonlocal cleanup_thread
        cleanup_thread = threading.Thread(
            target=_cleanup_loop,
            name="sam3d-file-cleanup",
            daemon=True,
        )
        cleanup_thread.start()

    @app.on_event("shutdown")
    def _stop_cleanup_worker() -> None:
        cleanup_stop_event.set()
        if cleanup_thread is not None and cleanup_thread.is_alive():
            cleanup_thread.join(timeout=1.0)

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
            logger.exception(
                "predict_body failed: inference_type=%s, use_mask=%s, has_bboxes=%s, has_masks=%s, has_cam_int=%s",
                req.inference_type,
                req.use_mask,
                req.bboxes is not None,
                req.masks is not None,
                req.cam_int is not None,
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": str(exc),
                    "type": type(exc).__name__,
                },
            ) from exc

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
            request_id = uuid4().hex
            request_dir = file_store.root_dir / request_id
            request_dir.mkdir(parents=True, exist_ok=True)

            files: list[dict[str, str]] = []
            gaussian = out.get("gaussian_splat")
            if gaussian is not None and hasattr(gaussian, "save_ply"):
                ply_path = request_dir / "object.ply"
                gaussian.save_ply(str(ply_path))
                ply_file_id = file_store.register(ply_path, "application/octet-stream")
                files.append(
                    {
                        "format": "ply",
                        "file_id": ply_file_id,
                        "filename": ply_path.name,
                        "download_url": f"/files/{ply_file_id}",
                    }
                )

            raw_outputs = out.get("raw")
            glb_obj = raw_outputs.get("glb") if isinstance(raw_outputs, dict) else None
            if glb_obj is not None and hasattr(glb_obj, "export"):
                glb_path = request_dir / "object.glb"
                glb_obj.export(str(glb_path))
                glb_file_id = file_store.register(glb_path, "model/gltf-binary")
                files.append(
                    {
                        "format": "glb",
                        "file_id": glb_file_id,
                        "filename": glb_path.name,
                        "download_url": f"/files/{glb_file_id}",
                    }
                )

            if not files:
                raise ValueError("No exportable files were produced for object prediction.")

            return {
                "task": "objects",
                "request_id": request_id,
                "files": files,
            }
        except Exception as exc:
            logger.exception(
                "predict_objects failed: has_mask=%s, seed=%s, has_pointmap=%s",
                req.mask is not None,
                req.seed,
                req.pointmap is not None,
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": str(exc),
                    "type": type(exc).__name__,
                },
            ) from exc

    @app.get("/files/{file_id}")
    def download_file(file_id: str):
        entry = file_store.pop(file_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="File not found or already downloaded.")

        file_path = Path(entry["path"])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File has expired.")

        return FileResponse(
            path=file_path,
            media_type=entry["media_type"],
            filename=entry["filename"],
            background=BackgroundTask(_cleanup_downloaded_file, str(file_path)),
        )

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
