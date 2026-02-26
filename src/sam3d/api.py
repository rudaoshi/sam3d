from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import SAM3DConfig
from .loaders import ensure_exists, load_body_symbols, load_objects_symbols


class SAM3D:
    """Unified interface for SAM 3D Body and SAM 3D Objects."""

    def __init__(self, config: SAM3DConfig):
        self.config = config
        self._body_estimator = None
        self._objects_inference = None
        self._objects_load_image = None

    @classmethod
    def from_defaults(
        cls,
        workspace_dir: str | Path,
        device: str = "cuda",
        compile_objects: bool = False,
    ) -> "SAM3D":
        config = SAM3DConfig.from_defaults(
            workspace_dir=workspace_dir,
            device=device,
            compile_objects=compile_objects,
        )
        return cls(config=config)

    def info(self) -> dict[str, Any]:
        return asdict(self.config)

    def _ensure_body(self) -> None:
        if self._body_estimator is not None:
            return

        ensure_exists(self.config.body_checkpoint_path, "SAM 3D Body checkpoint")
        ensure_exists(self.config.body_mhr_path, "SAM 3D Body MHR file")

        load_sam_3d_body, estimator_cls = load_body_symbols()
        model, model_cfg = load_sam_3d_body(
            checkpoint_path=str(self.config.body_checkpoint_path),
            device=self.config.device,
            mhr_path=str(self.config.body_mhr_path),
        )
        self._body_estimator = estimator_cls(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )

    def _ensure_objects(self) -> None:
        if self._objects_inference is not None:
            return

        ensure_exists(
            self.config.objects_pipeline_config_path,
            "SAM 3D Objects pipeline config",
        )

        inference_cls, load_image = load_objects_symbols()
        self._objects_inference = inference_cls(
            str(self.config.objects_pipeline_config_path),
            compile=self.config.compile_objects,
        )
        self._objects_load_image = load_image

    def predict_body(
        self,
        image: str | Path | np.ndarray,
        bboxes: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        cam_int: np.ndarray | None = None,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
    ) -> dict[str, Any]:
        self._ensure_body()
        outputs = self._body_estimator.process_one_image(
            str(image) if isinstance(image, Path) else image,
            bboxes=bboxes,
            masks=masks,
            cam_int=cam_int,
            bbox_thr=bbox_thr,
            nms_thr=nms_thr,
            use_mask=use_mask,
            inference_type=inference_type,
        )
        return {"task": "body", "instances": outputs}

    def predict_objects(
        self,
        image: str | Path | np.ndarray,
        mask: np.ndarray,
        seed: int | None = None,
        pointmap: np.ndarray | None = None,
    ) -> dict[str, Any]:
        self._ensure_objects()
        if isinstance(image, (str, Path)):
            image = self._objects_load_image(str(image))
        elif isinstance(image, Image.Image):
            image = np.asarray(image)

        outputs = self._objects_inference(image, mask, seed=seed, pointmap=pointmap)
        return {"task": "objects", "gaussian_splat": outputs.get("gs"), "raw": outputs}

    def predict(
        self,
        task: str,
        image: str | Path | np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        task_lower = task.lower()
        if task_lower == "body":
            return self.predict_body(image=image, **kwargs)
        if task_lower in {"object", "objects"}:
            return self.predict_objects(image=image, **kwargs)
        raise ValueError(f"Unsupported task={task}. Use 'body' or 'objects'.")
