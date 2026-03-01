from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _vendored_meta_root() -> Path:
    return (Path(__file__).resolve().parent / "third_party" / "meta").resolve()


def add_vendored_meta_to_syspath() -> None:
    root = _vendored_meta_root()
    ensure_exists(root / "sam_3d_body", "Vendored SAM 3D Body source")
    ensure_exists(root / "sam3d_objects", "Vendored SAM 3D Objects source")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def load_body_symbols():
    add_vendored_meta_to_syspath()
    module = importlib.import_module("sam_3d_body")
    load_sam_3d_body = getattr(module, "load_sam_3d_body")
    estimator_cls = getattr(module, "SAM3DBodyEstimator")
    return load_sam_3d_body, estimator_cls


class ObjectsInference:
    def __init__(self, config_path: str, compile: bool = False):
        add_vendored_meta_to_syspath()
        os.environ.setdefault("LIDRA_SKIP_INIT", "true")
        os.environ.setdefault("HYDRA_FULL_ERROR", "1")
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        config = OmegaConf.load(config_path)
        config.rendering_engine = "pytorch3d"
        config.compile_model = compile
        config.workspace_dir = str(Path(config_path).resolve().parent)
        self._pipeline = instantiate(config)

    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        return np.asarray(Image.open(path), dtype=np.uint8)

    def merge_mask_to_rgba(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask_u8 = mask.astype(np.uint8) * 255
        if mask_u8.ndim == 2:
            mask_u8 = mask_u8[..., None]
        h, w = image.shape[:2]
        mh, mw = mask_u8.shape[:2]
        if (mh, mw) != (h, w):
            from PIL import Image as _PILImage
            mask_2d = mask_u8.squeeze(-1) if mask_u8.ndim == 3 else mask_u8
            mask_2d = _PILImage.fromarray(mask_2d).resize((w, h), _PILImage.NEAREST)
            mask_u8 = np.asarray(mask_2d)[..., None]
        return np.concatenate([image[..., :3], mask_u8], axis=-1)

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        seed: int | None = None,
        pointmap: np.ndarray | None = None,
    ) -> dict:
        rgba_image = self.merge_mask_to_rgba(image, mask)
        return self._pipeline.run(
            rgba_image,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=pointmap,
        )


def load_objects_symbols():
    add_vendored_meta_to_syspath()
    return ObjectsInference, ObjectsInference.load_image
