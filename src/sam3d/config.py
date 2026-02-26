from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SAM3DConfig:
    body_model_root: Path
    objects_model_root: Path
    body_checkpoint_path: Path
    body_mhr_path: Path
    objects_pipeline_config_path: Path
    device: str = "cuda"
    compile_objects: bool = False

    @classmethod
    def from_defaults(
        cls,
        workspace_dir: str | Path,
        body_model_root_name: str = "sam-3d-body",
        objects_model_root_name: str = "sam-3d-objects",
        body_checkpoint_rel: str = "checkpoints/sam-3d-body-dinov3/model.ckpt",
        body_mhr_rel: str = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
        objects_pipeline_config_rel: str = "checkpoints/hf/pipeline.yaml",
        device: str = "cuda",
        compile_objects: bool = False,
    ) -> "SAM3DConfig":
        workspace = Path(workspace_dir).expanduser().resolve()
        body_model_root = workspace / body_model_root_name
        objects_model_root = workspace / objects_model_root_name
        return cls(
            body_model_root=body_model_root,
            objects_model_root=objects_model_root,
            body_checkpoint_path=body_model_root / body_checkpoint_rel,
            body_mhr_path=body_model_root / body_mhr_rel,
            objects_pipeline_config_path=objects_model_root / objects_pipeline_config_rel,
            device=device,
            compile_objects=compile_objects,
        )
