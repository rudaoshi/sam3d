from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Callable

Downloader = Callable[[str, str | None, Path | None], Path]
SourceName = str


def _require_modelscope() -> Downloader:
    try:
        from modelscope.hub.snapshot_download import snapshot_download  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: modelscope. Install with `pip install \"sam3d[download]\"`."
        ) from exc

    def _download(model_id: str, revision: str | None, cache_dir: Path | None) -> Path:
        kwargs: dict[str, str] = {"model_id": model_id}
        if revision:
            kwargs["revision"] = revision
        if cache_dir:
            kwargs["cache_dir"] = str(cache_dir)
        local = snapshot_download(**kwargs)
        return Path(local).resolve()

    return _download


def _require_huggingface() -> Downloader:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: huggingface_hub. Install with `pip install \"sam3d[download]\"`."
        ) from exc

    def _download(model_id: str, revision: str | None, cache_dir: Path | None) -> Path:
        kwargs: dict[str, object] = {
            "repo_id": model_id,
            "repo_type": "model",
            "local_dir_use_symlinks": False,
        }
        if revision:
            kwargs["revision"] = revision
        if cache_dir:
            kwargs["cache_dir"] = str(cache_dir)
        local = snapshot_download(**kwargs)
        return Path(local).resolve()

    return _download


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _find_first(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def _resolve_objects_payload(src: Path) -> Path:
    preferred = src / "checkpoints" / "hf"
    if (preferred / "pipeline.yaml").exists():
        return preferred
    direct = _find_first(src, "pipeline.yaml")
    if direct is None:
        raise RuntimeError(f"Could not find pipeline.yaml under {src}")
    return direct.parent


def deploy_body(
    download: Downloader,
    workspace_dir: Path,
    model_id: str,
    revision: str | None,
    cache_dir: Path | None,
) -> dict[str, str]:
    src = download(model_id, revision, cache_dir)
    ckpt = _find_first(src, "model.ckpt")
    cfg = _find_first(src, "model_config.yaml")
    mhr = _find_first(src, "mhr_model.pt")
    if not ckpt or not cfg or not mhr:
        raise RuntimeError(
            f"Body files incomplete under {src}. Need model.ckpt, model_config.yaml, mhr_model.pt."
        )

    dst = workspace_dir / "sam-3d-body" / "checkpoints" / "sam-3d-body-dinov3"
    assets = dst / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, dst / "model.ckpt")
    shutil.copy2(cfg, dst / "model_config.yaml")
    shutil.copy2(mhr, assets / "mhr_model.pt")

    return {
        "source": str(src),
        "checkpoint": str(dst / "model.ckpt"),
        "config": str(dst / "model_config.yaml"),
        "mhr": str(assets / "mhr_model.pt"),
    }


def deploy_objects(
    download: Downloader,
    workspace_dir: Path,
    model_id: str,
    revision: str | None,
    cache_dir: Path | None,
) -> dict[str, str]:
    src = download(model_id, revision, cache_dir)
    payload = _resolve_objects_payload(src)
    dst = workspace_dir / "sam-3d-objects" / "checkpoints" / "hf"
    _copy_tree(payload, dst)
    pipeline = dst / "pipeline.yaml"
    if not pipeline.exists():
        raise RuntimeError(f"Objects deploy succeeded but pipeline missing: {pipeline}")
    return {"source": str(src), "target_dir": str(dst), "pipeline": str(pipeline)}


def deploy_moge(
    download: Downloader,
    workspace_dir: Path,
    model_id: str,
    revision: str | None,
    cache_dir: Path | None,
) -> dict[str, str]:
    dst = workspace_dir / "sam-3d-objects" / "checkpoints" / "hf" / "moge-vitl"
    dst.mkdir(parents=True, exist_ok=True)
    src = download(model_id, revision, cache_dir)
    _copy_tree(src, dst)
    return {"source": str(src), "target_dir": str(dst)}


def patch_pipeline_moge_path(pipeline_path: Path, local_moge_dir: Path) -> None:
    text = pipeline_path.read_text(encoding="utf-8")
    new_line = f"    pretrained_model_name_or_path: {str(local_moge_dir)}"
    pattern = r"^(\s*pretrained_model_name_or_path:\s*).*$"
    new_text, count = re.subn(pattern, new_line, text, flags=re.MULTILINE)
    if count == 0:
        raise RuntimeError(f"pretrained_model_name_or_path not found in {pipeline_path}")
    pipeline_path.write_text(new_text, encoding="utf-8")


def _get_downloader(source: SourceName) -> Downloader:
    if source == "modelscope":
        return _require_modelscope()
    if source == "huggingface":
        return _require_huggingface()
    raise ValueError(f"Unsupported source: {source}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/deploy SAM3D model files into SAM3D workspace layout."
    )
    parser.add_argument("--workspace-dir", type=Path, required=True)
    parser.add_argument(
        "--body-source",
        choices=["modelscope", "huggingface"],
        default="modelscope",
    )
    parser.add_argument(
        "--objects-source",
        choices=["modelscope", "huggingface"],
        default="modelscope",
    )
    parser.add_argument(
        "--moge-source",
        choices=["modelscope", "huggingface"],
        default="huggingface",
    )
    parser.add_argument("--body-model-id", default="facebook/sam-3d-body-dinov3")
    parser.add_argument("--objects-model-id", default="facebook/sam-3d-objects")
    parser.add_argument("--moge-model-id", default="Ruicheng/moge-vitl")
    parser.add_argument("--body-revision", default=None)
    parser.add_argument("--objects-revision", default=None)
    parser.add_argument("--moge-revision", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--skip-body", action="store_true")
    parser.add_argument("--skip-objects", action="store_true")
    parser.add_argument("--skip-moge", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace_dir = args.workspace_dir.expanduser().resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] workspace_dir={workspace_dir}")

    if not args.skip_body:
        body_download = _get_downloader(args.body_source)
        print(f"[info] downloading body from {args.body_source}: {args.body_model_id}")
        body_info = deploy_body(
            download=body_download,
            workspace_dir=workspace_dir,
            model_id=args.body_model_id,
            revision=args.body_revision,
            cache_dir=args.cache_dir,
        )
        print("[ok] body deployed")
        for key, value in body_info.items():
            print(f"  - {key}: {value}")

    if not args.skip_objects:
        objects_download = _get_downloader(args.objects_source)
        print(f"[info] downloading objects from {args.objects_source}: {args.objects_model_id}")
        objects_info = deploy_objects(
            download=objects_download,
            workspace_dir=workspace_dir,
            model_id=args.objects_model_id,
            revision=args.objects_revision,
            cache_dir=args.cache_dir,
        )
        print("[ok] objects deployed")
        for key, value in objects_info.items():
            print(f"  - {key}: {value}")

    if not args.skip_moge:
        moge_download = _get_downloader(args.moge_source)
        print(f"[info] downloading moge from {args.moge_source}: {args.moge_model_id}")
        moge_info = deploy_moge(
            download=moge_download,
            workspace_dir=workspace_dir,
            model_id=args.moge_model_id,
            revision=args.moge_revision,
            cache_dir=args.cache_dir,
        )
        print("[ok] moge deployed")
        for key, value in moge_info.items():
            print(f"  - {key}: {value}")

        pipeline = workspace_dir / "sam-3d-objects" / "checkpoints" / "hf" / "pipeline.yaml"
        if pipeline.exists():
            patch_pipeline_moge_path(pipeline, Path(moge_info["target_dir"]))
            print(f"[ok] patched pipeline MoGe path: {pipeline}")
        else:
            print("[warn] pipeline.yaml not found; skipped MoGe path patch.")

    print("[done] deployment complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

