# sam3d

`sam3d` 提供 SAM 3D Body 和 SAM 3D Objects 的统一 Python 接口，目标是下载好模型后可直接调用。

## 1. 准备模型文件

### SAM 3D Body

示例路径约定：

- `sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt`
- `sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt`

### SAM 3D Objects

示例路径约定：

- `sam-3d-objects/checkpoints/hf/pipeline.yaml`

你也可以传入自定义绝对路径（`SAM3DConfig`）。

## 2. 安装

```bash
cd /Users/sun/Projects/embody/sam3d
pip install -e .
```

## 3. 使用

```python
import numpy as np
from sam3d import SAM3D

sam3d = SAM3D.from_defaults(
    workspace_dir="/Users/sun/Projects/embody/sam3d",
    device="cuda",
)

# 4.1 人体重建
body_result = sam3d.predict(
    task="body",
    image="/path/to/person.jpg",
)
print(len(body_result["instances"]))

# 4.2 物体重建
mask = np.zeros((1024, 1024), dtype=bool)  # 你的物体 mask
obj_result = sam3d.predict(
    task="objects",
    image="/path/to/object.png",
    mask=mask,
    seed=42,
)
obj_result["gaussian_splat"].save_ply("object.ply")
```

## 4. 接口设计说明

- 统一入口：`SAM3D.predict(task=...)`
- 任务拆分：`predict_body(...)` 和 `predict_objects(...)`
- 懒加载：首次调用对应任务时才初始化模型，减少启动负担
- 与官方实现兼容：底层直接复用 Meta 仓库里的推理入口

## 5. 第三方源码

当前包已 vendoring 下列源码（仅包含统一接口所需部分）：

- `sam_3d_body`（来自 `facebookresearch/sam-3d-body`）
- `sam3d_objects`（来自 `facebookresearch/sam-3d-objects`）

## 6. HTTP API 服务模式

安装服务依赖：

```bash
pip install "sam3d[service]"
```

启动服务：

```bash
sam3d-http --workspace-dir ~/.spwm/models --device cuda --host 0.0.0.0 --port 8000
```

接口：

- `GET /health`
- `GET /info`
- `POST /predict/body`
- `POST /predict/objects`

## 7. 模型下载脚本（HF / ModelScope）

安装下载依赖：

```bash
pip install "sam3d[download]"
```

默认（body/objects 从 ModelScope，moge 从 HuggingFace）：

```bash
sam3d-download --workspace-dir ~/.spwm/models
```

可分别指定来源：

```bash
sam3d-download \
  --workspace-dir ~/.spwm/models \
  --body-source modelscope \
  --objects-source modelscope \
  --moge-source huggingface
```

可覆盖模型 ID（例如全部走 HuggingFace）：

```bash
sam3d-download \
  --workspace-dir ~/.spwm/models \
  --body-source huggingface \
  --objects-source huggingface \
  --body-model-id <hf-body-model-id> \
  --objects-model-id <hf-objects-model-id>
```
