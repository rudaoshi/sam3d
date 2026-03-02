import numpy as np
from PIL import Image
from sam3d import SAM3D

sam3d = SAM3D.from_defaults(
    workspace_dir="/home/sun/Projects/embody/models",
    device="cuda",
)

# 4.1 人体重建
body_result = sam3d.predict(
    task="body",
    image="./image_body.png",
)
print(len(body_result["instances"]))

# 4.2 物体重建
image = np.asarray(Image.open("./image_objects.png"))
h, w = image.shape[:2]
mask = np.zeros((h, w), dtype=bool)
mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
obj_result = sam3d.predict(
    task="objects",
    image=image,
    mask=mask,
    seed=42,
)
obj_result["gaussian_splat"].save_ply("object.ply")