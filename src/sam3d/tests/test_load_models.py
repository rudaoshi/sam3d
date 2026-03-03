import os
import numpy as np
from PIL import Image
from sam3d import SAM3D

def test_load_models(workspace_dir):
    sam3d = SAM3D.from_defaults(
        workspace_dir=workspace_dir,
        device="cuda",
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 4.1 人体重建
    image_body_path = os.path.join(current_dir, "image_body.png")
    body_result = sam3d.predict(
        task="body",
        image=image_body_path,
    )
    print(len(body_result["instances"]))

    # 4.2 物体重建
    image_objects_path = os.path.join(current_dir, "image_objects.png")
    image = np.asarray(Image.open(image_objects_path))
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    obj_result = sam3d.predict(
        task="objects",
        image=image,
        mask=mask,
        seed=42,
    )
    
    object_ply_path = os.path.join(current_dir, "object.ply")
    obj_result["gaussian_splat"].save_ply(object_ply_path)