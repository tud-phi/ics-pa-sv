import cv2
import numpy as onp
from typing import Dict


def render_robot_cv2(
    rp: Dict[str, float], x_eb: onp.ndarray, x: onp.ndarray, height: int, width: int
) -> onp.ndarray:
    """
    Render the robot with OpenCV.
    Args:
        rp: Robot parameters dictionary
        x_eb: Elbow joint position of shape (2,)
        x: End effector position of shape (2,)
        height: Height of the image
        width: Width of the image
    Returns:
        img: Rendered image of shape (height, width, 3)
    """
    # initialize upscaled image
    img_up = 255 * onp.ones((height * 10, width * 10, 3), dtype=onp.uint8)
    height_up, width_up, _ = img_up.shape

    scale = ((height_up // 2) * 0.90) / (rp["l1"] + rp["l2"])

    # draw first link
    img_up = cv2.line(
        img_up,
        (width_up // 2, height_up // 2),
        (
            width_up // 2 + int(x_eb[0] * scale),
            height_up // 2 - int(x_eb[1] * scale),
        ),
        (0, 0, 255),
        15,
    )
    # draw second link
    img_up = cv2.line(
        img_up,
        (
            width_up // 2 + int(x_eb[0] * scale),
            height_up // 2 - int(x_eb[1] * scale),
        ),
        (width_up // 2 + int(x[0] * scale), height_up // 2 - int(x[1] * scale)),
        (255, 0, 0),
        15,
    )

    # downsample image
    img = cv2.resize(img_up, (height, width), interpolation=cv2.INTER_AREA)

    return img
