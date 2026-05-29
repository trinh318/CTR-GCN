import cv2
import numpy as np


COCO_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16)
]


def draw_keypoints(
        image,
        keypoints,
        conf_threshold=0.3
):
    """
    image: BGR image
    keypoints: [17, 3]
    """

    vis = image.copy()

    # draw joints
    for x, y, conf in keypoints:

        if conf < conf_threshold:
            continue

        cv2.circle(
            vis,
            (int(x), int(y)),
            4,
            (0, 255, 0),
            -1
        )

    # draw bones
    for i, j in COCO_SKELETON:

        if keypoints[i, 2] < conf_threshold:
            continue

        if keypoints[j, 2] < conf_threshold:
            continue

        x1, y1 = keypoints[i][:2]
        x2, y2 = keypoints[j][:2]

        cv2.line(
            vis,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 0, 0),
            2
        )

    return vis


def show_image(image, win_name="Pose"):
    cv2.imshow(win_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()