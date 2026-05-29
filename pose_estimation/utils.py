import os
import cv2
import numpy as np


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    return image


def get_image_files(folder):
    exts = (".png", ".jpg", ".jpeg")

    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(exts)
    ]

    return files


def normalize_keypoints(keypoints, width, height):
    """
    keypoints: [V, 3]
    return normalized [V, 3]
    """

    normalized = keypoints.copy()

    normalized[:, 0] = normalized[:, 0] / width
    normalized[:, 1] = normalized[:, 1] / height

    return normalized


def save_npy(data, save_path):
    create_dir(os.path.dirname(save_path))
    np.save(save_path, data)


def load_npy(path):
    return np.load(path)


def empty_keypoints(num_joints=17):
    return np.zeros((num_joints, 3), dtype=np.float32)
