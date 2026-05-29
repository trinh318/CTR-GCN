import os
import cv2
import numpy as np
from tqdm import tqdm

from yolov26_pose import YOLOv26Pose
from utils import (
    get_image_files,
    normalize_keypoints,
    save_npy,
    empty_keypoints,
    create_dir
)

from visualize import draw_keypoints


INPUT_ROOT = "../data/URFD"
OUTPUT_ROOT = "../data/URFD_pose"

CLASSES = ["adl", "fall"]

MODEL_PATH = "yolo26n-pose.pt"

SAVE_VIS = False


def process_sequence(
        sequence_dir,
        save_path,
        model
):
    image_files = get_image_files(sequence_dir)

    all_keypoints = []

    prev_keypoints = None

    for image_path in image_files:

        image = cv2.imread(image_path)

        h, w = image.shape[:2]

        keypoints = model.inference(image)

        # missing detection
        if keypoints is None:

            if prev_keypoints is not None:
                keypoints = prev_keypoints.copy()
            else:
                keypoints = empty_keypoints()

        # normalize x,y
        keypoints = normalize_keypoints(
            keypoints,
            width=w,
            height=h
        )

        prev_keypoints = keypoints.copy()

        all_keypoints.append(keypoints)

        # visualization
        if SAVE_VIS:

            vis = draw_keypoints(
                image,
                keypoints
            )

            cv2.imshow("Pose", vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    all_keypoints = np.array(
        all_keypoints,
        dtype=np.float32
    )

    save_npy(all_keypoints, save_path)

    print(f"Saved: {save_path}")
    print(f"Shape: {all_keypoints.shape}")


def main():

    create_dir(OUTPUT_ROOT)

    model = YOLOv26Pose(
        model_path=MODEL_PATH,
        device="cuda"
    )

    for cls_name in CLASSES:

        input_cls_dir = os.path.join(
            INPUT_ROOT,
            cls_name
        )

        output_cls_dir = os.path.join(
            OUTPUT_ROOT,
            cls_name
        )

        create_dir(output_cls_dir)

        sequences = sorted(os.listdir(input_cls_dir))

        for seq_name in tqdm(sequences):

            seq_dir = os.path.join(
                input_cls_dir,
                seq_name
            )

            save_path = os.path.join(
                output_cls_dir,
                f"{seq_name}.npy"
            )

            process_sequence(
                seq_dir,
                save_path,
                model
            )


if __name__ == "__main__":
    main()