from ultralytics import YOLO
import numpy as np


class YOLOv26Pose:

    def __init__(self, model_path="yolo26n-pose.pt", device="cuda"):
        self.model = YOLO(model_path)
        self.device = device

    def inference(self, image):
        """
        image: BGR image from OpenCV

        return:
            keypoints [17, 3]
            (x, y, conf)
        """

        results = self.model.predict(
            source=image,
            device=self.device,
            verbose=False
        )

        result = results[0]

        # no detection
        if result.keypoints is None:
            return None

        kpts_xy = result.keypoints.xy.cpu().numpy()
        kpts_conf = result.keypoints.conf.cpu().numpy()

        # no person
        if len(kpts_xy) == 0:
            return None

        # choose highest confidence person
        person_xy = kpts_xy[0]
        person_conf = kpts_conf[0]

        keypoints = np.concatenate(
            [
                person_xy,
                person_conf[:, None]
            ],
            axis=1
        )

        return keypoints.astype(np.float32)