import mediapipe
import numpy as np
from face import Face
from typing import List

class FaceLandmark:
    def __init__(self):
        self.detector = mediapipe.solutions.face_mesh.FaceMesh(
            max_num_faces=10,
            static_image_mode=False)

    def detect_face(self, image: np.ndarray) -> List[Face]:
        h, w = image.shape[:2]
        predictions = self.detector.process(image[:, :, ::-1])
        detected = []
        if predictions.multi_face_landmarks:
            for prediction in predictions.multi_face_landmarks:
                pts = np.array([(pt.x * w, pt.y * h)
                                for pt in prediction.landmark],
                               dtype=np.float64)
                bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                bbox = np.round(bbox).astype(np.int32)
                detected.append(Face(bbox, pts))
        return detected