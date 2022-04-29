from face import FacePartsName, Face
import torch
import numpy as np
import torchvision.transforms as T
from res_model import _load_model

class MyModel:
    def __init__(self):
        self.model = _load_model()
        self.transform = T.ToTensor()
    @torch.no_grad()
    def _run_model(self, face: Face) -> None:
        images = []
        head_poses = []
        for key in [FacePartsName.REYE, FacePartsName.LEYE]:
            eye = getattr(face, key.name.lower())
            image = eye.normalized_image
            normalized_head_pose = eye.normalized_head_rot2d
            if key == FacePartsName.REYE:
                image = image[:, ::-1].copy()
                normalized_head_pose *= np.array([1, -1])
            image = self.transform(image)
            images.append(image)
            head_poses.append(normalized_head_pose)
        images = torch.stack(images)
        head_poses = np.array(head_poses).astype(np.float32)
        head_poses = torch.from_numpy(head_poses)

        device = torch.device('cpu')
        images = images.to(device)
        head_poses = head_poses.to(device)
        predictions = self.model(images, head_poses)
        predictions = predictions.cpu().numpy()

        for i, key in enumerate([FacePartsName.REYE, FacePartsName.LEYE]):
            eye = getattr(face, key.name.lower())
            eye.normalized_gaze_angles = predictions[i]
            if key == FacePartsName.REYE:
                eye.normalized_gaze_angles *= np.array([1, -1])
            eye.angle_to_vector()
            eye.denormalize_gaze_vector()
