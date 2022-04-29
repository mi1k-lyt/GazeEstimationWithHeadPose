import enum
from typing import Optional
from scipy.spatial.transform import Rotation
import numpy as np
import cv2


class FacePartsName(enum.Enum):
    FACE = enum.auto()
    REYE = enum.auto()
    LEYE = enum.auto()

class FaceParts:
    def __init__(self, name: FacePartsName):
        self.name = name
        self.center: Optional[np.ndarray] = None
        self.head_pose_rot: Optional[Rotation] = None
        self.normalizing_rot: Optional[Rotation] = None
        self.normalized_head_rot2d: Optional[np.ndarray] = None
        self.normalized_image: Optional[np.ndarray] = None

        self.normalized_gaze_angles: Optional[np.ndarray] = None
        self.normalized_gaze_vector: Optional[np.ndarray] = None
        self.gaze_vector: Optional[np.ndarray] = None

    @property
    def distance(self) -> float:
        return np.linalg.norm(self.center)

    def angle_to_vector(self) -> None:
        pitch, yaw = self.normalized_gaze_angles
        self.normalized_gaze_vector = -np.array([
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw)
        ])

    def denormalize_gaze_vector(self) -> None:
        normalizing_rot = self.normalizing_rot.as_matrix()
        self.gaze_vector = self.normalized_gaze_vector @ normalizing_rot

    @staticmethod
    def vector_to_angle(vector: np.ndarray) -> np.ndarray:
        assert vector.shape == (3, )
        x, y, z = vector
        pitch = np.arcsin(-y)
        yaw = np.arctan2(-x, -z)
        return np.array([pitch, yaw])

class Eye(FaceParts):
    pass

class Face(FaceParts):
    def __init__(self, bbox: np.ndarray, landmarks: np.ndarray):
        super().__init__(FacePartsName.FACE)
        self.bbox = bbox
        self.landmarks = landmarks

        self.reye: Eye = Eye(FacePartsName.REYE)
        self.leye: Eye = Eye(FacePartsName.LEYE)

        self.head_position: Optional[np.ndarray] = None
        self.model3d: Optional[np.ndarray] = None

    @staticmethod
    def change_coordinate_system(euler_angles: np.ndarray) -> np.ndarray:
        return euler_angles * np.array([-1, 1, -1])

def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)

class HeadPoseNormalizer:
    def __init__(self):

        self.normalized_distance = 0.6

    def normalize(self, image: np.ndarray, eye_or_face: FaceParts) -> None:
        eye_or_face.normalizing_rot = self._compute_normalizing_rotation(
            eye_or_face.center, eye_or_face.head_pose_rot)
        self._normalize_image(image, eye_or_face)
        self._normalize_head_pose(eye_or_face)

    def _normalize_image(self, image: np.ndarray,
                         eye_or_face: FaceParts) -> None:
        # 相机参数（来自MPIIGAZE数据集）
        camera_mat = np.array([640., 0., 320.,
                               0., 640., 240.,
                               0., 0., 1.]).reshape(3, 3)
        # 标准化相机参数（来自MPIIGAZE数据集）
        norm_camera_mat = np.array([960., 0., 30,
                                    0., 960., 18.,
                                    0., 0., 1.]).reshape(3, 3)
        camera_matrix_inv = np.linalg.inv(camera_mat)
        normalized_camera_matrix = norm_camera_mat

        scale = self._get_scale_matrix(eye_or_face.distance)
        conversion_matrix = scale @ eye_or_face.normalizing_rot.as_matrix()

        projection_matrix = normalized_camera_matrix @ conversion_matrix @ camera_matrix_inv

        normalized_image = cv2.warpPerspective(
            image, projection_matrix,
            (60, 36))

        if eye_or_face.name in {FacePartsName.REYE, FacePartsName.LEYE}:
            normalized_image = cv2.cvtColor(normalized_image,
                                            cv2.COLOR_BGR2GRAY)
            normalized_image = cv2.equalizeHist(normalized_image)
        eye_or_face.normalized_image = normalized_image

    @staticmethod
    def _normalize_head_pose(eye_or_face: FaceParts) -> None:
        normalized_head_rot = eye_or_face.head_pose_rot * eye_or_face.normalizing_rot
        euler_angles2d = normalized_head_rot.as_euler('XYZ')[:2]
        eye_or_face.normalized_head_rot2d = euler_angles2d * np.array([1, -1])

    @staticmethod
    def _compute_normalizing_rotation(center: np.ndarray,
                                      head_rot: Rotation) -> Rotation:
        z_axis = _normalize_vector(center.ravel())
        head_rot = head_rot.as_matrix()
        head_x_axis = head_rot[:, 0]
        y_axis = _normalize_vector(np.cross(z_axis, head_x_axis))
        x_axis = _normalize_vector(np.cross(y_axis, z_axis))
        return Rotation.from_matrix(np.vstack([x_axis, y_axis, z_axis]))

    def _get_scale_matrix(self, distance: float) -> np.ndarray:
        return np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, self.normalized_distance / distance],
        ],
                        dtype=np.float)

