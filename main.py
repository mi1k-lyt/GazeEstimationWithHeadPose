import cv2
import numpy as np
from face_landmark import FaceLandmark, Face
from face_model import FaceModelMediaPipe
from face import FacePartsName, HeadPoseNormalizer
from mymodel import MyModel
from typing import Tuple


# 生成人脸检测器
facelandmark = FaceLandmark()
# 生成人脸关键点检测器
face_model = FaceModelMediaPipe()
# 透视变化
headpose_normalizer = HeadPoseNormalizer()
# 生成深度学习模型
gaze_estimation_model = MyModel()

# 图片处理过程
def process_image(image):
    # 相机参数（来自MPIIGAZE数据集）
    camera_mat = np.array([640., 0., 320.,
                           0., 640., 240.,
                           0., 0., 1.]).reshape(3, 3)
    # 相机畸变（来自MPIIGAZE数据集）
    dist = np.array([0., 0., 0., 0., 0.]).reshape(-1, 1)
    # 对图像进行反畸变操作
    undistorted = cv2.undistort(
        image, camera_mat, dist
    )
    # 区分展示的画面和实际处理画面
    show_image = image.copy()
    # 人脸检测
    faces = facelandmark.detect_face(undistorted)
    for face in faces:
        # 人脸关键点检测
        face_model.estimate_head_pose(face)
        face_model.compute_3d_pose(face)
        face_model.compute_face_eye_centers(face)
        # 视线估计
        estimate_gaze(undistorted, face)
        # 画人脸区域
        cv2.rectangle(show_image,
                      tuple(face.bbox[0]),
                      tuple(face.bbox[1]),
                      (0, 255, 0),
                      2)
        # 画视线方向
        for key in [FacePartsName.REYE, FacePartsName.LEYE]:
            eye = getattr(face, key.name.lower())
            p0 = eye.center
            p1 = eye.center + 0.05 * eye.gaze_vector
            assert p0.shape == p1.shape == (3,)
            p3d = np.vstack([p0, p1])
            p2d = project_p(p3d)
            pt0 = convert_pt(p2d[0])
            pt1 = convert_pt(p2d[1])
            cv2.line(show_image, pt0, pt1, (0, 0, 255), 2, cv2.LINE_AA)
            pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
            print('pitch:', pitch, '---yaw:', yaw)

    return show_image



def convert_pt(point: np.ndarray) -> Tuple[int, int]:
    return tuple(np.round(point).astype(np.int).tolist())

def project_p(p3d: np.ndarray) -> np.ndarray:
    # 相机参数（来自MPIIGAZE数据集）
    camera_mat = np.array([640., 0., 320.,
                           0., 640., 240.,
                           0., 0., 1.]).reshape(3, 3)
    # 相机畸变（来自MPIIGAZE数据集）
    dist = np.array([0., 0., 0., 0., 0.]).reshape(-1, 1)
    assert p3d.shape[1] == 3
    rvec = np.zeros(3, dtype=np.float)
    tvec = np.zeros(3, dtype=np.float)
    points2d, _ = cv2.projectPoints(p3d, rvec, tvec,
                                    camera_mat, dist)

    return points2d.reshape(-1, 2)

# 视线估计过程
def estimate_gaze(image: np.ndarray, face: Face) -> None:

    for key in [FacePartsName.REYE, FacePartsName.LEYE]:
        eye = getattr(face, key.name.lower())
        headpose_normalizer.normalize(image, eye)

    gaze_estimation_model._run_model(face)

def wait_key()->bool:
    key = cv2.waitKey(1) & 0xff
    if key in {27, ord('q')}:
        return True
    return False



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    stop = wait_key()
    if stop:
        break
    ok, frame = cap.read()
    if not ok:
        break
    show_image = process_image(frame)
    cv2.imshow('By liyuntao', show_image)

cap.release()



