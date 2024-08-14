import time
import cv2
import sys
import os
import argparse
from os.path import dirname, abspath
import json
import numpy as np
import os.path as osp
import pickle

sys.path.append(abspath(dirname(dirname(__file__))))


from easycalib.utils.frankaAPI.franka_manip import MoveGroupPythonInterface
import imageio

from datetime import datetime
from scipy.spatial.transform import Rotation as R

from easycalib.utils.camera_caliberation.realsense_api import RealSenseAPI

from easyhec_client_wrapper import SAVE_TAG as EASYHEC_SAVE_TAG
from easycalib_demo_client import SAVE_TAG as EASYCALIB_SAVE_TAG

from easycalib.utils.setup_logger import setup_logger
logger = setup_logger(__name__)


def caliberate_camera_with_chessboards(imgpath):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    rows = 5
    columns = 7
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    chessboard_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    ret, corners = cv2.findChessboardCorners(gray, (columns, rows), chessboard_flags)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (columns, rows), corners2, ret)
        cv2.imshow("img", img)
        cv2.imwrite(
            f"chessboard_corners_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", img
        )
        cv2.waitKey(1500)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    camera = {}

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    for variable in ["ret", "mtx", "dist", "rvecs", "tvecs"]:
        camera[variable] = eval(variable)

    with open("camera.json", "w") as f:
        json.dump(camera, f, indent=4, cls=NumpyEncoder)

    cv2.destroyAllWindows()


def caliberate_with_single_aruco_marker(type="DICT_5X5_1000"):
    drawaxes = True

    # logger.info("[INFO] detecting '{}' tags...".format(type))
    arucoDict = create_dict(name=type, offset=0)
    arucoParams = cv2.aruco.DetectorParameters()

    logger.info("[INFO] starting video stream...")

    time.sleep(2.0)

    dist = np.zeros((1, 5))
    _rvec = None
    _tvec = None

    frame, K = RealSenseAPI.capture_data()
    logger.info("The intrinsics is \n{}".format(K))
    (corners, ids, rejected) = cv2.aruco.detectMarkers(
        frame, arucoDict, parameters=arucoParams
    )
    # logger.info(corners, ids)
    logger.info("That belons to dict of {}".format(key))
    # markerLength = 0.022445345 ## inner
    markerLength = 0.0318  # padding
    # markerLength = 0.0395123234 ## outer
    if len(corners) > 0:
        if drawaxes:
            for i in range(0, len(ids)):
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], markerLength, K, dist
                )
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, markerLength)

                if ids[i] == 29:
                    _rvec = rvec
                    _tvec = tvec
        ids = ids.flatten()
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0)

    cv2.destroyAllWindows()
    # tvec and rvec is the marker coordiante in the camera's coordinate!!!!!!!! YES its TRUE
    logger.info("The translation is {} and the rotation is {}".format(_tvec, _rvec))
    return _tvec, _rvec


def create_dict(name, offset):
    dict_id = name if isinstance(name, int) else getattr(cv2.aruco, f"{name}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    aruco_dict.bytesList = aruco_dict.bytesList[offset:]
    return aruco_dict


def caliberate_with_charuco_board(type="DICT_5X5_1000"):
    refine_detected_markers = True
    aruco_detector_Params = cv2.aruco.DetectorParameters()
    aruco_refine_Params = cv2.aruco.RefineParameters()
    # create aruco predefiend dictionary from multical repository.
    aruco_marker_offset = 0
    arucoDict = create_dict(name=type, offset=aruco_marker_offset)
    aruco_detector = cv2.aruco.ArucoDetector(
        arucoDict, aruco_detector_Params, aruco_refine_Params
    )

    marker_num_x = 5
    marker_num_y = 5
    marker_separation_x = 0.45
    marker_separation_y = 0.55
    marker_size = (marker_num_x, marker_num_y)
    marker_separation = (marker_separation_x, marker_separation_y)

    dist = np.zeros((1, 5))
    rvec = None
    tvec = None
    # board = cv2.aruco.GridBoard(size=marker_size, markerLength=marker_length, markerSeparation=marker_separation, dictionary=arucoDict)
    # FROM MULTICAL REPO.
    board = cv2.aruco.CharucoBoard((10, 10), 0.040, 0.032, arucoDict)
    board.setLegacyPattern(True)

    frame, K = RealSenseAPI.capture_data()
    (detected_corners, detected_ids, rejected_corners) = aruco_detector.detectMarkers(
        frame
    )
    if refine_detected_markers:
        (detected_corners, detected_ids, rejected_corners, _) = (
            aruco_detector.refineDetectedMarkers(
                image=frame,
                board=board,
                detectedCorners=detected_corners,
                detectedIds=detected_ids,
                rejectedCorners=rejected_corners,
                cameraMatrix=K,
                distCoeffs=dist,
            )
        )
    detected_marker_result = cv2.aruco.drawDetectedMarkers(
        frame, detected_corners, detected_ids
    )
    cv2.imshow("detected_marker_result", detected_marker_result)
    """
	void cv::aruco::Board::matchImagePoints 	( 	InputArrayOfArrays  	detectedCorners,
		InputArray  	detectedIds,
		OutputArray  	objPoints,
		OutputArray  	imgPoints
	) 		const
	Python:
		cv.aruco.Board.matchImagePoints(	detectedCorners, detectedIds[, objPoints[, imgPoints]]	) -> 	objPoints, imgPoints
	reference: https://docs.opencv.org/4.x/d4/db2/classcv_1_1aruco_1_1Board.html
	"""
    """
		cv.aruco.estimatePoseCharucoBoard(	charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, rvec, tvec[, useExtrinsicGuess]	) -> 	retval, rvec, tvec
		instead, use estimatePoseCharucoBoard python bindings
	"""
    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        detected_corners, detected_ids, frame, board, cameraMatrix=K, distCoeffs=dist
    )

    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        K,
        dist,
        np.empty(1),
        np.empty(1),
        useExtrinsicGuess=False,
    )

    detected_charuco = cv2.aruco.drawDetectedCornersCharuco(
        frame, charuco_corners, charuco_ids, (255, 0, 0)
    )

    cv2.imshow("detected_charcuco", detected_charuco)

    painted_frame_axes = cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.1)
    cv2.imshow("predicted_rvec_tvec", painted_frame_axes)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    return rvec, tvec


def camera_pose_from_one_aruco_marker():
    """
    cv::Mat cameraMatrix, distCoeffs;
    // You can read camera parameters from tutorial_camera_params.yml
    readCameraParameters(cameraParamsFilename, cameraMatrix, distCoeffs); // This function is implemented in aruco_samples_utility.hpp
    std::vector<cv::Vec3d> rvecs, tvecs;
    // Set coordinate system
    cv::Mat objPoints(4, 1, CV_32FC3);
    ...
    // Calculate pose for each marker
    for (int i = 0; i < nMarkers; i++) {
    solvePnP(objPoints, corners.at(i), cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
    }
    """
    logger.info()


def save_realsense_img(save_dir):
    rgb, K = RealSenseAPI.capture_data()
    index = len([os.listdir(save_dir)])
    imageio.imwrite(os.path.join(save_dir, f"{index:06d}.png"), rgb)


def point(extrinsics):
    """
    the extrinsic is the camera2robot transforms matrix
    """
    # tvec, rvec = aruco_marker()
    # logger.info(f"Detecting single arucoMarker instance using aruco_marker(), tvec:{tvec}, rvec:{rvec}")
    rvec, tvec = caliberate_with_charuco_board(type="DICT_5X5_1000")
    tvec, rvec = caliberate_with_single_aruco_marker(type="DICT_5X5_1000")
    logger.info(
        f"Detecting arucoMarkerBoard using aruco_marker_board(), tvec:{tvec},tvec's shape: {tvec.shape}, rvec:{rvec},rvec's shape: {rvec.shape}"
    )
    # tvec: aruco_marker's translatino coordinate in the camera's system, concatenate it with np.eye(1) => (4,1) camera coorcdinate.
    camera_coor = np.concatenate([tvec, np.eye(1)], axis=0)

    robot_coor = np.dot(extrinsics, camera_coor)
    assert robot_coor.shape == (4, 1)
    demo = MoveGroupPythonInterface()
    logger.info(
        "In the robot frame: the robot coordinate is {} {} {}".format(
            robot_coor[0][0], robot_coor[1][0], robot_coor[2][0]
        )
    )
    # robot_coor[2][0] = max(0.004, robot_coor[2][0])
    demo.approach_goal(x=robot_coor[0][0], y=robot_coor[1][0], z=robot_coor[2][0])
    time.sleep(0.5)
    demo.go_to_rest_pose()


def RT2matrix(
    tvec=None,
    rvec=None,
):
    assert len(rvec) == 4 and len(tvec) == 3
    rot = R.from_quat((rvec[0], rvec[1], rvec[2], rvec[3]))
    rot = rot.as_matrix()
    tvec = np.array(tvec).reshape(3, 1)
    logger.debug("The rot is \n{}".format(rot))
    logger.debug("The tvec is \n{}".format(tvec))
    ext = np.concatenate([rot, tvec], axis=1)
    ext = np.concatenate([ext, np.array([[0, 0, 0, 1]])], axis=0)
    logger.info("The extrinscs is \n{}".format(ext))
    return ext


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=int, default="/mnt/public/datasets/DROID/sim/Robotflow/main/gaussian07_27_19_12_48", help="path where the caliberation result reside."
    )
    args = parser.parse_args()

    kalib_caliberate_save_path = osp.join(args.path, EASYCALIB_SAVE_TAG, "%s_outputs" % EASYCALIB_SAVE_TAG, "pnp_inference_res.pkl")
    kalib_caliberate_res = pickle.load(open(kalib_caliberate_save_path, "rb"))
    easyhec_caliberate_res_save_path = osp.join(args.path, EASYHEC_SAVE_TAG, "%s_outputs" % EASYHEC_SAVE_TAG, "caliberate_res.pkl")
    easyhec_caliberate_res = pickle.load(open(easyhec_caliberate_res_save_path, "rb"))
    avg_trans_err, avg_rot_err, avg_reproj_err, pred_camera_to_robot_matrix = kalib_caliberate_res["avg_trans_err"], kalib_caliberate_res["avg_rot_err"], kalib_caliberate_res["avg_reprojection_err"], kalib_caliberate_res["pnp_transform_predicted_mats"]
    _, _, _, history_Tc_c2b = easyhec_caliberate_res["avg_trans_err"], easyhec_caliberate_res["avg_rot_err"], easyhec_caliberate_res["avg_trans_err"], easyhec_caliberate_res["avg_reprojection_err"], easyhec_caliberate_res["history_Tc_c2b"]
    pred_mat = pred_camera_to_robot_matrix[-1]
    for extrinsic in [pred_mat]:
        logger.info(extrinsic)
        point(extrinsics=extrinsic)


"""
reference: https://docs.opencv.org/4.9.0/df/d4a/tutorial_charuco_detection.html(Detect CharucoBoardWithCaliberatoinPose)
"""
