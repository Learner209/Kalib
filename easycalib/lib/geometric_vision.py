import cv2
import numpy as np
from pyrr import Quaternion
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import itertools
import torch
import itertools
from easycalib.utils.setup_logger import setup_logger

logger = setup_logger(__name__)


def convert_rvec_to_quaternion(rvec):
    """Convert rvec (which is log quaternion) to quaternion"""
    theta = np.sqrt(
        rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]
    )  # in radians
    raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

    # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
    quaternion = Quaternion.from_axis_rotation(raxis, theta)
    quaternion.normalize()
    return quaternion


def zhangzhengyou_caliberate(
        canonical_points,
        projections,
        camera_K,
        image_size,
        dist_coeffs=np.array([]),
        return_rvec_tvec=False,
        flags=cv2.CALIB_USE_EXTRINSIC_GUESS,
):
    # try:
    canonical_points = canonical_points.astype(np.float32)[np.newaxis, :, :]
    projections = projections.astype(np.float32)[np.newaxis, :, np.newaxis, :]
    retval, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        canonical_points,
        projections,
        imageSize=image_size,
        cameraMatrix=camera_K,
        distCoeffs=dist_coeffs,
        flags=flags,
    )

    # logger.info(f"rvecs are {rvecs} and tvecs are {tvecs}.")
    tvec = tvecs[0]
    rvec = rvecs[0]
    translation = tvec[:, 0]
    quaternion = convert_rvec_to_quaternion(rvec[:, 0])

    total_error = 0
    for i in range(len(canonical_points)):
        points_pixel_repro, _ = cv2.projectPoints(
            canonical_points[i], rvecs[i], tvecs[i], camera_K, dist_coeffs
        )
        error = cv2.norm(projections[i], points_pixel_repro, cv2.NORM_L2) / len(
            points_pixel_repro
        )
        total_error += error
    reprojection_err = total_error / len(canonical_points)
    # logger.info(f"The reprojection error is {reprojection_err}.")

    if return_rvec_tvec:
        return retval, rvec, tvec, K, reprojection_err
    else:
        return retval, translation, quaternion, K, reprojection_err


def solve_pnp(
        canonical_points,
        projections,
        camera_K,
        refinement=True,
        dist_coeffs=np.array([]),
        method=cv2.SOLVEPNP_ITERATIVE,
        return_rvec_tvec=False,
):
    try:
        pnp_retval, rvec, tvec = cv2.solvePnP(
            canonical_points,
            projections,
            camera_K,
            distCoeffs=np.zeros((4, 1)),
            flags=method,
        )
        if refinement:
            pnp_retval, rvec, tvec = cv2.solvePnP(
                canonical_points,
                projections,
                camera_K,
                np.zeros((4, 1)),
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=True,
                rvec=rvec,
                tvec=tvec,
            )
        translation = tvec[:, 0]
        quaternion = convert_rvec_to_quaternion(rvec[:, 0])

        # Reproject the object points to the image plane
        reprojected_points, _ = cv2.projectPoints(
            canonical_points, rvec, tvec, camera_K, dist_coeffs
        )
        # Compute the reprojection error
        reprojection_error = projections - reprojected_points.squeeze()
        reprojection_error = np.mean(np.linalg.norm(reprojection_error, axis=1))
    except Exception as e:
        logger.info(f"Exception thrown when solving PNP: {e}.")
        pnp_retval = False
        translation = None
        quaternion = None
        reprojection_error = None
    if return_rvec_tvec:
        return pnp_retval, rvec, tvec, reprojection_error
    else:
        return pnp_retval, translation, quaternion, reprojection_error


def solve_pnp_ransac(
        canonical_points,
        projections,
        camera_K,
        method=cv2.SOLVEPNP_EPNP,
        inlier_thresh_px=5.0,  # this is the threshold for each point to be considered an inlier
        dist_coeffs=np.array([]),
        return_rvec_tvec=False,
):
    # Use cv2's PNP solver
    try:
        pnp_retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            canonical_points,
            projections,
            camera_K,
            np.zeros((4, 1)),
            reprojectionError=inlier_thresh_px,
            flags=method,
        )

        translation = tvec[:, 0]
        quaternion = convert_rvec_to_quaternion(rvec[:, 0])

        # Reproject the object points to the image plane
        reprojected_points, _ = cv2.projectPoints(
            canonical_points, rvec, tvec, camera_K, dist_coeffs
        )
        # Compute the reprojection error
        reprojection_error = projections - reprojected_points.squeeze()
        reprojection_error = np.mean(np.linalg.norm(reprojection_error, axis=1))
        # print(f"The reprojection error is {reprojection_error}")

    except:
        logger.info(f"Exception thrown when solving PNP RANSAC: {e}.")
        pnp_retval = False
        translation = None
        quaternion = None
        reprojection_error = None
        inliers = None

    if return_rvec_tvec:
        return pnp_retval, rvec, tvec, reprojection_error
    else:
        return pnp_retval, translation, quaternion, reprojection_error


def rtvec_to_matrix(translation, quaternion):
    transform = np.eye(4)
    transform[:3, :3] = quaternion.matrix33.tolist()
    transform[:3, -1] = translation

    return transform


def calculate_rotation_axis_and_angle(R1, R2):
    """
    Calculate the rotation axis and angle (in radians) for the rotation from R1 to R2.

    Parameters:
    R1 (numpy.ndarray): The first rotation matrix.
    R2 (numpy.ndarray): The second rotation matrix.

    Returns:
    tuple: A tuple containing the rotation axis and the rotation angle in radians.
    """
    # Calculate the rotation matrix from R1 to R2
    R_combined = np.dot(np.linalg.inv(R1), R2)

    # Convert the rotation matrix to axis-angle representation
    rotation = R.from_matrix(R_combined)
    rot_vec = rotation.as_rotvec()

    # Calculate the angle (magnitude of the rotation vector)
    angle = np.linalg.norm(rot_vec)

    # Normalize the rotation vector to get the rotation axis
    axis = rot_vec / angle if angle != 0 else rot_vec

    return axis, angle


def calculate_similarity(predicted_local_to_world_matrix, gt_local_to_world_matrix):
    """
    Calculate the similarity (translation and rotation error) between two transformations.

    Parameters:
    translation (numpy.ndarray): The translation vector of shape [3, 1].
    quaternion (numpy.ndarray): The quaternion of shape [4, 1], order x, y, z, w.
    local_to_world_matrix (numpy.ndarray): The transformation matrix of shape [4, 4].
    gt_local_to_world_matrix (numpy.ndarray): Ground truth local-to-world matrix.

    Returns:
    tuple: A tuple containing the translation error and the rotation error in radians.
    """
    # Calculate translation error
    translation_error = (
        predicted_local_to_world_matrix[:3, 3] - gt_local_to_world_matrix[:3, 3]
    )
    # Calculate rotation error
    _, rotation_error = calculate_rotation_axis_and_angle(
        gt_local_to_world_matrix[:3, :3], predicted_local_to_world_matrix[:3, :3]
    )

    return translation_error, rotation_error


def unity_left_handed_to_right_coord(left_handed_matrix, conversion_mat_1=None, conversion_mat_2=None):
    """
    Convert Unity's left-handed coordinate system to a right-handed 4x4 matrix.
    Unity's convention: x-right, y-up, z-forward (left-handed).
    Target convention: x-right, y-forward, z-up (right-handed).

    Args:
        left_handed_matrix (np.array): A size 4x4 transformation matrix.

    Returns:
        np.array: The converted right-handed 4x4 matrix.
    """
    S_l = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    S_w = np.array([
        [-1, 0, 0], [0, 0, 1], [0, -1, 0]
    ])

    new_matrix = np.copy(left_handed_matrix)

    rot_mat = S_l @ left_handed_matrix[:3, :3] @ np.linalg.inv(S_w)
    new_matrix[:3, :3] = rot_mat

    trans_vec = S_l @ left_handed_matrix[:3, 3]
    new_matrix[:3, 3] = trans_vec

    return new_matrix


def left_handed_coord2right_handed_coord(left_handed_matrix):
    aux_multiply_matrix = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )

    # Check if input is a NumPy array
    if isinstance(left_handed_matrix, np.ndarray):
        return np.matmul(aux_multiply_matrix, np.matmul(left_handed_matrix, aux_multiply_matrix))

    # Check if input is a tensor (PyTorch)
    elif isinstance(left_handed_matrix, (torch.Tensor)):
        aux_multiply_matrix_tensor = (
            torch.tensor(aux_multiply_matrix)
        )
        return left_handed_matrix @ aux_multiply_matrix_tensor.float().to(
            left_handed_matrix.device
        )

    # Check if input is a list
    elif isinstance(left_handed_matrix, list):
        left_handed_matrix_np = np.array(left_handed_matrix)
        result = np.matmul(left_handed_matrix_np, aux_multiply_matrix)
        return result.tolist()

    else:
        raise TypeError("Input must be a NumPy array, tensor, or list")


def render_debug_coordinates(transformations, gt_transform):
    """
    Render multiple debug 3D coordinates using Open3D.

    Args:
    transformations (list of numpy.ndarray): A list of 4x4 transformation matrices.

    This function creates a coordinate frame for each transformation matrix and renders them in a 3D scene.
    """
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2.5, origin=[0, 0, 0]
    )
    coordinate_frame.transform(gt_transform)
    vis.add_geometry(coordinate_frame)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )
    vis.add_geometry(coordinate_frame)

    for idx, trans in enumerate(transformations):
        # if idx == 1:
        # break
        # origin = trans[:3, 3]
        # Create a mesh coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        # Apply the transformation to the coordinate frame
        coordinate_frame.transform(trans)
        # Add the coordinate frame to the visualizer
        vis.add_geometry(coordinate_frame)

    # Render the scene
    vis.run()
    vis.destroy_window()


def generate_transform_matrices():
    axes = [0, 1, 2]  # Corresponds to x, y, z
    permutations = list(itertools.permutations(axes))
    negations = list(itertools.product([-1, 1], repeat=3))

    matrices = []
    for perm_ind, perm in enumerate(permutations):
        for neg_ind, neg in enumerate(negations):
            matrix = np.zeros((4, 4))
            for i, (axis, sign) in enumerate(zip(perm, neg)):
                matrix[i, axis] = sign
            # print("%d, %d" % (perm_ind, neg_ind,))
            matrix[3, 3] = 1  # Set the homogeneous coordinate to 1
            matrices.append(matrix)

    return matrices


def generate_transform_matrices():
    axes = [0, 1, 2]  # Corresponds to x, y, z
    permutations = list(itertools.permutations(axes))
    negations = list(itertools.product([-1, 1], repeat=3))

    matrices = []
    for perm in permutations:
        for neg in negations:
            matrix = np.zeros((3, 3))
            for i, (axis, sign) in enumerate(zip(perm, neg)):
                matrix[i, axis] = sign
            matrices.append(matrix)

    return matrices


if __name__ == "__main__":
    # Generate all matrices
    all_matrices = generate_transform_matrices()

    # Print the number of matrices and some examples
    print(f"Generated {len(all_matrices)} matrices.")

    print(all_matrices[46])
    print(all_matrices[10])
    # for i, matrix in enumerate(all_matrices):  # Print first 5 matrices as examples
    #     print(f"Matrix {i + 1}:\n{matrix}\n")
