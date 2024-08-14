import os
import cv2
import numpy as np
from enum import Enum
from tqdm import tqdm
import pickle
from scipy.spatial.transform import Rotation as R
from .geometric_vision import solve_pnp, solve_pnp_ransac, zhangzhengyou_caliberate, rtvec_to_matrix, calculate_similarity, render_debug_coordinates, left_handed_coord2right_handed_coord
from easycalib.utils.setup_logger import setup_logger

logger = setup_logger(__name__)

CALIBERATION_METHODS = Enum("methods", ["PNP", "ZHANGZHENGYOU", "EASYHEC"])


def caliberate_camera(
		image_objs,
		json_objs,
		output_dir,
		num_manipulator_keypoints,
		keypoint_ids,
		caliberate_method=CALIBERATION_METHODS.PNP,
		sliding_window_step=20,
		verbose=True,
		var_x=1.0,
		var_y=1.0,
		pnp_refinement=False,
		pnp_flag=cv2.SOLVEPNP_ITERATIVE,
		use_pnp_ransac=False,
		render_open3d=False,
		return_sample_result=False,
		has_gt=False,
		zhangzhengyou_flags=cv2.CALIB_USE_INTRINSIC_GUESS,
		**kwargs,
):
	os.makedirs(output_dir, exist_ok=True)
	logger.info(f"Running PNP algorithm with chosen keypoints: {keypoint_ids}")

	found_dataset = []
	video_frames = []
	for img_obj, json_obj in tqdm(zip(image_objs, json_objs)):
		if img_obj.shape[-1] == 4:
			img_obj = img_obj[:, :, :3]
		video_frames.append(img_obj)
		json_obj = json_obj["objects"][0]
		kp_objs = np.asarray(json_obj["keypoints"])[np.asarray(keypoint_ids)]
		robot_data = {
			"local_to_world_matrix": json_obj["local_to_world_matrix"],
			"camera_intrinsics": json_obj["camera_intrinsics"],
			"img_obj": img_obj,
			"gripper_cartesian_position": np.asarray([kp_obj["location"] for kp_obj in kp_objs]),
			"inference_gripper_proj_loc": np.asarray([kp_obj["predicted_location"] for kp_obj in kp_objs]),
			"gt_gripper_proj_loc": np.asarray([kp_obj["projected_location"] for kp_obj in kp_objs]),
			"joint_positions": json_obj["joint_positions"],
		}

		found_dataset.append(robot_data)

	overall_length = len(image_objs)

	if has_gt:
		gt_world_to_local_matrix = np.array(found_dataset[0]["local_to_world_matrix"])
	else:
		gt_world_to_local_matrix = np.eye(4)

	camera_K = np.array(found_dataset[0]["camera_intrinsics"])
	H, W, C = found_dataset[0]["img_obj"].shape

	# PNP analysis
	pnp_attempts_successful = []
	poses_xyzxyzw = []
	pnp_trans_errors = []
	pnp_rot_errors = []
	pnp_reprojection_errs = []
	pnp_transform_predicted_mats = []

	for ind in range(0, overall_length - sliding_window_step + 1):
		# logger.info(f"Processing frame {ind} to {ind + sliding_window_step - 1}.")
		pnp_inference_frame_inds = range(ind, ind + sliding_window_step)
		kp_projs_est_pnp = [
			found_dataset[datum_idx]["inference_gripper_proj_loc"]
			for datum_idx in pnp_inference_frame_inds
		]
		kp_projs_est_pnp = np.vstack(kp_projs_est_pnp).astype(np.float64)
		kp_projs_est_pnp[:, 0] += np.random.normal(
			0, var_x, kp_projs_est_pnp[:, 0].shape
		)
		kp_projs_est_pnp[:, 1] += np.random.normal(
			0, var_y, kp_projs_est_pnp[:, 1].shape
		)

		kp_projs_gt_pnp = [
			found_dataset[datum_idx]["gt_gripper_proj_loc"]
			for datum_idx in pnp_inference_frame_inds
		]
		kp_projs_gt_pnp = np.array(kp_projs_gt_pnp).astype(np.float64)

		kp_pos_gt_pnp = [
			found_dataset[datum_idx]["gripper_cartesian_position"]
			for datum_idx in pnp_inference_frame_inds
		]
		kp_pos_gt_pnp = np.vstack(kp_pos_gt_pnp).astype(np.float64)
		# logger.info(
		#     f"kp_projs_est_pnp: {kp_projs_est_pnp}, kp_pos_gt_pnp: {kp_pos_gt_pnp}, camera_K: {camera_K}"
		# )

		if caliberate_method == CALIBERATION_METHODS.PNP:
			# logger.info(f"Inferencing camera pose using PNP inference from opencv.")
			if not use_pnp_ransac:
				# logger.info(f"Inferencing camera pose using PNP inference from opencv.")
				ret_val, translation, quaternion, reprojection_err = (
					solve_pnp(
						kp_pos_gt_pnp,
						kp_projs_est_pnp,
						camera_K.copy(),
						refinement=pnp_refinement,
						method=pnp_flag,
					)
				)  # ! num_of_joints x 3, num_of_joints x 2, 3 x 3
			else:
				ret_val, translation, quaternion, reprojection_err = (
					solve_pnp_ransac(
						kp_pos_gt_pnp,
						kp_projs_est_pnp,
						camera_K.copy(),
						method=pnp_flag,
					)
				)
				# ! translation: (3, ), quaternion: (4, ).
			pnp_attempts_successful.append(ret_val)
		elif caliberate_method == CALIBERATION_METHODS.ZHANGZHENGYOU:
			ret_val, translation, quaternion, predicted_K, reprojection_err = (
				zhangzhengyou_caliberate(
					kp_pos_gt_pnp,
					kp_projs_est_pnp,
					camera_K.copy(),
					image_size=(H, W),
					flags=zhangzhengyou_flags,
				)
			)
			logger.info(
				f"The predicted K is \n {predicted_K}, while the camera K g.t is \n {camera_K}"
			)
		elif caliberate_method == CALIBERATION_METHODS.EASYHEC:
			Tc_c2b = np.array(kwargs["Tc_c2b"])
			logger.info(f"Caliberating method: EasyHeC, caliberated result: {Tc_c2b}")
			ret_val = True
			translation = Tc_c2b[:3, 3]
			rotation = R.from_matrix(Tc_c2b[:3, :3])
			quaternion = rotation.as_quat()
			reproject_err = None
		else:
			raise ValueError("Invalid caliberation method.")

		trans_err = -999.99
		rot_err = -999.99

		if ret_val:
			poses_xyzxyzw.append(translation.tolist() + quaternion.tolist())

			predicted_world_to_local_matrix = rtvec_to_matrix(
				translation, quaternion
			)
			if has_gt:
				trans_err, rot_err = calculate_similarity(
					predicted_world_to_local_matrix, gt_world_to_local_matrix
				)

				if verbose:
					logger.info(
						f"The gt_transforms matrix is \n {gt_world_to_local_matrix} \n the predicted matrix is {predicted_world_to_local_matrix} \n"
					)
					logger.info(f"the trans_error is {trans_err}, and the rot_err is {rot_err}")

			if verbose:
				logger.info(f"the pred_Tc_c2b is {predicted_world_to_local_matrix}")
			pnp_transform_predicted_mats.append(predicted_world_to_local_matrix)

		else:
			poses_xyzxyzw.append([-999.99] * num_manipulator_keypoints)

		pnp_trans_errors.append(trans_err)
		pnp_rot_errors.append(rot_err)
		pnp_reprojection_errs.append(reprojection_err)

	if render_open3d and has_gt:
		render_debug_coordinates(
			pnp_transform_predicted_mats, gt_world_to_local_matrix
		)

	pnp_trans_errors = np.vstack(pnp_trans_errors)
	pnp_rot_errors = np.vstack(pnp_rot_errors)
	pnp_reprojection_errs = np.vstack(pnp_reprojection_errs)

	average_trans_err = np.mean(pnp_trans_errors, axis=0)
	average_rot_err = np.mean(pnp_rot_errors, axis=0)
	average_reprojection_err = np.mean(pnp_reprojection_errs)
	pnp_transform_predicted_mats = np.stack(pnp_transform_predicted_mats)

	save_obj = {
		"avg_trans_err": average_trans_err.tolist(),
		"avg_rot_err": average_rot_err.item(),
		"avg_reprojection_err": average_reprojection_err.item(),
		"pnp_transform_predicted_mats": pnp_transform_predicted_mats.tolist(),
		"gt_local_to_world_matrix": gt_world_to_local_matrix.tolist(), 
		"camera_K": camera_K.tolist(),
		"joint_positions": found_dataset[0]["joint_positions"],
		"poses_xyzxyzw": poses_xyzxyzw,
		"width": W,
		"height": H,
	}

	if return_sample_result:
		with open(os.path.join(output_dir, "pnp_inference_res.pkl"), "wb") as f:
			pickle.dump(save_obj, f)
		return (
			average_trans_err,
			average_rot_err,
			average_reprojection_err,
			pnp_transform_predicted_mats,
		)
	return average_trans_err, average_rot_err, average_reprojection_err
