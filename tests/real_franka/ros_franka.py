import time
import yaml
import subprocess
import random
import json
import argparse
import torch
import os.path as osp
import os
import cv2
import imageio
import sys
from easycalib.utils.camera_caliberation.realsense_api import RealSenseAPI
from easycalib.utils.frankaAPI.franka_manip import MoveGroupPythonInterface
from easycalib.utils.utilities import overlay_mask_on_img, compute_forward_kinematics, render_mask, suppress_stdout, nostdout, time_block, namespace_to_dict, is_jsonable
from easycalib.utils.utilities import (
	extract_keypoints_from_panda_positions,
	in_cubic_coordinate,
	image_generation_context,
	signal_handler,
	set_random_seed
)

sim_franka_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(sim_franka_path)

from sim_franka.sim_franka_config import sim_franka_default_config
from sim_franka.sim_franka_config import (
	HEIGHT,
	WIDTH,
	FOV,
	CAMERA_INTRINSICS,
)
from real_franka_config import TIME_STEPS, TIME_FREQ, MOVE_SPEED, MAX_JOINT_DIFF, JOINT_POSITION_GAUSSIAN_NOISE, MOVE_STEPS_PER_DIRECTION, REST_POSE
from scipy.spatial.transform import Rotation as R
from easycalib.lib.geometric_vision import left_handed_coord2right_handed_coord, unity_left_handed_to_right_coord, generate_transform_matrices
from datetime import datetime

import numpy as np

from easycalib.utils.setup_logger import setup_logger
logger = setup_logger(__name__)
import signal

np.random.seed(2096)


def cli(save_name=None):
	args, ROBOT_DATA = sim_franka_default_config(save_name)
	logger.info(f"URDF file: {args.urdf_path}, Mesh files: {args.mesh_paths}")
	args.data_save_path = save_name
	args.debug_save_path = os.path.join(args.data_save_path, "debug")
	paths = {'data_save_path': args.data_save_path, 'debug_save_path': args.debug_save_path}

	# capture data
	franka_arm = MoveGroupPythonInterface()
	franka_arm.go_to_rest_pose()
	logger.info(f"Franka's arm has been moved to pose goal: {0}")

	with image_generation_context(logger, paths):
		try:
			os.makedirs(args.data_save_path, exist_ok=True)
			os.makedirs(args.debug_save_path, exist_ok=True)

			# rest pose configuration
			prev_joint_goal = cur_joint_goal = np.array(
				REST_POSE
			)
			franka_arm.go_to_pose(joint_goal=cur_joint_goal)

			OUTER_TIME_STEPS = TIME_STEPS // TIME_FREQ
			INNER_TIME_STEPS = TIME_FREQ // MOVE_STEPS_PER_DIRECTION

			for i in range(0, OUTER_TIME_STEPS):

				distance = MAX_JOINT_DIFF + 0.1
				while distance > MAX_JOINT_DIFF:
					noise = np.random.normal(0, JOINT_POSITION_GAUSSIAN_NOISE, 7)
					cur_joint_goal = prev_joint_goal + noise
					distance = np.linalg.norm(cur_joint_goal - prev_joint_goal)

				for m in range(0, INNER_TIME_STEPS):
					cur_joint_goal = prev_joint_goal + noise

					for j in range(0, MOVE_STEPS_PER_DIRECTION):
						index = i * MOVE_STEPS_PER_DIRECTION * INNER_TIME_STEPS + m * MOVE_STEPS_PER_DIRECTION + j

						joint_goal = (cur_joint_goal - prev_joint_goal) / MOVE_STEPS_PER_DIRECTION * (
							j + 1
						) + prev_joint_goal

						logger.info(f"Planning to go to position: {cur_joint_goal}")

						franka_arm.go_to_pose(joint_goal=joint_goal)
						cur_eef_pos = franka_arm.get_current_end_effector_position()
						_, cur_joint_val = franka_arm.get_joint_states()
						cur_joint_val = cur_joint_val[:7]

						logger.info(
							f"Franka's arm has been moved to cur_position: {cur_eef_pos}"
						)

						rgb = np.ones([HEIGHT, WIDTH, 3], dtype=np.uint8) * 255
						K = CAMERA_INTRINSICS
						# rgb, K=RealSenseAPI.capture_data()
						cv2.imshow("realsense_captured_data", rgb.astype(np.uint8))
						cv2.waitKey(20)

						image = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
						imageio.imwrite(osp.join(paths["data_save_path"], f"{index:06d}.rgb.jpg"), image)

						ROBOT_DATA["objects"][0]["camera_intrinsics"] = K.tolist()
						ROBOT_DATA["objects"][0]["eef_pos"] = cur_eef_pos
						ROBOT_DATA["objects"][0]["joint_positions"] = cur_joint_val
						ROBOT_DATA["objects"][0]["cartesian_position"] = ROBOT_DATA["objects"][0]["eef_pos"]

						for ptidx, pt in enumerate(range(1)):
							if ptidx >= len(ROBOT_DATA["objects"][0]["keypoints"]):
								ROBOT_DATA["objects"][0]["keypoints"].append({"location": None, "name": None, "projected_location": None, "predicted_location": None})

							ROBOT_DATA["objects"][0]["keypoints"][ptidx]["location"] = (
								cur_eef_pos
							)
							ROBOT_DATA["objects"][0]["keypoints"][ptidx]["name"] = args.keypoint_friendly_names[ptidx]
							ROBOT_DATA["objects"][0]["keypoints"][ptidx]["projected_location"] = [-999.99, -999.99]
							ROBOT_DATA["objects"][0]["keypoints"][ptidx]["predicted_location"] = [-999.99, -999.99]

						with open(os.path.join(args.data_save_path, f"{index:06d}.json"), "w") as json_file:
							json.dump(ROBOT_DATA, json_file, indent=4)

					prev_joint_goal = cur_joint_goal

			cv2.destroyAllWindows()

			os_env = os.environ.copy()
			# TODO: when this ffmpeg command line is run in the terminal, it works, but in here, it fails, any idea ?
			ffmpeg_comma = f"ffmpeg -framerate 10 -pattern_type glob -i '{os.path.join(os.getcwd(), args.data_save_path)}/00*.rgb.jpg' -c:v libx264 -pix_fmt yuv420p {os.path.join(os.getcwd(), args.data_save_path)}/outputs.mp4"
			process = subprocess.run(
				ffmpeg_comma,
				shell=True,
				env=os_env,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True,
			)

		except KeyboardInterrupt:
			logger.info("Caught KeyboardInterrupt during image generation")
			raise
		# except Exception as e:
		# 	logger.info(str(e))
		# 	raise


if __name__ == "__main__":
	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGQUIT, signal_handler)

	save_name = datetime.now().strftime("%m_%d_%H_%M_%S")
	save_name = osp.join("./dataset/real/franka", save_name)
	os.makedirs(osp.dirname(save_name), exist_ok=True)
	cli(save_name=save_name)
