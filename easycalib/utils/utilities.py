import argparse
import math
import numpy as np
import torch
import os.path as osp
from argparse import Namespace
import argparse
from PIL import Image
import matplotlib as mpl
import functools

mpl.use("tkagg")

import numpy as np
import pybullet as p
import pybullet_data
import cv2
import os
import sys
import io
import json
from contextlib import contextmanager
from easycalib.utils.nvdiffrast_renderer import NVDiffrastRenderApiHelper
import warnings
import subprocess

import matplotlib.pyplot as plt
from easycalib.utils.os_utils import TemporaryDirectory, atomic_directory_setup
from scipy.spatial.transform import Rotation as R
from easycalib.utils.setup_logger import setup_logger
import shutil
import contextlib
import logging
import inspect
import time
import multiprocessing as mp
import random

logger = setup_logger(__name__)


def set_random_seed(seed):
	assert isinstance(
		seed, int
	), 'Expected "seed" to be an integer, but it is "{}".'.format(type(seed))
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


@contextmanager
def multiprocessing_pool(processes):
	pool = mp.Pool(processes)
	try:
		yield pool
	finally:
		pool.close()
		pool.join()


@contextmanager
def time_block(label, logger):
	start = time.time()
	try:
		# caller_frame = inspect.currentframe().f_back
		# caller_info = inspect.getframeinfo(caller_frame)
		yield
	finally:
		end = time.time()
		logger.info(f"{label}: {end - start:.6f} seconds")


def disturb_Tc_c2b(ori_mat: np.ndarray, translation_noise: float = 0.2, rotation_noise: float = 10):
	"""add noise to the translation and rotation component of a 4x4 local_to_world_matrix: Tc_c2b.

	Args:
																																																																																																																																	ori_mat (np.ndarray): the 4x4 local_to_world_matrix

	Returns:
																																																																																																																																	np.ndarray: the disturbed np.ndarray of the local_to_world_matrix
	"""
	assert isinstance(ori_mat, np.ndarray) and ori_mat.ndim == 2 and ori_mat.shape[0] == 4 and ori_mat.shape[1] == 4, "The mat to be disturbed is {}".format(ori_mat)

	disturbed_matrix = ori_mat.copy()
	disturbed_matrix[:3, 3] += np.random.normal(0, translation_noise, 3)
	rot = R.from_matrix(disturbed_matrix[:3, :3])
	rpy = rot.as_euler('zxy', degrees=True)
	disturbed_rpy = rpy + np.random.normal(0, rotation_noise, 3)
	disturbed_rot = R.from_euler('zxy', disturbed_rpy, degrees=True)
	disturbed_rot = disturbed_rot.as_matrix()
	disturbed_matrix[:3, :3] = disturbed_rot

	return disturbed_matrix


def run_grounded_sam(frame_save_path: str, mask_save_path: str = None, text_prompt: str = None, grounded_sam_script: str = None, grounded_sam_config: str = None, grounded_sam_checkpoint_path: str = None, sam_checkpoint_path: str = None, grounded_sam_repo_path: str = None, save=False, device_id=0):
	"""
	frame_save_path: the file path of the rgb image.
	mask_save_path: where to save the infered mask. (If the parent directory of mask_save_path doesn't exist, create recursively using os.makedirs())
	"""
	# Change current CUDA_VISIBLE_DEVICES for grounded-SAM inference.
	cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else None
	os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

	# Change current working directory for Grounded-Segment-Anything inference.
	pwd = os.getcwd()
	os.chdir(grounded_sam_repo_path)

	with TemporaryDirectory(cleanup=False) as grounded_sam_output_dir:
		os_env = os.environ.copy()
		# print(grounded_sam_output_dir)
		grounded_sam_comma = f"proxychains4 python {grounded_sam_script} --input_image {frame_save_path} --config {grounded_sam_config} --grounded_checkpoint {grounded_sam_checkpoint_path} --text_prompt '{text_prompt}' --sam_checkpoint {sam_checkpoint_path} --output_dir {grounded_sam_output_dir} --text_threshold 1.25 --box_threshold 0.3 --device 'cuda'"
		# logger.warning(f"Runing command: {grounded_sam_comma}")
		process = subprocess.run(
			grounded_sam_comma,
			shell=True,
			env=os_env,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
		)
		logger.debug("STDOUT:", process.stdout)
		logger.debug("STDERR:", process.stderr)

		grounded_sam_mask_path = os.path.join(grounded_sam_output_dir, "mask.jpg")
		if osp.exists(grounded_sam_mask_path):
			grounded_sam_mask = Image.open(grounded_sam_mask_path)
			grounded_sam_mask = np.array(grounded_sam_mask)

			assert grounded_sam_mask.ndim == 2
			grounded_sam_mask = Image.fromarray(grounded_sam_mask)
			if save:
				os.makedirs(osp.dirname(mask_save_path), exist_ok=True)
				grounded_sam_mask.save(mask_save_path)
			grounded_sam_mask = np.asarray(grounded_sam_mask).astype(np.uint8)
			# show the grounded sam model inference path.
			# TODO: show inspects possible way to open an image, but fails to call xdg-open when no available utils are found.
			# grounded_sam_mask.show()
		else:
			grounded_sam_mask = None

	# Change back to the original working directory.
	os.chdir(pwd)
	# Change back to the original CUDA_VISIBLE_DEVICES.JK,
	if cuda_visible_devices is not None:
		os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
	else:
		del os.environ["CUDA_VISIBLE_DEVICES"]
	return grounded_sam_mask


def render_mask(urdf_path, mesh_paths, pose, K, H, W, link_positions, show=False):
	renderer = NVDiffrastRenderApiHelper(mesh_paths, K, pose, H, W)
	_, link_trans_mats = compute_forward_kinematics(
		urdf_path,
		link_positions,
		link_indices=list(range(6)),
		return_pose=True,
	)
	mask = renderer.render_mask(link_poses=link_trans_mats)
	mask = mask.detach().cpu().numpy()
	if show:
		plt.imshow(mask.astype(np.uint8))
		plt.show()
	return mask


@contextmanager
def nostdout():
	save_stdout = sys.stdout
	sys.stdout = io.BytesIO()
	yield
	sys.stdout = save_stdout


@contextmanager
def suppress_stdout():
	# TODO: suppressing stdout may arise many issues: like uvicorn logging or python pdb debugging.
	# Currently deprecated.
	raise DeprecationWarning("Suppresss stdout function is currently deprecated as it can cause a lot of issues.")
	fd = sys.stdout.fileno()

	def _redirect_stdout(to):
		sys.stdout.close()  # + implicit flush()
		os.dup2(to.fileno(), fd)  # fd writes to 'to' file
		sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

	with os.fdopen(os.dup(fd), "w") as old_stdout:
		with open(os.devnull, "w") as file:
			_redirect_stdout(to=file)
		try:
			yield  # allow code to be run with the redirected stdout
		finally:
			_redirect_stdout(to=old_stdout)  # restore stdout.
			# buffering and flags such as
			# CLOEXEC may be different


def overlay_multiple_masks_on_img(
	ori_img, masks, rgbs, labels, alpha, img_save_path, show=True, save_to_disk=False, window_name=None
):
	# Ensure the shape of image is (H, W, 3)
	if len(ori_img.shape) == 2:
		ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
	if ori_img.shape[2] == 4:
		ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGRA2BGR)

	assert len(masks) == len(rgbs) == len(labels), "The number of masks, rgbs, and labels should be the same."
	colored_masks = []
	all_contours = []
	for ind, (mask, rgb, label) in enumerate(zip(masks, rgbs, labels)):
		# if mask.dtype != np.uint8:
		mask = (mask * 255).astype(np.uint8)
		contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		all_contours.append(contours)
		colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
		colored_mask = np.where(colored_mask > 0, rgb, colored_mask).astype(np.uint8)
		colored_masks.append(colored_mask)

	# Overlay the masks with transparency
	overlay_img = ori_img
	for ind, (colored_mask, label) in enumerate(zip(colored_masks, labels)):
		overlay_img = cv2.addWeighted(overlay_img, 1, colored_mask, alpha, 0)
		cv2.drawContours(overlay_img, all_contours[ind], -1, rgbs[ind], 3)

	overlay_img = annotate_image_with_random_number_of_texts(overlay_img, rgbs, labels)

	# Display the image
	if show:
		if window_name is None:
			cv2.imshow("overlaid_masks", overlay_img)
		else:
			cv2.imshow(window_name, overlay_img)
		cv2.waitKey(20)  # Wait for a key press to close the window
		# cv2.destroyAllWindows()

	# Optionally, save the result
	if save_to_disk:
		cv2.imwrite(img_save_path, overlay_img)
	return overlay_img.astype(np.uint8)


def annotate_image_with_random_number_of_texts(image, rgbs, texts, text_offset=100):
	thickness = 2
	font_scale = 0.5
	font = cv2.FONT_HERSHEY_SIMPLEX

	# dimensions of the rectangles
	rect_width, rect_height = 50, 20

	left_x, right_x = image.shape[1] - rect_width - 10, image.shape[1] - 10
	up_y = 10
	annotated_image = image.copy()
	for ind, (rgb, text) in enumerate(zip(rgbs, texts)):
		if text == '':
			continue
		down_y = up_y + rect_height
		cv2.rectangle(annotated_image, (left_x, up_y), (right_x, down_y), rgb, thickness)
		cv2.putText(annotated_image, text, (left_x - text_offset, down_y), font, font_scale, rgb, thickness)
		up_y = down_y + 10
	return annotated_image


def overlay_mask_on_img(
	img, mask1, mask2, rgb1, rgb2, alpha, img_save_path, show=True, save_to_disk=False, window_name=None
):
	# Ensure the shape of image is (H, W, 3)
	if len(img.shape) == 2:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	if img.shape[2] == 4:
		img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

	# Normalize the masks to ensure they are binary
	mask1 = (mask1 * 255).astype(np.uint8)
	mask2 = (mask2 * 255).astype(np.uint8)

	# Color the masks - different colors for visibility, rgb1 for mask1, rgb2 for mask2
	colored_mask1 = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR)
	colored_mask1 = np.where(colored_mask1 > 0, rgb1, colored_mask1).astype(np.uint8)

	colored_mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
	colored_mask2 = np.where(colored_mask2 > 0, rgb2, colored_mask2).astype(np.uint8)

	# Overlay the masks with transparency
	overlay_img = cv2.addWeighted(img, 1, colored_mask1, alpha, 0)
	overlay_img = cv2.addWeighted(overlay_img, 1, colored_mask2, alpha, 0)
	overlay_img = annotate_image(overlay_img, rgb1, rgb2, "gt", "pred")

	# Display the image
	if show:
		if window_name is None:
			cv2.imshow("gt v.s. pred", overlay_img)
		else:
			cv2.imshow(window_name, overlay_img)
		cv2.waitKey(20)  # Wait for a key press to close the window
		# cv2.destroyAllWindows()

	# Optionally, save the result
	if save_to_disk:
		cv2.imwrite(img_save_path, overlay_img)
	return overlay_img.astype(np.uint8)


@functools.lru_cache(maxsize=None)
def load_urdf_in_pybullet(urdf_file, echo=False):
	physicsClient = p.connect(p.DIRECT)
	p.setAdditionalSearchPath(
		pybullet_data.getDataPath()
	)
	if echo:
		print(f"The loaded urdf file is {urdf_file}")

	# with suppress_stdout():
	robotId = p.loadURDF(urdf_file)  # Load the URDF file
	return robotId


def compute_forward_kinematics(
	urdf_file, joint_positions, link_indices=None, return_pose=False, echo=False
):
	robotId = load_urdf_in_pybullet(urdf_file, echo)
	# Set joint positions
	for joint_index, position in enumerate(joint_positions):
		p.resetJointState(robotId, jointIndex=joint_index, targetValue=position)

	# Compute forward kinematics
	link_positions = []
	if return_pose:
		link_mats = []

	if echo:
		print(f"the number of joints in panda.urdf is {p.getNumJoints(robotId)}")
		print(f"the number of links in panda.urdf is {p.getNumJoints(robotId)}")

	for joint_index in range(p.getNumJoints(robotId)):
		link_state = p.getLinkState(
			robotId, joint_index, computeForwardKinematics=True
		)
		link_position, link_quat = list(link_state[0]), list(link_state[1])
		if return_pose:
			rot_mat = p.getMatrixFromQuaternion(link_quat)
			rot_mat = np.array(rot_mat).reshape(3, 3)
			trans_mat = np.eye(4)
			trans_mat[:3, 3] = link_position
			trans_mat[:3, :3] = rot_mat
			link_mats.append(trans_mat)
		link_positions.append(np.array(link_position))

	link_positions = np.stack(link_positions)
	if return_pose:
		link_mats = np.stack(link_mats)
		return link_positions, link_mats
	if link_indices is not None:
		link_positions = link_positions[link_indices]
	return link_positions


def annotate_image(image, rgb1, rgb2, text1, text2, text_offset=100):
	thickness = 2
	font_scale = 0.5
	font = cv2.FONT_HERSHEY_SIMPLEX

	# dimensions of the rectangles
	rect_width, rect_height = 50, 20

	# positioning the first rectangle for 'ground-truth'
	top_right_x1 = image.shape[1] - rect_width - 10  # 10 pixels padding from the edge
	top_right_y1 = 10  # 10 pixels padding from the top
	bottom_left_x1 = image.shape[1] - 10
	bottom_left_y1 = top_right_y1 + rect_height

	# drawing the first rectangle
	cv2.rectangle(
		image,
		(top_right_x1, top_right_y1),
		(bottom_left_x1, bottom_left_y1),
		rgb1,
		thickness,
	)

	# adding text beside the first rectangle
	cv2.putText(
		image,
		text1,
		(top_right_x1 - text_offset, bottom_left_y1),
		font,
		font_scale,
		rgb1,
		thickness,
	)

	# positioning the second rectangle for 'co-tracker inference'
	top_right_x2 = top_right_x1
	top_right_y2 = bottom_left_y1 + 10  # 10 pixels below the first rectangle
	bottom_left_x2 = bottom_left_x1
	bottom_left_y2 = top_right_y2 + rect_height

	# drawing the second rectangle
	cv2.rectangle(
		image,
		(top_right_x2, top_right_y2),
		(bottom_left_x2, bottom_left_y2),
		rgb2,
		thickness,
	)

	# adding text beside the second rectangle
	cv2.putText(
		image,
		text2,
		(top_right_x1 - text_offset, bottom_left_y2),
		font,
		font_scale,
		rgb2,
		thickness,
	)

	return image


def in_planar(position, cubic_coordinate_list):
	"""
	Check if the position is in the planar region defined by the cubic_coordinate_list.
	"""
	x_bool = position[0] == cubic_coordinate_list[0][0]
	y_bool = position[1] < np.max(cubic_coordinate_list[:, 1]) and position[1] > np.min(
		cubic_coordinate_list[:, 1]
	)
	z_bool = position[2] < np.max(cubic_coordinate_list[:, 2]) and position[2] > np.min(
		cubic_coordinate_list[:, 2]
	)
	return x_bool and y_bool and z_bool


def extract_keypoints_from_panda_positions(positions):
	"""
	Extract keypoints from panda positions.
	"""
	return positions[10:11]
	# return positions[0:13]


def in_cubic_coordinate(position, cubic_coordinate_list):
	"""
	Check if the position is in the cubic region defined by the cubic_coordinate_list.
	"""
	x_bool = position[0] < np.max(cubic_coordinate_list[:, 0]) and position[0] > np.min(
		cubic_coordinate_list[:, 0]
	)
	y_bool = position[1] < np.max(cubic_coordinate_list[:, 1]) and position[1] > np.min(
		cubic_coordinate_list[:, 1]
	)
	z_bool = position[2] < np.max(cubic_coordinate_list[:, 2]) and position[2] > np.min(
		cubic_coordinate_list[:, 2]
	)
	return x_bool and y_bool and z_bool


def calculate_camera_intrinsics_given_fov_x(fov_x, width, height):
	"""
	Calculate camera intrinsics matrix K from horizontal FOV and image dimensions,
	automatically adjusting for vertical FOV based on aspect ratio.
	return_type: torch.Tensor float64
	"""
	# Convert FOV from degrees to radians
	fov_x_rad = math.radians(fov_x)
	fov_y_rad = 2 * math.atan(math.tan(fov_x_rad / 2) * (height / width))

	# Calculate focal lengths
	f_x = width / (2 * math.tan(fov_x_rad / 2))
	f_y = height / (2 * math.tan(fov_y_rad / 2))

	# Calculate optical center
	c_x = width / 2
	c_y = height / 2

	# Form the camera intrinsics matrix
	K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]], dtype=np.float64)
	K = torch.from_numpy(K).double()  # Ensuring the tensor is of type float64
	return K


def calculate_camera_intrinsics_given_fov_y(fov_y, width, height):
	"""
	Calculate camera intrinsics matrix K from verical FOV and image dimensions,
	automatically adjusting for vertical FOV based on aspect ratio.
	return_type: torch.Tensor float64
	"""
	# Convert FOV from degrees to radians
	fov_y_rad = math.radians(fov_y)
	fov_x_rad = 2 * math.atan(math.tan(fov_y_rad / 2) * (width / height))

	# Calculate focal lengths
	f_x = width / (2 * math.tan(fov_x_rad / 2))
	f_y = height / (2 * math.tan(fov_y_rad / 2))

	# Calculate optical center
	c_x = width / 2
	c_y = height / 2

	# Form the camera intrinsics matrix
	K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]], dtype=np.float64)
	K = torch.from_numpy(K).double()  # Ensuring the tensor is of type float64
	return K


def calculate_camera_intrinsics(fov_x, fov_y, width, height):
	"""
	Calculate camera intrinsics matrix K from FOV and image dimensions.
	Useful when conducting synthetic data generation when getting CAMERA_INTRINSICS from Unity.
	return_type: torch.Tensor float64
	"""
	# Convert FOV from degrees to radians
	fov_x_rad = math.radians(fov_x)
	fov_y_rad = math.radians(fov_y)

	# Calculate focal lengths
	f_x = width / (2 * math.tan(fov_x_rad / 2))
	f_y = height / (2 * math.tan(fov_y_rad / 2))

	# Calculate optical center
	c_x = width / 2
	c_y = height / 2

	# Form the camera intrinsics matrix
	K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
	K = torch.from_numpy(K).float()
	return K


def merge_two_dicts(starting_dict: dict, updater_dict: dict) -> dict:
	"""
	Starts from base starting dict and then adds the remaining key values from updater replacing the values from
	the first starting/base dict with the second updater dict.

	For later: how does d = {**d1, **d2} replace collision?

	:param starting_dict:
	:param updater_dict:
	:return:
	"""
	new_dict: dict = starting_dict.copy()  # start with keys and values of starting_dict
	new_dict.update(
		updater_dict
	)  # modifies starting_dict with keys and values of updater_dict
	return new_dict


def is_jsonable(x):
	try:
		json.dumps(x)
		return True
	except (TypeError, OverflowError):
		return False


def namespace_to_dict(namespace: argparse.Namespace) -> dict:
	return {
		attr: getattr(namespace, attr)
		for attr in dir(namespace)
		if not attr.startswith("_")
	}


def merge_args(args1: Namespace, args2: Namespace) -> Namespace:
	"""

	ref: https://stackoverflow.com/questions/56136549/how-can-i-merge-two-argparse-namespaces-in-python-2-x
	:param args1: the default args: Namespace
	:param args2: the override args: Namespace
	:return:
	"""
	# - the merged args
	# The vars() function returns the __dict__ attribute to values of the given object e.g {field:value}.
	default_dict = namespace_to_dict(args1)
	override_dict = namespace_to_dict(args2)
	merged_key_values_for_namespace: dict = merge_two_dicts(default_dict, override_dict)
	args = Namespace(**merged_key_values_for_namespace)
	return args


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ("yes", "true", "t", "y", "1"):
		return True
	elif v.lower() in ("no", "false", "f", "n", "0"):
		return False
	else:
		raise argparse.ArgumentTypeError("Boolean value expected.")


def compute_iou(mask1: np.ndarray, mask2: np.ndarray):
	"""
	Compute the Intersection over Union (IoU) of two image masks.

	Parameters:
	- mask1 (np.array): First mask image as a numpy array.
	- mask2 (np.array): Second mask image as a numpy array.

	Returns:
	- float: IoU score.
	"""
	# Ensure the masks are boolean
	mask1_bool = mask1.astype(bool)
	mask2_bool = mask2.astype(bool)

	# Calculate intersection and union
	intersection = np.logical_and(mask1_bool, mask2_bool)
	union = np.logical_or(mask1_bool, mask2_bool)

	# Compute IoU
	iou = np.sum(intersection) / np.sum(union)

	return iou


def signal_handler(signal_received, frame):
	logger.warning(f"Signal {signal_received} received, terminating the process.")
	# logger.debug(f"Current line number: {frame.f_lineno}")
	# logger.debug(f"Current function: {frame.f_code.co_name}")
	# logger.debug(f"Local variables: {frame.f_locals}")
	# logger.debug(f"Global variables: {frame.f_globals}")
	# logger.debug(f"Reference to the previous frame: {frame.f_back}")
	# logger.debug(f"Code object that represents the compiles: {frame.f_code}")
	sys.exit(0)


@contextlib.contextmanager
def image_generation_context(logger, paths):
	try:
		logger.debug("Starting image generation...")
		yield
	except Exception as e:
		logger.error(f"An error occurred during image generation: {str(e)}, removing broken directory:{paths['data_save_path']},{paths['debug_save_path']}")
		shutil.rmtree(paths['data_save_path'], ignore_errors=True)
		shutil.rmtree(paths['debug_save_path'], ignore_errors=True)
		raise
	finally:
		logger.debug("Image generation completed.")


if __name__ == "__main__":
	# Example usage
	fov_x = 60  # horizontal FOV in degrees
	fov_y = 45  # vertical FOV in degrees
	width = 1024  # image width in pixels
	height = 768  # image height in pixels

	K = calculate_camera_intrinsics(fov_x, fov_y, width, height)
	print(K)

	# p = compute_forward_kinematics(
	#     urdf_file="assets/franka/urdf/franka_another.urdf",
	#     joint_positions=[0, 0, 0, 0, 0, 0, 0]
	# )
	# Setup and use the logger
	# logger = setup_logger()
	# logger.debug("This is a debug message")
	# logger.info("This is an info message")
	# logger.warning("This is a warning message")
	# logger.error("This is an error message")
	# logger.critical("This is a critical message")
