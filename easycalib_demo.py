import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
from enum import Enum
import numpy as np
import os
import torch
import argparse
import numpy as np
from multiprocessing import Pool
torch.multiprocessing.set_start_method('spawn', force=True)

from easycalib.utils.utilities import overlay_mask_on_img, compute_forward_kinematics, render_mask, suppress_stdout, nostdout, time_block, namespace_to_dict, is_jsonable
import imageio
import subprocess
import numpy as np
import os.path as osp
from glob import glob
import cv2
import functools
from easycalib.utils.utilities import (
	merge_two_dicts,
	namespace_to_dict,
	time_block
)
from easycalib.utils.setup_logger import setup_logger
from easycalib.lib.easycalib_analysis import caliberate_camera, CALIBERATION_METHODS
from easycalib.utils.point_drawer import PointDrawer
from easycalib.config.parse_demo_argument import parse_easycalib_default_args
from easycalib.utils.utilities import run_grounded_sam
import pickle

import matplotlib as mpl

mpl.use("tkagg")
import matplotlib.pyplot as plt

from tqdm import tqdm

logger = setup_logger(__name__)

COTRACKER_INSTALLED=True
try:
	from easycalib.utils.co_tracker_predictor_wrapper import CoTrackerPredictorWrapper
except ImportError:
	COTRACKER_INSTALLED = False
	logger.warning("CoTracker is not installed, running w/o cotracker support.")

SPATIAL_TRACKER_INSTALLED=True
try:
	from easycalib.utils.spatial_tracker_predictor_wrapper import parse_spatracker_args, spatracker_predict
except ImportError:
	SPATIAL_TRACKER_INSTALLED = False
	logger.warning("Spatial tracker is not installed, running w/o spatial_tracker support.")

KEYPOINT_TRACKING_METHODS = Enum("methods", ["COTRACKER", "SPATIAL_TRACKER", "DINO_TRACKER"])
KEYPOINT_TRACKING_DICT = {"cotracker": KEYPOINT_TRACKING_METHODS.COTRACKER, "spatial_tracker": KEYPOINT_TRACKING_METHODS.SPATIAL_TRACKER, "dino_tracker": KEYPOINT_TRACKING_METHODS.DINO_TRACKER}

SAVE_TAG = "Kalib"

class Kalib:

	def __init__(self, args):
		override_args = {"namespace": namespace_to_dict(args)}
		parsed_args = self.parse_argument(override_args)
		self._args = parsed_args
		self.root_dir = self._args.root_dir
		self.saved_name = None
		self.override_args = None
		self.model_inference_path = None
		self.video: torch.Tensor

		self.found_image_paths = None
		self.found_json_paths = None
		self.img_objs = None
		self.json_objs = None

		self.end_effector_position = None
		self.first_frame_segm_mask = None
		self.first_frame = None
		self.cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.Tc_c2b = None

	# region
	@ staticmethod
	def _post_process_and_render_mask(index, img_path, args, H, W, local_to_world_matrices, camera_intrinsics, qposes, pred_trans, gt_mask_save_dir, pred_img_save_dir, overlaid_img_save_dir):
		# ! This function should only be used inside Kalib, as multiprocessing requires pickable objects, so this function cannot be a local function (e.g, a class func for Kalib class.)
		"""
		use nvdiffrast renderer to render gt_mask, pred_mask and overlay gt mask and pred_mask on original rgb image.
		"""
		with nostdout():
			gt_mask = render_mask(args.urdf_path, args.mesh_paths, local_to_world_matrices[index], np.array(camera_intrinsics[index]), H, W, qposes[index])
			gt_mask_save_path = osp.join(gt_mask_save_dir, "%s.png" % osp.splitext(osp.basename(img_path))[0])
			pred_mask_save_path = osp.join(pred_img_save_dir, "%s.png" % osp.splitext(osp.basename(img_path))[0])
			pred_mask = render_mask(args.urdf_path, args.mesh_paths, np.array(pred_trans), np.array(camera_intrinsics[index]), H, W, qposes[index])

		cv2.imwrite(gt_mask_save_path, (gt_mask * 255).astype(np.uint8))
		cv2.imwrite(pred_mask_save_path, (pred_mask * 255).astype(np.uint8))

		overlay_img = overlay_mask_on_img(
			cv2.imread(img_path),
			gt_mask,
			pred_mask,
			rgb1=(255, 0, 255),
			rgb2=(0, 255, 255),
			alpha=0.5,
			show=False,
			save_to_disk=True,
			img_save_path=osp.join(overlaid_img_save_dir, "%0.4d.png" % index),
		)
	# endregion


	# region
	@ staticmethod
	def _render_all_masks(found_image_paths, found_json_paths, img_objs, json_objs, pred_local_to_world_matrix, args, save_tag):
		"""
		TODO: Determine whether the nvdiffrast renderere takes local_to_world_matrix/world_to_local_matrix as input.
		"""
		qposes = [
			single_json_obj["objects"][0]["joint_positions"]
			for single_json_obj in json_objs
		]
		camera_intrinsics = [
			single_json_obj["objects"][0]["camera_intrinsics"]
			for single_json_obj in json_objs
		]
		local_to_world_matrices = [
			single_json_obj["objects"][0]["local_to_world_matrix"] if args.has_gt else pred_local_to_world_matrix
			for single_json_obj in json_objs
		]
		H, W = cv2.imread(found_image_paths[0]).shape[:2]
		# device = torch.device("cuda:%d" % args.renderer_device_id) if torch.cuda.is_available() else "cpu"

		overlaid_img_save_dir = osp.join(osp.dirname(found_image_paths[0]), save_tag, "%s_rendered_mask" % save_tag)
		os.makedirs(overlaid_img_save_dir, exist_ok=True)
		pred_img_save_dir = osp.join(osp.dirname(found_image_paths[0]), save_tag, "%s_pred_mask" % save_tag)
		os.makedirs(pred_img_save_dir, exist_ok=True)
		gt_mask_save_dir = osp.join(osp.dirname(found_image_paths[0]), save_tag, "%s_gt_mask" % save_tag)
		os.makedirs(gt_mask_save_dir, exist_ok=True)

		tasks = [(i, path, args, H, W, local_to_world_matrices, camera_intrinsics, qposes, pred_local_to_world_matrix, gt_mask_save_dir, pred_img_save_dir, overlaid_img_save_dir)
				for i, path in enumerate(found_image_paths)]

		logger.debug(f"Use multiprocessing.Pool to parallel mask rendering routine.")

		cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else None
		os.environ["CUDA_VISIBLE_DEVICES"] = str(args.renderer_device_id)
		with time_block("Finish mask rendering routine. ", logger):
			with Pool(processes=25) as pool:
				pool.starmap(Kalib._post_process_and_render_mask, tasks)

		if cuda_visible_devices is not None:
			os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
		else:
			del os.environ["CUDA_VISIBLE_DEVICES"]

		os_env = os.environ.copy()

		ffmpeg_comma = f"ffmpeg -framerate 10 -pattern_type glob -y -i '{osp.join(gt_mask_save_dir,'*.png')}' -c:v libx264 -pix_fmt yuv420p {osp.join(gt_mask_save_dir, 'outputs.mp4')}"
		process = subprocess.run(
			ffmpeg_comma,
			shell=True,
			env=os_env,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
		)

		ffmpeg_comma = f"ffmpeg -framerate 10 -pattern_type glob -y -i '{osp.join(pred_img_save_dir,'*.png')}' -c:v libx264 -pix_fmt yuv420p {osp.join(pred_img_save_dir, 'outputs.mp4')}"
		process = subprocess.run(
			ffmpeg_comma,
			shell=True,
			env=os_env,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
		)
	# endregion


	# region
	def parse_argument(self, namespace: dict):
		# logger.info(f"Receiving override argparse.Namespace object: {namespace}")
		default_args = namespace_to_dict(parse_easycalib_default_args(cli_input=False))
		override_args = namespace["namespace"]
		merged_args = merge_two_dicts(default_args, override_args)
		merged_args = argparse.Namespace(**merged_args)
		return merged_args
	# endregion

	def post_process(self):
		args = self._args
		# render the pred mask on the inference result from EasyHeC.
		if self.found_image_paths is None or self.found_json_paths is None or self.img_objs is None or self.json_objs is None:
			self.img_objs, self.json_objs, self.found_image_paths, self.found_json_paths = Kalib.find_data_in_dir(
				self.root_dir, args.cut_off
			)

		with open(self.pnp_res_save_path, "rb") as f:
			"""
			save_obj = {
			"avg_trans_err": average_trans_err.tolist(),
			"avg_rot_err": average_rot_err.item(),
			"avg_reprojection_err": average_reprojection_err.item(),
			"pnp_transform_predicted_mats": pnp_transform_predicted_mats.tolist(),
			"gt_local_to_world_matrix": gt_local_to_world_matrix.tolist(),
			}
			"""
			pnp_inference_res_save_obj = pickle.load(f)
			avg_trans_err = pnp_inference_res_save_obj["avg_trans_err"]
			avg_rot_err = pnp_inference_res_save_obj["avg_rot_err"]
			avg_reproj_err = pnp_inference_res_save_obj["avg_reprojection_err"]
			pnp_trans_predicted_mats = pnp_inference_res_save_obj[
				"pnp_transform_predicted_mats"
			]
			# gt_local_to_world_matrix = pnp_inference_res_save_obj[
			#     "gt_local_to_world_matrix"
			# ]
			# camera_K = pnp_inference_res_save_obj["camera_K"]

			# camera_K = np.array(camera_K)
			# gt_local_to_world_matrix = np.array(gt_local_to_world_matrix)
			pnp_trans_predicted_mats = np.array(pnp_trans_predicted_mats)

			if args.render_mask:
				pred_world_to_local_matrix = (
					pnp_trans_predicted_mats[0]
					if len(pnp_trans_predicted_mats) > 1
					else pnp_trans_predicted_mats
				)

				logger.info(
					f"Avg_trans_err: {avg_trans_err}, Avg_rot_err: {avg_rot_err}, Avg_reproj_err: {avg_reproj_err}, root_dir: {self.root_dir}."
				)
				Kalib._render_all_masks(self.found_image_paths, self.found_json_paths, self.img_objs, self.json_objs, pred_world_to_local_matrix, args, SAVE_TAG)
		# region

	# region
	@ functools.lru_cache(maxsize=None)
	@ staticmethod
	def find_data_in_dir(dir_path, cut_off):
		image_suffixes = ["rgb.jpg", "jpg", "png"]
		suffix_hits = {suffix: 0 for suffix in image_suffixes}

		for suffix in image_suffixes:
			matches = glob(osp.join(dir_path, f"*.{suffix}"))
			suffix_hits[suffix] = len(matches)

		sorted_suffixes = sorted(suffix_hits.items(), key=lambda x: x[1], reverse=True)

		main_suffix = sorted_suffixes[0][0]
		found_image_paths = sorted(
			glob(osp.join(dir_path, f"*.{main_suffix}"))
		)
		found_json_paths = sorted(glob(osp.join(dir_path, "*.json")))

		if cut_off != -1:
			found_image_paths = found_image_paths[: cut_off]
			found_json_paths = found_json_paths[: cut_off]

		img_insts = []
		json_objs = []

		for img_path, json_path in tqdm(zip(found_image_paths, found_json_paths), desc="Reading image paths and json paths."):

			img_inst = cv2.imread(img_path)
			if img_inst.shape[-1] == 4:
				img_inst = img_inst[:, :, :3]
			img_insts.append(img_inst.astype(np.uint8))
			with open(json_path, "r") as j:
				json_obj = json.loads(j.read())
				json_objs.append(json_obj)

		return img_insts, json_objs, found_image_paths, found_json_paths
	# endregion

	def mask_end_effector_position(self, first_frame_path):
		# run PointDrawer to obtain the end_effector_position
		args = self._args
		tracking_kpts = np.array(args.keypoint_friendly_names)[np.array(args.keypoint_ids)]
		rgb = imageio.imread_v2(first_frame_path)
		pointdrawer = PointDrawer(
			sam_checkpoint=args.sam_checkpoint_path,
			sam_model_type=args.sam_type,
			window_name="End Effector Position Annotation",
			dry_run=True,
		)
		rgb = rgb[..., :3]
		eef_pos, labels, mask = pointdrawer.run(rgb)
		if eef_pos.shape[0] != len(args.keypoint_ids):
			raise Exception("The number of keypoints annotated is not equal to the number of keypoints in the config file.")
		# self.end_effector_position = np.mean(eef_pos, axis=0).tolist()  # N x 2
		self.end_effector_position = eef_pos.tolist()
		return self.end_effector_position

	def preprocess(self, save_tag):
		"""
		Skip the main routine if the pnp_inference_res.pkl is founded in the correct path.
		"""

		self.pnp_res_save_path = osp.join(
			self.root_dir,
			save_tag,
			"%s_outputs" % save_tag,
			"pnp_inference_res.pkl",
		)
		args = self._args

		if not osp.exists(self.pnp_res_save_path):
			img_objs, json_objs, found_image_paths, found_json_paths = Kalib.find_data_in_dir(
				self.root_dir, args.cut_off
			)
			first_frame_path = found_image_paths[0]

			self.found_image_paths = found_image_paths
			self.found_json_paths = found_json_paths
			self.img_objs = img_objs
			self.json_objs = json_objs

			if len(found_image_paths) == 0 or len(found_json_paths) == 0:
				raise Exception("Couldn't found any images or json files in %s" % self.root_dir)
			if len(found_image_paths) != len(found_json_paths):
				raise Exception("The number of image paths and json files doesn't match! Directory: %s" % self.root_dir)

			logger.debug("First frame path: %s" % first_frame_path)
			# cv2.imshow("first_frame", imageio.imread_v2(first_frame_path))
			# cv2.waitKey(0)
			# key = cv2.waitKey(0)
			# if key == ord("n"):
			#     return
			# else:
			#     cv2.destroyAllWindows()

			self.end_effector_position = self.mask_end_effector_position(first_frame_path)
		
			with time_block("Processing payload data from local disk ", logger):
				img_objs, json_objs, found_image_paths, found_json_paths = Kalib.find_data_in_dir(
					self.root_dir, args.cut_off
				)
				first_frame_path = found_image_paths[0]

				self.len_frame = len(found_image_paths)
				self.found_image_paths = found_image_paths
				self.found_json_paths = found_json_paths

				assert (
					len(found_image_paths) > 0
					and len(found_json_paths) > 0
					and len(found_image_paths) == len(found_json_paths)
				), "Error: No image or json files found in the directory, or the number of images and json files are not equal."

				video_frames = np.stack(img_objs)
				video = (
					torch.from_numpy(video_frames).permute(0, 3, 1, 2)[None].float()
				)  # batch x times x channel x height x width

				self.model_inference_path = osp.join(
					self.root_dir,
					SAVE_TAG,
					"saved_frames_visibility",
				)
				self.first_frame = cv2.imread(first_frame_path).astype(np.uint8)

				if args.use_segm_mask:
					mask_save_dir = osp.join(self.root_dir, SAVE_TAG, "first_frame_mask_segm")
					os.makedirs(mask_save_dir, exist_ok=True)
					self.mask_save_path = osp.join(mask_save_dir, "mask.png")

					if args.use_grounded_sam:
						first_frame_segm_mask = self.run_grounded_sam(
							frame_save_path=first_frame_path,
							mask_save_path=self.mask_save_path,
							device_id=args.mask_inference_device_id,
						)
						if first_frame_segm_mask is None:
							logger.warning("Grounded_SAM inference failed to find valid mask for this frame. Input prompt: Robot arm.")
					elif args.use_sam:
						pointdrawer = PointDrawer(
							sam_checkpoint=args.sam_checkpoint_path,
							sam_model_type=args.sam_type,
							window_name="First Frame Mask segmentation",
							device=torch.device("cuda:%d" % args.mask_inference_device_id if torch.cuda.is_available() else torch.device("cpu"))
						)
						_, _, mask = pointdrawer.run(cv2.imread(first_frame_path))
						first_frame_segm_mask = (mask * 255).astype(np.uint8)
					if first_frame_segm_mask is not None:
						cv2.imwrite(self.mask_save_path, first_frame_segm_mask)
				else:
					first_frame_segm_mask = None

				(
					self.video,
					self.img_objs,
					self.json_objs,
					self.first_frame_segm_mask,
				) = (
					video,
					img_objs,
					json_objs,
					first_frame_segm_mask,
				)
			return False
		return True

	@time_block("Grounded-segment-Anything inference w.r.t first frame mask ", logger)
	def run_grounded_sam(self, frame_save_path: str, mask_save_path: str = None, device_id=0):
		args = self._args
		# Change current working directory for Grounded-Segment-Anything inference.
		mask = run_grounded_sam(frame_save_path, mask_save_path, args.text_prompt, args.grounded_sam_script, args.grounded_sam_config, args.grounded_sam_checkpoint_path,
								args.sam_checkpoint_path,
								args.grounded_sam_repo_path, device_id=device_id)
		return mask


	def kpt_tracking(self):
		args = self._args
		device_id = args.tracking_device_id
		cuda_device = (
			torch.device(f"cuda:{device_id}")
			if torch.cuda.is_available()
			else torch.device("cpu")
		)
		video = self.video

		with time_block("Kpt tracking routine. ", logger):
			first_frame = video[0, 0].permute(1, 2, 0).int()
			first_frame = first_frame.detach().cpu().numpy().astype(np.uint8)

			end_effector_position = self.end_effector_position

			queries: torch.Tensor = torch.FloatTensor(end_effector_position)
			queries = queries.unsqueeze(0)  # 1 x N x 2

			end_effector_position = torch.cat(
				[torch.ones_like(queries[..., 0:1]) * 0, queries], dim=2
			)  # ! 1 x N x 3
			# logger.info(end_effector_position.shape)
			# end_effector_position = end_effector_position.unsqueeze(0)  # ! 1 x N x 2
			logger.info(f"the end_effector_positions is {end_effector_position}")

			if args.use_segm_mask and self.first_frame_segm_mask is not None:
				segm_mask = np.array(self.first_frame_segm_mask)
				segm_mask = torch.from_numpy(segm_mask)[None, None]
			else:
				segm_mask = None

			keypoint_tracking_method = KEYPOINT_TRACKING_DICT[args.keypoint_tracking_method.lower()]
			if keypoint_tracking_method == KEYPOINT_TRACKING_METHODS.COTRACKER:
				if not COTRACKER_INSTALLED:
					raise Exception("Cotracker is not installed, however, the kpt-tracking method is set to cotracker.")

				end_effector_position = end_effector_position.to(cuda_device)
				if args.checkpoint is not None:
					model = CoTrackerPredictorWrapper(checkpoint=args.checkpoint, device=cuda_device)
				else:
					model = torch.hub.load("facebookresearch/co-tracker", "cotracker2")

				model = model.to(cuda_device)
				video = video.to(cuda_device)
				segm_mask = segm_mask.to(cuda_device) if segm_mask is not None else None

				pred_tracks, pred_visibility = model(
					video=video,
					grid_size=args.grid_size,
					grid_query_frame=args.grid_query_frame,
					backward_tracking=args.backward_tracking,
					queries=end_effector_position,
					segm_mask=segm_mask,
				)

				pred_tracks = pred_tracks.detach().cpu().numpy()
				pred_visibility = pred_visibility.detach().cpu().numpy()

			elif keypoint_tracking_method == KEYPOINT_TRACKING_METHODS.SPATIAL_TRACKER:
				if not SPATIAL_TRACKER_INSTALLED:
					raise Exception("SpaTracker is not installed, however, the kpt-tracking method is set to spatial_tracker.")

				torch.cuda.empty_cache()
				# Change current working directory for Grounded-Segment-Anything inference.
				pwd = os.getcwd()
				os.chdir(args.spatial_tracker_repo_path)

				cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else None
				os.environ["CUDA_VISIBLE_DEVICES"] = str(args.tracking_device_id)

				spatracker_args = parse_spatracker_args()
				segm_mask = segm_mask.cpu().numpy() if segm_mask is not None else None

				with Pool(processes=1) as pool:
					res = pool.starmap(spatracker_predict, [(spatracker_args, video, end_effector_position, segm_mask)])
				pred_tracks, pred_visibility = res[0][0], res[0][1]

				if cuda_visible_devices is not None:
					os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
				else:
					del os.environ["CUDA_VISIBLE_DEVICES"]

				os.chdir(pwd)

			elif keypoint_tracking_method == KEYPOINT_TRACKING_METHODS.DINO_TRACKER:
				raise Exception("Dino-tracker method is currently not supported !")

			json_stored_path = self.root_dir
			os.makedirs(self.model_inference_path, exist_ok=True)

			tracker_data_save_path = osp.join(self.model_inference_path, "tracker_data")
			tracker_img_save_path = osp.join(self.model_inference_path, "tracker_img")
			os.makedirs(tracker_data_save_path, exist_ok=True)
			os.makedirs(tracker_img_save_path, exist_ok=True)

			np.save(osp.join(tracker_data_save_path, "pred_tracks"), pred_tracks)
			np.save(osp.join(tracker_data_save_path, "pred_visibility"), pred_visibility)

			pred_tracks = pred_tracks[0, :, : len(args.keypoint_ids)]  # N x num_keypoints
			pred_visibilities = pred_visibility[0, :, : len(args.keypoint_ids)]  # N x num_keypoints

			all_pred_err_float = []
			all_pred_err_int = []

			img_objs = self.img_objs
			json_objs = self.json_objs

		with time_block("Saving kpt tracking inference result to annotated images and json objs. ", logger):
			for frame_idx, (single_img, single_json_obj, frame, visibility) in tqdm(
					enumerate(zip(img_objs, json_objs, pred_tracks, pred_visibilities))
			):
				json_keypoints = np.asarray(single_json_obj["objects"][0]["keypoints"])
				json_keypoints = json_keypoints[np.asarray(args.keypoint_ids)]

				pred_err_float_per_frame = []
				pred_err_int_per_frame = []

				for point_idx, (point, vis_flag) in enumerate(zip(frame, visibility)):
					color = (255, 255, 0) if vis_flag else (255, 0, 0)
					single_img = cv2.circle(
						single_img,
						(int(point[0]), int(point[1])),
						radius=5,
						color=color,
						thickness=-1,
					)
					single_img = cv2.circle(
						single_img,
						(int(json_keypoints[point_idx]["projected_location"][0]), int(json_keypoints[point_idx]["projected_location"][1])),
						radius=5,
						color=(0, 0, 255),
						thickness=-1,
					)
					point = point.tolist()
					json_keypoints[point_idx]["predicted_location"] = [point[0], point[1]]

					if args.has_gt:
						pred_err_float_per_frame.append(
							json_keypoints[point_idx]["projected_location"]
							- np.array(point)
						)
						pred_err_int_per_frame.append(
							json_keypoints[point_idx]["projected_location"]
							- np.array([int(point[0]), int(point[1])])
						)

				if args.has_gt:
					pred_err_int_per_frame = np.stack(pred_err_int_per_frame)
					pred_err_float_per_frame = np.stack(pred_err_float_per_frame)

					all_pred_err_float.append(pred_err_float_per_frame)
					all_pred_err_int.append(pred_err_int_per_frame)

				# Construct the path for saving the image
				save_path = osp.join(
					tracker_img_save_path,
					f"{frame_idx:05d}.png",
				)
				cv2.imwrite(save_path, single_img)

				single_json_obj_save_path = osp.join(
					json_stored_path, f"{frame_idx:06d}.json"
				)
				with open(single_json_obj_save_path, "w") as j:
					json.dump(single_json_obj, j, indent=4)

			if args.has_gt:
				all_pred_err_float = np.stack(all_pred_err_float)
				all_pred_err_int = np.stack(all_pred_err_int)

			os_env = os.environ.copy()
			dir_path = self.root_dir
			ffmpeg_comma = f"ffmpeg -framerate 10 -pattern_type glob -y -i '{dir_path}/*.rgb.jpg' -c:v libx264 -pix_fmt yuv420p {dir_path}/outputs.mp4"
			process = subprocess.run(
				ffmpeg_comma,
				shell=True,
				env=os_env,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True,
			)

			ffmpeg_comma = f"ffmpeg -framerate 10 -pattern_type glob -y -i '{osp.join(tracker_img_save_path, '*.png')}' -c:v libx264 -pix_fmt yuv420p {osp.join(tracker_img_save_path, f'outputs_{args.keypoint_tracking_method}.mp4')}"
			process = subprocess.run(
				ffmpeg_comma,
				shell=True,
				env=os_env,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True,
			)

		if args.has_gt:
			fig, axs = plt.subplots(len(args.keypoint_ids), 2, layout="constrained")
			dataset_len = len(img_objs)
			if len(args.keypoint_ids) == 1:
				axs = [axs]
			for ax_ind, ax in enumerate(axs):

				axs[ax_ind][0].plot(
					np.linspace(0, dataset_len, dataset_len),
					all_pred_err_float[:, ax_ind, 0],
					color="green",
					label="float",
				)
				axs[ax_ind][1].plot(
					np.linspace(0, dataset_len, dataset_len),
					all_pred_err_int[:, ax_ind, 1],
					color="red",
					label="int",
				)
				axs[ax_ind][0].plot(
					np.linspace(0, dataset_len, dataset_len),
					np.tile(np.mean(all_pred_err_float[:, ax_ind, 0]), dataset_len),
					color="green",
					label="float",
				)
				axs[ax_ind][1].plot(
					np.linspace(0, dataset_len, dataset_len),
					np.tile(np.mean(all_pred_err_int[:, ax_ind, 1]), dataset_len),
					color="red",
					label="int",
				)
				axs[ax_ind][0].set_title("pred_error_x")
				axs[ax_ind][1].set_title("pred_error_y")

			fig.suptitle("pred_error v.s. time_steps")

			plt.legend()
			plt.show()

	def run_caliberate_camera(self):
		args = self._args
		output_dir = osp.join(self.root_dir, SAVE_TAG, "%s_outputs" % SAVE_TAG)
		os.makedirs(output_dir, exist_ok=True)
		sliding_window_step = (
			self.len_frame
			- args.win_len
		)
		# initialize caliberate related method options.
		zhangzhengyou_flags = None
		caliberate_method = None
		if args.caliberate_method.upper() == "PNP":
			caliberate_method = CALIBERATION_METHODS.PNP
		elif args.caliberate_method.upper() == "ZHANGZHENGYOU":
			if args.intrinsics_guess:
				zhangzhengyou_flags = cv2.CALIB_USE_INTRINSIC_GUESS
			else:
				zhangzhengyou_flags = None
			caliberate_method = CALIBERATION_METHODS.ZHANGZHENGYOU
		elif args.caliberate_method.upper() == "EASYHEC":
			caliberate_method = CALIBERATION_METHODS.EASYHEC
		else:
			raise NotImplementedError(
				f"Caliberation method {args.caliberate_method} is not implemented yet."
			)
		with time_block("Perspective-n-Points routine for camera caliberation procedure. ", logger):
			(
				avg_trans_err,
				avg_rot_err,
				avg_reproj_err,
				_,
			) = caliberate_camera(
				image_objs=self.img_objs,
				json_objs=self.json_objs,
				output_dir=output_dir,
				num_manipulator_keypoints=args.num_keypoints,
				keypoint_ids=args.keypoint_ids,
				sliding_window_step=sliding_window_step,
				verbose=args.verbose,
				caliberate_method=caliberate_method,
				var_x=args.var_x,
				var_y=args.var_y,
				render_open3d=False,
				pnp_refinement=args.pnp_refinement,
				pnp_flag=args.pnp_flag,
				use_pnp_ransac=args.use_pnp_ransac,
				has_gt=args.has_gt,
				return_sample_result=True,
				zhangzhengyou_flags=zhangzhengyou_flags,
				nvdiffrast_render=False,
				urdf_path=args.urdf_path
			)
		if args.verbose and args.has_gt:
			logger.info(
				f"avg-trans-err:{avg_trans_err}, avg-rot-err:{avg_rot_err}, avg-rproj-err:{avg_reproj_err}"
			)


if __name__ == "__main__":
	args = parse_easycalib_default_args(cli_input=True)
	kalib = Kalib(args)
	calibrated = kalib.preprocess(SAVE_TAG)
	if not calibrated:
		kalib.kpt_tracking()
		kalib.run_caliberate_camera()
	kalib.post_process()

