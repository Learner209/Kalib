import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import warnings
# from easycalib_demo_browser import EasyCalibServerWrapper
from easycalib_demo_client import EasyCalibClientWrapper
from easycalib.config.parse_argument import parse_easycalib_default_args
from easycalib.utils.point_drawer import PointDrawer
from easycalib.utils.utilities import overlay_mask_on_img, compute_forward_kinematics, render_mask, merge_args
from easycalib.utils.make_plot.good_takes import SIM_GOOD_TAKE_PATHS, DROID_GOOD_TAKE_PATHS
from glob import glob
from scipy.spatial.transform import Rotation as R
from third_party.droid.droid.data_processing.timestep_processing import TimestepProcesser
from third_party.droid.droid.trajectory_utils.misc import load_trajectory
from third_party.droid.droid.data_loading.trajectory_sampler import crawler
from tqdm import tqdm
import argparse


import imageio
import json
import os.path as osp
import shutil
import sys
import numpy as np
import torch
from tqdm import tqdm
from easycalib.utils.setup_logger import setup_logger
logger = setup_logger(__name__)

CONFIGURED_CAMERAS = [
    "varied_camera_1_left",
    "varied_camera_1_right",
    "varied_camera_2_left",
    "varied_camera_2_right",
]
# region


class TrajectorySampler:
    def __init__(
            self,
            all_folderpaths,
            recording_prefix="",
            traj_loading_kwargs={},
            timestep_filtering_kwargs={},
            image_transform_kwargs={},
            camera_kwargs={},
    ):
        self._all_folderpaths = all_folderpaths
        self.recording_prefix = recording_prefix
        self.traj_loading_kwargs = traj_loading_kwargs
        self.timestep_processer = TimestepProcesser(
            **timestep_filtering_kwargs, image_transform_kwargs=image_transform_kwargs
        )
        self.camera_kwargs = camera_kwargs

    def fetch_samples(self, traj_ind: int = None):
        folderpath = self._all_folderpaths[traj_ind]

        filepath = os.path.join(folderpath, "trajectory.h5")
        recording_folderpath = os.path.join(folderpath, "recordings", self.recording_prefix)
        if not os.path.exists(recording_folderpath):
            recording_folderpath = None

        traj_samples = load_trajectory(
            filepath,
            recording_folderpath=recording_folderpath,
            camera_kwargs=self.camera_kwargs,
            **self.traj_loading_kwargs,
        )

        processed_traj_samples = [self.timestep_processer.forward(t) for t in traj_samples]

        return processed_traj_samples, folderpath
# endregion

# region


class TrajectoryDataset():
    def __init__(self, trajectory_sampler, all_folderpaths):
        self._trajectory_sampler = trajectory_sampler
        self._trajectory_len = len(all_folderpaths)
        self._trajectory_cnt = 0

    def _refresh_generator(self):
        # Check if the trajectory count has reached the limit
        if self._trajectory_cnt >= self._trajectory_len:
            raise StopIteration

        # Fetch new samples and increment the trajectory counter
        new_samples, folder_path = self._trajectory_sampler.fetch_samples(traj_ind=self._trajectory_cnt)
        print("Examining the folderpath: ", folder_path)
        self._trajectory_cnt += 1

        # Return the new samples without creating a separate generator
        return folder_path, new_samples

    def __iter__(self):
        return self

    def __next__(self):
        # This will fetch and return the entire batch of samples from _refresh_generator
        if self._trajectory_cnt < self._trajectory_len:
            return self._refresh_generator()
        else:
            raise StopIteration
# endregion

# region


def convert_raw_extrinsics_to_mat(raw_data):
    # Preprocessing of the extrinsics
    pos = raw_data[0:3]
    rot_mat = R.from_euler("xyz", raw_data[3:6]).as_matrix()

    raw_data = np.zeros((4, 4))
    raw_data[:3, :3] = rot_mat
    raw_data[:3, 3] = pos
    raw_data[3, 3] = 1.0
    raw_data = np.linalg.inv(raw_data)
    return raw_data
# endregion


def forward(processed_timestep):
    extrinsics_dict = processed_timestep["extrinsics_dict"]
    intrinsics_dict = processed_timestep["intrinsics_dict"]
    # import pdb; pdb.set_trace()
    # processed_timestep["observation"] contains keys: 'cartesian_position', 'gripper_position', 'joint_positions', 'joint_torques_computed', 'joint_velocities', 'motor_torques_measured', 'prev_command_successful', 'prev_controller_latency_ms', 'prev_joint_torques_computed', 'prev_joint_torques_computed_safened'

    obs = {
        "robot_state/cartesian_position": processed_timestep["observation"]["robot_state"]["cartesian_position"][:3],
        "robot_state/joint_positions": processed_timestep["observation"]["robot_state"]["joint_positions"],
        "robot_state/gripper_position": [
            processed_timestep["observation"]["robot_state"]["gripper_position"]
        ],  # wrap as array, raw data is single float
        "camera/img/varied_camera_1_left_img": processed_timestep["observation"]["camera"]["image"]["varied_camera"][0],
        "camera/img/varied_camera_1_right_img": processed_timestep["observation"]["camera"]["image"]["varied_camera"][1],
            "camera/img/varied_camera_2_left_img": processed_timestep["observation"]["camera"]["image"]["varied_camera"][2],
            "camera/img/varied_camera_2_right_img": processed_timestep["observation"]["camera"]["image"]["varied_camera"][3],
            "camera/extrinsics/varied_camera_1_left": convert_raw_extrinsics_to_mat(extrinsics_dict["varied_camera"][0]),
            "camera/extrinsics/varied_camera_1_right": convert_raw_extrinsics_to_mat(extrinsics_dict["varied_camera"][1]),
            "camera/extrinsics/varied_camera_2_left": convert_raw_extrinsics_to_mat(extrinsics_dict["varied_camera"][2]),
            "camera/extrinsics/varied_camera_2_right": convert_raw_extrinsics_to_mat(extrinsics_dict["varied_camera"][3]),
            "camera/intrinsics/varied_camera_1_left": intrinsics_dict["varied_camera"][0],
            "camera/intrinsics/varied_camera_1_right": intrinsics_dict["varied_camera"][1],
            "camera/intrinsics/varied_camera_2_left": intrinsics_dict["varied_camera"][2],
            "camera/intrinsics/varied_camera_2_right": intrinsics_dict["varied_camera"][3],
    }

    # set item of obs as np.array
    for k in obs:
        obs[k] = np.array(obs[k])

    return obs

# region


def parse_local_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--send_to_remote_machine", action="store_true", help="Send the generated frames to remote machine"
    )
    parser.add_argument(
        "--config_path", type=str, default="./easycalib/config/template_config.json", help="path to the config file"
    )
    parser.add_argument(
        "--annotate_whole_dataset", action="store_true", help="whether to annotate whole dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="the root dir of the dataset, will iteratively find sub-folder in it that contains any valid .svo camera recording file(s).",
    )
    parser.add_argument(
        "--data_save_path",
        type=str,
        help="",
    )
    # debugging purposes
    parser.add_argument(
        "--debug", action="store_true", help="whether to use pdb.set_trace() for debugging"
    )
    default_args = parser.parse_args()
    override_args = parse_easycalib_default_args(cli_input=False)
    args = merge_args(default_args, override_args)
    return args
# endregion

# region


def extract_droid_data_and_save_to_disk(args):

    all_folderpaths = crawler(args.data_dir)

    traj_sampler = TrajectorySampler(all_folderpaths, recording_prefix="SVO", timestep_filtering_kwargs={"gripper_action_space": ["cartesian_position"]})
    traj_dataset = TrajectoryDataset(traj_sampler, all_folderpaths)
    data_iterator = iter(traj_dataset)

    # Forward every sub-dataset in the collected data folder.
    while True:
        try:
            folder_path, timesteps = next(data_iterator)

            data_save_path = osp.join(args.data_save_path, folder_path)
            logger.info(f"folder_path: {folder_path}")
            os.makedirs(data_save_path, exist_ok=True)

            save_paths = {key: osp.join(data_save_path, key) for key in CONFIGURED_CAMERAS}
            save_imgs = {key: [] for key in CONFIGURED_CAMERAS}
            debug_save_imgs = {key: [] for key in CONFIGURED_CAMERAS}

            logger.info(f"Saving frames to {save_paths}")

            flag = 0
            for save_path in save_paths.values():
                if osp.exists(save_path):
                    flag += 1
                os.makedirs(save_path, exist_ok=True)
                os.makedirs(osp.join(save_path, "debug"), exist_ok=True)
            if flag == len(save_paths):
                logger.info("All save paths for %s exist, skip." % folder_path)
                continue

            # Forward every timestep in the sub-dataset.
            for ind, processed_timestep in tqdm(enumerate(timesteps), desc=f"Processing {folder_path}"):
                # if ind == 1:
                # break
                processed_timestep = forward(processed_timestep)

                joint_positions = processed_timestep["robot_state/joint_positions"]
                raw_eef_pos = processed_timestep["robot_state/cartesian_position"]
                eefpos_from_joint_positions = compute_forward_kinematics(
                    urdf_file=args.urdf_path, joint_positions=joint_positions
                )
                eefpos_from_joint_positions = eefpos_from_joint_positions[-1]
                eefpos = np.array(eefpos_from_joint_positions)
                robot_joints = eefpos[None]

                # Forward every camera: available options: 'varied_camera_1_left', 'varied_camera_1_right', 'varied_camera_2_left', 'varied_camera_2_right'
                for camera in CONFIGURED_CAMERAS:
                    camera_intrinsics = processed_timestep["camera/intrinsics/" + camera]
                    camera_extrinsics = processed_timestep["camera/extrinsics/" + camera]

                    projected_eefpos = np.hstack([eefpos, 1])
                    projected_eefpos = np.dot(camera_extrinsics, projected_eefpos)
                    projected_eefpos = np.dot(camera_intrinsics, projected_eefpos[:3])
                    projected_eefpos = projected_eefpos[:2] / projected_eefpos[2]

                    projected_raw_eefpos = np.hstack([raw_eef_pos, 1])
                    projected_raw_eefpos = np.dot(camera_extrinsics, projected_raw_eefpos)
                    projected_raw_eefpos = np.dot(camera_intrinsics, projected_raw_eefpos[:3])
                    projected_raw_eefpos = projected_raw_eefpos[:2] / projected_raw_eefpos[2]

                    save_imgs[camera] = processed_timestep["camera/img/" + camera + "_img"]
                    debug_img = np.copy(save_imgs[camera])
                    debug_img = cv2.circle(debug_img, tuple(projected_eefpos.astype(int)), 5, (0, 0, 255), -1)
                    debug_img = cv2.circle(debug_img, tuple(projected_raw_eefpos.astype(int)), 5, (0, 255, 255), -1)

                    debug_save_imgs[camera] = debug_img

                    cv2.imwrite(osp.join(save_paths[camera], f"{ind:06d}_{camera}.png"), save_imgs[camera])
                    cv2.imwrite(
                        osp.join(save_paths[camera], "debug", f"{ind:06d}_{camera}.png"), debug_save_imgs[camera]
                    )

                    ROBOT_DATA["objects"][0]["eef_pos"] = eefpos.tolist()
                    ROBOT_DATA["objects"][0]["camera_intrinsics"] = camera_intrinsics.tolist()
                    ROBOT_DATA["objects"][0]["local_to_world_matrix"] = camera_extrinsics.tolist()
                    ROBOT_DATA["objects"][0]["joint_positions"] = processed_timestep[
                        "robot_state/joint_positions"
                    ].tolist()
                    ROBOT_DATA["objects"][0]["cartesian_position"] = processed_timestep[
                        "robot_state/cartesian_position"
                    ].tolist()

                    for ptidx, pt in enumerate(robot_joints):
                        ROBOT_DATA["objects"][0]["keypoints"][ptidx]["location"] = eefpos.tolist()
                        ROBOT_DATA["objects"][0]["keypoints"][ptidx]["name"] = "panda_left_finger"
                        ROBOT_DATA["objects"][0]["keypoints"][ptidx]["projected_location"] = projected_eefpos.tolist()
                        ROBOT_DATA["objects"][0]["keypoints"][ptidx]["predicted_location"] = [-999.0, -999.0]

                    json.dump(ROBOT_DATA, open(osp.join(save_paths[camera], f"{ind:06d}.json"), "w"))

        except StopIteration:
            logger.info("End of dataset")
            break
        except Exception as e:
            logger.error(e)
# endregion


def render_debug_mask_for_droid_dataset(args):

    all_folderpaths = crawler(args.data_dir)

    for folder_path in all_folderpaths:

        data_save_paths = osp.join(args.data_save_path, folder_path)
        render_mask_save_path = osp.join(data_save_paths, "rendered_mask_debug")
        os.makedirs(render_mask_save_path, exist_ok=True)
        data_save_paths = {key: osp.join(data_save_paths, key) for key in CONFIGURED_CAMERAS}
        filter_key = CONFIGURED_CAMERAS[0]
        found_image_paths, _ = EasyCalibClientWrapper.find_data_in_dir(
            data_save_paths[filter_key], -1
        )
        for ind in tqdm(range(len(found_image_paths)), desc=f"Processing {folder_path}"):

            robot_data = json.load(open(osp.join(data_save_paths[filter_key], f"{ind:06d}.json"), "r"))
            cam_img = cv2.imread(osp.join(data_save_paths[filter_key], f"{ind:06d}_{filter_key}.png"))

            gt_local_to_world_matrix = np.array(robot_data["objects"][0]["local_to_world_matrix"])
            camera_K = np.array(robot_data["objects"][0]["camera_intrinsics"])

            H, W = cam_img.shape[:2]
            pose = robot_data["objects"][0]["joint_positions"]

            gt_mask = render_mask(args.urdf_path, args.mesh_paths, gt_local_to_world_matrix, camera_K, H, W, pose)
            pred_mask = render_mask(args.urdf_path, args.mesh_paths, gt_local_to_world_matrix, camera_K, H, W, pose)
            overlay_img = overlay_mask_on_img(
                cam_img,
                gt_mask,
                pred_mask,
                rgb1=(255, 0, 255),
                rgb2=(0, 255, 255),
                alpha=0.5,
                show=False,
                save_to_disk=False,
                img_save_path=None,
            )
            cv2.imwrite(osp.join(render_mask_save_path, f"{ind:06d}.png"), overlay_img)


def filter_out_droid_dataset(args):

    all_folderpaths = crawler(args.data_dir)

    for folder_path in all_folderpaths:

        data_save_paths = osp.join(args.data_save_path, folder_path)
        render_mask_save_path = osp.join(data_save_paths, "rendered_mask_debug")
        # data_save_paths = {key: osp.join(data_save_paths, key) for key in CONFIGURED_CAMERAS}
        # filter_key = CONFIGURED_CAMERAS[0]
        # found_image_paths, _ = EasyCalibClientWrapper.find_data_in_dir(
        #     render_mask_save_path
        # )
        if not osp.exists(render_mask_save_path):
            raise Exception("No valid path, trying to find: %s" % render_mask_save_path)

        rendered_mask_debug_imgs = glob(osp.join(render_mask_save_path, "*.png"))

        if len(rendered_mask_debug_imgs) <= 0:
            raise Exception("No rendered mask debug images found in the folder %s" % render_mask_save_path)
        for img_path in rendered_mask_debug_imgs:

            cam_img = cv2.imread(img_path)

            cv2.imshow(
                "%s" % render_mask_save_path, cam_img
            )
            key = cv2.waitKey(1)
            if key == ord("b"):
                break

        c = input("Whether to remove folder %s? (y/n)" % folder_path)
        if c == "y":
            shutil.rmtree(folder_path)
        cv2.destroyAllWindows()


def annotate_gt_mask_with_sam(local_arg, takes=None):

    if takes is None:
        takes = crawler(local_arg.data_dir)

    def _annotate_gt_mask_with_sam(local_arg, first_frame_save_path, pred_sam_mask_save_path):
        rgb = imageio.imread_v2(first_frame_save_path)
        pointdrawer = PointDrawer(
            sam_checkpoint=local_arg.sam_checkpoint_path,
            sam_model_type=local_arg.sam_type,
            window_name="First Frame Mask segmentation",
        )
        rgb = rgb[..., :3]
        _, _, mask = pointdrawer.run(rgb)

        cv2.imwrite(pred_sam_mask_save_path, (mask * 255).astype(np.uint8))

    for folder_path in takes:
        logger.info("Procesing folderpaths: %s" % folder_path)
        img_objs, json_objs, found_image_paths, found_json_paths = EasyCalibClientWrapper.find_data_in_dir(
            folder_path, -1
        )
        first_frame_path = found_image_paths[0]
        first_frame_basename = osp.basename(first_frame_path)
        os.makedirs(osp.join(folder_path, "labeled_mask"), exist_ok=True)
        pred_sam_mask_save_path = osp.join(folder_path, "labeled_mask", "%s" % first_frame_basename)
        if not osp.exists(pred_sam_mask_save_path):
            _annotate_gt_mask_with_sam(local_arg, first_frame_path, pred_sam_mask_save_path)


if __name__ == "__main__":
    co_tracker_client_wrapper_args = parse_local_args()

    with open(co_tracker_client_wrapper_args.config_path) as f:
        ROBOT_DATA = json.load(f)

    CO_TRACKER_CLIENT_WRAPPER = EasyCalibClientWrapper(
        server_ip=co_tracker_client_wrapper_args.browser_ip,
        server_port=co_tracker_client_wrapper_args.browser_port,
        args=co_tracker_client_wrapper_args,
    )

    # Extract data from the dataset and save to disk: SAVE_PATHS is first initialized as a global variable.
    extract_droid_data_and_save_to_disk(co_tracker_client_wrapper_args)
    # render_debug_mask_for_droid_dataset(co_tracker_client_wrapper_args)
    # filter_out_droid_dataset(co_tracker_client_wrapper_args)
    # annotate_gt_mask_with_sam(co_tracker_client_wrapper_args, takes=DROID_GOOD_TAKE_PATHS)
