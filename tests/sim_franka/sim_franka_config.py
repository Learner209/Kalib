import os
import sys
import numpy as np
import argparse
from easycalib.utils.utilities import (
    calculate_camera_intrinsics,
    calculate_camera_intrinsics_given_fov_x,
    calculate_camera_intrinsics_given_fov_y,
    extract_keypoints_from_panda_positions,
    in_cubic_coordinate,
)
import json

np.random.seed(2024)


HEIGHT = 1080
WIDTH = 1920
FOV = 60
# CAMERA_INTRINSICS = calculate_camera_intrinsics(FOV, FOV, WIDTH, HEIGHT)
# CAMERA_INTRINSICS = calculate_camera_intrinsics_given_fov_x(FOV, WIDTH, HEIGHT)
CAMERA_INTRINSICS = calculate_camera_intrinsics_given_fov_y(FOV, WIDTH, HEIGHT)

TIME_STEPS = 300
TIME_FREQ = 30
MOVE_SPEED = 0.05


def sim_franka_default_config(save_name=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="debug option using debugpy"
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="./easycalib/config/template_config.json",
        help="path to the config file",
    )
    parser.add_argument(
        "--manipulator_config_path",
        type=str,
        default="./easycalib/config/franka_config.json",
    )
    args = parser.parse_args()
    with open(args.config_path) as f:
        ROBOT_DATA = json.load(f)

    # ! load .yaml manipulator config files.
    assert os.path.exists(args.manipulator_config_path)
    with open(args.manipulator_config_path, "r") as file:
        config = json.load(file)

        # Extract manipulator name
        manipulator_name = config["manipulator"]["name"]

        urdf_path = config["manipulator"]["urdf_path"]
        keypoints = config["manipulator"]["keypoints"]
        mesh_paths = config["manipulator"]["mesh_paths"]
        num_keypoints = len(keypoints)

        # Add extracted data to args
        args.urdf_path = urdf_path
        args.mesh_paths = mesh_paths
        args.manipulator_name = manipulator_name
        args.keypoint_friendly_names = keypoints
        args.num_keypoints = num_keypoints
        # print(keypoints, urdf_path, mesh_paths)

    args.data_save_path = os.path.join("./saved_frames", save_name)
    os.makedirs(args.data_save_path, exist_ok=True)
    os.makedirs(os.path.join(args.data_save_path, "debug"), exist_ok=True)
    args.debug_save_path = os.path.join(args.data_save_path, "debug")
    return args, ROBOT_DATA
