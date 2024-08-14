import subprocess
import cv2
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv
import pyrfuniverse.attributes as attr
import os
import sys
import numpy as np
import pybullet as p
import pybullet_data
from tqdm import tqdm

import torch
torch.multiprocessing.set_start_method('spawn', force=True)
from easycalib.utils.utilities import overlay_mask_on_img, compute_forward_kinematics, render_mask, suppress_stdout, nostdout, time_block, namespace_to_dict, is_jsonable
from easycalib.utils.utilities import (
    extract_keypoints_from_panda_positions,
    in_cubic_coordinate,
    image_generation_context,
    signal_handler,
    set_random_seed
)
from scipy.spatial.transform import Rotation as R
from easycalib.lib.geometric_vision import left_handed_coord2right_handed_coord, unity_left_handed_to_right_coord, generate_transform_matrices
import os.path as osp
import math
import json
from datetime import datetime
from sim_franka_config import (
    HEIGHT,
    WIDTH,
    FOV,
    CAMERA_INTRINSICS,
    TIME_STEPS,
    MOVE_SPEED,
    TIME_FREQ,
    sim_franka_default_config,
)
from easycalib.utils.setup_logger import setup_logger
from multiprocessing import Pool
logger = setup_logger(__name__)
import signal


import numpy as np


def generate_camera_positions(radius, n_positions):
    # Angles for circular positioning around the origin
    angles = np.linspace(0, 2 * np.pi, n_positions, endpoint=False)

    camera_positions = []
    eye_targets = []
    # ys = [0.2,0.4,0.8,1.0]
    # eye_tar_ys = [0.9,0.7,0.4,0.2]
    ys = [-0.4, -0.2, 0.6, 1.2, 0.4]
    eye_tar_ys = [0.55, 0.55, 0.4, 0.2, 0.7]

    for angle in angles:
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)

        for ind, (y, eye_tar_y) in enumerate(zip(ys, eye_tar_ys)):
            position = np.array([x, y, z])
            eye_target = np.array([0.0, eye_tar_y, 0.65])
            camera_positions.append(position)
            eye_targets.append(eye_target)

    return camera_positions, eye_targets


GOOD_CAMERA_EYE_TARGETS = np.array(
    [[0.3, 0.2, 0.4], [0.3, 0.5, 0.4], [0.3, 0.2, 0.4], [0.3, 0.9, 0.4], [0.3, 0.4, 0.4], [0.3, 0.4, 0.4], [0.3, 0.4, 0.4], [0.3, 0.4, 0.4], [0.3, 0.6, 0.4], [0.3, 0.2, 0.4], [0.3, 0.8, 0.4], [0.2, 0.4, 0.4], [0.2, 0.2, 0.4], [0.1, 0.4, 0.4], [0.3, 0.3, 0.4], [0.3, 0.9, 0.4], [0.3, 0.3, 0.4], [0.3, 0.3, 0.4], [0.3, 0.5, 0.4], [0.3, 0.5, 0.4]])
GOOD_CAMERA_POSITIONS = np.array(
    [[1.2, 1.0, 0.4], [1.2, 0.1, 0.4], [1.9, 1.1, 1.0], [1.9, 0.2, 1.0], [0.8, 0.6, 1.0], [1.0, 0.6, 0.45], [1.0, 0.6, 0.1], [1.1, 1.3, -0.1], [1.1, 0.2, -0.1], [0.6, 1.1, -0.6], [0.6, 0.2, -0.6], [-0.5, 0.9, 0.5], [-0.5, 1.4, 0.5], [0.0, -0.4, 0.4], [0.5, 1.1, 1.2], [0.5, 0.2, 1.2], [-0.2, 1.1, 1.2], [-0.5, 1.1, 1.0], [1.0, 0.8, 0.5], [1.4, 0.8, 0.3]]
)
logger.info(f"GOOD_CAMERA_EYE_TARGETS:{GOOD_CAMERA_EYE_TARGETS.shape},GOOD_CAMERA_POSITIONS:{GOOD_CAMERA_POSITIONS.shape}")


def cli(camera_position=None, camera_lookAt_target=None, save_name=None, rfu_port=5004):
    args, ROBOT_DATA = sim_franka_default_config(save_name)
    logger.info(f"URDF file: {args.urdf_path}, Mesh files: {args.mesh_paths}")
    args.data_save_path = save_name
    args.debug_save_path = os.path.join(args.data_save_path, "debug")
    paths = {'data_save_path': args.data_save_path, 'debug_save_path': args.debug_save_path}

    with image_generation_context(logger, paths):
        try:
            os.makedirs(args.data_save_path, exist_ok=True)
            os.makedirs(args.debug_save_path, exist_ok=True)

            gt_local_to_world_matrices = []
            gt_3d_points = []
            gt_2d_projections = []

            # ! load into panda.urdf files.
            env = RFUniverseBaseEnv(assets=["franka_panda"], port=rfu_port)
            env.SetTimeStep(0.005)
            franka = env.LoadURDF(
                path=args.urdf_path,
                axis="y",
                native_ik=True,
                id=12345678,
            )
            # franka = env.InstanceObject(name="franka_panda",id=123456,attr_type=attr.ControllerAttr)
            franka.EnabledNativeIK(True)
            franka.SetIKTargetOffset(position=[0, 0, 0])

            env.step()
            gripper = env.GetAttr(12345678)
            gripper.GripperClose()
            env.step()

            camera = env.InstanceObject(name="Camera", id=1299456, attr_type=attr.CameraAttr)
            camera.SetTransform(position=camera_position, rotation=[0, 0, 0])  # rotation has no effect since camera.LookAT is called after.
            camera.LookAt(target=camera_lookAt_target)
            env.step()

            S_w = np.array([
                [0, 0, 1], [-1, 0, 0], [0, 1, 0]
            ])
            rCameraPosition = S_w @ np.asarray(camera_position)
            rCameraLookAtTarget = S_w @ np.asarray(camera_lookAt_target)
            pro_vec = p.computeViewMatrix(rCameraPosition, rCameraLookAtTarget, [0, 0, 1])
            pro_mat = np.array(pro_vec).reshape([4, 4])
            pCameraExtrinsics = pro_mat.transpose()  # !output is a world_to_local_matrix
            opengl_2_opencv = np.array([
                [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]
            ])
            pCameraExtrinsics = opengl_2_opencv @ pCameraExtrinsics

            local_to_world_matrix = np.array(camera.data["local_to_world_matrix"])

            cur_position = prev_position = np.asarray([0.0, 0.5, 0.6])
            vert_ind = 0

            basic_motion_vec = np.asarray([0, 0, 0])
            x_max, x_min = 0.7, -0.7
            y_max, y_min = 0.8, 0.3
            z_max, z_min = 0.9, 0.4
            cubic_coordinate_list = np.zeros((8, 3))

            cubic_coordinate_list[:4, 0] = x_max
            cubic_coordinate_list[4:, 0] = x_min

            cubic_coordinate_list[:2, 1] = cubic_coordinate_list[4:6, 1] = y_min
            cubic_coordinate_list[2:4, 1] = cubic_coordinate_list[6:8, 1] = y_max

            cubic_coordinate_list[::2, 2] = z_min
            cubic_coordinate_list[1::2, 2] = z_max

            tar_vert = cubic_coordinate_list[vert_ind]

            for i in range(-2, TIME_STEPS):
                env.step()

                if i % TIME_FREQ == 0:
                    vert_ind = np.random.randint(0, cubic_coordinate_list.shape[0])

                basic_motion_vec = tar_vert - cur_position

                if np.linalg.norm(basic_motion_vec) <= 0.5 * MOVE_SPEED:
                    vert_ind = (vert_ind + 1) % cubic_coordinate_list.shape[0]
                    tar_vert = cubic_coordinate_list[vert_ind]
                    basic_motion_vec = tar_vert - cur_position

                basic_motion_vec = (
                    MOVE_SPEED * basic_motion_vec / np.sqrt(np.sum(basic_motion_vec**2))
                )

                cur_position = prev_position + basic_motion_vec
                prev_position = cur_position

                # logger.info(f"The robot arm is to be moved to {cur_position}")
                franka.IKTargetDoMove(
                    position=cur_position,
                    duration=0.001,
                    speed_based=False,
                )
                franka.WaitDo()
                env.step()
                camera.GetRGB(width=WIDTH, height=HEIGHT, fov=FOV)

                env.step()
                if i < 0:
                    continue

                image = np.frombuffer(camera.data["rgb"], dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join(args.data_save_path, f"{i:06d}.rgb.jpg"), image)
                cv2.imshow("raw_image", image)
                cv2.waitKey(2)

                # continue

                local_to_world_matrix = np.array(camera.data["local_to_world_matrix"])
                # print(local_to_world_matrix)
                world_to_local_matrix = np.linalg.inv(local_to_world_matrix)

                joint_positions = np.array(
                    extract_keypoints_from_panda_positions(franka.data["positions"])
                )  # n x 3
                # logger.debug(f"The length of joint_positions are {len(joint_positions)}, the joint positions are \n{joint_positions}")
                joint_positions = np.concatenate(
                    [joint_positions, np.ones_like(joint_positions[..., :1])], axis=-1
                )  # n x 4
                joint_positions = joint_positions.transpose()  # 4 x n
                joint_proj = world_to_local_matrix[:3] @ joint_positions  # 3 x n

                joint_proj[1, :] *= -1
                joint_proj = CAMERA_INTRINSICS.float() @ joint_proj  # 3 x n
                joint_proj = joint_proj.permute(1, 0)  # n x 3 (n = 9)

                robot_joints = joint_proj
                robot_joints = robot_joints / robot_joints[:, 2:3]
                robot_joints = robot_joints[:, :2]

                ROBOT_DATA["objects"][0]["location"] = robot_joints[0].tolist()
                ROBOT_DATA["objects"][0]["camera_intrinsics"] = CAMERA_INTRINSICS.tolist()
                ROBOT_DATA["objects"][0]["local_to_world_matrix"] = (
                    pCameraExtrinsics.tolist()
                )
                lCartesian_positions = np.asarray(extract_keypoints_from_panda_positions(franka.data["positions"]))
                rCartesian_positions = (S_w @ lCartesian_positions.T).T
                rCartesian_positions = rCartesian_positions.tolist()
                ROBOT_DATA["objects"][0]["eef_pos"] = rCartesian_positions
                ROBOT_DATA["objects"][0]["joint_positions"] = np.radians(np.asarray(franka.data["joint_positions"])).tolist()
                ROBOT_DATA["objects"][0]["cartesian_position"] = ROBOT_DATA["objects"][0]["eef_pos"]
                ROBOT_DATA["objects"][0]["sim_camera_positions"] = rCameraPosition.tolist()
                ROBOT_DATA["objects"][0]["sim_camera_eye_target"] = rCameraLookAtTarget.tolist()

                # ! calculate 3d points and 2d projection points and calculate pnp solutions.

                gt_2d_projections.append(robot_joints[None])
                gt_local_to_world_matrices.append(local_to_world_matrix[None])

                for ptidx, pt in enumerate(robot_joints):
                    color = (0, 255, 0)
                    image = cv2.circle(
                        image,
                        (int(pt[0]), int(pt[1])),
                        radius=3,
                        color=color,
                        thickness=-1,
                    )
                    if ptidx >= len(ROBOT_DATA["objects"][0]["keypoints"]):
                        ROBOT_DATA["objects"][0]["keypoints"].append({"location": None, "name": None, "projected_location": None, "predicted_location": None})
                    ROBOT_DATA["objects"][0]["keypoints"][ptidx]["location"] = (
                        rCartesian_positions[ptidx]
                    )
                    ROBOT_DATA["objects"][0]["keypoints"][ptidx]["name"] = args.keypoint_friendly_names[ptidx]
                    ROBOT_DATA["objects"][0]["keypoints"][ptidx]["projected_location"] = (
                        pt.tolist()
                    )
                    ROBOT_DATA["objects"][0]["keypoints"][ptidx]["predicted_location"] = (
                        pt.tolist()
                    )

                if args.debug:
                    cv2.imwrite(os.path.join(args.debug_save_path, f"debug_{i:06d}.png"), image)
                    cv2.imshow("debug_image", image)
                    cv2.waitKey(20)

                with open(os.path.join(args.data_save_path, f"{i:06d}.json"), "w") as json_file:
                    json.dump(ROBOT_DATA, json_file, indent=4)

                # joint_positions = [math.radians(position) for position in franka.data["joint_positions"]]

                # gt_mask = render_mask(args.urdf_path, args.mesh_paths, pCameraExtrinsics, np.array(CAMERA_INTRINSICS), HEIGHT, WIDTH, joint_positions)
                # cv2.imshow("image", (gt_mask*255).astype(np.uint8))
                # cv2.waitKey(10)

            # gt_3d_points = np.concatenate(gt_3d_points, axis=0)
            # gt_2d_projections = np.concatenate(gt_2d_projections, axis=0)
            # if args.debug:
            # 	np.save(os.path.join(args.debug_save_path, "gt_3d_points"), gt_3d_points)
            # 	np.save(os.path.join(args.debug_save_path, "gt_2d_projections"), gt_2d_projections)
            # 	np.save(
            # 		os.path.join(args.debug_save_path, "gt_local_to_world_matrices"),
            # 		gt_local_to_world_matrices,
            # 	)

            os_env = os.environ.copy()
            ffmpeg_comma = f"ffmpeg -framerate 10 -pattern_type glob -i '{os.path.join(os.getcwd(), args.data_save_path)}/00*.rgb.jpg' -c:v libx264 -pix_fmt yuv420p {os.path.join(os.getcwd(), args.data_save_path)}/outputs.mp4"
            logger.info(ffmpeg_comma)
            process = subprocess.run(
                ffmpeg_comma,
                shell=True,
                env=os_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            env.step()

        except KeyboardInterrupt:
            logger.info("Caught KeyboardInterrupt during image generation")
            raise
        # except Exception as e:
        # 	logger.info(str(e))
        # 	raise


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)

    # SAVE_NAMES = [datetime.now().strftime("%m_%d_%H_%M_%S")] * len(GOOD_CAMERA_EYE_TARGETS)
    # SAVE_NAMES = ["./dataset/rfu/gaussian" + save_name for save_name in SAVE_NAMES]

    # tasks = [(cam_pos.tolist(), cam_eye_target.tolist(), save_name, 5004+i) for i, (cam_pos, cam_eye_target, save_name) in enumerate(zip(GOOD_CAMERA_POSITIONS, GOOD_CAMERA_EYE_TARGETS, SAVE_NAMES))]
    # logger.debug(f"Use multiprocessing.Pool to parallel mask rendering routine.")

    # with time_block("Finish mask rendering routine. ", logger):
    # 	with Pool(processes=8) as pool:
    # 		pool.starmap(cli, tasks)
    cam_poss, cam_eye_targets = generate_camera_positions(radius=1.6, n_positions=5)
    print(cam_poss, cam_eye_targets)

    for ind in range(len(cam_eye_targets)):
        save_name = datetime.now().strftime("%m_%d_%H_%M_%S")
        save_name = "./dataset/rfu/gaussian" + save_name
        camera_position = cam_poss[ind]
        camera_eye_target = cam_eye_targets[ind]
        set_random_seed(2024 + ind)
        cli(save_name=save_name, camera_position=camera_position.tolist(), camera_lookAt_target=camera_eye_target.tolist(), rfu_port=5004 + ind)
        # sys.exit(0)
