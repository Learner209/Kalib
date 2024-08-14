import argparse
from datetime import datetime
from easycalib.utils.utilities import str2bool
import cv2
import os
import json


def parse_easycalib_default_args(cli_input: bool = True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", default="./saved_frames", help="root directory where json files and rgb images reside."
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/cotracker2.pth",
        help="the path to cotracker ckpt",
    )
    parser.add_argument("--grid_size", type=int, default=100, help="Regular grid size in kpt-tracking module")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame (kpt_tracking module)",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward(kpt_tracking module)",
    )
    parser.add_argument(
        "--manipulator_config_path",
        type=str,
        default="easycalib/config/franka_config.json",
        help="config mesh paths, urdf paths, keypoints_names and manipulator name in this file."
    )
    # segment-anything related configuration
    parser.add_argument(
        "--use_sam",
        action="store_true",
        help="whether to use Segment-Anything for first frame mask segmentation",
    )
    parser.add_argument(
        "--sam_checkpoint_path",
        type=str,
        default="./pretrained_checkpoints/sam_vit_h_4b8939.pth",
        help="SAM checkpoint to inference the mask.",
    )
    parser.add_argument(
        "--sam_type",
        type=str,
        default="vit_h",
        help="whether to use Grounded-Segment-Anything model instead of SAM.",
    )
    # grounded_sam_model configuration
    parser.add_argument(
        "--use_grounded_sam",
        action="store_true",
        help="whether to use Grounded-Segment-Anything model instead of SAM. (exclusive with --use-sam option)",
    )
    parser.add_argument(
        "--grounded_sam_repo_path",
        type=str,
        default="./third_party/grounded_segment_anything",
        help="path to grounded-sam repository.",
    )
    parser.add_argument(
        "--easyhec_repo_path",
        type=str,
        default="./third_party/easyhec",
        help="path to easyhec repository.",
    )
    parser.add_argument(
        "--spatial_tracker_repo_path",
        type=str,
        default="./third_party/spatial_tracker",
        help="path to spatial tracker repo.",
    )
    parser.add_argument("--text_prompt", type=str, default="robot arm", help="the text prompt to grouded-sam for foreground mask inference.")
    parser.add_argument(
        "--grounded_sam_script", type=str, default="grounded_sam_demo.py"
    )
    parser.add_argument(
        "--grounded_sam_config",
        type=str,
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--grounded_sam_checkpoint_path", type=str, default="groundingdino_swint_ogc.pth"
    )

    # mask generating option
    parser.add_argument(
        "--save_model_inference",
        type=bool,
        default=True,
        help="whether to save kpt-tracking module inference outputs to predetermined path.",
    )
    parser.add_argument(
        "--model_inference_path",
        type=str,
        default="",
        help="Path to kpt-tracking module outputs path.",
    )
    parser.add_argument(
        "--use_segm_mask",
        type=str2bool,
        default=False,
        help="whether to use foreground mask in kpt-tracking module(only available in cotracker and spatial-tracker)",
    )
    # camera caliberation methods options
    parser.add_argument(
        "--caliberate_method",
        type=str,
        default="pnp",
        choices=["pnp", "zhangzhengyou", "easyhec"],
        help="caliberation method to use.",
    )
    parser.add_argument(
        "--win_len",
        type=int,
        default=1,
        help="sliding window approach is taken, so win_len corresponds to how many iterations are finally executed.",
    )
    parser.add_argument(
        "--var_x", type=int, default=0, help="variance of noise added to x axis."
    )
    parser.add_argument(
        "--var_y", type=int, default=0, help="variance of noise added to y axis."
    )
    parser.add_argument(
        "--pnp_refinement",
        type=str2bool,
        help="whether to use pnp_refinement refined on coarse estimation of camera extrinsics.",
    )
    parser.add_argument(
        "--pnp_flag",
        type=str,
        default="iterative",
        choices=["iterative", "epnp"],
        help="pnp flag method.",
    )
    parser.add_argument(
        "--use_pnp_ransac", type=str2bool, help="whether to use pnp_ransac algorithm"
    )
    parser.add_argument(
        "--intrinsics_guess",
        type=str2bool,
        default=True,
        help="whether to use intrinsics_guess in zhangzhengyou alg.",
    )
    # local configuration
    parser.add_argument("--verbose", action="store_true", help="output sys log")
    parser.add_argument(
        "--has_gt",
        action="store_true",
        help="has ground_truth labels for 2D predicted location.",
    )

    parser.add_argument(
        "--render_mask", action="store_true", help="render mask in both pred and gt."
    )
    parser.add_argument("--tracking_device_id", type=int, default=1, help="keypoint tracking device_id")
    parser.add_argument("--renderer_device_id", type=int, default=2, help="nvdiffrast renderer device_id")
    parser.add_argument("--mask_inference_device_id", type=int, default=1, help="mask inference(SAM\Grounded-SAM) device id")
    parser.add_argument("--cut_off", type=int, default=-1, help="cut_off for found images and jsons, to prevent cuda OOM.")
    parser.add_argument("--keypoint_ids", type=int, default=[8], nargs="+", help="the keypoints we want to track.")
    parser.add_argument("--skip_rendering", action="store_true", help="skip rendering ")
    parser.add_argument("--keypoint_tracking_method", type=str, choices=["cotracker", "spatial_tracker", "dino_tracker"], default="spatial_tracker", help="keypoint tracking methods")

    if cli_input:
        args = parser.parse_args()
    else:
        args = parser.parse_args([])

    args.pnp_flag = (
        cv2.SOLVEPNP_ITERATIVE if args.pnp_flag == "iterative" else cv2.SOLVEPNP_EPNP
    )

    # ! load .yaml manipulator config files.
    assert os.path.exists(args.manipulator_config_path)
    with open(args.manipulator_config_path, "r") as file:
        config = jsonsrc_calib.load(file)

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

    if hasattr(args, "saved_name"):
        current_time_formatted = datetime.now().strftime("'%Y-%m-%d-%H-%M-%S")
        args.saved_name = current_time_formatted

    return args