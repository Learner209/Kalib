<p align="center">
  <h3 align="center"><strong>Kalib: <br>Markerless</br> Hand-Eye Calibration with Keypoint Tracking</strong></h3>

<p align="center">
    <a href="https://github.com/ElectronicElephant">Tutian Tang</a><sup>1</sup>,
    <a href="https://github.com/Learner209">Minghao Liu</a><sup>1</sup>,
    <a href="https://wenqiangx.github.io/">Wenqiang Xu</a><sup>1</sup>,
    <a href="https://www.mvig.org/">CeWu Lu</a><sup>1</sup><span class="note">*</span>,
    <br>
    <br>
    <sup>*</sup>Corresponding authors.
    <br>
    <sup>1</sup>Shanghai Jiao Tong University
    <br>
</p>



<div align="center">

<img src="https://img.shields.io/badge/Python-v3-E97040?logo=python&logoColor=white" /> &nbsp;&nbsp;&nbsp;&nbsp;
<img alt="powered by Pytorch" src="https://img.shields.io/badge/PyTorch-❤️-F8C6B5?logo=pytorch&logoColor=white"> &nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://img.shields.io/badge/Conda-Supported-lightgreen?style=social&logo=anaconda" /> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://sites.google.com/view/hand-eye-kalib'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLearner209%2FKalib&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</div>

<!-- # **Kalib**: Markerless Hand-Eye Calibration with Keypoint Tracking

The official code repository for our paper: **Kalib: Markerless Hand-Eye Calibration with Keypoint Tracking.** -->
## ❖ Contents

- [Introduction](#introduction)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset Conventions](#-dataset-conventions)

## ❖ Introduction:

Hand-eye calibration involves estimating the transformation between the camera and the robot. Traditional methods rely on fiducial markers, involving much manual labor and careful setup.
Recent advancements in deep learning offer markerless techniques, but they present challenges, including the need for retraining networks for each robot, the requirement of accurate mesh models for data generation, and the need to address the sim-to-real gap.
In this letter, we propose **Kalib**, an automatic and universal markerless hand-eye calibration pipeline that leverages the generalizability of visual foundation models to eliminate these barriers.
In each calibration process, **Kalib** uses keypoint tracking and proprioceptive sensors to estimate the transformation between a robot's coordinate space and its corresponding points in camera space.
Our method does not require training new networks or access to mesh models. Through evaluations in simulation environments and the real-world dataset DROID, **Kalib** demonstrates superior accuracy compared to recent baseline methods.
This approach provides an effective and flexible calibration process for various robot systems by simplifying setup and removing dependency on precise physical markers.

🤗 Please cite [Kalib](https://github.com/Learner209/Kalib) in your publications if it helps with your work. Please star🌟 this repo to help others notice **Kalib** if you think it is useful. Thank you! 
😉

## ❖ Installation

We run on `Ubuntu 22.04 LTS` with a system configured with $2\times$ NVIDIA RTX A40 GPU.

1. Use conda to create a env for **Kalib** and activate it.

```bash
conda create -n kalib python==3.10
conda activate kalib
pip install -r requirements.txt
```

2. Download SAM checkpoints.
```bash
sam_ckpts_dir="./pretrained_checkpoints"
wget -P "$sam_ckpts_dir" "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
wget -P "$sam_ckpts_dir" "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
wget -P "$sam_ckpts_dir" "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
```
> Note: If you wanna specify the sam_checkpoints path(the default is *sam_vit_l*), plz modify default value in [sam_type](./easycalib/config/parse_demo_argument.py#L49) and [sam_checkpoint_path](./easycalib/config/parse_demo_argument.py#L43) or pass it as an argument.

3. Grounded-SAM installation.
Please refer to [Grounded-SAM readme](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/README.md).
Additionally, link the downloaded SAM checkpoints to Grounded-SAM root path as Grounded-SAM are using SAM ckpts too.

```bash
ln -sf $PWD/pretrained_checkpoints ./third_party/grounded_segment_anything/
```

4. Install spatial tracker.

Please refer to [Spatracker readme](https://github.com/henry123-boy/SpaTracker/blob/main/README.md)

Additionally, download the `SpaT_final.pth` checkpoints into `./third_party/spatial_tracker/checkpoints/` directory.

5. (Optional) Install cotracker.
Please refer to [CoTracker readme](https://github.com/facebookresearch/co-tracker/blob/main/README.md)

6. Install **Kalib** as a package.
```bash
cd <your-project-root-directory>
pip install -e .
```

## ❖ Usage

1. Running the camera calibration pipeline.
```bash
easyhec_repo_path=$PWD/third_party/easyhec/
grounded_sam_repo_path=$PWD/third_party/grounded_segment_anything/
spatial_tracker_repo_path=$PWD/third_party/spatial_tracker/
dataset_dir=<your dataset dir>
python easycalib_demo.py  --root_dir $dataset_dir --use_segm_mask true --caliberate_method pnp --pnp_refinement true --use_pnp_ransac false --use_grounded_sam --has_gt --win_len 1 --verbose --render_mask --easyhec_repo_path $easyhec_repo_pah --grounded_sam_repo_path $grounded_sam_repo_path --spatial_tracker_repo_path $spatial_tracker_repo_path --cut_off 300 --renderer_device_id 0 --tracking_device_id 0 --mask_inference_device_id 0 --keypoint_ids 0
```
Parameters:

* `root_dir`: where you stored all your video frames and json data.
* `use_segm_mask`: whether to use foreground mask to guide kpt tracking module.
* `use_grounded_sam`: whether to use Grounded-SAM to automatically generate robot arm mask.
* `cut_off`: only process first $cut_off frames for both computational efficiency and kpt tracking stability.
* `renderer_device_id`, `tracking_device_id`, `mask_inference_device_id`, the gpu_id for rendering mask, tracking annotated TCP and use SAM/Grounded-SAM to inference foreground mask.
* `keypoint_ids`: choose what keypoints to be tracked. The keypoint configurations are specified in [franka_config.json](./easycalib/config/franka_config.json)


2. Running the synthetic data generation pipeline. 
We use pyrfuniverse as our simulation environment, please refer to [pyrfuniverse](https://github.com/robotflow-initiative/pyrfuniverse) for more details.

```bash
python ./tests/sim_franka/test_gaussian_trajectory.py
```

## ❖ Dataset Conventions
The video frames can be saved to `.png` or `.jpg` format, along with which an accompanying json file should be stored. For alignment between corresponding video frame and json file, they should be sorted alphabetically in ascending order.
A template json format for specifying the robot configurations at the same timestamp with its image counterpart is in [template_config.json](./easycalib/config/template_config.json).

A more elaborated example is as follows:

```json
{
    "objects": [
        {
            "class": "panda",
            "visibility": 1,
            "location": [
                854.9748197663477,
                532.4341293247742
            ],
            "camera_intrinsics": [ 
                [
                    935.3074360871939,
                    0.0,
                    960.0
                ],
                [
                    0.0,
                    935.3074360871938,
                    540.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            "local_to_world_matrix": [ // The gt local_to_world_matrix, only valid when gt data is present.
                [
                    0.936329185962677,
                    0.3511234521865845,
                    0.0,
                    -0.26919466257095337
                ],
                [
                    0.1636243760585785,
                    -0.4363316595554352,
                    -0.8847835659980774,
                    -0.01939260959625244
                ],
                [
                    -0.3106682598590851,
                    0.8284486532211304,
                    -0.4660024046897888,
                    2.3973233699798584
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            "keypoints": [
                {
                    "name": "panda_link_0",
                    "location": [
                        2.421438694000244e-08,
                        8.149072527885437e-10,
                        -5.587935447692871e-08
                    ],
                    "projected_location": [ // The projected 2D keypoints locations on the images, only valid when gt data is present.
                        854.9748197663477,
                        532.4341293247742
                    ],
                    "predicted_location": [ // The predicted 2D keypoints locations by kpt-tracking module.
                        854.9748197663477,
                        532.4341293247742
                    ]
                },
                // other keypoints
            ],
            "eef_pos": [
                [
                    2.421438694000244e-08,
                    8.149072527885437e-10,
                    -5.587935447692871e-08
                ]
                ... 
            ],
            "joint_positions": [
                1.4787984922025454,
                -0.6394085992873211,
                -1.1422850521276044,
                -1.4485166195536359,
                -0.5849469549952007,
                1.3101860404224674,
                0.2957148441498494,
                0.0
            ],
            "cartesian_position": [
                [
                    2.421438694000244e-08,
                    8.149072527885437e-10,
                    -5.587935447692871e-08
                ],
                ...
            ],
            // other auxiliary keys for debugging purposes.
        }
    ]
}
```

### ✨Stars/forks/issues/PRs are all welcome!

## ❖ Last but Not Least

If you have any additional questions or have interests in collaboration,please feel free to contact me at [Tutian Tang](tttang@sjtu.edu.cn), [Minghao Liu](lmh209@sjtu.edu.cn), [Wenqiang Xu](vinjohn@sjtu.edu.cn) 😃.