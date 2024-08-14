import sys
import os
from easydict import EasyDict as edict
from glob import glob

import torch
import torch.nn.functional as F
from base64 import b64encode
import numpy as np
from PIL import Image
import cv2
import argparse
from moviepy.editor import ImageSequenceClip
import torchvision.transforms as transforms
from easycalib.utils.setup_logger import setup_logger

logger = setup_logger(__name__)


# -------- import cotracker -------------
from third_party.spatial_tracker.models.cotracker.utils.visualizer import Visualizer, read_video_from_path
from third_party.spatial_tracker.models.cotracker.predictor import CoTrackerPredictor

# -------- import spatialtracker -------------
from third_party.spatial_tracker.models.spatracker.predictor import SpaTrackerPredictor
from third_party.spatial_tracker.models.spatracker.utils.visualizer import Visualizer, read_video_from_path

# -------- import Depth Estimator -------------
from third_party.spatial_tracker.mde import MonoDEst

# set the arguments


def parse_spatracker_args(cli_input=False):
    parser = argparse.ArgumentParser()
    # add the video and segmentation
    parser.add_argument('--root', type=str, default='./assets', help='path to the video')
    parser.add_argument('--vid_name', type=str, default='breakdance', help='path to the video')
    # set the gpu
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    # set the model
    parser.add_argument('--model', type=str, default='cotracker', help='model name')
    # set the downsample factor
    parser.add_argument('--downsample', type=float, default=1, help='downsample factor')
    parser.add_argument('--grid_size', type=int, default=40, help='grid size')
    # set the results outdir
    parser.add_argument('--outdir', type=str, default='./vis_results', help='output directory')
    # set the fps
    parser.add_argument('--fps', type=float, default=1, help='fps')
    # draw the track length
    parser.add_argument('--len_track', type=int, default=10, help='len_track')
    parser.add_argument('--fps_vis', type=int, default=15, help='len_track')
    # crop the video
    parser.add_argument('--crop', action='store_true', help='whether to crop the video')
    parser.add_argument('--crop_factor', type=float, default=1, help='whether to crop the video')
    # backward tracking
    parser.add_argument('--backward', action='store_true', help='whether to backward the tracking')
    # if visualize the support points
    parser.add_argument('--vis_support', action='store_true', help='whether to visualize the support points')
    # query frame
    parser.add_argument('--query_frame', type=int, default=0, help='query frame')
    # set the visualized point size
    parser.add_argument('--point_size', type=int, default=3, help='point size')
    # take the RGBD as input
    parser.add_argument('--rgbd', action='store_true', help='whether to take the RGBD as input')

    if not cli_input:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args


def spatracker_predict(args, video, queries, segm_mask):

    video = video.cuda()
    queries = queries.cuda()

    fps_vis = args.fps_vis

    # set input
    root_dir = args.root
    vid_dir = os.path.join(root_dir, args.vid_name + '.mp4')
    outdir = args.outdir
    os.path.exists(outdir) or os.makedirs(outdir)
    # set the paras
    grid_size = args.grid_size
    model_type = args.model
    downsample = args.downsample
    # set the gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # read the video
    # video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    transform = transforms.Compose([
        transforms.CenterCrop((int(384 * args.crop_factor),
                               int(512 * args.crop_factor))),
    ])
    _, T, _, H, W = video.shape
    if segm_mask is None:
        segm_mask = np.ones((H, W), dtype=np.uint8)
        logger.info("No segmentation mask provided. Computing tr > ")
        segm_mask = transform(torch.from_numpy(segm_mask[None, None]))[0, 0].numpy()
    _, _, _, H, W = video.shape
    # adjust the downsample factor
    if H > W:
        downsample = max(downsample, 640 // H)
    elif H < W:
        downsample = max(downsample, 960 // W)
    else:
        downsample = max(downsample, 640 // H)

    video = F.interpolate(video[0], scale_factor=downsample,
                          mode='bilinear', align_corners=True)[None]
    vidLen = video.shape[1]
    idx = torch.range(0, vidLen - 1, args.fps).long()
    video = video[:, idx]
    # save the first image
    img0 = video[0, 0].permute(1, 2, 0).detach().cpu().numpy()

    # cv2.imwrite(os.path.join(outdir, f'{args.vid_name}_ref.png'), img0[:, :, ::-1])
    # cv2.imwrite(os.path.join(outdir, f'{args.vid_name}_seg.png'), segm_mask * 255)

    S_lenth = 12       # [8, 12, 16] choose one you want
    model = SpaTrackerPredictor(
        checkpoint=os.path.join(
            './checkpoints/spaT_final.pth',
        ),
        interp_shape=(384, 512),
        seq_length=S_lenth
    )
    if torch.cuda.is_available():
        model = model.cuda()
        video = video.cuda()
        queries = queries.cuda()
        # segm_mask = segm_mask.cuda()

    cfg = edict({
        "mde_name": "zoedepth_nk"
    })

    MonoDEst_O = MonoDEst(cfg)
    # MonoDEst_O = MonoDEst_O.cuda()
    MonoDEst_M = MonoDEst_O.model
    # MonoDEst_M = MonoDEst_M.cuda()
    MonoDEst_M.eval()
    depths = None

    pred_tracks, pred_visibility, T_Firsts = (
        model(video, video_depth=depths,
              grid_size=grid_size, backward_tracking=args.backward,
              depth_predictor=MonoDEst_M, grid_query_frame=args.query_frame, queries=queries,
              segm_mask=torch.from_numpy(segm_mask)[None, None].cuda(), wind_length=S_lenth)
    )
    return pred_tracks[..., :2].detach().cpu().numpy(), pred_visibility.detach().cpu().numpy()

    vis = Visualizer(save_dir=outdir, grayscale=True,
                     fps=fps_vis, pad_value=0, linewidth=args.point_size,
                     tracks_leave_trace=args.len_track)
    msk_query = (T_Firsts == args.query_frame)
    # visualize the all points
    if args.vis_support:
        video_vis = vis.visualize(video=video, tracks=pred_track[..., :2],
                                  visibility=pred_visi,
                                  filename=args.vid_name + "_spatracker")
    else:
        pred_track = pred_track[:, :, msk_query.squeeze()]
        pred_visi = pred_visi[:, :, msk_query.squeeze()]
        video_vis = vis.visualize(video=video, tracks=pred_track[..., :2],
                                  visibility=pred_visi,
                                  filename=args.vid_name + "_spatracker")

    # vis the first queried video
    img0 = video_vis[0, 0].permute(1, 2, 0).detach().cpu().numpy()
    cv2.imwrite(os.path.join(outdir, f'{args.vid_name}_ref_query.png'), img0[:, :, ::-1])
    # save the tracks
    tracks_vis = pred_track[0].detach().cpu().numpy()
    np.save(os.path.join(outdir, f'{args.vid_name}_{args.model}_tracks.npy'), tracks_vis)
    # save the video
    wide_list = list(video.unbind(1))
    wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
    clip = ImageSequenceClip(wide_list, fps=60)
    save_path = os.path.join(outdir, f'{args.vid_name}_vid.mp4')
    clip.write_videofile(save_path, codec="libx264", fps=25, logger=None)
    print(f"Original Video saved to {save_path}")

    T = pred_track[0].shape[0]
    # save the 3d trajectories
    xyzt = pred_track[0].cpu().numpy()   # T x N x 3
    intr = np.array([[W, 0.0, W // 2],
                     [0.0, W, H // 2],
                     [0.0, 0.0, 1.0]])
    xyztVis = xyzt.copy()
    xyztVis[..., 2] = 1.0

    xyztVis = np.linalg.inv(intr[None, ...]) @ xyztVis.reshape(-1, 3, 1)  # (TN) 3 1
    xyztVis = xyztVis.reshape(T, -1, 3)  # T N 3
    xyztVis[..., 2] *= xyzt[..., 2]

    pred_tracks2d = pred_track[0][:, :, :2]
    S1, N1, _ = pred_tracks2d.shape
    video2d = video[0]  # T C H W
    H1, W1 = video[0].shape[-2:]
    pred_tracks2dNm = pred_tracks2d.clone()
    pred_tracks2dNm[..., 0] = 2 * (pred_tracks2dNm[..., 0] / W1 - 0.5)
    pred_tracks2dNm[..., 1] = 2 * (pred_tracks2dNm[..., 1] / H1 - 0.5)
    color_interp = torch.nn.functional.grid_sample(video2d, pred_tracks2dNm[:, :, None, :],
                                                   align_corners=True)

    color_interp = color_interp[:, :, :, 0].permute(0, 2, 1).cpu().numpy().astype(np.uint8)
    colored_pts = np.concatenate([xyztVis, color_interp], axis=-1)
    np.save(f'{outdir}/{args.vid_name}_3d.npy', colored_pts)

    print(f"3d colored tracks to {outdir}/{args.vid_name}_3d.npy")

    return pred_track, pred_visi


if __name__ == "__main__":

    spatracker_args = parse_spatracker_args()
    img_paths = "/mnt/public/datasets/DROID/PennPAL/2023-06-15/Thu_Jun_15_17:04:05_2023/varied_camera_1_left/*.png"
    img_paths = sorted(glob(img_paths))
    img_objs = [cv2.imread(img_path) for img_path in img_paths]
    video_frames = np.stack(img_objs)  # N x H x W x C
    video = (
        torch.from_numpy(video_frames).permute(0, 3, 1, 2)[None].float()
    )  # batch x times x channel x height x width
    queries = torch.tensor([[[0.0000, 1071.1674, 173.7271]]])
    queries = queries.cuda()
    video = video.cuda()
    output_dir = "/mnt/homes/minghao/robotflow/easy_calib/third_party/spatial_tracker/outputs"
    pred_tracks, pred_visibility = spatracker_predict(spatracker_args, video, queries=queries, segm_mask=cv2.imread("/mnt/public/datasets/DROID/Robotflow/gaussian07_26_23_08_02/EasyCalib/first_frame_mask_segm/mask.png"))
    logger.info(f"pred_tracks's shape: {pred_tracks.shape}, pred_visibility's shape:{pred_visibility.shape}")
