import numpy as np
import nvdiffrast.torch as dr
import matplotlib

import trimesh

import sys
import numpy as np
import json
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import trimesh
from easycalib.utils.setup_logger import setup_logger
logger = setup_logger(__name__)


class NVDiffrastRenderApiHelper(nn.Module):
    """
    Render the mask of a urdf articulated object given camera pose and intrinsics.
    """

    def __init__(self, mesh_paths, K, Tc_c2b, H, W):
        super().__init__()
        for link_idx, mesh_path in enumerate(mesh_paths):
            mesh = trimesh.load(osp.expanduser(mesh_path), force="mesh")
            vertices = torch.from_numpy(mesh.vertices).cuda().float()
            faces = torch.from_numpy(mesh.faces).cuda().int()
            self.register_buffer(f"vertices_{link_idx}", vertices)
            self.register_buffer(f"faces_{link_idx}", faces)
        self.nlinks = len(mesh_paths)
        self.H, self.W = H, W
        self.renderer = NVDiffrastRenderer([self.H, self.W])
        self.K = K
        self.Tc_c2b = Tc_c2b

    def render_mask(self, link_poses):
        Tc_c2b = self.Tc_c2b
        all_frame_all_link_si = []
        K = self.K

        all_link_si = []
        for link_idx in range(self.nlinks):
            Tc_c2l = Tc_c2b @ link_poses[link_idx]
            verts, faces = (
                getattr(self, f"vertices_{link_idx}"),
                getattr(self, f"faces_{link_idx}"),
            )
            si = self.renderer.render_mask(verts, faces, K=K, object_pose=Tc_c2l)
            all_link_si.append(si)
        all_link_si = torch.stack(all_link_si).sum(0).clamp(max=1)
        all_frame_all_link_si.append(all_link_si)
        all_frame_all_link_si = torch.stack(all_frame_all_link_si)
        all_frame_all_link_si = all_frame_all_link_si.squeeze(0)

        return all_frame_all_link_si

    @staticmethod
    def compute_mask_iou(mask_pred, mask_gt):
        """
        Compute the Intersection over Union (IoU) of two binary masks.

        Parameters:
            mask1 (np.array): First binary mask.
            mask2 (np.array): Second binary mask.

        Returns:
            float: the computed IoU value.
        """
        mask_pred = mask_pred.astype(bool)
        mask_gt = mask_gt.astype(bool)

        intersection = np.logical_and(mask_pred, mask_gt)

        union = np.logical_or(mask_pred, mask_gt)

        iou = intersection.sum() / union.sum()

        return iou


def K_to_projection(K, H, W, n=0.001, f=10.0):
    fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    proj = torch.tensor([[2 * fu / W, 0, -2 * cu / W + 1, 0],
                         [0, 2 * fv / H, 2 * cv / H - 1, 0],
                         [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                         [0, 0, -1, 0]]).cuda().float()
    return proj


def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

class NVDiffrastRenderer:
    """
    Render the mask of a object given camera pose and intrinsics.
    """

    def __init__(self, image_size):
        """
        image_size: H,W
        """
        # self.
        self.H, self.W = image_size
        self.resolution = image_size
        blender2opencv = (
            torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            .float()
            .cuda()
        )
        self.opencv2blender = torch.inverse(blender2opencv)
        self.glctx = dr.RasterizeCudaContext()

    def render_mask(self, verts, faces, K, object_pose, anti_aliasing=True):
        """
        @param verts: N,3, torch.tensor, float, cuda
        @param faces: M,3, torch.tensor, int32, cuda
        @param K: 3,3 torch.tensor, float ,cuda
        @param object_pose: 4,4 torch.tensor, float, cuda
        @return: mask: 0 to 1, HxW torch.cuda.FloatTensor
        """
        K = torch.from_numpy(K).float()
        object_pose = torch.from_numpy(object_pose).cuda().float()

        proj = K_to_projection(K, self.H, self.W)
        proj = proj.float().cuda()

        pose = self.opencv2blender @ object_pose
        # pose = object_pose
        pos_clip = transform_pos(proj @ pose, verts)

        rast_out, _ = dr.rasterize(
            self.glctx, pos_clip, faces, resolution=self.resolution
        )
        if anti_aliasing:
            vtx_color = torch.ones(verts.shape, dtype=torch.float, device=verts.device)
            color, _ = dr.interpolate(vtx_color[None, ...], rast_out, faces)
            color = dr.antialias(color, rast_out, pos_clip, faces)
            mask = color[0, :, :, 0]
        else:
            mask = rast_out[0, :, :, 2] > 0
        mask = torch.flip(mask, dims=[0])
        return mask


def main():
    import pdb

    pdb.set_trace()

    # code.interact(local=locals())
    pose = np.array(
        [
            [
                0.66342535733919,
                -0.7466611919051226,
                -0.048619540744332466,
                -0.5192662186991392,
            ],
            [
                -0.08094655933770384,
                -0.007022207492831332,
                -0.9966937057759094,
                0.4669590753305032,
            ],
            [
                0.7438510938156613,
                0.6651674624519373,
                -0.06509836499095036,
                0.6149715416001105,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).astype(np.float32)
    K = np.array(
        [
            [522.212890625, 0.0, 649.6121215820312],
            [0.0, 522.212890625, 366.571533203125],
            [0.0, 0.0, 1.0],
        ]
    )
    H, W = 720, 1280
    # load json config from path: franka_config_path
    franka_config_path = "./cotracker/config/franka_config.json"
    with open(franka_config_path, "r") as f:
        cfg = json.load(f)
    mesh_paths = cfg["manipulator"]["mesh_paths"]
    urdf_path = cfg["manipulator"]["urdf_path"]
    renderer = NVDiffrastRenderApiHelper(mesh_paths, K, pose, H, W)
    rest_pose_wrt_link_pos = np.array(
        [
            0.0468769297003746,
            -0.9893064498901367,
            0.050184402614831924,
            -2.5965023040771484,
            0.03965343162417412,
            1.876589059829712,
            0.047545723617076874,
        ]
    )
    # rest_pose_wrt_world, rest_mat_wrt_world = compute_forward_kinematics(
    #     urdf_path, rest_pose_wrt_link_pos, link_indices=list(range(8)), return_pose=True
    # )
    # mask = renderer.render_mask(link_poses=rest_mat_wrt_world)
    # plt.imshow(mask.detach().cpu())
    # plt.show()


if __name__ == "__main__":
    main()
