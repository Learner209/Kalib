"""
A class to wrap the CoTrackerPredictor class from the CoTracker library.
"""

import torch
import torch.nn.functional as F

from third_party.cotracker.cotracker.predictor import CoTrackerPredictor
from third_party.cotracker.cotracker.models.core.model_utils import smart_cat, get_points_on_a_grid
from third_party.cotracker.cotracker.models.build_cotracker import build_cotracker


class CoTrackerPredictorWrapper(CoTrackerPredictor):
    """A class to wrap the CoTrackerPredictor class from the Co-Tracker library.
    Added functionality: concat the segmentation mask queries with user-defined queries.
    Original Implementation: if the user queries are not provided, the model will generate queries on a grid.
    """

    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def _compute_sparse_tracks(
        self,
        video,
        queries,
        segm_mask=None,
        grid_size=0,
        add_support_grid=False,
        grid_query_frame=0,
        backward_tracking=False,
    ):
        B, T, C, H, W = video.shape
        assert B == 1

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear", align_corners=True)
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        if queries is not None:
            B, N, D = queries.shape
            assert D == 3
            queries = queries.clone()
            queries[:, :, 1:] *= queries.new_tensor(
                [
                    (self.interp_shape[1] - 1) / (W - 1),
                    (self.interp_shape[0] - 1) / (H - 1),
                ]
            )
            queries = queries.to(self.device)
            grid_pts = get_points_on_a_grid(grid_size, self.interp_shape, device=video.device)
            if segm_mask is not None:
                segm_mask = F.interpolate(segm_mask, tuple(self.interp_shape), mode="nearest")
                point_mask = segm_mask[0, 0][
                    (grid_pts[0, :, 1]).round().long().cpu(),
                    (grid_pts[0, :, 0]).round().long().cpu(),
                ].bool()
                grid_pts = grid_pts[:, point_mask]

            mask_queries = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
                dim=2,
            ).repeat(B, 1, 1)
            mask_queries = mask_queries.to(self.device)
            # ! Combine mask queries and original queries.
            queries = torch.cat(
                [queries, mask_queries],
                dim=1
            )

        elif grid_size > 0:
            grid_pts = get_points_on_a_grid(grid_size, self.interp_shape, device=video.device)
            if segm_mask is not None:
                segm_mask = F.interpolate(segm_mask, tuple(self.interp_shape), mode="nearest")
                point_mask = segm_mask[0, 0][
                    (grid_pts[0, :, 1]).round().long().cpu(),
                    (grid_pts[0, :, 0]).round().long().cpu(),
                ].bool()
                grid_pts = grid_pts[:, point_mask]

            queries = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
                dim=2,
            ).repeat(B, 1, 1)

        if add_support_grid:
            grid_pts = get_points_on_a_grid(
                self.support_grid_size, self.interp_shape, device=video.device
            )
            grid_pts = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)
            grid_pts = grid_pts.repeat(B, 1, 1)
            grid_pts = grid_pts.to(self.device)
            queries = torch.cat([queries, grid_pts], dim=1)

        queries = queries.to(self.device)
        self.model = self.model.to(self.device)

        # with torch.autocast(device_type=self.device, dtype=torch.float32):
        tracks, visibilities, __ = self.model.forward(video=video, queries=queries, iters=6)

        if backward_tracking:
            tracks, visibilities = self._compute_backward_tracks(
                video, queries, tracks, visibilities
            )
            if add_support_grid:
                queries[:, -self.support_grid_size**2:, 0] = T - 1
        if add_support_grid:
            tracks = tracks[:, :, : -self.support_grid_size**2]
            visibilities = visibilities[:, :, : -self.support_grid_size**2]
        thr = 0.9
        visibilities = visibilities > thr

        # correct query-point predictions
        # see https://github.com/facebookresearch/co-tracker/issues/28

        # TODO: batchify
        for i in range(len(queries)):
            queries_t = queries[i, : tracks.size(2), 0].to(torch.int64)
            arange = torch.arange(0, len(queries_t))

            # overwrite the predictions with the query points
            tracks[i, queries_t, arange] = queries[i, : tracks.size(2), 1:]

            # correct visibilities, the query points should be visible
            visibilities[i, queries_t, arange] = True

        tracks *= tracks.new_tensor(
            [(W - 1) / (self.interp_shape[1] - 1), (H - 1) / (self.interp_shape[0] - 1)]
        )
        return tracks, visibilities
