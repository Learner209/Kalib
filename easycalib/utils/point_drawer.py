import imageio
import subprocess
from easycalib.utils.setup_logger import setup_logger
import numpy as np
import cv2
import torch


logger = setup_logger(__name__)

class PointDrawer(object):
    def __init__(
            self,
            window_name="Point Drawer",
            screen_scale=1.0,
            sam_checkpoint="",
            sam_model_type="vit_h",
            dry_run=False,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        self.window_name = window_name  # Name for our window
        self.done = False  # Flag signalling we're done
        self.current = (0, 0)  # Current position, so we can draw the line-in-progress
        self.points = np.empty((0, 2))  # List of points defining our polygon
        self.labels = np.empty([0], dtype=int)
        self.screen_scale = screen_scale * 1.2
        self.dry_run = dry_run

        ################  ↓ Step : build SAM  ↓  ##############
        if not dry_run:
            from segment_anything import sam_model_registry, SamPredictor

            sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            self.predictor = SamPredictor(sam)

        self.mask = None

    def on_mouse(self, event, x, y, flags, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done:  # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                label = 0
            else:
                label = 1
            logger.debug(
                f"Adding point #{len(self.points)} with position({x},{y}), label {label}"
            )
            self.points = np.concatenate((self.points, np.array([[x, y]])), axis=0)
            self.labels = np.concatenate((self.labels, np.array([label])), axis=0)

            self.detect()

    def detect(self):
        input_point = self.points / self.ratio
        input_label = self.labels.astype(int)

        if not self.dry_run:
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            maxidx = np.argmax(scores)
            mask = masks[maxidx]
            self.mask = mask.copy()

    def run(self, rgb):
        if not self.dry_run:
            self.predictor.set_image(rgb)

        # initialize self.mask as all black, return all-black mask if no points are clicked.
        self.mask = np.zeros(rgb.shape[:2], dtype=bool)

        image_to_show = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        output = subprocess.check_output(["xrandr"]).decode("utf-8")
        current_mode = [line for line in output.splitlines() if "*" in line][0]

        screen_width, screen_height = [
            int(x) for x in current_mode.split()[0].split("x")
        ]
        scale = self.screen_scale
        screen_w = int(screen_width / scale)
        screen_h = int(screen_height / scale)

        image_h, image_w = image_to_show.shape[:2]
        ratio = min(screen_w / image_w, screen_h / image_h)
        self.ratio = ratio
        target_size = (int(image_w * ratio), int(image_h * ratio))
        image_to_show = cv2.resize(image_to_show, target_size)

        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, image_to_show)
        cv2.waitKey(20)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while not self.done:
            tmp = image_to_show.copy()
            tmp = cv2.circle(
                tmp, self.current, radius=2, color=(0, 0, 255), thickness=-1
            )
            if self.points.shape[0] > 0:
                for ptidx, pt in enumerate(self.points):
                    color = (0, 255, 0) if self.labels[ptidx] == 1 else (0, 0, 255)
                    tmp = cv2.circle(
                        tmp,
                        (int(pt[0]), int(pt[1])),
                        radius=5,
                        color=color,
                        thickness=-1,
                    )
            if self.mask is not None:
                mask_to_show = cv2.resize(
                    self.mask.astype(np.uint8), target_size
                ).astype(bool)
                tmp = tmp / 255.0
                tmp[mask_to_show] *= 0.5
                tmp[mask_to_show] += 0.5
                tmp = (tmp * 255).astype(np.uint8)
            cv2.imshow(self.window_name, tmp)
            if cv2.waitKey(50) == 27:  # ESC hit
                self.done = True
        cv2.destroyAllWindows()
        self.points = self.points / self.ratio
        return self.points, self.labels, self.mask


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        default="./assets/sim_demo_1_first_frame.png",
        help="path to a video",
    )

    args = parser.parse_args()

    img_path = args.img_path
    basename = os.path.splitext(os.path.basename(img_path))[-2]
    ext = os.path.splitext(os.path.basename(img_path))[-1]
    savename = basename + "_mask" + ext
    save_path = os.path.join(os.path.dirname(img_path), savename)

    rgb = imageio.imread_v2(img_path)
    pointdrawer = PointDrawer(
        sam_checkpoint="./pretrained_checkpoints/sam_vit_b_01ec64.pth"
    )
    rgb = rgb[..., :3]
    points, labels, mask = pointdrawer.run(rgb)
    print(points, labels, mask.shape)

    cv2.imwrite(save_path, (mask * 255).astype(np.uint8))
    print()


if __name__ == "__main__":
    import os

    main()
