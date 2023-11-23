import numpy as np
import cv2
import torch
import torch.nn.functional as F
import rembg
from tqdm import tqdm
import os

from gs_renderer import Renderer, MiniCam
from cam_utils import orbit_camera, OrbitCamera

class SplatModelTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.device = torch.device("cuda")
        self.renderer = Renderer(sh_degree=opt.sh_degree)
        self.optimizer = None
        self.step = 0

        assert self.opt.input is not None, "Input image is required"
        self.load_input(self.opt.input)

        self.renderer.initialize(num_pts=opt.num_pts)

        
    def load_input(self, file):
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)
        
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        self.input_img = self.input_img[..., ::-1].copy()
    
    def prepare_train(self):
        self.step = 0

        self.renderer.gaussians.training_setup(self.opt)
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(pose, self.opt.ref_size, self.opt.ref_size, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

        self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        from guidance.zero123_utils import Zero123
        self.guidance_zero123 = Zero123(self.device)

        with torch.no_grad():
            self.guidance_zero123.get_img_embeds(self.input_img_torch)

    def train_step(self):
        self.step += 1
        step_ratio = min(1, self.step / self.opt.iters)
        self.renderer.gaussians.update_learning_rate(self.step)

        loss = 0

        cur_cam = self.fixed_cam
        out = self.renderer.render(cur_cam)

        image = out["image"].unsqueeze(0)
        loss = loss + 10000 * step_ratio * F.mse_loss(image, self.input_img_torch)

        mask = out["alpha"].unsqueeze(0)
        loss = loss + 1000 * step_ratio * F.mse_loss(mask, self.input_mask_torch)

        render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
        images = []
        vers, hors, radii = [], [], []
        min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
        max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)

        for _ in range(self.opt.batch_size):
            ver = np.random.randint(min_ver, max_ver)
            hor = np.random.randint(-180, 180)
            radius = 0

            vers.append(ver)
            hors.append(hor)
            radii.append(radius)

            pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)

            cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

            bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
            out = self.renderer.render(cur_cam, bg_color=bg_color)

            image = out["image"].unsqueeze(0)
            images.append(image)
        
        images = torch.cat(images, dim=0)

        loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio=step_ratio if self.opt.anneal_timestep else None)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
            viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
            self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if self.step % self.opt.densification_interval == 0:
                self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.1, extent=4, max_screen_size=1)
            
            if self.step % self.opt.opacity_reset_interval == 0:
                self.renderer.gaussians.reset_opacity()

        torch.cuda.synchronize()
    
    def train(self, iters):
        self.prepare_train()
        for _ in tqdm(range(iters)):
            self.train_step()
        self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        self.save_model()
    
    def save_model(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
        path = os.path.join(self.opt.outdir, self.opt.save_path + "_model.ply")
        self.renderer.gaussians.save_ply(path)


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    trainer = SplatModelTrainer(opt)
    trainer.train(opt.iters)
