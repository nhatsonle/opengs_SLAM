import time
from collections import OrderedDict

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
from utils.dust3r_utils import get_result, get_scale


def _sync_cuda_if_available():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class FrontEnd(mp.Process):
    def __init__(self, config, d3r_model):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False            
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False

        self.d3r_model = d3r_model
        self.last_color = None          # last frame RGB
        self.pts3d = None               # last frame pointcloud
        self.imgs = None
        self.mask = None
        self.matches_im0 = None
        self.matches_im1 = None
        self.matches_3d0 = None
        self.scale = 1                  # Scale factor computed using median, not enabled
        self.scale1 = 1                 # Scale factor computed using mean, used for scale correction
        self.theta = 0                  # Camera angle diff from last keyframe
        self.depth_builder = None
        self.local_depth_gaussians = OrderedDict()
        
    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]
        self.profile_timing = self.config["Results"].get("profile_timing", True)

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]
        self.max_frames = self.config["Results"].get("max_frames", None)

        tracking_cfg = self.config.get("Tracking", {})
        self.pose_init_mode = tracking_cfg.get("pose_init", "constant_velocity")
        self.use_dust3r_every_frame = tracking_cfg.get("use_dust3r_every_frame", False)
        self.wait_for_keyframe_backend = tracking_cfg.get(
            "wait_for_keyframe_backend", True
        )
        self.use_depth_local_map = tracking_cfg.get("use_depth_local_map", True)
        self.depth_tracking_buffer = int(tracking_cfg.get("depth_tracking_buffer", 3))
        self.depth_min = float(tracking_cfg.get("depth_min", 0.1))
        self.depth_max = float(tracking_cfg.get("depth_max", 80.0))

        self.mapping_opt_cfg = self.config.get("MappingOptimization", {})
        self.selective_keyframe_insertion = self.mapping_opt_cfg.get(
            "selective_keyframe_insertion", False
        )

    def _profile_start(self):
        if not self.profile_timing:
            return None
        _sync_cuda_if_available()
        return time.perf_counter()

    def _profile_end(self, name, start, frame_idx=None):
        if start is None:
            return
        _sync_cuda_if_available()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if frame_idx is None:
            Log(f"{name}: {elapsed_ms:.2f} ms", tag="Profile")
        else:
            Log(f"frame {frame_idx} {name}: {elapsed_ms:.2f} ms", tag="Profile")

    def _dilate_insert_mask(self, mask):
        dilation = int(self.mapping_opt_cfg.get("insertion_mask_dilate", 0))
        if dilation <= 0:
            return mask
        kernel = 2 * dilation + 1
        mask_float = mask.float()[None, None]
        dilated = torch.nn.functional.max_pool2d(
            mask_float, kernel_size=kernel, stride=1, padding=dilation
        )
        return dilated[0, 0] > 0

    def _build_keyframe_insert_mask(self, cur_frame_idx, viewpoint):
        if not self.selective_keyframe_insertion or self.gaussians is None:
            return None, {}

        profile_start = self._profile_start()
        global_render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )
        self._profile_end(
            "keyframe_global_render_for_insert_mask", profile_start, cur_frame_idx
        )
        if global_render_pkg is None:
            return None, {}

        profile_start = self._profile_start()
        gt_img = viewpoint.original_image.cuda()
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        valid_rgb = gt_img.sum(dim=0) > rgb_boundary_threshold
        insert_mask = torch.zeros_like(valid_rgb, dtype=torch.bool)
        score_map = torch.zeros_like(gt_img[0], dtype=torch.float32)
        score_cfg = self.mapping_opt_cfg.get("insert_score", {})
        w_cov = float(score_cfg.get("coverage_weight", 1.0))
        w_rgb = float(score_cfg.get("rgb_weight", 1.0))
        w_depth = float(score_cfg.get("depth_weight", 1.0))
        score_th_high = float(score_cfg.get("threshold_high", 1.0))
        score_th_low = float(score_cfg.get("threshold_low", 0.7))

        opacity = global_render_pkg["opacity"].detach().squeeze()
        opacity_th = float(self.mapping_opt_cfg.get("opacity_insert_th", 0.35))
        if self.mapping_opt_cfg.get("use_render_coverage_mask", True):
            insert_mask = torch.logical_or(insert_mask, opacity < opacity_th)
            cov_gap = torch.clamp(opacity_th - opacity, min=0.0) / max(opacity_th, 1e-6)
            score_map = score_map + w_cov * cov_gap

        rgb_residual_mean = 0.0
        rgb_residual = None
        if self.mapping_opt_cfg.get("use_rgb_residual_mask", True):
            render_rgb = global_render_pkg["render"].detach()
            rgb_residual = torch.abs(render_rgb - gt_img).mean(dim=0)
            rgb_residual_th = float(
                self.mapping_opt_cfg.get("rgb_residual_insert_th", 0.08)
            )
            insert_mask = torch.logical_or(insert_mask, rgb_residual > rgb_residual_th)
            score_map = score_map + w_rgb * (rgb_residual / max(rgb_residual_th, 1e-6))
            if valid_rgb.any():
                rgb_residual_mean = rgb_residual[valid_rgb].mean().item()

        depth_residual_mean = 0.0
        rel_depth_residual = None
        if self.mapping_opt_cfg.get("use_depth_residual_mask", True):
            dataset_depth = self._filtered_dataset_depth(viewpoint.depth)
            if dataset_depth is not None:
                dataset_depth_t = torch.from_numpy(dataset_depth).to(
                    device=gt_img.device, dtype=torch.float32
                )
                render_depth = global_render_pkg["depth"].detach().squeeze()
                valid_depth = torch.logical_and(dataset_depth_t > 0, render_depth > 0)
                valid_depth = torch.logical_and(valid_depth, valid_rgb)
                if valid_depth.any():
                    rel_depth_residual = torch.abs(render_depth - dataset_depth_t) / (
                        dataset_depth_t + 1e-6
                    )
                    depth_th = float(
                        self.mapping_opt_cfg.get(
                            "depth_residual_insert_ratio_th", 0.08
                        )
                    )
                    insert_mask = torch.logical_or(
                        insert_mask,
                        torch.logical_and(rel_depth_residual > depth_th, valid_depth),
                    )
                    normalized_depth_residual = rel_depth_residual / max(depth_th, 1e-6)
                    score_map = score_map + w_depth * normalized_depth_residual * valid_depth.float()
                    depth_residual_mean = rel_depth_residual[valid_depth].mean().item()

        # unified unexplained-region score + hysteresis gating
        high_mask = score_map > score_th_high
        low_mask = score_map > score_th_low
        insert_mask = torch.logical_or(insert_mask, high_mask)
        insert_mask = torch.logical_and(insert_mask, low_mask)
        insert_mask = torch.logical_and(insert_mask, valid_rgb)
        insert_mask = self._dilate_insert_mask(insert_mask)
        insert_mask = torch.logical_and(insert_mask, valid_rgb)

        insert_pixels = int(insert_mask.count_nonzero().item())
        valid_pixels = max(int(valid_rgb.count_nonzero().item()), 1)
        coverage_ratio = (
            float((opacity[valid_rgb] >= opacity_th).float().mean().item())
            if valid_rgb.any()
            else 0.0
        )
        insert_ratio = insert_pixels / valid_pixels
        insert_info = {
            "insert_pixels": insert_pixels,
            "valid_pixels": valid_pixels,
            "insert_ratio": insert_ratio,
            "coverage_ratio": coverage_ratio,
            "mean_rgb_residual": rgb_residual_mean,
            "mean_depth_residual": depth_residual_mean,
            "score_mean": float(score_map[valid_rgb].mean().item()) if valid_rgb.any() else 0.0,
            "score_high_pixels": int(high_mask.count_nonzero().item()),
        }
        if self.profile_timing:
            Log(
                "keyframe_insert_mask: "
                f"pixels={insert_pixels}/{valid_pixels} "
                f"insert_ratio={insert_ratio:.4f} "
                f"coverage={coverage_ratio:.4f} "
                f"mean_rgb={rgb_residual_mean:.4f} "
                f"mean_depth_rel={depth_residual_mean:.4f}",
                tag="Profile",
            )
        self._profile_end("keyframe_insert_mask_build", profile_start, cur_frame_idx)
        return insert_mask.detach().cpu().numpy(), insert_info

    def _filtered_dataset_depth(self, depth):
        if depth is None:
            return None
        depth_np = np.asarray(depth, dtype=np.float32).copy()
        valid = np.isfinite(depth_np)
        valid = np.logical_and(valid, depth_np >= self.depth_min)
        valid = np.logical_and(valid, depth_np <= self.depth_max)
        depth_np[~valid] = 0.0
        if np.count_nonzero(depth_np) == 0:
            return None
        return depth_np

    def _ensure_depth_builder(self):
        if self.depth_builder is None and self.gaussians is not None:
            self.depth_builder = GaussianModel(
                self.gaussians.max_sh_degree, config=self.config
            )
            self.depth_builder.active_sh_degree = self.gaussians.active_sh_degree
        return self.depth_builder

    def _build_depth_gaussian_tensors(self, viewpoint):
        depth_map = self._filtered_dataset_depth(viewpoint.depth)
        if depth_map is None:
            return None
        depth_builder = self._ensure_depth_builder()
        if depth_builder is None:
            return None
        with torch.no_grad():
            fused_point_cloud, features, scales, rots, opacities = (
                depth_builder.create_pcd_from_image(
                    viewpoint, init=False, depthmap=depth_map
                )
            )
        if fused_point_cloud.shape[0] == 0:
            return None
        return {
            "xyz": fused_point_cloud.detach(),
            "features_dc": features[:, :, 0:1].transpose(1, 2).contiguous().detach(),
            "features_rest": features[:, :, 1:].transpose(1, 2).contiguous().detach(),
            "scaling": scales.detach(),
            "rotation": rots.detach(),
            "opacity": opacities.detach(),
        }

    def _store_local_depth_gaussian(self, frame_idx, viewpoint):
        if not self.use_depth_local_map or self.depth_tracking_buffer <= 0:
            return
        tensors = self._build_depth_gaussian_tensors(viewpoint)
        if tensors is None:
            return
        self.local_depth_gaussians[frame_idx] = tensors
        while len(self.local_depth_gaussians) > self.depth_tracking_buffer:
            self.local_depth_gaussians.popitem(last=False)

    def _make_tracking_gaussians(self):
        if self.gaussians is None or len(self.local_depth_gaussians) == 0:
            return self.gaussians

        tracking_gaussians = GaussianModel(
            self.gaussians.max_sh_degree, config=self.config
        )
        tracking_gaussians.active_sh_degree = self.gaussians.active_sh_degree

        local_entries = list(self.local_depth_gaussians.values())
        tracking_gaussians._xyz = torch.cat(
            [self.gaussians._xyz] + [entry["xyz"] for entry in local_entries], dim=0
        )
        tracking_gaussians._features_dc = torch.cat(
            [self.gaussians._features_dc]
            + [entry["features_dc"] for entry in local_entries],
            dim=0,
        )
        tracking_gaussians._features_rest = torch.cat(
            [self.gaussians._features_rest]
            + [entry["features_rest"] for entry in local_entries],
            dim=0,
        )
        tracking_gaussians._scaling = torch.cat(
            [self.gaussians._scaling] + [entry["scaling"] for entry in local_entries],
            dim=0,
        )
        tracking_gaussians._rotation = torch.cat(
            [self.gaussians._rotation]
            + [entry["rotation"] for entry in local_entries],
            dim=0,
        )
        tracking_gaussians._opacity = torch.cat(
            [self.gaussians._opacity] + [entry["opacity"] for entry in local_entries],
            dim=0,
        )
        n_points = tracking_gaussians._xyz.shape[0]
        tracking_gaussians.max_radii2D = torch.zeros(n_points, device=self.device)
        tracking_gaussians.unique_kfIDs = torch.zeros(n_points).int()
        tracking_gaussians.n_obs = torch.zeros(n_points).int()
        return tracking_gaussians

    def _initialize_pose(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        if self.pose_init_mode == "constant_velocity" and (
            cur_frame_idx - 2 * self.use_every_n_frames
        ) in self.cameras:
            prev2 = self.cameras[cur_frame_idx - 2 * self.use_every_n_frames]
            w2c_prev = getWorld2View2(prev.R, prev.T)
            w2c_prev2 = getWorld2View2(prev2.R, prev2.T)
            w2c_pred = w2c_prev @ torch.linalg.inv(w2c_prev2) @ w2c_prev
        else:
            w2c_pred = getWorld2View2(prev.R, prev.T)
        viewpoint.update_RT(w2c_pred[:3, :3], w2c_pred[:3, 3])

    def _estimate_median_depth(self, render_pkg, viewpoint):
        try:
            return get_median_depth(render_pkg["depth"], render_pkg["opacity"])
        except Exception:
            depth_map = self._filtered_dataset_depth(viewpoint.depth)
            if depth_map is None:
                return torch.tensor(1.0, device=self.device)
            valid_depth = depth_map[depth_map > 0]
            if valid_depth.size == 0:
                return torch.tensor(1.0, device=self.device)
            return torch.tensor(np.median(valid_depth), device=self.device)

    def _run_keyframe_dust3r(self, viewpoint, reference_frame_idx):
        reference = self.cameras.get(reference_frame_idx)
        reference_image = None if reference is None else reference.original_image
        if reference_image is None:
            reference_image = self.last_color
        if reference_image is None:
            return False

        try:
            (
                _,
                pts3d,
                imgs,
                matches_im0,
                matches_im1,
                matches_3d0,
                confidence_masks,
            ) = get_result(
                viewpoint.original_image,
                reference_image,
                model=self.d3r_model,
                device=self.device,
            )
        except Exception as exc:
            Log("DUSt3R keyframe estimation failed, using depth fallback:", exc)
            self.pts3d = None
            self.imgs = None
            self.mask = None
            return False

        if (
            self.matches_im1 is not None
            and self.matches_im0 is not None
            and self.matches_3d0 is not None
        ):
            try:
                scale1, scale = get_scale(
                    self.matches_im1,
                    self.matches_im0,
                    matches_im1,
                    matches_im0,
                    self.matches_3d0,
                    matches_3d0,
                )
                if np.isfinite(scale) and scale > 0:
                    self.scale = self.scale * scale
                if np.isfinite(scale1) and scale1 > 0:
                    self.scale1 = self.scale1 * scale1
            except Exception as exc:
                Log("Adaptive scale matching failed, keeping previous scale:", exc)

        self.pts3d = pts3d
        self.imgs = imgs
        self.mask = (
            confidence_masks
            if self.mapping_opt_cfg.get("use_dust3r_confidence_mask", True)
            else None
        )
        if self.mask is not None:
            mask_pixels = sum(int(m.detach().cpu().numpy().sum()) for m in self.mask)
            if self.profile_timing:
                Log(f"dust3r_confidence_mask_pixels: {mask_pixels}", tag="Profile")
        self.matches_im0 = matches_im0
        self.matches_im1 = matches_im1
        self.matches_3d0 = matches_3d0
        return True
        
    def add_new_keyframe(
        self, cur_frame_idx, depth=None, opacity=None, init=False, insert_mask=None
    ):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        if len(self.kf_indices) > 0:
            last_kf = self.kf_indices[-1]
            viewpoint_last = self.cameras[last_kf]
            R_last = viewpoint_last.R
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        R_now = viewpoint.R
        # Compute angle diff from previous frame
        if len(self.kf_indices) > 1:
            R_now = R_now.to(torch.float32)
            R_last = R_last.to(torch.float32)
            R_diff = torch.matmul(R_last.T, R_now)
            trace_R_diff = torch.trace(R_diff)
            theta_rad = torch.acos((trace_R_diff - 1) / 2)
            theta_deg = torch.rad2deg(theta_rad)
            self.theta = theta_deg
        # print("angle diff is:",self.theta)
        ### MonoGS Gaussian init depth, not used
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]    
        dataset_depth = self._filtered_dataset_depth(viewpoint.depth)
        if dataset_depth is not None:
            dataset_depth = dataset_depth.copy()
            dataset_depth[~valid_rgb.cpu().numpy()[0]] = 0.0
            if insert_mask is not None and insert_mask.shape == dataset_depth.shape:
                dataset_depth[~insert_mask] = 0.0
            return dataset_depth
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2]) 
                initial_depth += torch.randn_like(initial_depth) * 0.3            
            else:      
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:   
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2     
                    )

                initial_depth[~valid_rgb] = 0
                if insert_mask is not None and insert_mask.shape == initial_depth.shape[-2:]:
                    insert_mask_t = torch.from_numpy(insert_mask).to(
                        device=initial_depth.device, dtype=torch.bool
                    )
                    initial_depth = initial_depth * insert_mask_t.unsqueeze(0).to(
                        initial_depth.dtype
                    )
            return initial_depth.cpu().numpy()[0]

        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)      
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()      # (C, H, W), not used!

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)      
        self.reset = False
    
    def tracking(self, cur_frame_idx, viewpoint):
        self._initialize_pose(cur_frame_idx, viewpoint)
        tracking_gaussians = self._make_tracking_gaussians()
        # pose optimization
        opt_params = []     
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)
        render_pkg = None
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, tracking_gaussians, self.pipeline_params, self.background
            )
            if render_pkg is None:
                break
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint) 

            if tracking_itr % 10 == 0:             
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break

        global_render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )
        if global_render_pkg is not None:
            self.median_depth = self._estimate_median_depth(global_render_pkg, viewpoint)
            if render_pkg is None:
                render_pkg = global_render_pkg
            else:
                render_pkg["n_touched"] = global_render_pkg["n_touched"]
                render_pkg["visibility_filter"] = global_render_pkg[
                    "visibility_filter"
                ]
                render_pkg["radii"] = global_render_pkg["radii"]
        else:
            self.median_depth = self._estimate_median_depth(render_pkg, viewpoint)

        return render_pkg
    
    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)        
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)                   
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])        
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check       
    
    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:        
                to_remove.append(kf_idx)
        # Remove earliest keyframe with overlap below threshold
        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))
        # If the window count exceeds the limit, remove the frame farthest from the current frame.
        # The distance is weighted to favor deleting the candidate with the highest inverse distance to the other candidates.
        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)
        #print("current keyframe ",cur_frame_idx,'window is ',window)
        return window, removed_frame
    ### Exchange info with backend via following functions
    # Request new keyframe; enqueue related info to backend
    def request_keyframe(
        self, cur_frame_idx, viewpoint, current_window, depthmap, insert_info=None
    ):
        msg = [
            "keyframe",
            cur_frame_idx,
            viewpoint,
            current_window,
            depthmap,
            self.pts3d,
            self.imgs,
            self.mask,
            self.scale1,
            self.theta,
            insert_info or {},
        ]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1
    # Request initialization; enqueue related info to backend.
    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map, self.pts3d, self.imgs, self.mask, self.scale1]
        self.backend_queue.put(msg)
        self.requested_init = True
    # Sync data from backend (3D Gaussians, occlusion-aware visibility, keyframe info)
    def sync_backend(self, data):
        self.gaussians = data[1]
        if self.depth_builder is not None:
            self.depth_builder.active_sh_degree = self.gaussians.active_sh_degree
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()
            
    # Main loop: process messages in frontend and backend queues; perform tracking, keyframe management;
    # synchronize data, clean up resources, and save results
    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(       
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)      
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():        
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():         # Check if frontend_queue is empty; if so, start processing the current frame
                tic.record()
                if cur_frame_idx >= len(self.dataset) or (
                    self.max_frames is not None and cur_frame_idx >= self.max_frames
                ):
                    # Finish the frontend process
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break
              
                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.requested_keyframe > 0 and (
                    self.single_thread
                    or self.wait_for_keyframe_backend
                    or not self.initialized
                ):
                    time.sleep(0.01)
                    continue
                
                profile_start = self._profile_start()
                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)
                self._profile_end("load_frame", profile_start, cur_frame_idx)

                self.cameras[cur_frame_idx] = viewpoint
        
                if self.reset:
                    self.last_color = self.cameras[cur_frame_idx].original_image
                    self.pts3d = None
                    self.imgs = None
                    self.matches_im0 = None
                    self.matches_im1 = None
                    self.matches_3d0 = None
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                profile_start = self._profile_start()
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                self._profile_end("tracking", profile_start, cur_frame_idx)
                self.last_color = self.cameras[cur_frame_idx].original_image
                if render_pkg is None:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue
                profile_start = self._profile_start()
                self._store_local_depth_gaussian(cur_frame_idx, viewpoint)
                self._profile_end(
                    "local_depth_gaussian_creation", profile_start, cur_frame_idx
                )
    
                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
                
                if self.config["Results"]["use_gui"]:
                    profile_start = self._profile_start()
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=clone_obj(self.gaussians),
                            current_frame=viewpoint,
                            keyframes=keyframes,
                            kf_window=current_window_dict,
                        )
                    )
                    self._profile_end(
                        "clone_gui_gaussian_packet", profile_start, cur_frame_idx
                    )
                
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval    
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,        
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:      
                    create_kf = check_time and create_kf
                if create_kf:       
                    reference_keyframe_idx = last_keyframe_idx
                    insert_mask, insert_info = self._build_keyframe_insert_mask(
                        cur_frame_idx, viewpoint
                    )
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )       
                    depth_map = self.add_new_keyframe(      
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                        insert_mask=insert_mask,
                    )
                    profile_start = self._profile_start()
                    self._run_keyframe_dust3r(viewpoint, reference_keyframe_idx)
                    self._profile_end(
                        "dust3r_keyframe_inference", profile_start, cur_frame_idx
                    )
                    Log("new keyframe: ", cur_frame_idx)
                    self.request_keyframe(   
                        cur_frame_idx,
                        viewpoint,
                        self.current_window,
                        depth_map,
                        insert_info,
                    )
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1              

                if (                    # Perform trajectory evaluation when certain conditions are met
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()      
                if create_kf:
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:       # If the frontend queue contains messages from the backend, process them
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    profile_start = self._profile_start()
                    self.sync_backend(data)
                    self._profile_end("frontend_sync_backend", profile_start)

                elif data[0] == "keyframe":
                    profile_start = self._profile_start()
                    self.sync_backend(data)
                    self._profile_end("frontend_sync_keyframe", profile_start)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    profile_start = self._profile_start()
                    self.sync_backend(data)
                    self._profile_end("frontend_sync_init", profile_start)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
