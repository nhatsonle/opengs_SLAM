import random
import time

import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping


def _sync_cuda_if_available():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class BackEnd(mp.Process):
    def __init__(self, config, save_dir=None):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False
        self.save_dir = save_dir

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0    # Total iterations
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        
        self.pcd_scale = 1          # scale factor
        self.theta = 0              # Camera angle diff from last keyframe

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]
        self.profile_timing = self.config["Results"].get("profile_timing", True)

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
        self.mapping_opt_cfg = self.config.get("MappingOptimization", {})
        self.max_new_gaussians_per_kf = self.mapping_opt_cfg.get(
            "max_new_gaussians_per_kf", None
        )
        self.adaptive_mapping_iters = self.mapping_opt_cfg.get(
            "adaptive_mapping_iters", False
        )
        self.recent_density_keyframes = int(
            self.mapping_opt_cfg.get("recent_density_keyframes", 3)
        )
        self.prune_recent_observation_th = int(
            self.mapping_opt_cfg.get("prune_min_observations", 3)
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

    def _log_profile(self, message):
        if self.profile_timing:
            Log(message, tag="Profile")

    def _recent_gaussian_mask(self, current_window):
        if self.gaussians is None or self.gaussians.unique_kfIDs.numel() == 0:
            return None

        if not self.initialized:
            return self.gaussians.unique_kfIDs >= 0

        recent_kf_count = max(1, self.recent_density_keyframes)
        sorted_window = sorted(current_window, reverse=True)
        if len(sorted_window) == 0:
            return self.gaussians.unique_kfIDs >= 0

        threshold_idx = min(recent_kf_count - 1, len(sorted_window) - 1)
        recent_threshold = sorted_window[threshold_idx]
        return self.gaussians.unique_kfIDs >= recent_threshold

    def _select_mapping_iters(self, new_gaussians, base_iters, insert_info):
        if not self.adaptive_mapping_iters:
            return base_iters
        coverage_ratio = insert_info.get("coverage_ratio") if insert_info else None
        low_coverage_th = float(
            self.mapping_opt_cfg.get("low_coverage_full_mapping_th", 0.55)
        )
        if coverage_ratio is not None and coverage_ratio < low_coverage_th:
            selected_iters = base_iters
        elif new_gaussians < int(
            self.mapping_opt_cfg.get("small_insert_gaussians", 5000)
        ):
            selected_iters = min(
                base_iters, int(self.mapping_opt_cfg.get("small_insert_iters", 20))
            )
        elif new_gaussians < int(
            self.mapping_opt_cfg.get("medium_insert_gaussians", 30000)
        ):
            selected_iters = min(
                base_iters, int(self.mapping_opt_cfg.get("medium_insert_iters", 50))
            )
        else:
            selected_iters = base_iters
        self._log_profile(
            "adaptive_mapping_iters: "
            f"new_gaussians={new_gaussians} "
            f"coverage={coverage_ratio if coverage_ratio is not None else 'n/a'} "
            f"iters={selected_iters}"
        )
        return selected_iters

    # In MonoGS, initialize Gaussians and add to the current scene (not enabled)
    def add_next_kf(
        self,
        frame_idx,
        viewpoint,
        init=False,
        scale=2.0,
        depth_map=None,
        insert_mask=None,
        max_points=None,
    ):
        before = self.gaussians.get_xyz.shape[0]
        self.gaussians.extend_from_pcd_seq(
            viewpoint,
            kf_id=frame_idx,
            init=init,
            scale=scale,
            depthmap=depth_map,
            insert_mask=insert_mask,
            max_points=max_points,
        )
        return self.gaussians.get_xyz.shape[0] - before

    # Initialize Gaussians via pointmap and add to the current scene   
    def add_next_kf_dust3r(
        self,
        frame_idx,
        pts3d,
        imgs,
        T,
        mask=None,
        init=False,
        scale=1,
        max_points=None,
    ):
        before = self.gaussians.get_xyz.shape[0]
        fused_point_cloud, features, scales, rots, opacities = (
            self.gaussians.create_pcd_from_dust3r(
                pts3d,
                imgs,
                T,
                frame_idx,
                self.save_dir,
                scale,
                mask,
                init=init,
                max_points=max_points,
            )
        )
        self.gaussians.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, frame_idx
        )
        return self.gaussians.get_xyz.shape[0] - before

    def reset(self):
        self.iteration_count = 0
        self.iteration_count1 = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()
            
    # Initialize SLAM map by optimizing Gaussian parameters over multiple iterations
    def initialize_map(self, cur_frame_idx, viewpoint):
        profile_start = self._profile_start()
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            self.iteration_count1 += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, viewpoint, depth=depth,initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(  
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(               
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:  
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()                         
                self.gaussians.optimizer.zero_grad(set_to_none=True)    

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()   
        Log("Initialized map")
        self._profile_end("backend_initialize_map", profile_start, cur_frame_idx)
        return render_pkg
    
    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        profile_start = self._profile_start()
        profile_frame_idx = current_window[0] if len(current_window) > 0 else None
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)            
        for cam_idx, viewpoint in self.viewpoints.items(): 
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)
                
        for _ in range(iters):
            self.iteration_count += 1
            self.iteration_count1 += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []               
            visibility_filter_acm = []                      
            radii_acm = []                                
            n_touched_acm = []                              

            keyframes_opt = []        

            for cam_idx in range(len(current_window)):      
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (                                         
                    image,
                    viewspace_point_tensor,               
                    visibility_filter,                     
                    radii,                                  
                    depth,                                  
                    opacity,                                
                    n_touched,                            
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, viewpoint, depth=depth
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)     

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:     
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, viewpoint, depth=depth
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            # Compute isotropic loss and add it to the total loss
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 1.0 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}         
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()
                    
                # Only prune on the last iteration and when we have full window
                if prune:      
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if self.mapping_opt_cfg.get("smart_pruning", False):
                            opacity_th = float(
                                self.mapping_opt_cfg.get(
                                    "prune_opacity_th", self.gaussian_th
                                )
                            )
                            recent_mask = self._recent_gaussian_mask(current_window)
                            if recent_mask is None:
                                recent_mask = torch.ones_like(
                                    self.gaussians.unique_kfIDs, dtype=torch.bool
                                )
                            low_observation = (
                                self.gaussians.n_obs <= self.prune_recent_observation_th
                            )
                            low_opacity = (
                                self.gaussians.get_opacity.detach().cpu().squeeze()
                                < opacity_th
                            )
                            prune_reason = torch.logical_or(
                                low_observation, low_opacity
                            )
                            too_large = torch.zeros_like(low_observation)
                            if self.mapping_opt_cfg.get(
                                "prune_large_screen_radius", True
                            ):
                                too_large = (
                                    self.gaussians.max_radii2D.detach().cpu()
                                    > self.size_threshold
                                )
                                prune_reason = torch.logical_or(
                                    prune_reason, too_large
                                )
                            to_prune = torch.logical_and(recent_mask, prune_reason)
                            self._log_profile(
                                "gaussian_prune_count: "
                                f"recent={int(recent_mask.count_nonzero().item())} "
                                f"low_opacity={int(low_opacity.count_nonzero().item())} "
                                f"low_observation={int(low_observation.count_nonzero().item())} "
                                f"too_large={int(too_large.count_nonzero().item())} "
                                f"final={int(to_prune.count_nonzero().item())}"
                            )
                        else:
                            if prune_mode == "odometry":
                                to_prune = self.gaussians.n_obs < 3
                                # make sure we don't split the gaussians, break here.
                            if prune_mode == "slam":
                                # only prune keyframes which are relatively new
                                sorted_window = sorted(current_window, reverse=True)
                                mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                                if not self.initialized:
                                    mask = self.gaussians.unique_kfIDs >= 0
                                to_prune = torch.logical_and(
                                    self.gaussians.n_obs <= prune_coviz, mask
                                )
                        if to_prune is not None and self.monocular:         
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (               
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    self._profile_end(
                        f"backend_mapping(iters={iters}, prune={prune})",
                        profile_start,
                        profile_frame_idx,
                    )
                    return False
          
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )
                
                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    density_mask = None
                    if self.mapping_opt_cfg.get("smart_pruning", False):
                        density_mask = self._recent_gaussian_mask(current_window)
                        if density_mask is not None:
                            self._log_profile(
                                "recent_density_control: "
                                f"selected={int(density_mask.count_nonzero().item())}/"
                                f"{density_mask.numel()}"
                            )
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                        density_mask=density_mask.cuda()
                        if density_mask is not None
                        else None,
                    )
                    gaussian_split = True

                ## Reset opacity every fixed number of iterations
                #if (self.iteration_count % self.gaussian_reset) == 0 and (
                #    not update_gaussian
                #):
                #    Log("Resetting the opacity of non-visible Gaussians")
                #    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                #    gaussian_split = True
               
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        self._profile_end(
            f"backend_mapping(iters={iters}, prune={prune})",
            profile_start,
            profile_frame_idx,
        )
        return gaussian_split
    
    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())      
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]    
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if iteration % 500 == 0 and iteration <= 15000:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        0.005,
                        self.gaussian_extent,
                        None,
                    )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(20000)
                if iteration % 1000 == 0:
                    self.gaussians.oneupSHdegree()
        Log("Map refinement done")
    
    def push_to_frontend(self, tag=None):
        profile_start = self._profile_start()
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"
            
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)
        self._profile_end(f"backend_clone_sync_gaussians[{tag}]", profile_start)
    # Main loop: process messages from the backend queue, perform map optimization, color refinement, 
    # initialization, and keyframe management; synchronize data and push to the frontend
    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue
                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:         
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    self.scale = 1
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    pts3d = data[5]
                    imgs = data[6]
                    mask = data[7]
                    self.scale = data[8]
                    self.theta = data[9]
                    insert_info = data[10] if len(data) > 10 else {}
                    theta_value = (
                        self.theta.item() if hasattr(self.theta, "item") else float(self.theta)
                    )
                    ## adjust the cumulative iterations for Adaptive Learning Rate Adjustment
                    if theta_value >= 2:
                        self.iteration_count = self.iteration_count * (1-np.sqrt(theta_value / 90))
                        self.iteration_count = int(self.iteration_count)
                        self.gaussians.update_learning_rate(self.iteration_count)
                    #print("current keyframe:", cur_frame_idx, "cumulative iterations:", self.iteration_count)

                    T_np = np.linalg.inv(getWorld2View2(viewpoint.R,viewpoint.T).cpu().numpy())
                    T = torch.from_numpy(T_np).to(self.device)
                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    gaussian_count_before = self.gaussians.get_xyz.shape[0]
                    profile_start = self._profile_start()
                    new_gaussians = 0
                    if pts3d is not None and imgs is not None:
                        try:
                            new_gaussians = self.add_next_kf_dust3r(
                                cur_frame_idx,
                                pts3d,
                                imgs,
                                T,
                                mask,
                                scale=self.scale,
                                max_points=self.max_new_gaussians_per_kf,
                            )
                        except Exception as exc:
                            Log(
                                "DUSt3R keyframe insertion failed, using depth fallback:",
                                exc,
                            )
                            new_gaussians = self.add_next_kf(
                                cur_frame_idx,
                                viewpoint,
                                depth_map=depth_map,
                                max_points=self.max_new_gaussians_per_kf,
                            )
                    else:
                        new_gaussians = self.add_next_kf(
                            cur_frame_idx,
                            viewpoint,
                            depth_map=depth_map,
                            max_points=self.max_new_gaussians_per_kf,
                        )
                    self._profile_end(
                        "backend_keyframe_gaussian_insertion",
                        profile_start,
                        cur_frame_idx,
                    )
                    gaussian_count_after = self.gaussians.get_xyz.shape[0]
                    self._log_profile(
                        f"frame {cur_frame_idx} gaussian_insert_count: {new_gaussians}"
                    )
                    self._log_profile(
                        "frame "
                        f"{cur_frame_idx} gaussian_count_before_after: "
                        f"{gaussian_count_before}->{gaussian_count_after}"
                    )
    
                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    else:
                        iter_per_kf = self._select_mapping_iters(
                            new_gaussians, iter_per_kf, insert_info
                        )
                    for cam_idx in range(len(self.current_window)):    
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:        
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
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
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
