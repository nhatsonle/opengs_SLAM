import torch
import torch.nn.functional as F


def _resize_map_to(target_hw, *maps):
    """Resize one or more [1,H,W] or [H,W] float maps to target (H, W) via bilinear."""
    th, tw = target_hw
    out = []
    for m in maps:
        if m is None:
            out.append(None)
            continue
        x = m
        if x.dim() == 2:
            x = x.unsqueeze(0)
        # x is [C,H,W]
        if x.shape[-2:] != (th, tw):
            x = F.interpolate(
                x.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False
            ).squeeze(0)
        out.append(x)
    return out if len(out) > 1 else out[0]


def compute_coverage_residual_mask(
    target_hw,
    opacity,
    render_rgb,
    gt_image,
    render_depth=None,
    dust3r_depth=None,
    opacity_threshold=0.5,
    rgb_residual_threshold=0.1,
    use_depth_residual=False,
    depth_residual_threshold=0.1,
    min_opacity_floor=0.1,
):
    """Build a [H,W] bool insertion mask over the DUSt3R pointmap grid (target_hw).

    A pixel is flagged for new-Gaussian insertion when the current map either
    fails to cover it (low rendered opacity), renders it with the wrong color
    (high RGB residual), or — optionally — gets its geometry wrong (high depth
    residual). All render-space maps are resized to the pointmap resolution so
    the mask aligns with pts3d/imgs/confidence masks.

    Returns a torch bool tensor of shape target_hw on the same device as inputs.
    """
    th, tw = target_hw

    opacity_r = _resize_map_to((th, tw), opacity)  # [1,H,W]
    render_rgb_r, gt_rgb_r = _resize_map_to((th, tw), render_rgb, gt_image)  # [3,H,W]

    opacity_r = opacity_r.squeeze(0)  # [H,W]

    # Vùng chưa được map phủ đủ (opacity thấp).
    need_coverage = opacity_r < opacity_threshold

    # Vùng render sai màu so với ground truth.
    rgb_residual = torch.abs(render_rgb_r - gt_rgb_r).mean(dim=0)  # [H,W]
    need_appearance = rgb_residual > rgb_residual_threshold

    insertion = need_coverage | need_appearance

    if use_depth_residual and render_depth is not None and dust3r_depth is not None:
        depth_r, dust3r_depth_r = _resize_map_to(
            (th, tw), render_depth, dust3r_depth
        )
        depth_r = depth_r.squeeze(0)
        dust3r_depth_r = dust3r_depth_r.squeeze(0)
        valid_depth = (depth_r > 0) & (dust3r_depth_r > 0)
        depth_residual = torch.abs(depth_r - dust3r_depth_r)
        need_depth = (depth_residual > depth_residual_threshold) & valid_depth
        insertion = insertion | need_depth

    # Sàn an toàn: vùng gần như không có Gaussian thì luôn chèn, bất kể residual.
    insertion = insertion | (opacity_r < min_opacity_floor)

    return insertion


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]

def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)

def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err

def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)

def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    #l1 = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()

def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()

def get_loss_mapping(config, image,  viewpoint, depth=None, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)

def get_loss_mapping_rgb(config, image, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()

def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()

def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
