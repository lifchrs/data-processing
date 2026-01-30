#!/usr/bin/env python3
"""
Visualize SAM3D reconstructed object using Gaussian splat in the original image frame.

This is a DUPLICATED version of `visualize_sam3d_output.py`.

Key fix vs the original:
- Use the **exact same pose application utilities** as `sam-3d-objects`:
  - `pytorch3d.transforms.quaternion_to_matrix` (SAM3D quaternions are w,x,y,z)
  - `sam3d_objects.data.dataset.tdfy.transforms_3d.compose_transform`
  - `sam3d_objects.pipeline.layout_post_optimization_utils.denormalize_f`

This eliminates the common "angle is completely wrong" issue that comes from mixing
SciPy quaternion conventions / transpose assumptions with SAM3D's PyTorch3D convention.
"""

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["LIDRA_SKIP_INIT"] = "true"  # avoid importing missing `sam3d_objects.init`

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import binary_dilation

import sys

# Allow importing `sam3d_objects.*` utilities from the vendored repo.
_SAM3D_REPO_ROOT = os.path.join(os.path.dirname(__file__), "sam-3d-objects")
if _SAM3D_REPO_ROOT not in sys.path:
    sys.path.insert(0, _SAM3D_REPO_ROOT)


def denormalize_intrinsics(norm_K: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Match `sam3d_objects/pipeline/layout_post_optimization_utils.py:denormalize_f`.

    `norm_K` stores fx, fy, cx, cy normalized by (width, height).
    """
    K = np.array(norm_K, dtype=np.float32, copy=True)
    K[0, 0] *= width   # fx
    K[1, 1] *= height  # fy
    K[0, 2] *= width   # cx
    K[1, 2] *= height  # cy
    return K


def transform_local_to_camera_l2c(
    xyz_local: np.ndarray,
    rotation_wxyz: np.ndarray,
    translation: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """
    Apply SAM3D's local-to-camera (l2c) similarity transform.

    IMPORTANT:
    - Output points stay in SAM3D/PyTorch3D camera convention (x left, y up, z inward).
    """
    import torch
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    pts_local = torch.from_numpy(np.asarray(xyz_local, dtype=np.float32)).unsqueeze(0)  # (1, N, 3)
    quat = torch.from_numpy(np.asarray(rotation_wxyz, dtype=np.float32).reshape(1, 4))  # (1, 4) wxyz
    trans = torch.from_numpy(np.asarray(translation, dtype=np.float32).reshape(1, 3))   # (1, 3)
    sc = torch.from_numpy(np.asarray(scale, dtype=np.float32).reshape(1, 3))            # (1, 3)

    R_l2c = quaternion_to_matrix(quat)  # (1, 3, 3)
    tfm_l2c = compose_transform(scale=sc, rotation=R_l2c, translation=trans)
    pts_cam = tfm_l2c.transform_points(pts_local)[0].cpu().numpy()
    return pts_cam


def project_points_opencv_from_pytorch3d_cam(
    xyz_cam_p3d: np.ndarray, intrinsics_px: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project points assuming xyz are in PyTorch3D camera coords:
      x left, y up, z inwards (positive).

    Convert to OpenCV-like camera coords for pinhole projection:
      x right, y down, z forward  ==> flip x and y.
    """
    fx, fy = float(intrinsics_px[0, 0]), float(intrinsics_px[1, 1])
    cx, cy = float(intrinsics_px[0, 2]), float(intrinsics_px[1, 2])

    xyz = np.asarray(xyz_cam_p3d, dtype=np.float32)
    depths = xyz[:, 2].copy()

    x = -xyz[:, 0]
    y = -xyz[:, 1]
    z = depths

    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    return np.stack([u, v], axis=-1), depths


def rotate_points_about_camera_y(points_cam: np.ndarray, angle_rad: float, center: np.ndarray) -> np.ndarray:
    """
    Rotate points in camera space around the camera Y axis by `angle_rad`,
    but around a provided `center` so the object stays in place.

    Camera convention here is PyTorch3D: x left, y up, z inward.
    """
    pts = np.asarray(points_cam, dtype=np.float32)
    c = np.asarray(center, dtype=np.float32).reshape(1, 3)
    pts0 = pts - c

    ca = float(np.cos(angle_rad))
    sa = float(np.sin(angle_rad))
    # Y-axis rotation (right-handed) for row vectors
    # [x', y', z'] = [x*ca + z*sa, y, -x*sa + z*ca]
    x = pts0[:, 0]
    y = pts0[:, 1]
    z = pts0[:, 2]
    xr = x * ca + z * sa
    zr = -x * sa + z * ca
    out = np.stack([xr, y, zr], axis=-1) + c
    return out


def render_gaussian_splat(uv: np.ndarray,
                          depths: np.ndarray,
                          colors: np.ndarray,
                          opacities: np.ndarray,
                          image_shape: tuple[int, int],
                          mask_filter: np.ndarray | None = None):
    """Render Gaussian splat as a simple alpha-blended point cloud with optional mask filtering."""
    img_h, img_w = image_shape

    # Filter valid points
    valid = (depths > 0.01) & (depths < 10.0)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < img_w)
    valid &= (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    valid &= (opacities > 0.1)

    # Additional mask filter - only keep points projecting near the mask
    if mask_filter is not None:
        u_int = np.clip(uv[:, 0].astype(int), 0, img_w - 1)
        v_int = np.clip(uv[:, 1].astype(int), 0, img_h - 1)
        in_mask_region = mask_filter[v_int, u_int]
        valid &= in_mask_region

    uv = uv[valid]
    depths = depths[valid]
    colors_valid = colors[valid]
    opacities_valid = opacities[valid]

    # Sort by depth (back to front)
    sort_idx = np.argsort(-depths)
    uv = uv[sort_idx]
    depths = depths[sort_idx]
    colors_valid = colors_valid[sort_idx]
    opacities_valid = opacities_valid[sort_idx]

    rendered = np.zeros((img_h, img_w, 3), dtype=np.float32)
    depth_map = np.zeros((img_h, img_w), dtype=np.float32)
    alpha_map = np.zeros((img_h, img_w), dtype=np.float32)

    u_int = uv[:, 0].astype(int)
    v_int = uv[:, 1].astype(int)

    for i in range(len(u_int)):
        x, y = u_int[i], v_int[i]
        alpha = float(opacities_valid[i])
        color = colors_valid[i]
        d = float(depths[i])

        rendered[y, x] = rendered[y, x] * (1 - alpha) + color * alpha
        if depth_map[y, x] == 0 or d < depth_map[y, x]:
            depth_map[y, x] = d
        alpha_map[y, x] = min(1.0, alpha_map[y, x] + alpha)

    rendered = (np.clip(rendered, 0, 1) * 255).astype(np.uint8)
    mask = alpha_map > 0.1
    return rendered, depth_map, mask


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


def main():
    # Paths
    SAM3D_OUTPUT_PATH = "/home/ap977/GRILL/new_project/tv_output/sam3d_output.npy"
    IMAGE_PATH = "/home/ap977/GRILL/new_project/tv.jpg"
    MASK_PATH = "/home/ap977/GRILL/new_project/tv_output/mask.npy"
    ATTEMPTS_DIR = "/home/ap977/GRILL/new_project/tv_output/attempts_sam3d_utils"

    os.makedirs(ATTEMPTS_DIR, exist_ok=True)
    attempt_num = len([f for f in os.listdir(ATTEMPTS_DIR) if f.startswith("attempt_")]) // 3 + 1

    # Load data
    print(f"=== Attempt {attempt_num} (sam-3d-objects utils) ===")
    data = np.load(SAM3D_OUTPUT_PATH, allow_pickle=True).item()
    image = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    img_h, img_w = image.shape[:2]
    gt_mask = np.load(MASK_PATH).astype(bool)

    # Extract pose (SAM3D: local -> camera)
    rotation = data["rotation"].squeeze()
    translation = data["translation"].squeeze()
    scale = data["scale"].squeeze()
    intrinsics_norm = data["intrinsics"]

    # Get Gaussian data
    gs_xyz = data["gs_xyz"]
    gs_features = data["gs_features"].squeeze()
    gs_opacity = data["gs_opacity"].squeeze()

    # Convert features/opacities to display-friendly values
    colors = np.clip(gs_features * 0.5 + 0.5, 0, 1)
    opacities = 1.0 / (1.0 + np.exp(-gs_opacity))
    print(f"Gaussian splat: {gs_xyz.shape[0]} points")

    # Use sam-3d-objects' intrinsics denormalization to match their camera model.
    from sam3d_objects.pipeline.layout_post_optimization_utils import denormalize_f
    intrinsics_px = denormalize_f(intrinsics_norm, img_h, img_w)

    # Apply pose using sam-3d-objects utilities (this is the critical "angle" fix).
    xyz_cam = transform_local_to_camera_l2c(gs_xyz, rotation, translation, scale)
    # "Back vs front" is ambiguous for a thin object (TV). If it looks like you're seeing the back,
    # a 180° yaw around the object's center in camera space typically fixes it.
    # We'll render BOTH and save them, so you can confirm visually.
    center = xyz_cam.mean(axis=0)
    xyz_cam_flip = rotate_points_about_camera_y(xyz_cam, np.pi, center=center)

    dilated_mask = binary_dilation(gt_mask, iterations=20)

    def _render_variant(tag: str, xyz_cam_variant: np.ndarray):
        uv, depths = project_points_opencv_from_pytorch3d_cam(xyz_cam_variant, intrinsics_px)
        rendered_color, depth_map, silhouette = render_gaussian_splat(
            uv, depths, colors, opacities, image.shape[:2], mask_filter=dilated_mask
        )
        iou = compute_iou(gt_mask, silhouette)
        return uv, depths, rendered_color, depth_map, silhouette, iou

    print("Rendering with mask filter...")
    _, _, rendered_color_a, depth_map_a, silhouette_a, iou_a = _render_variant("orig", xyz_cam)
    _, _, rendered_color_b, depth_map_b, silhouette_b, iou_b = _render_variant("yaw180", xyz_cam_flip)

    # Compute IoU
    print(f"IoU (orig):   {iou_a:.4f} ({iou_a*100:.2f}%)")
    print(f"IoU (yaw180): {iou_b:.4f} ({iou_b*100:.2f}%)")

    # Choose which one to show as the default overlay:
    # Prefer yaw180 if it is close in IoU (it usually will be).
    use_yaw180 = True if abs(iou_b - iou_a) < 0.02 else (iou_b > iou_a)
    chosen_tag = "yaw180" if use_yaw180 else "orig"
    rendered_color = rendered_color_b if use_yaw180 else rendered_color_a
    depth_map = depth_map_b if use_yaw180 else depth_map_a
    silhouette = silhouette_b if use_yaw180 else silhouette_a
    iou = iou_b if use_yaw180 else iou_a

    print("\n=== Results (chosen) ===")
    print(f"Chosen variant: {chosen_tag}")
    print(f"Ground truth mask pixels: {gt_mask.sum()}")
    print(f"Rendered silhouette pixels: {silhouette.sum()}")
    print(f"IoU: {iou:.4f} ({iou*100:.2f}%)")

    # Visualizations
    output = image.copy()
    output[silhouette] = rendered_color[silhouette]
    side_by_side = np.hstack([image, output])

    mask_vis = np.zeros_like(image)
    mask_vis[gt_mask] = [0, 255, 0]
    mask_vis[silhouette] = [255, 0, 0]
    mask_vis[np.logical_and(gt_mask, silhouette)] = [255, 255, 0]

    # Save chosen + also save the other variant for inspection
    output_path = os.path.join(ATTEMPTS_DIR, f"attempt_{attempt_num:02d}_{chosen_tag}_overlay.jpg")
    comparison_path = os.path.join(ATTEMPTS_DIR, f"attempt_{attempt_num:02d}_{chosen_tag}_comparison.jpg")
    mask_vis_path = os.path.join(ATTEMPTS_DIR, f"attempt_{attempt_num:02d}_{chosen_tag}_masks.jpg")

    other_tag = "orig" if chosen_tag == "yaw180" else "yaw180"
    other_rendered = rendered_color_a if chosen_tag == "yaw180" else rendered_color_b
    other_sil = silhouette_a if chosen_tag == "yaw180" else silhouette_b
    other_depth = depth_map_a if chosen_tag == "yaw180" else depth_map_b

    cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    cv2.imwrite(comparison_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_vis_path, cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))

    print("\nSaved:")
    print(f"  {comparison_path}")
    print(f"  {mask_vis_path}")

    # Save the alternative variant for quick A/B checking
    output_other = image.copy()
    output_other[other_sil] = other_rendered[other_sil]
    side_by_side_other = np.hstack([image, output_other])
    mask_vis_other = np.zeros_like(image)
    mask_vis_other[gt_mask] = [0, 255, 0]
    mask_vis_other[other_sil] = [255, 0, 0]
    mask_vis_other[np.logical_and(gt_mask, other_sil)] = [255, 255, 0]

    output_path_other = os.path.join(ATTEMPTS_DIR, f"attempt_{attempt_num:02d}_{other_tag}_overlay.jpg")
    comparison_path_other = os.path.join(ATTEMPTS_DIR, f"attempt_{attempt_num:02d}_{other_tag}_comparison.jpg")
    mask_vis_path_other = os.path.join(ATTEMPTS_DIR, f"attempt_{attempt_num:02d}_{other_tag}_masks.jpg")
    cv2.imwrite(output_path_other, cv2.cvtColor(output_other, cv2.COLOR_RGB2BGR))
    cv2.imwrite(comparison_path_other, cv2.cvtColor(side_by_side_other, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_vis_path_other, cv2.cvtColor(mask_vis_other, cv2.COLOR_RGB2BGR))

    print(f"  {comparison_path_other}")
    print(f"  {mask_vis_path_other}")

    if silhouette.sum() > 0:
        print(f"Depth range: {depth_map[silhouette].min():.3f} to {depth_map[silhouette].max():.3f}")
    if other_sil.sum() > 0:
        print(f"Depth range ({other_tag}): {other_depth[other_sil].min():.3f} to {other_depth[other_sil].max():.3f}")


if __name__ == "__main__":
    main()

