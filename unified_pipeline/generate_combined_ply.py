import argparse
import os
import numpy as np
import cv2
import glob
import trimesh
from pathlib import Path
from plyfile import PlyData


# ── OBJ export helper ────────────────────────────────────────────────────────

def export_glb_as_obj(glb_path, obj_path, transform=None):
    """Load a GLB mesh and export it as Wavefront OBJ with vertex colors.

    Standard OBJ doesn't support per-vertex color, so we bake vertex colors
    into a texture atlas (via trimesh) and write the companion .mtl + .png.
    If the mesh has no visual data we fall back to a plain OBJ.

    Args:
        glb_path: Path to the source GLB file (triangle mesh with materials).
        obj_path: Destination OBJ path.
        transform: Optional 4x4 similarity transform to apply to vertices.
    """
    mesh = trimesh.load(glb_path, force="mesh")
    if transform is not None:
        mesh.vertices = (transform[:3, :3] @ mesh.vertices.T).T + transform[:3, 3]

    # If the mesh carries vertex colors, bake them into a texture so that
    # standard OBJ viewers (Blender, MeshLab, …) display colour correctly.
    if mesh.visual.kind == "vertex":
        from PIL import Image as PILImage

        vertex_colors = np.asarray(mesh.visual.vertex_colors)[:, :3]  # (V, 3)

        # Unwrap UVs — trimesh can auto-parametrise
        try:
            import xatlas
            vmapping, faces_new, uvs = xatlas.parametrize(
                mesh.vertices, mesh.faces)
            # Remap vertex colors to the new vertex order
            vertex_colors = vertex_colors[vmapping]
            mesh = trimesh.Trimesh(
                vertices=mesh.vertices[vmapping], faces=faces_new, process=False)
        except ImportError:
            # Fallback: simple per-face UV packing (one small square per face)
            n_faces = len(mesh.faces)
            # Grid layout for per-face squares
            grid_size = int(np.ceil(np.sqrt(n_faces)))
            tex_size = max(grid_size * 4, 64)
            texture = np.ones((tex_size, tex_size, 3), dtype=np.uint8) * 128

            uvs = np.zeros((n_faces * 3, 2), dtype=np.float64)
            new_faces = np.arange(n_faces * 3).reshape(-1, 3)
            new_verts = mesh.vertices[mesh.faces.flatten()]
            face_colors = vertex_colors[mesh.faces]  # (F, 3, 3)

            cell = tex_size / grid_size
            for fi in range(n_faces):
                row, col = divmod(fi, grid_size)
                # Fill a small cell in the texture with avg face colour
                avg_col = face_colors[fi].mean(axis=0).astype(np.uint8)
                r0, r1 = int(row * cell), int((row + 1) * cell)
                c0, c1 = int(col * cell), int((col + 1) * cell)
                texture[r0:r1, c0:c1] = avg_col
                # UVs pointing to centre of cell
                cx = (col + 0.5) / grid_size
                cy = 1.0 - (row + 0.5) / grid_size
                uvs[fi * 3] = [cx, cy]
                uvs[fi * 3 + 1] = [cx, cy]
                uvs[fi * 3 + 2] = [cx, cy]

            tex_img = PILImage.fromarray(texture)
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=tex_img,
                baseColorFactor=[255, 255, 255, 255],
            )
            mesh = trimesh.Trimesh(
                vertices=new_verts, faces=new_faces,
                visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
                process=False,
            )

    mesh.export(obj_path)
    print(f"Saved OBJ mesh: {obj_path}")


def remove_occluded_points(points_world, intrinsics, pose_c2w, image_shape,
                           depth_tolerance=0.005):
    """
    Remove points occluded by closer points from the camera's viewpoint.

    Projects all points to 2D, builds a z-buffer (min depth per pixel), and
    keeps only points within `depth_tolerance` (relative) of the frontmost
    point at each pixel.  Points behind the camera or outside the image are
    also removed.

    Args:
        points_world: (N, 3) world-space points
        intrinsics:   3x3 camera intrinsics
        pose_c2w:     4x4 camera-to-world pose
        image_shape:  (H, W) resolution used for the z-buffer
        depth_tolerance: relative tolerance (0.02 = 2 %).  A point at depth d
            is kept if d <= min_depth_at_pixel * (1 + depth_tolerance).

    Returns:
        keep: boolean array of shape (N,)
    """
    H, W = image_shape
    N = len(points_world)

    # World → camera
    R_w2c = pose_c2w[:3, :3].T
    t_w2c = -R_w2c @ pose_c2w[:3, 3]
    points_cam = (R_w2c @ points_world.T).T + t_w2c

    z = points_cam[:, 2]
    in_front = z > 0

    # Project to pixel coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x_2d = np.full(N, -1, dtype=np.int32)
    y_2d = np.full(N, -1, dtype=np.int32)
    x_2d[in_front] = (fx * points_cam[in_front, 0] / z[in_front] + cx).astype(np.int32)
    y_2d[in_front] = (fy * points_cam[in_front, 1] / z[in_front] + cy).astype(np.int32)

    in_bounds = in_front & (x_2d >= 0) & (x_2d < W) & (y_2d >= 0) & (y_2d < H)
    valid = np.where(in_bounds)[0]

    # Build z-buffer (minimum depth per pixel)
    depth_buffer = np.full((H, W), np.inf, dtype=np.float64)
    np.minimum.at(depth_buffer, (y_2d[valid], x_2d[valid]), z[valid])

    # Keep points within tolerance of frontmost depth at their pixel
    keep = np.zeros(N, dtype=bool)
    keep[valid] = z[valid] <= depth_buffer[y_2d[valid], x_2d[valid]] * (1.0 + depth_tolerance)

    return keep


def filter_points_by_mask(points_world, intrinsics, pose_c2w, image_shape, mask_2d):
    """
    Filter 3D world points to only those that project within a 2D binary mask.

    Args:
        points_world: (N, 3) array of world-space points
        intrinsics: 3x3 camera intrinsics matrix
        pose_c2w: 4x4 camera-to-world transformation matrix
        image_shape: (H, W) tuple
        mask_2d: (H, W) boolean mask - only keep points projecting into True pixels

    Returns:
        keep_indices: boolean array of shape (N,), True for points to keep
    """
    H, W = image_shape
    N = len(points_world)
    keep = np.zeros(N, dtype=bool)

    # Transform points to camera space
    R_c2w = pose_c2w[:3, :3]
    t_c2w = pose_c2w[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w

    points_cam = (R_w2c @ points_world.T).T + t_w2c  # (N, 3)

    # Points behind camera are not visible
    in_front = points_cam[:, 2] > 0
    if not in_front.any():
        return keep

    # Project to image space
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x_2d = (fx * points_cam[:, 0] / points_cam[:, 2] + cx).astype(np.int32)
    y_2d = (fy * points_cam[:, 1] / points_cam[:, 2] + cy).astype(np.int32)

    # Check bounds and mask membership (vectorized)
    in_bounds = in_front & (x_2d >= 0) & (x_2d < W) & (y_2d >= 0) & (y_2d < H)
    valid_idx = np.where(in_bounds)[0]
    if len(valid_idx) > 0:
        keep[valid_idx] = mask_2d[y_2d[valid_idx], x_2d[valid_idx]]

    return keep


def project_points_to_mask(points_world, intrinsics, pose_c2w, image_shape, dilate_pixels=0):
    """
    Project 3D world points to a 2D binary mask.

    Args:
        points_world: (N, 3) array of world-space points
        intrinsics: 3x3 camera intrinsics matrix
        pose_c2w: 4x4 camera-to-world transformation matrix
        image_shape: (H, W) tuple for output mask size
        dilate_pixels: optional dilation to expand the mask

    Returns:
        mask: (H, W) boolean mask where projected points land
    """
    H, W = image_shape

    # Transform points to camera space
    R_c2w = pose_c2w[:3, :3]
    t_c2w = pose_c2w[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w

    points_cam = (R_w2c @ points_world.T).T + t_w2c  # (N, 3)

    # Filter points behind camera
    valid = points_cam[:, 2] > 0
    points_cam = points_cam[valid]

    if len(points_cam) == 0:
        return np.zeros((H, W), dtype=bool)

    # Project to image space
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x_2d = (fx * points_cam[:, 0] / points_cam[:, 2] + cx).astype(np.int32)
    y_2d = (fy * points_cam[:, 1] / points_cam[:, 2] + cy).astype(np.int32)

    # Filter points within image bounds
    valid = (x_2d >= 0) & (x_2d < W) & (y_2d >= 0) & (y_2d < H)
    x_2d = x_2d[valid]
    y_2d = y_2d[valid]

    # Create mask
    mask = np.zeros((H, W), dtype=bool)
    mask[y_2d, x_2d] = True

    # Fill gaps in the mask using morphological closing (fills interior holes)
    # dilate_pixels controls the kernel size for closing
    if dilate_pixels > 0:
        # Close to fill interior gaps between sparse projected points
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_pixels*2+1, dilate_pixels*2+1))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel).astype(bool)

    return mask



def render_color_from_points(points_world, colors, intrinsics, pose_c2w, image_shape):
    """
    Render a color image from world-space points using z-buffering.

    Args:
        points_world: (N, 3) array of world-space points
        colors: (N, 3) array of RGB colors (0-255)
        intrinsics: 3x3 camera intrinsics matrix
        pose_c2w: 4x4 camera-to-world transformation matrix
        image_shape: (H, W) tuple for output image size

    Returns:
        color_image: (H, W, 3) RGB image
    """
    H, W = image_shape

    # Transform points to camera space
    R_c2w = pose_c2w[:3, :3]
    t_c2w = pose_c2w[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w

    points_cam = (R_w2c @ points_world.T).T + t_w2c  # (N, 3)

    # Filter points behind camera
    valid = points_cam[:, 2] > 0
    points_cam = points_cam[valid]
    colors_valid = colors[valid]

    if len(points_cam) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    # Project to image space
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x_2d = (fx * points_cam[:, 0] / points_cam[:, 2] + cx).astype(np.int32)
    y_2d = (fy * points_cam[:, 1] / points_cam[:, 2] + cy).astype(np.int32)
    z = points_cam[:, 2]

    # Filter points within image bounds
    valid = (x_2d >= 0) & (x_2d < W) & (y_2d >= 0) & (y_2d < H)
    x_2d = x_2d[valid]
    y_2d = y_2d[valid]
    z = z[valid]
    colors_valid = colors_valid[valid]

    if len(z) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    # Sort by depth (furthest first so closest overwrites)
    order = np.argsort(-z)
    x_2d = x_2d[order]
    y_2d = y_2d[order]
    colors_valid = colors_valid[order]

    # Render color image
    color_image = np.zeros((H, W, 3), dtype=np.uint8)
    color_image[y_2d, x_2d] = colors_valid[:, :3]

    return color_image


def render_depth_from_points(points_world, intrinsics, pose_c2w, depth_shape):
    """
    Render a depth image from world-space points using z-buffering.

    Args:
        points_world: (N, 3) array of world-space points
        intrinsics: 3x3 camera intrinsics matrix
        pose_c2w: 4x4 camera-to-world transformation matrix
        depth_shape: (H, W) tuple for output depth map size

    Returns:
        depth_map: (H, W) depth image
    """
    H, W = depth_shape

    # Transform points to camera space
    R_c2w = pose_c2w[:3, :3]
    t_c2w = pose_c2w[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w

    points_cam = (R_w2c @ points_world.T).T + t_w2c  # (N, 3)

    # Filter points behind camera
    valid = points_cam[:, 2] > 0
    points_cam = points_cam[valid]

    if len(points_cam) == 0:
        return np.zeros((H, W), dtype=np.float32)

    # Project to image space
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x_2d = (fx * points_cam[:, 0] / points_cam[:, 2] + cx).astype(np.int32)
    y_2d = (fy * points_cam[:, 1] / points_cam[:, 2] + cy).astype(np.int32)
    z = points_cam[:, 2]

    # Filter points within image bounds
    valid = (x_2d >= 0) & (x_2d < W) & (y_2d >= 0) & (y_2d < H)
    x_2d = x_2d[valid]
    y_2d = y_2d[valid]
    z = z[valid]

    if len(z) == 0:
        return np.zeros((H, W), dtype=np.float32)

    # Sort by depth (furthest first so closest overwrites)
    order = np.argsort(-z)
    x_2d = x_2d[order]
    y_2d = y_2d[order]
    z = z[order]

    # Render depth map
    depth_map = np.zeros((H, W), dtype=np.float32)
    depth_map[y_2d, x_2d] = z

    return depth_map


def unproject_depth(depth, intrinsics, pose_c2w, color=None, exclude_mask=None):
    """
    Unproject depth map to 3D world points.

    Args:
        depth: (H, W) depth map
        intrinsics: 3x3 camera intrinsics
        pose_c2w: 4x4 camera-to-world pose
        color: optional (H, W, 3) color image
        exclude_mask: optional (H, W) boolean mask - True pixels will be excluded
    """
    H, W = depth.shape
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Mask valid depth
    valid = depth > 0

    # Apply exclusion mask if provided
    if exclude_mask is not None:
        valid = valid & ~exclude_mask

    z = depth[valid]
    x = x[valid]
    y = y[valid]
    
    # Unproject to camera space
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z
    
    points_cam = np.stack([X, Y, Z], axis=-1) # (N, 3)
    
    # Transform to world space
    # P_world = R_c2w * P_cam + t_c2w
    # pose_c2w is 4x4
    
    R = pose_c2w[:3, :3]
    t = pose_c2w[:3, 3]
    
    points_world = (R @ points_cam.T).T + t
    
    colors = None
    if color is not None:
        colors = color[valid]
        # Ensure RGB
        if colors.shape[-1] == 3:
            pass # already 3 channels
        elif colors.shape[-1] == 4:
            colors = colors[:, :3]
            
    return points_world, colors

def sh_to_rgb(sh_dc):
    """
    Convert spherical harmonics DC component to RGB colors.

    Args:
        sh_dc: numpy array of shape (N, 3) with f_dc_0, f_dc_1, f_dc_2

    Returns:
        RGB colors as uint8 array of shape (N, 3)
    """
    # Apply sigmoid: 1 / (1 + exp(-x))
    rgb = 1.0 / (1.0 + np.exp(-sh_dc))
    # Scale to 0-255 range
    rgb = (rgb * 255).astype(np.uint8)
    return rgb

def main():
    parser = argparse.ArgumentParser(description="Generate combined PLY files (Scene + Objects)")
    parser.add_argument("--ttt3r_out", type=str, required=True, help="Path to TTT3R output directory")
    parser.add_argument("--object_dir", type=str, default=None, help="Path to object reconstruction directory (e.g., output_tracked_objects/last_4_seconds)")
    parser.add_argument("--output_dir", type=str, default="depth", help="Directory to save PLY files")
    parser.add_argument("--contrast", action="store_true", help="Use bright purple color for objects instead of natural colors")
    parser.add_argument("--stride", type=int, default=1, help="Stride used for frame processing (mapped to original video frame indices)")
    parser.add_argument("--no_object_transform", action="store_true",
        help="Skip object rescaling/repositioning - use raw SAM3D world coordinates")
    parser.add_argument("--render_replaced_depth", action="store_true", default=True,
        help="Excise original object from scene using mask, insert reconstruction, and render depth image")
    parser.add_argument("--no_render_replaced_depth", action="store_false", dest="render_replaced_depth",
        help="Disable excision/insertion rendering")
    parser.add_argument("--mask_mode", type=str,
        choices=["original", "reconstructed", "intersection", "all"],
        default="intersection",
        help="Mask excision mode: 'original' (SAM3D mask), 'reconstructed' (projected recon footprint), "
             "'intersection' (AND of both), 'all' (generate all three)")
    parser.add_argument("--vitra_annotation", type=str, default=None,
        help="Path to VITRA annotation .npy file (used for frame filtering)")
    parser.add_argument("--skip_visualizations", action="store_true",
        help="Skip rendering PNG depth/color visualizations (saves ~2-5s per frame)")
    parser.add_argument("--export_obj", action="store_true", default=True,
        help="Export object reconstructions as Wavefront OBJ with texture (default: True)")
    parser.add_argument("--no_export_obj", action="store_false", dest="export_obj",
        help="Disable OBJ export of object reconstructions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load VITRA annotation if provided (for frame filtering)
    vitra_ann = None
    if args.vitra_annotation:
        if not os.path.exists(args.vitra_annotation):
            print(f"WARNING: VITRA annotation not found: {args.vitra_annotation}")
        else:
            vitra_ann = np.load(args.vitra_annotation, allow_pickle=True).item()
            print(f"Loaded VITRA annotation: {len(vitra_ann.get('video_decode_frame', []))} frames")

    # Get depth files
    depth_dir = os.path.join(args.ttt3r_out, "depth")
    color_dir = os.path.join(args.ttt3r_out, "color")
    camera_dir = os.path.join(args.ttt3r_out, "camera")

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))

    print(f"Found {len(depth_files)} frames.")

    # Filter out margin frames (frames outside the VITRA annotation range)
    if vitra_ann is not None:
        vdf = vitra_ann.get('video_decode_frame')
        if vdf is not None:
            from rescale_to_metric import _rebase_vdf
            rebased_vdf = _rebase_vdf(np.asarray(vdf), args.vitra_annotation)
            vdf_set = set(int(v) for v in rebased_vdf)
            before = len(depth_files)
            depth_files = [
                dp for dp in depth_files
                if int(os.path.splitext(os.path.basename(dp))[0]) in vdf_set
            ]
            if len(depth_files) < before:
                print(f"Excluded {before - len(depth_files)} margin frames "
                      f"(keeping {len(depth_files)} annotated frames)")

    for depth_path in depth_files:
        basename = os.path.splitext(os.path.basename(depth_path))[0]
        
        # Load depth
        depth_map = np.load(depth_path)
        
        # Load color
        color_path = os.path.join(color_dir, f"{basename}.png")
        if not os.path.exists(color_path):
             # Try jpg
             color_path = os.path.join(color_dir, f"{basename}.jpg")
        
        if os.path.exists(color_path):
            color_img = cv2.imread(color_path)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            # Resize color to match depth if needed
            if color_img.shape[:2] != depth_map.shape:
                color_img = cv2.resize(color_img, (depth_map.shape[1], depth_map.shape[0]))
        else:
            color_img = None
            
        # Load camera
        cam_path = os.path.join(camera_dir, f"{basename}.npz")
        if not os.path.exists(cam_path):
            print(f"Camera file not found for {basename}, skipping.")
            continue
            
        cam_data = np.load(cam_path)
        intrinsics = cam_data['intrinsics']
        pose_c2w = cam_data['pose']

        # Load object mask for excision if render_replaced_depth is enabled
        exclude_mask = None
        if args.render_replaced_depth and args.object_dir is not None:
            obj_frame_idx = int(basename) * args.stride
            # Check for mask in object subdirectories
            obj_subdirs = sorted(glob.glob(os.path.join(args.object_dir, "object_*")))
            if not obj_subdirs:
                obj_subdirs = [args.object_dir]

            # Combine masks from all objects
            for obj_subdir in obj_subdirs:
                mask_binary_path = os.path.join(obj_subdir, f"frame_{obj_frame_idx:04d}_mask.npy")
                if os.path.exists(mask_binary_path):
                    obj_mask = np.load(mask_binary_path)
                    # Resize mask to match depth map
                    if obj_mask.shape != depth_map.shape:
                        obj_mask = cv2.resize(obj_mask.astype(np.uint8),
                                              (depth_map.shape[1], depth_map.shape[0]),
                                              interpolation=cv2.INTER_NEAREST).astype(bool)
                    if exclude_mask is None:
                        exclude_mask = obj_mask
                    else:
                        exclude_mask = exclude_mask | obj_mask

        # 1. Scene Point Cloud (with optional object excision)
        scene_pts, scene_colors = unproject_depth(depth_map, intrinsics, pose_c2w, color_img, exclude_mask)

        # Create Trimesh object for scene
        scene_pcd = trimesh.PointCloud(scene_pts, colors=scene_colors)

        if exclude_mask is not None:
            excised_count = exclude_mask.sum()
            print(f"  Excised {excised_count} pixels from scene using object mask")
        
        # 2. Object Reconstructions Point Cloud
        object_pcd_list = []
        if args.object_dir is not None:
            # Check for object subdirectories (object_0, object_1, etc.)
            obj_subdirs = sorted(glob.glob(os.path.join(args.object_dir, "object_*")))
            
            # If no subdirectories found, assume flat structure (single object in root)
            if not obj_subdirs:
                obj_subdirs = [args.object_dir]

            for obj_subdir in obj_subdirs:
                # Calculate original video frame index for SAM3D
                # TTT3R outputs are usually 0, 1, 2... which correspond to video frames 0, stride, 2*stride...
                obj_frame_idx = int(basename) * args.stride
                
                obj_ply_path = os.path.join(obj_subdir, f"frame_{obj_frame_idx:04d}.ply")
                mask_binary_path = os.path.join(obj_subdir, f"frame_{obj_frame_idx:04d}_mask.npy")
                pose_path = os.path.join(obj_subdir, f"frame_{obj_frame_idx:04d}_pose.npy")

                if os.path.exists(obj_ply_path):
                    # Use PlyData to manually extract Gaussian splat properties
                    plydata = PlyData.read(obj_ply_path)

                    # Extract vertices (already in world coordinates from track_objects.py)
                    obj_pts_world = np.stack([
                        np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])
                    ], axis=1)

                    M = None  # similarity transform, computed below if needed
                    if args.no_object_transform:
                        # Use raw SAM3D world coordinates - no transformation
                        obj_pts_transformed = obj_pts_world
                        print(f"  Using raw SAM3D world coordinates (no transform)")
                    else:
                        # Load and process mask to compute transformation
                        mask = np.load(mask_binary_path)

                        # Resize mask to match depth map if needed
                        if mask.shape != depth_map.shape:
                            mask = cv2.resize(mask.astype(np.uint8),
                                                (depth_map.shape[1], depth_map.shape[0]),
                                                interpolation=cv2.INTER_NEAREST).astype(bool)

                        # Find masked region to sample depth
                        y_coords, x_coords = np.where(mask)
                        if len(y_coords) == 0:
                            print(f"  Warning: Empty mask for {obj_ply_path}, skipping")
                            continue

                        mask_center_y = int(np.median(y_coords))
                        mask_center_x = int(np.median(x_coords))
                        mask_h = y_coords.max() - y_coords.min()
                        mask_w = x_coords.max() - x_coords.min()
                        mask_diag_pix = np.sqrt(mask_h**2 + mask_w**2)

                        # Use median depth across the mask (robust to edges/background)
                        mask_depths = depth_map[mask]
                        valid_mask_depths = mask_depths[mask_depths > 0]
                        if len(valid_mask_depths) == 0:
                            print(f"  Warning: No valid depth in mask for {obj_ply_path}, skipping")
                            continue
                        ttt3r_depth = float(np.median(valid_mask_depths))

                        # Compute Transformation Matrix M
                        # 1. Analyze Object in Camera Space
                        R_c2w = pose_c2w[:3, :3]
                        t_c2w = pose_c2w[:3, 3]
                        R_w2c = R_c2w.T
                        t_w2c = -R_w2c @ t_c2w

                        obj_pts_cam = (R_w2c @ obj_pts_world.T).T + t_w2c
                        obj_centroid_cam = obj_pts_cam.mean(axis=0)
                        obj_pts_centered = obj_pts_cam - obj_centroid_cam

                        obj_min, obj_max = obj_pts_centered.min(axis=0), obj_pts_centered.max(axis=0)
                        obj_diag_cam = np.sqrt((obj_max[0]-obj_min[0])**2 + (obj_max[1]-obj_min[1])**2)

                        if obj_diag_cam < 1e-6:
                            print("  Warning: Object has near-zero size, skipping.")
                            continue

                        # 2. Compute Scale and Target Position
                        f_avg = (intrinsics[0,0] + intrinsics[1,1]) / 2.0
                        target_physical_diag = (mask_diag_pix * ttt3r_depth) / f_avg
                        scale_factor = target_physical_diag / obj_diag_cam

                        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
                        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                        target_center_cam = np.array([
                            (mask_center_x - cx) * ttt3r_depth / fx,
                            (mask_center_y - cy) * ttt3r_depth / fy,
                            ttt3r_depth
                        ])

                        # 3. Construct Similarity Transform Matrix (World -> World)
                        # P_w_final = scale * P_w + [(1 - scale) * t_c2w + R_c2w @ (target_center - scale * centroid)]
                        M = np.eye(4)
                        M[:3, :3] = scale_factor * np.eye(3)
                        M[:3, 3] = (1 - scale_factor) * t_c2w + R_c2w @ (target_center_cam - scale_factor * obj_centroid_cam)

                        np.save(pose_path, M)
                        print(f"  Computed and saved pose to {pose_path}")

                        # Apply Transformation
                        obj_pts_transformed = (M[:3, :3] @ obj_pts_world.T).T + M[:3, 3]

                    # Color object points by projecting to 2D and sampling the original image
                    if args.contrast:
                        obj_colors = np.tile([0, 0, 255], (len(obj_pts_transformed), 1))
                    elif color_img is not None:
                        # Project transformed world points to camera space, then to 2D
                        R_w2c_obj = pose_c2w[:3, :3].T
                        t_w2c_obj = -R_w2c_obj @ pose_c2w[:3, 3]
                        obj_cam = (R_w2c_obj @ obj_pts_transformed.T).T + t_w2c_obj
                        obj_z = obj_cam[:, 2]
                        valid_z = obj_z > 0
                        H_img, W_img = color_img.shape[:2]
                        ox = np.full(len(obj_pts_transformed), -1, dtype=np.int32)
                        oy = np.full(len(obj_pts_transformed), -1, dtype=np.int32)
                        fx_o, fy_o = intrinsics[0, 0], intrinsics[1, 1]
                        cx_o, cy_o = intrinsics[0, 2], intrinsics[1, 2]
                        ox[valid_z] = (fx_o * obj_cam[valid_z, 0] / obj_z[valid_z] + cx_o).astype(np.int32)
                        oy[valid_z] = (fy_o * obj_cam[valid_z, 1] / obj_z[valid_z] + cy_o).astype(np.int32)
                        in_bounds = valid_z & (ox >= 0) & (ox < W_img) & (oy >= 0) & (oy < H_img)
                        # Default: mid-gray fallback for out-of-bounds
                        obj_colors = np.tile([128, 128, 128], (len(obj_pts_transformed), 1))
                        if in_bounds.any():
                            obj_colors[in_bounds] = color_img[oy[in_bounds], ox[in_bounds]]
                    else:
                        obj_colors = np.tile([255, 255, 0], (len(obj_pts_transformed), 1))

                    obj_pcd = trimesh.PointCloud(obj_pts_transformed, colors=obj_colors)
                    object_pcd_list.append(obj_pcd)
                    print(f"  Loaded and transformed object from {obj_ply_path}")

                    if args.export_obj:
                        # Load the GLB triangle mesh (already in world coords from track_objects.py)
                        # and apply the same similarity transform used for the Gaussian splat points.
                        obj_glb_path = os.path.join(obj_subdir, f"frame_{obj_frame_idx:04d}.glb")
                        obj_obj_path = os.path.join(obj_subdir, f"frame_{obj_frame_idx:04d}.obj")
                        if os.path.exists(obj_glb_path):
                            export_glb_as_obj(obj_glb_path, obj_obj_path, transform=M)
                        else:
                            print(f"  Warning: GLB not found at {obj_glb_path}, skipping OBJ export")
                else:
                    print(f"  No object reconstruction for frame {basename}")

        # Combine all point clouds (default: simple concatenation, overridden below by intersection if available)
        combined_pcd = scene_pcd
        for obj_pcd in object_pcd_list:
            combined_pcd = combined_pcd + obj_pcd

        # Save replaced scene PLY if requested (scene with object excised + reconstruction inserted)
        if args.render_replaced_depth and len(object_pcd_list) > 0:
            # Get depth range from original for consistent visualization
            orig_valid = depth_map > 0
            depth_min = depth_map[orig_valid].min() if orig_valid.any() else 0
            depth_max = depth_map[orig_valid].max() if orig_valid.any() else 1

            # Block A: Original mask excision
            if args.mask_mode in ("original", "all"):
                scene_pts_orig_excised, scene_colors_orig_excised = unproject_depth(
                    depth_map, intrinsics, pose_c2w, color_img, exclude_mask=exclude_mask
                )

                # Save excised scene only (object region removed, no reconstruction)
                excised_colors = scene_colors_orig_excised if scene_colors_orig_excised is not None else np.tile([128, 128, 128], (len(scene_pts_orig_excised), 1))
                excised_pcd = trimesh.PointCloud(scene_pts_orig_excised, colors=excised_colors)
                excised_ply_path = os.path.join(args.output_dir, f"{basename}_excised.ply")
                excised_pcd.export(excised_ply_path)
                print(f"Saved excised scene PLY: {excised_ply_path}")

                # Filter reconstructed object points to only those projecting within original mask
                orig_recon_colors = np.concatenate([
                    np.asarray(obj_pcd.colors)[:, :3] if obj_pcd.colors is not None else np.tile([255, 255, 0], (len(obj_pcd.vertices), 1))
                    for obj_pcd in object_pcd_list
                ], axis=0)
                orig_recon_pts = np.concatenate([np.asarray(obj_pcd.vertices) for obj_pcd in object_pcd_list], axis=0)
                orig_keep = filter_points_by_mask(orig_recon_pts, intrinsics, pose_c2w, depth_map.shape, exclude_mask)
                filtered_orig_recon_pts = orig_recon_pts[orig_keep]
                filtered_orig_recon_colors = orig_recon_colors[orig_keep]
                print(f"Original mask filter: kept {orig_keep.sum()}/{len(orig_recon_pts)} reconstructed points")

                # Create point cloud with: excised scene + filtered reconstructed objects
                combined_points = np.concatenate([scene_pts_orig_excised, filtered_orig_recon_pts], axis=0)
                combined_colors = np.concatenate([excised_colors, filtered_orig_recon_colors], axis=0)

                # Save as PLY
                replaced_pcd = trimesh.PointCloud(combined_points, colors=combined_colors)
                replaced_ply_path = os.path.join(args.output_dir, f"{basename}_replaced.ply")
                replaced_pcd.export(replaced_ply_path)
                print(f"Saved replaced scene PLY: {replaced_ply_path}")

                if not args.skip_visualizations:
                    # Render excised depth
                    excised_depth = render_depth_from_points(scene_pts_orig_excised, intrinsics, pose_c2w, depth_map.shape)
                    excised_depth_vis = excised_depth.copy()
                    valid_mask = excised_depth_vis > 0
                    if valid_mask.any():
                        excised_depth_vis[valid_mask] = (excised_depth_vis[valid_mask] - depth_min) / (depth_max - depth_min + 1e-8) * 255
                        excised_depth_vis = np.clip(excised_depth_vis, 0, 255).astype(np.uint8)
                        excised_depth_vis = cv2.applyColorMap(excised_depth_vis, cv2.COLORMAP_VIRIDIS)
                        excised_depth_vis[~valid_mask] = 0
                        excised_png_path = os.path.join(args.output_dir, f"{basename}_excised_depth.png")
                        cv2.imwrite(excised_png_path, excised_depth_vis)

                    # Render replaced depth
                    replaced_depth = render_depth_from_points(combined_points, intrinsics, pose_c2w, depth_map.shape)
                    replaced_depth_vis = replaced_depth.copy()
                    valid_mask = replaced_depth_vis > 0
                    if valid_mask.any():
                        replaced_depth_vis[valid_mask] = (replaced_depth_vis[valid_mask] - depth_min) / (depth_max - depth_min + 1e-8) * 255
                        replaced_depth_vis = np.clip(replaced_depth_vis, 0, 255).astype(np.uint8)
                        replaced_depth_vis = cv2.applyColorMap(replaced_depth_vis, cv2.COLORMAP_VIRIDIS)
                        replaced_depth_vis[~valid_mask] = 0
                        replaced_png_path = os.path.join(args.output_dir, f"{basename}_replaced_depth.png")
                        cv2.imwrite(replaced_png_path, replaced_depth_vis)

                    # Save original depth for comparison
                    orig_depth_vis = depth_map.copy()
                    if orig_valid.any():
                        orig_depth_vis[orig_valid] = (orig_depth_vis[orig_valid] - depth_min) / (depth_max - depth_min + 1e-8) * 255
                        orig_depth_vis = np.clip(orig_depth_vis, 0, 255).astype(np.uint8)
                        orig_depth_vis = cv2.applyColorMap(orig_depth_vis, cv2.COLORMAP_VIRIDIS)
                        orig_depth_vis[~orig_valid] = 0
                        orig_png_path = os.path.join(args.output_dir, f"{basename}_original_depth.png")
                        cv2.imwrite(orig_png_path, orig_depth_vis)

                    # Render color images
                    excised_color = render_color_from_points(scene_pts_orig_excised, excised_colors, intrinsics, pose_c2w, depth_map.shape)
                    cv2.imwrite(os.path.join(args.output_dir, f"{basename}_excised_color.png"), cv2.cvtColor(excised_color, cv2.COLOR_RGB2BGR))

                    replaced_color = render_color_from_points(combined_points, combined_colors, intrinsics, pose_c2w, depth_map.shape)
                    cv2.imwrite(os.path.join(args.output_dir, f"{basename}_replaced_color.png"), cv2.cvtColor(replaced_color, cv2.COLOR_RGB2BGR))

                    print(f"Saved depth PNGs: {basename}_excised_depth.png, {basename}_replaced_depth.png, {basename}_original_depth.png")
                    print(f"Saved color PNGs: {basename}_excised_color.png, {basename}_replaced_color.png")

            # Compute recon_mask (needed for both "reconstructed" and "intersection" modes)
            recon_mask = None
            obj_points_combined = []
            for obj_pcd in object_pcd_list:
                obj_points_combined.append(np.asarray(obj_pcd.vertices))
            if obj_points_combined:
                recon_obj_pts = np.concatenate(obj_points_combined, axis=0)
                recon_mask = project_points_to_mask(recon_obj_pts, intrinsics, pose_c2w, depth_map.shape, dilate_pixels=0)

            # Block B: Reconstructed footprint excision
            if args.mask_mode in ("reconstructed", "all") and recon_mask is not None:
                # Unproject original depth with reconstructed object region excluded
                scene_pts_recon_excised, scene_colors_recon_excised = unproject_depth(
                    depth_map, intrinsics, pose_c2w, color_img, exclude_mask=recon_mask
                )

                # Save PLY: scene with reconstructed object footprint removed
                recon_excised_colors = scene_colors_recon_excised if scene_colors_recon_excised is not None else np.tile([128, 128, 128], (len(scene_pts_recon_excised), 1))
                recon_excised_pcd = trimesh.PointCloud(scene_pts_recon_excised, colors=recon_excised_colors)
                recon_excised_ply_path = os.path.join(args.output_dir, f"{basename}_excised_by_recon.ply")
                recon_excised_pcd.export(recon_excised_ply_path)
                print(f"Saved excised-by-recon PLY: {recon_excised_ply_path}")

                # Save PLY: excised by recon + reconstructed object inserted
                pts_list = [scene_pts_recon_excised, recon_obj_pts]
                colors_list = [recon_excised_colors] + [
                    np.asarray(obj_pcd.colors)[:, :3] if obj_pcd.colors is not None else np.tile([255, 255, 0], (len(obj_pcd.vertices), 1))
                    for obj_pcd in object_pcd_list
                ]

                recon_replaced_pts = np.concatenate(pts_list, axis=0)
                recon_replaced_colors = np.concatenate(colors_list, axis=0)
                recon_replaced_pcd = trimesh.PointCloud(recon_replaced_pts, colors=recon_replaced_colors)
                recon_replaced_ply_path = os.path.join(args.output_dir, f"{basename}_replaced_by_recon.ply")
                recon_replaced_pcd.export(recon_replaced_ply_path)
                print(f"Saved replaced-by-recon PLY: {recon_replaced_ply_path}")

                if not args.skip_visualizations:
                    # Render depth PNGs for recon-based excision
                    recon_excised_depth = render_depth_from_points(scene_pts_recon_excised, intrinsics, pose_c2w, depth_map.shape)
                    recon_excised_depth_vis = recon_excised_depth.copy()
                    valid_mask = recon_excised_depth_vis > 0
                    if valid_mask.any():
                        recon_excised_depth_vis[valid_mask] = (recon_excised_depth_vis[valid_mask] - depth_min) / (depth_max - depth_min + 1e-8) * 255
                        recon_excised_depth_vis = np.clip(recon_excised_depth_vis, 0, 255).astype(np.uint8)
                        recon_excised_depth_vis = cv2.applyColorMap(recon_excised_depth_vis, cv2.COLORMAP_VIRIDIS)
                        recon_excised_depth_vis[~valid_mask] = 0
                        cv2.imwrite(os.path.join(args.output_dir, f"{basename}_excised_by_recon_depth.png"), recon_excised_depth_vis)

                    recon_replaced_depth = render_depth_from_points(recon_replaced_pts, intrinsics, pose_c2w, depth_map.shape)
                    recon_replaced_depth_vis = recon_replaced_depth.copy()
                    valid_mask = recon_replaced_depth_vis > 0
                    if valid_mask.any():
                        recon_replaced_depth_vis[valid_mask] = (recon_replaced_depth_vis[valid_mask] - depth_min) / (depth_max - depth_min + 1e-8) * 255
                        recon_replaced_depth_vis = np.clip(recon_replaced_depth_vis, 0, 255).astype(np.uint8)
                        recon_replaced_depth_vis = cv2.applyColorMap(recon_replaced_depth_vis, cv2.COLORMAP_VIRIDIS)
                        recon_replaced_depth_vis[~valid_mask] = 0
                        cv2.imwrite(os.path.join(args.output_dir, f"{basename}_replaced_by_recon_depth.png"), recon_replaced_depth_vis)

                    # Render color images
                    recon_excised_color = render_color_from_points(scene_pts_recon_excised, recon_excised_colors, intrinsics, pose_c2w, depth_map.shape)
                    cv2.imwrite(os.path.join(args.output_dir, f"{basename}_excised_by_recon_color.png"), cv2.cvtColor(recon_excised_color, cv2.COLOR_RGB2BGR))

                    recon_replaced_color = render_color_from_points(recon_replaced_pts, recon_replaced_colors, intrinsics, pose_c2w, depth_map.shape)
                    cv2.imwrite(os.path.join(args.output_dir, f"{basename}_replaced_by_recon_color.png"), cv2.cvtColor(recon_replaced_color, cv2.COLOR_RGB2BGR))

                    print(f"Saved recon-based depth PNGs: {basename}_excised_by_recon_depth.png, {basename}_replaced_by_recon_depth.png")
                    print(f"Saved recon-based color PNGs: {basename}_excised_by_recon_color.png, {basename}_replaced_by_recon_color.png")

            # Block C: Intersection excision (tightest)
            if args.mask_mode in ("intersection", "all") and recon_mask is not None and exclude_mask is not None:
                intersection_mask = exclude_mask & recon_mask

                # Unproject original depth with intersection mask excluded
                scene_pts_inter_excised, scene_colors_inter_excised = unproject_depth(
                    depth_map, intrinsics, pose_c2w, color_img, exclude_mask=intersection_mask
                )

                # Save PLY: scene with intersection mask removed
                inter_excised_colors = scene_colors_inter_excised if scene_colors_inter_excised is not None else np.tile([128, 128, 128], (len(scene_pts_inter_excised), 1))
                inter_excised_pcd = trimesh.PointCloud(scene_pts_inter_excised, colors=inter_excised_colors)
                inter_excised_ply_path = os.path.join(args.output_dir, f"{basename}_excised_by_intersection.ply")
                inter_excised_pcd.export(inter_excised_ply_path)
                print(f"Saved excised-by-intersection PLY: {inter_excised_ply_path}")

                # Filter reconstructed object points to only those projecting within intersection mask
                recon_obj_colors = np.concatenate([
                    np.asarray(obj_pcd.colors)[:, :3] if obj_pcd.colors is not None else np.tile([255, 255, 0], (len(obj_pcd.vertices), 1))
                    for obj_pcd in object_pcd_list
                ], axis=0)
                keep_mask = filter_points_by_mask(recon_obj_pts, intrinsics, pose_c2w, depth_map.shape, intersection_mask)
                filtered_recon_pts = recon_obj_pts[keep_mask]
                filtered_recon_colors = recon_obj_colors[keep_mask]
                print(f"Intersection filter: kept {keep_mask.sum()}/{len(recon_obj_pts)} reconstructed points")

                # Save PLY: excised by intersection + filtered reconstructed object inserted
                inter_replaced_pts = np.concatenate([scene_pts_inter_excised, filtered_recon_pts], axis=0)
                inter_replaced_colors = np.concatenate([inter_excised_colors, filtered_recon_colors], axis=0)
                inter_replaced_pcd = trimesh.PointCloud(inter_replaced_pts, colors=inter_replaced_colors)
                inter_replaced_ply_path = os.path.join(args.output_dir, f"{basename}_replaced_by_intersection.ply")
                inter_replaced_pcd.export(inter_replaced_ply_path)
                print(f"Saved replaced-by-intersection PLY: {inter_replaced_ply_path}")

                if not args.skip_visualizations:
                    # Render depth PNGs for intersection-based excision
                    inter_excised_depth = render_depth_from_points(scene_pts_inter_excised, intrinsics, pose_c2w, depth_map.shape)
                    inter_excised_depth_vis = inter_excised_depth.copy()
                    valid_mask = inter_excised_depth_vis > 0
                    if valid_mask.any():
                        inter_excised_depth_vis[valid_mask] = (inter_excised_depth_vis[valid_mask] - depth_min) / (depth_max - depth_min + 1e-8) * 255
                        inter_excised_depth_vis = np.clip(inter_excised_depth_vis, 0, 255).astype(np.uint8)
                        inter_excised_depth_vis = cv2.applyColorMap(inter_excised_depth_vis, cv2.COLORMAP_VIRIDIS)
                        inter_excised_depth_vis[~valid_mask] = 0
                        cv2.imwrite(os.path.join(args.output_dir, f"{basename}_excised_by_intersection_depth.png"), inter_excised_depth_vis)

                    inter_replaced_depth = render_depth_from_points(inter_replaced_pts, intrinsics, pose_c2w, depth_map.shape)
                    inter_replaced_depth_vis = inter_replaced_depth.copy()
                    valid_mask = inter_replaced_depth_vis > 0
                    if valid_mask.any():
                        inter_replaced_depth_vis[valid_mask] = (inter_replaced_depth_vis[valid_mask] - depth_min) / (depth_max - depth_min + 1e-8) * 255
                        inter_replaced_depth_vis = np.clip(inter_replaced_depth_vis, 0, 255).astype(np.uint8)
                        inter_replaced_depth_vis = cv2.applyColorMap(inter_replaced_depth_vis, cv2.COLORMAP_VIRIDIS)
                        inter_replaced_depth_vis[~valid_mask] = 0
                        cv2.imwrite(os.path.join(args.output_dir, f"{basename}_replaced_by_intersection_depth.png"), inter_replaced_depth_vis)

                    # Render color images
                    inter_excised_color = render_color_from_points(scene_pts_inter_excised, inter_excised_colors, intrinsics, pose_c2w, depth_map.shape)
                    cv2.imwrite(os.path.join(args.output_dir, f"{basename}_excised_by_intersection_color.png"), cv2.cvtColor(inter_excised_color, cv2.COLOR_RGB2BGR))

                    inter_replaced_color = render_color_from_points(inter_replaced_pts, inter_replaced_colors, intrinsics, pose_c2w, depth_map.shape)
                    cv2.imwrite(os.path.join(args.output_dir, f"{basename}_replaced_by_intersection_color.png"), cv2.cvtColor(inter_replaced_color, cv2.COLOR_RGB2BGR))

                    print(f"Saved intersection-based depth PNGs: {basename}_excised_by_intersection_depth.png, {basename}_replaced_by_intersection_depth.png")
                    print(f"Saved intersection-based color PNGs: {basename}_excised_by_intersection_color.png, {basename}_replaced_by_intersection_color.png")

                # Use intersection-replaced as the main output
                combined_pcd = inter_replaced_pcd

        # Remove occluded points (behind closer points from camera perspective)
        pts = np.asarray(combined_pcd.vertices)
        cols = np.asarray(combined_pcd.colors)
        keep = remove_occluded_points(pts, intrinsics, pose_c2w, depth_map.shape)
        n_before = len(pts)
        pts = pts[keep]
        cols = cols[keep]
        combined_pcd = trimesh.PointCloud(pts, colors=cols)
        print(f"  Occlusion culling: {n_before} → {len(pts)} points "
              f"({n_before - len(pts)} removed, {100*(n_before-len(pts))/max(n_before,1):.1f}%)")

        # Save depth map as .npy (final output with object reconstructions composited in)
        if len(object_pcd_list) > 0:
            # Re-render depth from the combined (scene + object) point cloud
            final_depth = render_depth_from_points(pts, intrinsics, pose_c2w, depth_map.shape)
        else:
            # No objects — use intermediate depth as-is
            final_depth = depth_map
        depth_npy_path = os.path.join(args.output_dir, f"{basename}.npy")
        np.save(depth_npy_path, final_depth.astype(np.float32))
        print(f"Saved {depth_npy_path}")

if __name__ == "__main__":
    main()
