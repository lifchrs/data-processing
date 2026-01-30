import argparse
import os
import numpy as np
import cv2
import glob
import trimesh
from pathlib import Path
from plyfile import PlyData

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
    parser = argparse.ArgumentParser(description="Generate combined PLY files (Scene + Hand + Objects)")
    parser.add_argument("--ttt3r_out", type=str, required=True, help="Path to TTT3R output directory")
    parser.add_argument("--wilor_out", type=str, required=True, help="Path to WiLoR output directory")
    parser.add_argument("--object_dir", type=str, default=None, help="Path to object reconstruction directory (e.g., output_tracked_objects/last_4_seconds)")
    parser.add_argument("--output_dir", type=str, default="ply_output", help="Directory to save PLY files")
    parser.add_argument("--contrast", action="store_true", help="Use bright purple color for objects instead of natural colors")
    parser.add_argument("--stride", type=int, default=1, help="Stride used for frame processing (mapped to original video frame indices)")
    parser.add_argument("--no_object_transform", action="store_true",
        help="Skip object rescaling/repositioning - use raw SAM3D world coordinates")
    parser.add_argument("--render_replaced_depth", action="store_true",
        help="Excise original object from scene using mask, insert reconstruction, and render depth image")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get depth files
    depth_dir = os.path.join(args.ttt3r_out, "depth")
    color_dir = os.path.join(args.ttt3r_out, "color")
    camera_dir = os.path.join(args.ttt3r_out, "camera")
    
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))[:14]
    
    print(f"Found {len(depth_files)} frames.")

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
        
        # 2. Hand Point Cloud
        # Look for hand meshes: {basename}_{hand_idx}.obj
        hand_files = glob.glob(os.path.join(args.wilor_out, f"{basename}_*.obj"))
        # Exclude scaled files
        hand_files = [f for f in hand_files if not f.endswith('_scaled.obj')]
        
        hand_pts_list = []
        hand_colors_list = []
        
        for hf in hand_files:
            # Load hand mesh
            hand_mesh = trimesh.load(hf)
            
            # The hand mesh vertices are already in camera space (v + cam_t) as saved by WiLoR
            # BUT WiLoR applies a 180 deg rotation around X for rendering (OpenGL convention).
            # TTT3R/Dust3r uses OpenCV convention (Y down, Z forward).
            # We need to undo that rotation to get back to OpenCV camera frame.
            
            rot_fix = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            hand_mesh.apply_transform(rot_fix)
            
            # Now hand_mesh.vertices are in camera frame (OpenCV convention)
            # Transform to world space WITHOUT scaling yet
            hand_mesh_world = hand_mesh.copy()
            hand_mesh_world.apply_transform(pose_c2w)
            
            # Get camera origin in world coordinates
            cam_origin_world = pose_c2w[:3, 3]
            
            # Compute WiLoR hand distance from camera origin
            hand_center_world = hand_mesh_world.vertices.mean(axis=0)
            wilor_distance = np.linalg.norm(hand_center_world - cam_origin_world)
            
            # Now unproject a hand pixel using TTT3R depth to get reference distance
            # We need the hand's 2D location - we can approximate from the mesh center
            # Project hand center to 2D
            hand_center_cam = hand_mesh.vertices.mean(axis=0)
            
            # Project to 2D: x_2d = fx * X/Z + cx
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]
            
            x_2d = fx * hand_center_cam[0] / hand_center_cam[2] + cx
            y_2d = fy * hand_center_cam[1] / hand_center_cam[2] + cy
            
            # Sample TTT3R depth at this location
            x_int = int(np.clip(x_2d, 0, depth_map.shape[1] - 1))
            y_int = int(np.clip(y_2d, 0, depth_map.shape[0] - 1))
            ttt3r_depth = depth_map[y_int, x_int]
            
            # Unproject this pixel to world space using TTT3R depth
            point_cam = np.array([
                (x_2d - cx) * ttt3r_depth / fx,
                (y_2d - cy) * ttt3r_depth / fy,
                ttt3r_depth
            ])
            
            # Transform to world
            point_world = pose_c2w[:3, :3] @ point_cam + cam_origin_world
            
            # Compute TTT3R distance from camera origin
            ttt3r_distance = np.linalg.norm(point_world - cam_origin_world)
            
            # Compute scale factor
            scale_factor = ttt3r_distance / wilor_distance if wilor_distance > 0 else 1.0
            
            print(f"  Hand scaling: WiLoR dist={wilor_distance:.3f}m, TTT3R dist={ttt3r_distance:.3f}m, scale={scale_factor:.3f}x")
            
            # Scale the hand from camera origin
            # Move to camera origin, scale, move back
            hand_mesh_world.vertices = (hand_mesh_world.vertices - cam_origin_world) * scale_factor + cam_origin_world

            # Sample dense points from mesh surface instead of just using vertices
            num_samples = 50000  # Dense sampling for good visualization
            sampled_pts, face_indices = hand_mesh_world.sample(num_samples, return_index=True)
            hand_pts_list.append(sampled_pts)

            # Color for hand - sample colors at the sampled points
            N_samples = len(sampled_pts)
            if args.contrast:
                # Bright Pink as requested for contrast mode
                hand_colors_list.append(np.tile([255, 105, 180], (N_samples, 1)))
            else:
                # Use existing mesh colors if available, otherwise default skin tone
                if hasattr(hand_mesh_world.visual, 'vertex_colors') and hand_mesh_world.visual.vertex_colors is not None:
                    # Interpolate vertex colors at sampled face locations
                    vertex_colors = hand_mesh_world.visual.vertex_colors[:, :3]
                    faces = hand_mesh_world.faces[face_indices]
                    # Use barycentric interpolation (simplified: just use first vertex color of each face)
                    sampled_colors = vertex_colors[faces[:, 0]]
                    hand_colors_list.append(sampled_colors)
                else:
                    # Default generic skin tone
                    hand_colors_list.append(np.tile([224, 172, 105], (N_samples, 1)))
                
        if hand_pts_list:
            all_hand_pts = np.concatenate(hand_pts_list, axis=0)
            all_hand_colors = np.concatenate(hand_colors_list, axis=0)
            hand_pcd = trimesh.PointCloud(all_hand_pts, colors=all_hand_colors)

            # Save separate hand PLY
            hand_out_path = os.path.join(args.output_dir, f"{basename}_hand.ply")
            hand_pcd.export(hand_out_path)
            print(f"Saved {hand_out_path}")
        else:
            hand_pcd = None

        # 3. Object Reconstructions Point Cloud
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

                        # Sample depth at mask center
                        sample_y, sample_x = (mask_center_y, mask_center_x) if mask[mask_center_y, mask_center_x] else (y_coords[0], x_coords[0])
                        ttt3r_depth = depth_map[sample_y, sample_x]

                        if ttt3r_depth <= 0:
                            print(f"  Warning: Invalid depth ({ttt3r_depth}) for object at ({sample_x}, {sample_y}), skipping.")
                            continue

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

                    # Extract colors from SH DC components if available
                    if args.contrast:
                        # Use bright purple for high contrast (Note: [0, 0, 255] is Blue, kept original logic)
                        obj_colors = np.tile([0, 0, 255], (len(obj_pts_transformed), 1))
                    else:
                        try:
                            sh_dc = np.stack([
                                np.asarray(plydata.elements[0]["f_dc_0"]),
                                np.asarray(plydata.elements[0]["f_dc_1"]),
                                np.asarray(plydata.elements[0]["f_dc_2"])
                            ], axis=1)
                            obj_colors = sh_to_rgb(sh_dc)
                        except (KeyError, ValueError):
                            # Fallback: Yellow for objects if no color data
                            obj_colors = np.tile([255, 255, 0], (len(obj_pts_transformed), 1))

                    obj_pcd = trimesh.PointCloud(obj_pts_transformed, colors=obj_colors)
                    object_pcd_list.append(obj_pcd)
                    print(f"  Loaded and transformed object from {obj_ply_path}")
                else:
                    print(f"  No object reconstruction for frame {basename}")

        # Combine all point clouds
        combined_pcd = scene_pcd
        if hand_pcd is not None:
            combined_pcd = combined_pcd + hand_pcd
        for obj_pcd in object_pcd_list:
            combined_pcd = combined_pcd + obj_pcd

        # Save PLY
        out_path = os.path.join(args.output_dir, f"{basename}.ply")
        combined_pcd.export(out_path)
        print(f"Saved {out_path}")

        # Save replaced scene PLY if requested (scene with object excised + reconstruction inserted)
        if args.render_replaced_depth and len(object_pcd_list) > 0:
            # Save excised scene only (object region removed, no reconstruction)
            excised_colors = scene_colors if scene_colors is not None else np.tile([128, 128, 128], (len(scene_pts), 1))
            excised_pcd = trimesh.PointCloud(scene_pts, colors=excised_colors)
            excised_ply_path = os.path.join(args.output_dir, f"{basename}_excised.ply")
            excised_pcd.export(excised_ply_path)
            print(f"Saved excised scene PLY: {excised_ply_path}")

            # Create point cloud with: excised scene + reconstructed objects (no hands)
            all_points = [scene_pts]
            all_colors = [excised_colors]

            for obj_pcd in object_pcd_list:
                all_points.append(np.asarray(obj_pcd.vertices))
                obj_colors = np.asarray(obj_pcd.colors)[:, :3] if obj_pcd.colors is not None else np.tile([255, 255, 0], (len(obj_pcd.vertices), 1))
                all_colors.append(obj_colors)

            combined_points = np.concatenate(all_points, axis=0)
            combined_colors = np.concatenate(all_colors, axis=0)

            # Save as PLY
            replaced_pcd = trimesh.PointCloud(combined_points, colors=combined_colors)
            replaced_ply_path = os.path.join(args.output_dir, f"{basename}_replaced.ply")
            replaced_pcd.export(replaced_ply_path)
            print(f"Saved replaced scene PLY: {replaced_ply_path}")

            # Render depth images from point clouds
            # Get depth range from original for consistent visualization
            orig_valid = depth_map > 0
            depth_min = depth_map[orig_valid].min() if orig_valid.any() else 0
            depth_max = depth_map[orig_valid].max() if orig_valid.any() else 1

            # Render excised depth
            excised_depth = render_depth_from_points(scene_pts, intrinsics, pose_c2w, depth_map.shape)
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

            print(f"Saved depth PNGs: {basename}_excised_depth.png, {basename}_replaced_depth.png, {basename}_original_depth.png")

            # Create version excised by RECONSTRUCTED object's projected footprint
            # This removes the region where the reconstruction lands, not the original mask
            obj_points_combined = []
            for obj_pcd in object_pcd_list:
                obj_points_combined.append(np.asarray(obj_pcd.vertices))

            if obj_points_combined:
                recon_obj_pts = np.concatenate(obj_points_combined, axis=0)

                # Project reconstructed object to get its 2D footprint mask
                # Create mask with NO dilation/closing - only excise exactly where recon points project
                recon_mask = project_points_to_mask(recon_obj_pts, intrinsics, pose_c2w, depth_map.shape, dilate_pixels=0)

                # Also project hands to create hand exclusion mask
                if hand_pts_list:
                    hand_mask = project_points_to_mask(all_hand_pts, intrinsics, pose_c2w, depth_map.shape, dilate_pixels=0)
                    # Combine object and hand masks
                    combined_excision_mask = recon_mask | hand_mask
                else:
                    combined_excision_mask = recon_mask

                # Unproject original depth with reconstructed object AND hand regions excluded
                scene_pts_recon_excised, scene_colors_recon_excised = unproject_depth(
                    depth_map, intrinsics, pose_c2w, color_img, exclude_mask=combined_excision_mask
                )

                # Save PLY: scene with reconstructed object footprint removed
                recon_excised_colors = scene_colors_recon_excised if scene_colors_recon_excised is not None else np.tile([128, 128, 128], (len(scene_pts_recon_excised), 1))
                recon_excised_pcd = trimesh.PointCloud(scene_pts_recon_excised, colors=recon_excised_colors)
                recon_excised_ply_path = os.path.join(args.output_dir, f"{basename}_excised_by_recon.ply")
                recon_excised_pcd.export(recon_excised_ply_path)
                print(f"Saved excised-by-recon PLY: {recon_excised_ply_path}")

                # Save PLY: excised by recon + reconstructed object + hands inserted
                pts_list = [scene_pts_recon_excised, recon_obj_pts]
                colors_list = [recon_excised_colors] + [
                    np.asarray(obj_pcd.colors)[:, :3] if obj_pcd.colors is not None else np.tile([255, 255, 0], (len(obj_pcd.vertices), 1))
                    for obj_pcd in object_pcd_list
                ]

                # Add hands if available
                if hand_pts_list:
                    pts_list.append(all_hand_pts)
                    colors_list.append(all_hand_colors)

                recon_replaced_pts = np.concatenate(pts_list, axis=0)
                recon_replaced_colors = np.concatenate(colors_list, axis=0)
                recon_replaced_pcd = trimesh.PointCloud(recon_replaced_pts, colors=recon_replaced_colors)
                recon_replaced_ply_path = os.path.join(args.output_dir, f"{basename}_replaced_by_recon.ply")
                recon_replaced_pcd.export(recon_replaced_ply_path)
                print(f"Saved replaced-by-recon PLY: {recon_replaced_ply_path}")

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

if __name__ == "__main__":
    main()
