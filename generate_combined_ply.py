import argparse
import os
import numpy as np
import cv2
import glob
import trimesh
from pathlib import Path
from plyfile import PlyData

def unproject_depth(depth, intrinsics, pose_c2w, color=None):
    """
    Unproject depth map to 3D world points.
    """
    H, W = depth.shape
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Mask valid depth
    valid = depth > 0
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
        
        # 1. Scene Point Cloud
        scene_pts, scene_colors = unproject_depth(depth_map, intrinsics, pose_c2w, color_img)
        
        # Create Trimesh object for scene
        scene_pcd = trimesh.PointCloud(scene_pts, colors=scene_colors)
        
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
            
            # Use vertices from scaled mesh
            hand_pts_list.append(hand_mesh_world.vertices)
            
            # Color for hand
            N_v = len(hand_mesh.vertices)
            if args.contrast:
                # Bright Pink as requested for contrast mode
                hand_colors_list.append(np.tile([255, 105, 180], (N_v, 1)))
            else:
                # Use existing mesh colors if available, otherwise default skin tone
                if hasattr(hand_mesh.visual, 'vertex_colors') and len(hand_mesh.visual.vertex_colors) == N_v:
                    # Trimesh colors are usually RGBA, take RGB
                    hand_colors_list.append(hand_mesh.visual.vertex_colors[:, :3])
                else:
                    # Default generic skin tone
                    hand_colors_list.append(np.tile([224, 172, 105], (N_v, 1)))
                
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

                if os.path.exists(obj_ply_path) and os.path.exists(mask_binary_path):
                    # Use PlyData to manually extract Gaussian splat properties
                    plydata = PlyData.read(obj_ply_path)

                    # Extract vertices (already in world coordinates from track_objects.py)
                    obj_pts_world = np.stack([
                        np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])
                    ], axis=1)

                    # Load the binary mask
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

                    # Rescale and Reposition Object based on Mask and TTT3R Depth
                    # -----------------------------------------------------------
                    # The original object from sam3d might be at an arbitrary scale and location (often near origin).
                    # To fix this, we:
                    # 1. Transform the object to camera space and measure its canonical size (diagonal).
                    # 2. Measure the object's size in the 2D mask (diagonal pixels).
                    # 3. Determine the target physical size at the TTT3R depth that matches the mask size.
                    # 4. Calculate scale factor = Target Size / Canonical Size.
                    # 5. Position the object at the TTT3R depth, aligned with the mask center.

                    # A. Get Mask Properties (Target 2D Size & Location)
                    # We already have y_coords, x_coords from line 249
                    mask_center_y = int(np.median(y_coords))
                    mask_center_x = int(np.median(x_coords))

                    min_y, max_y = y_coords.min(), y_coords.max()
                    min_x, max_x = x_coords.min(), x_coords.max()
                    mask_h = max_y - min_y
                    mask_w = max_x - min_x
                    mask_diag_pix = np.sqrt(mask_h**2 + mask_w**2)

                    # B. Get Target Depth (TTT3R)
                    # Sample depth at the mask center
                    if mask[mask_center_y, mask_center_x]:
                        sample_x, sample_y = mask_center_x, mask_center_y
                    else:
                        # Fallback if median is hole (unlikely for convex shapes)
                        sample_y, sample_x = y_coords[0], x_coords[0]

                    ttt3r_depth = depth_map[sample_y, sample_x]

                    if ttt3r_depth <= 0:
                        print(f"  Warning: Invalid depth ({ttt3r_depth}) for object at ({sample_x}, {sample_y}), skipping.")
                        continue

                    # C. Analyze Object in Camera Space (Canonical Size)
                    # Transform world points back to camera reference frame
                    # P_cam = R^T * (P_world - t)
                    R_w2c = pose_c2w[:3, :3].T
                    t_w2c = -R_w2c @ pose_c2w[:3, 3]

                    obj_pts_cam = (R_w2c @ obj_pts_world.T).T + t_w2c 

                    # Center the object in camera space to measure pure size
                    obj_centroid_cam = obj_pts_cam.mean(axis=0)
                    obj_pts_centered = obj_pts_cam - obj_centroid_cam


                    # FIX: Coordinate System Mismatch
                    # We fixed track_objects.py to correctly output World Coordinates (handling GL->CV conversion).
                    # So we don't need manual flips here anymore.
                    # obj_pts_centered is now in correct OpenCV Camera Space (relative to centroid).


                    # Measure object size (diagonal of bounding box in X/Y plane)
                    # We focus on X/Y because that corresponds to the image plane mask
                    obj_min = obj_pts_centered.min(axis=0)
                    obj_max = obj_pts_centered.max(axis=0)
                    obj_w_cam = obj_max[0] - obj_min[0]
                    obj_h_cam = obj_max[1] - obj_min[1]
                    obj_diag_cam = np.sqrt(obj_w_cam**2 + obj_h_cam**2)

                    # D. Compute Scale Factor
                    if obj_diag_cam < 1e-6:
                        print("  Warning: Object has near-zero size, skipping.")
                        continue

                    # Target physical size = (Pixels * Depth) / FocalLength
                    f_avg = (intrinsics[0,0] + intrinsics[1,1]) / 2.0
                    target_physical_diag = (mask_diag_pix * ttt3r_depth) / f_avg

                    scale_factor = target_physical_diag / obj_diag_cam

                    print(f"  Object correction: Mask diag={mask_diag_pix:.1f}px, Depth={ttt3r_depth:.3f}m")
                    print(f"                   Org Size={obj_diag_cam:.3f}m, Target Size={target_physical_diag:.3f}m, Scale={scale_factor:.3f}")

                    # E. Position and Scale
                    # Scale the centered points
                    obj_pts_scaled_cam = obj_pts_centered * scale_factor

                    # Calculate target centroid in Camera space
                    # Project mask center to 3D: (u - cx)*Z/fx
                    cx = intrinsics[0, 2]
                    cy = intrinsics[1, 2]
                    fx = intrinsics[0, 0]
                    fy = intrinsics[1, 1]

                    target_X = (mask_center_x - cx) * ttt3r_depth / fx
                    target_Y = (mask_center_y - cy) * ttt3r_depth / fy
                    target_Z = ttt3r_depth

                    target_center_cam = np.array([target_X, target_Y, target_Z])

                    # Move scaled points to target position
                    obj_pts_final_cam = obj_pts_scaled_cam + target_center_cam

                    # F. Transform back to World Space
                    # P_world = R * P_cam + t
                    obj_pts_scaled = (pose_c2w[:3, :3] @ obj_pts_final_cam.T).T + pose_c2w[:3, 3]

                    # Extract colors from SH DC components if available
                    if args.contrast:
                        # Use bright purple for high contrast
                        obj_colors = np.tile([0, 0, 255], (len(obj_pts_scaled), 1))
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
                            obj_colors = np.tile([255, 255, 0], (len(obj_pts_scaled), 1))

                    obj_pcd = trimesh.PointCloud(obj_pts_scaled, colors=obj_colors)
                    object_pcd_list.append(obj_pcd)
                    print(f"  Loaded object from {obj_ply_path}")
                elif os.path.exists(obj_ply_path):
                    print(f"  Warning: Object PLY exists but mask not found at {mask_binary_path}")
            else:
                print(f"  No object reconstruction for frame {basename}")

        # Combine all point clouds
        combined_pcd = scene_pcd
        if hand_pcd is not None:
            combined_pcd = combined_pcd + hand_pcd
        for obj_pcd in object_pcd_list:
            combined_pcd = combined_pcd + obj_pcd

        # Save
        out_path = os.path.join(args.output_dir, f"{basename}.ply")
        combined_pcd.export(out_path)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
