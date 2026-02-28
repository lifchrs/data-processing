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


# ── MANO hand model ──────────────────────────────────────────────────────────

MANO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'mano_data', 'MANO_RIGHT_clean.npz')

MANO_LEFT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', 'mano_data', 'MANO_LEFT_clean.npz')


def load_mano_model(path=None):
    """Load the clean MANO model (no chumpy dependency)."""
    path = path or MANO_MODEL_PATH
    data = dict(np.load(path, allow_pickle=True))
    return data


def mano_forward(mano_data, beta, global_orient, hand_pose, transl, is_left=False):
    """
    MANO forward pass: shape + pose blend shapes + LBS.

    Args:
        mano_data: dict from MANO_RIGHT_clean.npz
        beta: (10,) shape parameters
        global_orient: (3, 3) wrist rotation matrix
        hand_pose: (15, 3, 3) finger joint rotation matrices
        transl: (3,) translation
        is_left: if True, mirror for left hand

    Returns:
        vertices: (778, 3) in camera space
        faces: (1538, 3) face indices
    """
    v_template = mano_data['v_template'].copy()  # (778, 3)
    shapedirs = mano_data['shapedirs'].copy()     # (778, 3, 10)
    posedirs = mano_data['posedirs'].copy()       # (778, 3, 135)
    J_regressor = mano_data['J_regressor']         # (16, 778)
    weights = mano_data['weights']                 # (778, 16)
    kintree = mano_data['kintree_table']           # (2, 16)
    faces = mano_data['f'].copy()                  # (1538, 3)

    if is_left:
        # Mirror x for left hand
        v_template[:, 0] *= -1
        shapedirs[:, 0, :] *= -1
        posedirs[:, 0, :] *= -1
        faces = faces[:, ::-1]  # reverse winding
        # Mirror rotations: S @ R @ S where S = diag(-1, 1, 1)
        S = np.diag([-1.0, 1.0, 1.0])
        global_orient = S @ global_orient @ S
        hand_pose = np.array([S @ r @ S for r in hand_pose])
        transl = transl.copy()
        transl[0] *= -1

    # 1. Shape blend shapes
    v_shaped = v_template + np.einsum('ijk,k->ij', shapedirs, beta)

    # 2. Rest-pose joints
    J = J_regressor @ v_shaped  # (16, 3)

    # 3. Full pose rotations: [global_orient, 15 finger joints]
    pose_rotmats = np.concatenate([global_orient[None], hand_pose], axis=0)  # (16, 3, 3)

    # 4. Pose blend shapes (exclude root joint)
    ident = np.tile(np.eye(3), (15, 1, 1))
    pose_feature = (pose_rotmats[1:] - ident).reshape(-1)  # (135,)
    v_posed = v_shaped + np.einsum('ijk,k->ij', posedirs, pose_feature)

    # 5. Linear Blend Skinning
    parents = kintree[0].astype(np.int64)

    local_T = np.zeros((16, 4, 4))
    for i in range(16):
        local_T[i, :3, :3] = pose_rotmats[i]
        local_T[i, :3, 3] = J[0] if i == 0 else J[i] - J[parents[i]]
        local_T[i, 3, 3] = 1.0

    global_T = np.zeros((16, 4, 4))
    global_T[0] = local_T[0]
    for i in range(1, 16):
        global_T[i] = global_T[parents[i]] @ local_T[i]

    # Remove rest-pose offset for proper skinning
    for i in range(16):
        pad_J = np.array([J[i, 0], J[i, 1], J[i, 2], 0.0])
        global_T[i, :, 3] -= global_T[i] @ pad_J

    # Blend transforms per vertex
    T = np.einsum('vj,jab->vab', weights, global_T)  # (778, 4, 4)
    v_homo = np.ones((778, 4))
    v_homo[:, :3] = v_posed
    v_out = np.einsum('vab,vb->va', T, v_homo)[:, :3]

    # Apply translation
    v_out += transl

    if is_left:
        v_out[:, 0] *= -1

    return v_out, faces


def _procrustes_align(src, tgt):
    """Rigid Procrustes: find R, t minimizing ||tgt - (R @ src.T).T - t||."""
    mu_s = src.mean(axis=0)
    mu_t = tgt.mean(axis=0)
    H = (src - mu_s).T @ (tgt - mu_t)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_t - R @ mu_s
    return R, t


def generate_hand_mesh(mano_data, vitra_hand, frame_idx, is_left=False,
                       mano_left_data=None):
    """
    Generate hand mesh vertices and faces for a specific frame.

    For VITRA data, all MANO params (including left hands) are stored in
    MANO_RIGHT convention.  However, left-hand joints_camspace was computed
    using MANO_LEFT with S@R@S correction on hand_pose.  We replicate that:
      - Left hands: use MANO_LEFT model + S@R@S on hand_pose
      - Right hands: use MANO_RIGHT model as-is
    Translation is computed by aligning the wrist joint to the stored
    joints_camspace[0] position (since transl_camspace is deprecated).

    Falls back to Procrustes alignment when MANO_LEFT model is unavailable.

    Args:
        mano_data: loaded MANO_RIGHT model dict
        vitra_hand: dict with VITRA hand params (beta, global_orient_camspace, etc.)
        frame_idx: index into the VITRA temporal dimension
        is_left: whether this is the left hand
        mano_left_data: loaded MANO_LEFT model dict (optional, enables correct
                        left-hand reconstruction without Procrustes)

    Returns:
        vertices: (778, 3) in camera space, or None if frame not valid
        faces: (1538, 3) face indices
    """
    kept = vitra_hand['kept_frames']
    if not kept[frame_idx]:
        return None, None

    beta = vitra_hand['beta']  # (10,)
    global_orient = vitra_hand['global_orient_camspace'][frame_idx]  # (3, 3)
    hand_pose = vitra_hand['hand_pose'][frame_idx]  # (15, 3, 3)

    if is_left and mano_left_data is not None:
        # VITRA left-hand convention: HaWoR corrected global_orient (Y,Z negate
        # in axis-angle) but did NOT correct hand_pose.  VITRA computed joints
        # using MANO_LEFT + S@R@S on hand_pose.  Replicate that here.
        S = np.diag([-1.0, 1.0, 1.0])
        hand_pose_corrected = np.array([S @ r @ S for r in hand_pose])

        # Compute wrist-aligned translation
        verts_zero, faces = mano_forward(
            mano_left_data, beta, global_orient, hand_pose_corrected,
            np.zeros(3), is_left=False
        )
        J_reg = mano_left_data['J_regressor']
        wrist_offset = J_reg[0] @ verts_zero  # wrist position with zero translation

        if 'joints_camspace' in vitra_hand:
            target_wrist = vitra_hand['joints_camspace'][frame_idx][0]
        else:
            target_wrist = wrist_offset  # no correction available

        transl = target_wrist - wrist_offset
        verts, faces = mano_forward(
            mano_left_data, beta, global_orient, hand_pose_corrected,
            transl, is_left=False
        )
    else:
        # Right hand (or left without MANO_LEFT model): use MANO_RIGHT directly
        # Compute wrist-aligned translation
        verts_zero, faces = mano_forward(
            mano_data, beta, global_orient, hand_pose,
            np.zeros(3), is_left=False
        )
        J_reg = mano_data['J_regressor']
        wrist_offset = J_reg[0] @ verts_zero

        if 'joints_camspace' in vitra_hand:
            target_wrist = vitra_hand['joints_camspace'][frame_idx][0]
        else:
            target_wrist = wrist_offset

        transl = target_wrist - wrist_offset
        verts, faces = mano_forward(
            mano_data, beta, global_orient, hand_pose,
            transl, is_left=False
        )

    return verts, faces


def sample_hand_colors(vertices, faces, intrinsics, color_img, num_samples=50000):
    """
    Sample points on hand mesh surface and color them from the image.

    Args:
        vertices: (778, 3) hand mesh vertices in camera space
        faces: (1538, 3) face indices
        intrinsics: 3x3 camera intrinsics
        color_img: (H, W, 3) RGB image
        num_samples: number of surface points to sample

    Returns:
        points: (num_samples, 3) sampled points in camera space
        colors: (num_samples, 3) RGB colors
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    points, face_idx = mesh.sample(num_samples, return_index=True)

    H, W = color_img.shape[:2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Project sampled points to 2D
    z = points[:, 2]
    valid_z = z > 0
    x_2d = np.full(len(points), -1, dtype=np.int32)
    y_2d = np.full(len(points), -1, dtype=np.int32)
    x_2d[valid_z] = (fx * points[valid_z, 0] / z[valid_z] + cx).astype(np.int32)
    y_2d[valid_z] = (fy * points[valid_z, 1] / z[valid_z] + cy).astype(np.int32)

    in_bounds = valid_z & (x_2d >= 0) & (x_2d < W) & (y_2d >= 0) & (y_2d < H)

    # Default skin tone
    colors = np.tile(np.array([224, 172, 105], dtype=np.uint8), (len(points), 1))

    # Sample actual image colors where projection is valid
    if in_bounds.any():
        colors[in_bounds] = color_img[y_2d[in_bounds], x_2d[in_bounds]]

    return points, colors


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


def rasterize_mesh_to_mask(vertices_cam, faces, intrinsics, image_shape):
    """
    Rasterize a 3D mesh to a solid 2D mask by filling projected triangles.

    Args:
        vertices_cam: (V, 3) mesh vertices in camera space
        faces: (F, 3) triangle face indices
        intrinsics: 3x3 camera intrinsics
        image_shape: (H, W) output mask size

    Returns:
        mask: (H, W) boolean mask with solid filled triangles
    """
    H, W = image_shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    z = vertices_cam[:, 2]
    valid = z > 0

    # Project all vertices to 2D
    pts_2d = np.zeros((len(vertices_cam), 2), dtype=np.float32)
    pts_2d[valid, 0] = (fx * vertices_cam[valid, 0] / z[valid] + cx)
    pts_2d[valid, 1] = (fy * vertices_cam[valid, 1] / z[valid] + cy)

    # Filter to faces where all 3 vertices are in front of camera
    face_valid = valid[faces[:, 0]] & valid[faces[:, 1]] & valid[faces[:, 2]]
    valid_faces = faces[face_valid]

    mask = np.zeros((H, W), dtype=np.uint8)
    if len(valid_faces) > 0:
        # Batch rasterize: gather triangle vertices as (N, 3, 2) int32 array
        triangles = pts_2d[valid_faces].astype(np.int32)  # (N, 3, 2)
        cv2.fillPoly(mask, triangles, 1)

    return mask.astype(bool)


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
    parser.add_argument("--wilor_out", type=str, default=None, help="Path to WiLoR output directory (optional)")
    parser.add_argument("--object_dir", type=str, default=None, help="Path to object reconstruction directory (e.g., output_tracked_objects/last_4_seconds)")
    parser.add_argument("--output_dir", type=str, default="ply_output", help="Directory to save PLY files")
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
        help="Path to VITRA annotation .npy file for MANO hand reconstruction")
    parser.add_argument("--mano_model", type=str, default=None,
        help="Path to MANO_RIGHT_clean.npz (default: auto-detect)")
    parser.add_argument("--skip_visualizations", action="store_true",
        help="Skip rendering PNG depth/color visualizations (saves ~2-5s per frame)")
    parser.add_argument("--export_obj", action="store_true", default=True,
        help="Export object reconstructions as Wavefront OBJ with texture (default: True)")
    parser.add_argument("--no_export_obj", action="store_false", dest="export_obj",
        help="Disable OBJ export of object reconstructions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load VITRA annotation and MANO model if provided
    vitra_ann = None
    mano_data = None
    mano_left_data = None
    if args.vitra_annotation:
        if not os.path.exists(args.vitra_annotation):
            print(f"WARNING: VITRA annotation not found: {args.vitra_annotation}")
        else:
            vitra_ann = np.load(args.vitra_annotation, allow_pickle=True).item()
            mano_data = load_mano_model(args.mano_model)
            # Load MANO_LEFT for correct left-hand reconstruction.
            # NOTE: MANO_LEFT_clean.npz must have the shapedirs[:, 0, :] *= -1 fix
            # baked in (smplx bug https://github.com/vchoutas/smplx/issues/48).
            left_model_path = MANO_LEFT_MODEL_PATH
            if os.path.exists(left_model_path):
                mano_left_data = load_mano_model(left_model_path)
                print(f"Loaded MANO_LEFT model for left-hand reconstruction")
            else:
                print(f"WARNING: MANO_LEFT_clean.npz not found at {left_model_path}, "
                      f"left hands will use Procrustes-only alignment")
            print(f"Loaded VITRA annotation: {len(vitra_ann.get('video_decode_frame', []))} frames")
            for hand_name in ['left', 'right']:
                if hand_name in vitra_ann:
                    kf = vitra_ann[hand_name]['kept_frames']
                    print(f"  {hand_name} hand: {kf.sum()}/{len(kf)} valid frames")

    # Get depth files
    depth_dir = os.path.join(args.ttt3r_out, "depth")
    color_dir = os.path.join(args.ttt3r_out, "color")
    camera_dir = os.path.join(args.ttt3r_out, "camera")

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))

    print(f"Found {len(depth_files)} frames.")

    # Pre-compute a global scale factor by comparing hand size in metric (VITRA) vs
    # TTT3R space.  For each valid frame we project VITRA joint pairs to 2D, read
    # TTT3R depth at those pixels, unproject into TTT3R camera space, and compare
    # the 3D distance to the known metric distance from joints_camspace.
    # Joint pairs: wrist(0) ↔ each fingertip (4,8,12,16,20).
    global_hand_scale = None
    vitra_intrinsics = vitra_ann.get('intrinsics', None) if vitra_ann is not None else None
    if vitra_ann is not None and mano_data is not None:
        vitra_frame_indices = vitra_ann.get('video_decode_frame', np.arange(100))
        scale_samples = []
        joint_pairs = [(0, 4), (0, 8), (0, 12), (0, 16), (0, 20)]  # wrist ↔ fingertips
        for hand_name in ['right', 'left']:
            if hand_name not in vitra_ann:
                continue
            vitra_hand = vitra_ann[hand_name]
            if 'joints_camspace' not in vitra_hand:
                continue
            for vi, vf in enumerate(vitra_frame_indices):
                if not vitra_hand['kept_frames'][vi]:
                    continue
                ref_depth_path = os.path.join(depth_dir, f"{vf:06d}.npy")
                ref_cam_path = os.path.join(camera_dir, f"{vf:06d}.npz")
                if not os.path.exists(ref_depth_path) or not os.path.exists(ref_cam_path):
                    continue
                ref_depth = np.load(ref_depth_path)
                ref_cam = np.load(ref_cam_path)
                ttt3r_K = ref_cam['intrinsics']
                H_d, W_d = ref_depth.shape

                # Build projection matrix: VITRA intrinsics scaled to depth map resolution
                if vitra_intrinsics is not None:
                    vitra_W = int(round(2 * vitra_intrinsics[0, 2]))
                    vitra_H = int(round(2 * vitra_intrinsics[1, 2]))
                    proj_fx = vitra_intrinsics[0, 0] * W_d / vitra_W
                    proj_fy = vitra_intrinsics[1, 1] * H_d / vitra_H
                    proj_cx = vitra_intrinsics[0, 2] * W_d / vitra_W
                    proj_cy = vitra_intrinsics[1, 2] * H_d / vitra_H
                else:
                    proj_fx, proj_fy = ttt3r_K[0, 0], ttt3r_K[1, 1]
                    proj_cx, proj_cy = ttt3r_K[0, 2], ttt3r_K[1, 2]

                joints_metric = vitra_hand['joints_camspace'][vi]  # (21, 3) metric

                for ja, jb in joint_pairs:
                    # Metric 3D distance between joint pair
                    metric_dist = np.linalg.norm(joints_metric[ja] - joints_metric[jb])
                    if metric_dist < 0.01:  # skip degenerate
                        continue

                    # Project both joints to 2D
                    pts_ok = True
                    ttt3r_pts = []
                    for jidx in [ja, jb]:
                        pt = joints_metric[jidx]
                        if pt[2] <= 0:
                            pts_ok = False
                            break
                        u = proj_fx * pt[0] / pt[2] + proj_cx
                        v = proj_fy * pt[1] / pt[2] + proj_cy
                        # Skip if outside image with margin
                        if u < 5 or u >= W_d - 5 or v < 5 or v >= H_d - 5:
                            pts_ok = False
                            break
                        ui, vi_ = int(round(u)), int(round(v))
                        d = ref_depth[vi_, ui]
                        if d <= 0:
                            pts_ok = False
                            break
                        # Unproject using TTT3R intrinsics
                        x3 = (ui - ttt3r_K[0, 2]) * d / ttt3r_K[0, 0]
                        y3 = (vi_ - ttt3r_K[1, 2]) * d / ttt3r_K[1, 1]
                        ttt3r_pts.append(np.array([x3, y3, d]))

                    if not pts_ok or len(ttt3r_pts) != 2:
                        continue

                    ttt3r_dist = np.linalg.norm(ttt3r_pts[0] - ttt3r_pts[1])
                    if ttt3r_dist > 0:
                        scale_samples.append(ttt3r_dist / metric_dist)

        if len(scale_samples) > 0:
            # IQR outlier filtering — remove samples outside 1.5×IQR
            arr = np.array(scale_samples)
            q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            filtered = arr[(arr >= lo) & (arr <= hi)]
            if len(filtered) > 0:
                global_hand_scale = float(np.median(filtered))
                print(f"Global hand scale: {global_hand_scale:.4f} "
                      f"(median of {len(filtered)}/{len(scale_samples)} samples after IQR filter, "
                      f"range [{filtered.min():.4f}, {filtered.max():.4f}])")
            else:
                global_hand_scale = float(np.median(arr))
                print(f"Global hand scale: {global_hand_scale:.4f} "
                      f"(median of {len(arr)} samples, IQR filter removed all — using raw)")
        else:
            print("WARNING: Could not compute global hand scale (no valid hand+depth overlap)")

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
        
        # 2. Hand Point Cloud (VITRA MANO or WiLoR)
        hand_pts_list = []
        hand_colors_list = []
        hand_meshes_cam = []  # list of (vertices_cam_scaled, faces) for rasterized masks

        # Compute VITRA-based intrinsics rescaled to depth map resolution for hand projection
        if vitra_intrinsics is not None:
            vitra_W = int(round(2 * vitra_intrinsics[0, 2]))
            vitra_H = int(round(2 * vitra_intrinsics[1, 2]))
            hand_K = np.array([
                [vitra_intrinsics[0, 0] * depth_map.shape[1] / vitra_W, 0, vitra_intrinsics[0, 2] * depth_map.shape[1] / vitra_W],
                [0, vitra_intrinsics[1, 1] * depth_map.shape[0] / vitra_H, vitra_intrinsics[1, 2] * depth_map.shape[0] / vitra_H],
                [0, 0, 1]
            ])
        else:
            hand_K = intrinsics

        if vitra_ann is not None and mano_data is not None:
            # VITRA-based hand reconstruction using MANO forward pass
            # TTT3R frame i corresponds to VITRA temporal index i (both
            # process the same clip sequentially).  video_decode_frame
            # contains original video frame numbers, NOT clip-local indices.
            vitra_idx = int(basename) * args.stride
            n_vitra = len(vitra_ann.get('video_decode_frame', []))
            if vitra_idx >= n_vitra:
                vitra_idx = None
            if vitra_idx is None:
                print(f"  Frame {basename} not in VITRA annotation")

            if vitra_idx is not None:
                for hand_name, is_left in [('right', False), ('left', True)]:
                    if hand_name not in vitra_ann:
                        continue
                    vitra_hand = vitra_ann[hand_name]
                    if vitra_idx >= len(vitra_hand['kept_frames']):
                        continue
                    if not vitra_hand['kept_frames'][vitra_idx]:
                        continue

                    # Generate hand mesh in camera space
                    hand_verts, hand_faces = generate_hand_mesh(
                        mano_data, vitra_hand, vitra_idx, is_left=is_left,
                        mano_left_data=mano_left_data
                    )
                    if hand_verts is None:
                        continue

                    # Sample dense points on mesh surface
                    mesh_tmp = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces, process=False)
                    hand_pts_cam, _ = mesh_tmp.sample(50000, return_index=True)

                    # Color: sample a single pixel at the hand centroid projection
                    # Use VITRA intrinsics (rescaled to color image) since MANO was fit with VITRA
                    if color_img is not None:
                        centroid_cam = hand_verts.mean(axis=0)
                        if centroid_cam[2] > 0:
                            H_c, W_c = color_img.shape[:2]
                            if vitra_intrinsics is not None:
                                vitra_W = int(round(2 * vitra_intrinsics[0, 2]))
                                vitra_H = int(round(2 * vitra_intrinsics[1, 2]))
                                fx_h = vitra_intrinsics[0, 0] * W_c / vitra_W
                                fy_h = vitra_intrinsics[1, 1] * H_c / vitra_H
                                cx_h = vitra_intrinsics[0, 2] * W_c / vitra_W
                                cy_h = vitra_intrinsics[1, 2] * H_c / vitra_H
                            else:
                                fx_h, fy_h = intrinsics[0, 0], intrinsics[1, 1]
                                cx_h, cy_h = intrinsics[0, 2], intrinsics[1, 2]
                            cx_px = int(fx_h * centroid_cam[0] / centroid_cam[2] + cx_h)
                            cy_px = int(fy_h * centroid_cam[1] / centroid_cam[2] + cy_h)
                            cx_px = np.clip(cx_px, 0, W_c - 1)
                            cy_px = np.clip(cy_px, 0, H_c - 1)
                            hand_color = color_img[cy_px, cx_px]
                        else:
                            hand_color = np.array([224, 172, 105], dtype=np.uint8)
                    else:
                        hand_color = np.array([224, 172, 105], dtype=np.uint8)
                    hand_colors = np.tile(hand_color, (len(hand_pts_cam), 1))

                    if args.contrast:
                        hand_colors = np.tile([255, 105, 180], (len(hand_pts_cam), 1))

                    # Transform hand from VITRA metric space to TTT3R camera
                    # space.  Wrist-anchored uniform scaling: anchor wrist at
                    # the correct TTT3R 3D position (from depth map), then
                    # uniformly scale the hand shape around it.  This preserves
                    # the 3D shape (no anisotropic distortion) while placing
                    # the hand at the correct depth and pixel location.
                    if global_hand_scale is not None and vitra_intrinsics is not None:
                        # Compute per-hand local scale from wrist depth
                        wrist_cam = vitra_hand['joints_camspace'][vitra_idx][0]
                        vW = int(round(2 * vitra_intrinsics[0, 2]))
                        vH = int(round(2 * vitra_intrinsics[1, 2]))
                        H_d, W_d = depth_map.shape
                        fxv = vitra_intrinsics[0, 0] * W_d / vW
                        fyv = vitra_intrinsics[1, 1] * H_d / vH
                        cxv = vitra_intrinsics[0, 2] * W_d / vW
                        cyv = vitra_intrinsics[1, 2] * H_d / vH

                        wu = int(np.clip(fxv * wrist_cam[0] / wrist_cam[2] + cxv, 0, W_d - 1))
                        wv = int(np.clip(fyv * wrist_cam[1] / wrist_cam[2] + cyv, 0, H_d - 1))
                        d_at_wrist = depth_map[wv, wu]

                        if d_at_wrist > 0 and wrist_cam[2] > 0:
                            local_scale = float(d_at_wrist / wrist_cam[2])
                        else:
                            local_scale = global_hand_scale

                        # Unproject wrist into TTT3R 3D space using TTT3R intrinsics
                        fxt, fyt = intrinsics[0, 0], intrinsics[1, 1]
                        cxt, cyt = intrinsics[0, 2], intrinsics[1, 2]
                        wrist_ttt3r = np.array([
                            (wu - cxt) * d_at_wrist / fxt,
                            (wv - cyt) * d_at_wrist / fyt,
                            d_at_wrist
                        ])

                        # Uniform scale hand shape around wrist, then place at TTT3R wrist
                        hand_pts_cam_scaled = (hand_pts_cam - wrist_cam) * local_scale + wrist_ttt3r
                        hand_verts_cam_scaled = (hand_verts - wrist_cam) * local_scale + wrist_ttt3r
                        print(f"    local_scale={local_scale:.4f} (depth@wrist={d_at_wrist:.4f}, vitra_z={wrist_cam[2]:.4f})")
                    elif global_hand_scale is not None:
                        hand_pts_cam_scaled = hand_pts_cam * global_hand_scale
                        hand_verts_cam_scaled = hand_verts * global_hand_scale
                    else:
                        hand_pts_cam_scaled = hand_pts_cam
                        hand_verts_cam_scaled = hand_verts

                    # Save scaled mesh for rasterized hand mask
                    hand_meshes_cam.append((hand_verts_cam_scaled, hand_faces))

                    # Transform to world space
                    hand_pts_world = (pose_c2w[:3, :3] @ hand_pts_cam_scaled.T).T + pose_c2w[:3, 3]
                    hand_pts_list.append(hand_pts_world)
                    hand_colors_list.append(hand_colors)
                    print(f"  {hand_name} hand: {len(hand_pts_world)} points")

        elif args.wilor_out is not None:
            # Legacy WiLoR-based hand reconstruction
            hand_files = glob.glob(os.path.join(args.wilor_out, f"{basename}_*.obj"))
            hand_files = [f for f in hand_files if not f.endswith('_scaled.obj')]

            for hf in hand_files:
                hand_mesh = trimesh.load(hf)
                rot_fix = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
                hand_mesh.apply_transform(rot_fix)

                hand_mesh_world = hand_mesh.copy()
                hand_mesh_world.apply_transform(pose_c2w)

                cam_origin_world = pose_c2w[:3, 3]
                hand_center_world = hand_mesh_world.vertices.mean(axis=0)
                wilor_distance = np.linalg.norm(hand_center_world - cam_origin_world)

                hand_center_cam = hand_mesh.vertices.mean(axis=0)
                fx = intrinsics[0, 0]
                fy = intrinsics[1, 1]
                cx_cam = intrinsics[0, 2]
                cy_cam = intrinsics[1, 2]
                x_2d = fx * hand_center_cam[0] / hand_center_cam[2] + cx_cam
                y_2d = fy * hand_center_cam[1] / hand_center_cam[2] + cy_cam
                x_int = int(np.clip(x_2d, 0, depth_map.shape[1] - 1))
                y_int = int(np.clip(y_2d, 0, depth_map.shape[0] - 1))
                ttt3r_depth = depth_map[y_int, x_int]

                point_cam = np.array([
                    (x_2d - cx_cam) * ttt3r_depth / fx,
                    (y_2d - cy_cam) * ttt3r_depth / fy,
                    ttt3r_depth
                ])
                point_world = pose_c2w[:3, :3] @ point_cam + cam_origin_world
                ttt3r_distance = np.linalg.norm(point_world - cam_origin_world)
                sf = ttt3r_distance / wilor_distance if wilor_distance > 0 else 1.0
                hand_mesh_world.vertices = (hand_mesh_world.vertices - cam_origin_world) * sf + cam_origin_world

                sampled_pts, face_indices = hand_mesh_world.sample(50000, return_index=True)
                hand_pts_list.append(sampled_pts)

                N_s = len(sampled_pts)
                if args.contrast:
                    hand_colors_list.append(np.tile([255, 105, 180], (N_s, 1)))
                elif hasattr(hand_mesh_world.visual, 'vertex_colors') and hand_mesh_world.visual.vertex_colors is not None:
                    vc = hand_mesh_world.visual.vertex_colors[:, :3]
                    hand_colors_list.append(vc[hand_mesh_world.faces[face_indices][:, 0]])
                else:
                    hand_colors_list.append(np.tile([224, 172, 105], (N_s, 1)))

        if hand_pts_list:
            all_hand_pts = np.concatenate(hand_pts_list, axis=0)
            all_hand_colors = np.concatenate(hand_colors_list, axis=0)
            hand_pcd = trimesh.PointCloud(all_hand_pts, colors=all_hand_colors)
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

        # Pre-compute hand mask once — reused in all excision blocks below
        # After uniform scaling, hand vertices are in TTT3R camera space → use TTT3R intrinsics
        cached_hand_mask = None
        if hand_pcd is not None:
            if hand_meshes_cam:
                cached_hand_mask = np.zeros(depth_map.shape, dtype=bool)
                for verts_cam, faces in hand_meshes_cam:
                    cached_hand_mask |= rasterize_mesh_to_mask(verts_cam, faces, intrinsics, depth_map.shape)
            else:
                cached_hand_mask = project_points_to_mask(all_hand_pts, intrinsics, pose_c2w, depth_map.shape, dilate_pixels=0)

        if hand_pcd is not None:
            # Apply hand intersection excision: excise scene under hand projection, insert hand
            scene_pts_hand_excised, scene_colors_hand_excised = unproject_depth(
                depth_map, intrinsics, pose_c2w, color_img, exclude_mask=cached_hand_mask
            )
            excised_colors = scene_colors_hand_excised if scene_colors_hand_excised is not None else np.tile([128, 128, 128], (len(scene_pts_hand_excised), 1))
            combined_pcd = trimesh.PointCloud(scene_pts_hand_excised, colors=excised_colors) + hand_pcd
            print(f"  Applied hand excision: removed {cached_hand_mask.sum()} scene pixels, inserted {len(all_hand_pts)} hand points")
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
                # Excise both object and hand regions from scene
                if cached_hand_mask is not None:
                    orig_excision_mask = exclude_mask | cached_hand_mask
                else:
                    orig_excision_mask = exclude_mask

                scene_pts_orig_excised, scene_colors_orig_excised = unproject_depth(
                    depth_map, intrinsics, pose_c2w, color_img, exclude_mask=orig_excision_mask
                )

                # Save excised scene only (object + hand regions removed, no reconstruction)
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

                # Create point cloud with: excised scene + filtered reconstructed objects + hands
                replaced_pts_list = [scene_pts_orig_excised, filtered_orig_recon_pts]
                replaced_colors_list = [excised_colors, filtered_orig_recon_colors]
                if hand_pts_list:
                    replaced_pts_list.append(all_hand_pts)
                    replaced_colors_list.append(all_hand_colors)
                combined_points = np.concatenate(replaced_pts_list, axis=0)
                combined_colors = np.concatenate(replaced_colors_list, axis=0)

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
                # Combine object reconstruction footprint with cached hand mask
                if cached_hand_mask is not None:
                    combined_excision_mask = recon_mask | cached_hand_mask
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
                # Add cached hand mask if available
                if cached_hand_mask is not None:
                    intersection_excision_mask = intersection_mask | cached_hand_mask
                else:
                    intersection_excision_mask = intersection_mask

                # Unproject original depth with intersection mask excluded
                scene_pts_inter_excised, scene_colors_inter_excised = unproject_depth(
                    depth_map, intrinsics, pose_c2w, color_img, exclude_mask=intersection_excision_mask
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

                # Save PLY: excised by intersection + filtered reconstructed object + hands inserted
                inter_pts_list = [scene_pts_inter_excised, filtered_recon_pts]
                inter_colors_list = [inter_excised_colors, filtered_recon_colors]

                # Add hands if available
                if hand_pts_list:
                    inter_pts_list.append(all_hand_pts)
                    inter_colors_list.append(all_hand_colors)

                inter_replaced_pts = np.concatenate(inter_pts_list, axis=0)
                inter_replaced_colors = np.concatenate(inter_colors_list, axis=0)
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

        # Rescale to metric using global hand-derived scale factor.
        # Computed once from the first valid VITRA frame, applied to ALL frames
        # for consistent scale (TTT3R is globally scale-consistent).
        if global_hand_scale is not None and global_hand_scale > 0:
            cam_origin = pose_c2w[:3, 3]
            pts = np.asarray(combined_pcd.vertices)
            cols = np.asarray(combined_pcd.colors)
            pts = (pts - cam_origin) / global_hand_scale + cam_origin
            combined_pcd = trimesh.PointCloud(pts, colors=cols)

        # Save main PLY
        out_path = os.path.join(args.output_dir, f"{basename}.ply")
        combined_pcd.export(out_path)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
