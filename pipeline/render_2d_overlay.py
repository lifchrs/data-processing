#!/usr/bin/env python3
"""
Full pipeline: SAM3 segmentation -> SAM3D reconstruction -> Visualization
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
sys.path.append("sam-3d-objects/notebook")
sys.path.insert(0, "sam-3d-objects/sam3d_objects")

import numpy as np
import cv2
import torch
import trimesh
import pyrender
from PIL import Image
from inference import Inference
from scipy.spatial.transform import Rotation as R
from transformers import Sam3VideoModel, Sam3VideoProcessor
from accelerate import Accelerator


def run_sam3_segmentation_single_image(image_path, prompt, device):
    """Run SAM3 segmentation on a single image."""
    print(f"Loading SAM3 model...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    # Load image as single-frame "video"
    image = Image.open(image_path).convert('RGB')
    video_frames = [image]

    print(f"Running SAM3 segmentation with prompt: '{prompt}'")
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    inference_session = processor.add_text_prompt(inference_session=inference_session, text=prompt)

    outputs_per_frame = {}
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session, max_frame_num_to_track=None
    ):
        processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
        outputs_per_frame[model_outputs.frame_idx] = processed_outputs

    # Get mask for frame 0
    masks = outputs_per_frame[0]['masks']
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    if len(masks) == 0 or masks[0].sum() == 0:
        raise ValueError(f"No mask found for prompt '{prompt}'")

    mask = masks[0].cpu().numpy().astype(bool)
    return mask, np.array(image)


def run_sam3d_inference(image, mask, output_path):
    """Run SAM3D inference and save output."""
    print("Running SAM3D inference...")
    config_path = "sam-3d-objects/checkpoints/hf/pipeline.yaml"
    inference = Inference(config_path, compile=False)
    output = inference(image, mask, seed=42)

    # Extract outputs and convert to numpy
    output_dict = {
        'intrinsics': output['intrinsics'].cpu().numpy(),
        'rotation': output['rotation'].detach().cpu().numpy(),
        'translation': output['translation'].detach().cpu().numpy(),
        'scale': output['scale'].detach().cpu().numpy(),
        'glb_vertices': output['glb'].vertices,
        'glb_faces': output['glb'].faces,
    }

    if hasattr(output['glb'].visual, 'vertex_colors'):
        output_dict['glb_vertex_colors'] = output['glb'].visual.vertex_colors

    # Also save Gaussian splat data for visualizer
    if 'gs' in output:
        gs = output['gs']
        try:
            output_dict['gs_xyz'] = gs.get_xyz.detach().cpu().numpy()
        except:
            output_dict['gs_xyz'] = gs._xyz.detach().cpu().numpy()
        if hasattr(gs, '_features_dc'):
            output_dict['gs_features'] = gs._features_dc.detach().cpu().numpy()
        if hasattr(gs, '_opacity'):
            output_dict['gs_opacity'] = gs._opacity.detach().cpu().numpy()
        if hasattr(gs, '_rotation'):
            output_dict['gs_rotation'] = gs._rotation.detach().cpu().numpy()
        if hasattr(gs, '_scaling'):
            output_dict['gs_scaling'] = gs._scaling.detach().cpu().numpy()
    if 'pointmap' in output:
        output_dict['pointmap'] = output['pointmap'].detach().cpu().numpy()

    np.save(output_path, output_dict, allow_pickle=True)
    print(f"Saved SAM3D output to {output_path}")
    return output_dict


def transform_glb_to_camera_space(vertices, rotation, translation, scale):
    """Transform GLB mesh vertices from Y-up to camera space."""
    vertices = vertices.copy()

    # Convert from Y-up to Z-up
    R_yup_to_zup = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    vertices = vertices @ R_yup_to_zup

    # Apply pose transformation
    rot_matrix = R.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]]).as_matrix()
    vertices = vertices * scale
    vertices = vertices @ rot_matrix  # Note: no transpose - matches pytorch3d convention
    vertices = vertices + translation.reshape(1, 3)

    # Convert to OpenGL frame
    vertices[:, 0] *= -1
    vertices[:, 2] *= -1

    return vertices


def render_mesh(mesh, intrinsics, image_shape):
    """Render mesh using pyrender."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=10.0)

    scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    scene.add(camera, pose=np.eye(4))
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=np.eye(4))

    renderer = pyrender.OffscreenRenderer(image_shape[1], image_shape[0])
    color, depth = renderer.render(scene)
    renderer.delete()

    return color, depth


def compute_iou(mask1, mask2):
    """Compute IoU between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def visualize(image, gt_mask, data, output_dir):
    """Visualize the reconstruction."""
    img_h, img_w = image.shape[:2]

    # Extract pose
    rotation = data['rotation'].squeeze()
    translation = data['translation'].squeeze()
    scale = data['scale'].squeeze()
    intrinsics_norm = data['intrinsics']

    # Convert intrinsics to pixel coords
    intrinsics = intrinsics_norm.copy()
    intrinsics[0, 0] *= img_w
    intrinsics[1, 1] *= img_h
    intrinsics[0, 2] *= img_w
    intrinsics[1, 2] *= img_h

    # Create and transform mesh
    vertices = data['glb_vertices']
    faces = data['glb_faces']
    vertex_colors = data.get('glb_vertex_colors', None)

    transformed_vertices = transform_glb_to_camera_space(vertices, rotation, translation, scale)

    mesh = trimesh.Trimesh(
        vertices=transformed_vertices,
        faces=faces,
        vertex_colors=vertex_colors
    )

    # Render
    print("Rendering...")
    rendered_color, depth = render_mesh(mesh, intrinsics, image.shape[:2])

    # Create visualization
    silhouette = depth > 0
    output = image.copy()
    output[silhouette] = rendered_color[silhouette]

    # Save outputs
    vis_path = os.path.join(output_dir, "visualization.jpg")
    cv2.imwrite(vis_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {vis_path}")

    comparison_path = os.path.join(output_dir, "comparison.jpg")
    side_by_side = np.hstack([image, output])
    cv2.imwrite(comparison_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
    print(f"Saved comparison to {comparison_path}")

    # Compute IoU
    iou = compute_iou(gt_mask, silhouette)
    print(f"\n=== Results ===")
    print(f"Ground truth mask pixels: {gt_mask.sum()}")
    print(f"Rendered silhouette pixels: {silhouette.sum()}")
    print(f"IoU: {iou:.4f} ({iou*100:.2f}%)")

    if silhouette.sum() > 0:
        print(f"Depth range: {depth[silhouette].min():.3f} to {depth[silhouette].max():.3f}")


def main():
    # Configuration
    IMAGE_PATH = "/home/ap977/GRILL/new_project/first_frame.png"
    PROMPT = "kettle being held"
    OUTPUT_DIR = "/home/ap977/GRILL/new_project/kettle_output"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")

    # Step 1: SAM3 Segmentation
    print("\n=== Step 1: SAM3 Segmentation ===")
    mask, image = run_sam3_segmentation_single_image(IMAGE_PATH, PROMPT, device)
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}, sum: {mask.sum()}")

    # Save mask
    mask_path = os.path.join(OUTPUT_DIR, "mask.npy")
    np.save(mask_path, mask)
    print(f"Saved mask to {mask_path}")

    # Save mask as image
    mask_img_path = os.path.join(OUTPUT_DIR, "mask.png")
    cv2.imwrite(mask_img_path, (mask * 255).astype(np.uint8))
    print(f"Saved mask image to {mask_img_path}")

    # Step 2: SAM3D Reconstruction
    print("\n=== Step 2: SAM3D Reconstruction ===")
    sam3d_output_path = os.path.join(OUTPUT_DIR, "sam3d_output.npy")
    data = run_sam3d_inference(image, mask, sam3d_output_path)

    # Step 3: Visualization
    print("\n=== Step 3: Visualization ===")
    visualize(image, mask, data, OUTPUT_DIR)

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
