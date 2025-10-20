"""
Example script demonstrating how to use FFB6D model with your own data.

This script shows:
1. How to initialize the model
2. What inputs the model expects
3. How to prepare data from RGB-D images
4. How to process model outputs for pose estimation

Requirements:
- RGB image (480x640)
- Depth image (480x640)
- Camera intrinsics (3x3 matrix)
"""

import torch
import numpy as np
from typing import Dict, Tuple
import cv2

from common import ConfigRandLA
from models.ffb6d_wrapper import FFB6DWrapper, create_model


def prepare_rgbd_data(
    rgb_path: str,
    depth_path: str,
    camera_intrinsics: np.ndarray,
    num_points: int = 12800
) -> Dict[str, torch.Tensor]:
    """
    Prepare input data from RGB-D images for FFB6D model.
    
    This is a simplified example. In practice, you'll need:
    - Proper depth preprocessing (filling, denoising)
    - Normal map estimation (using normalSpeed or other methods)
    - Sampling strategy for point selection
    - Pre-computed indexing tensors for fusion
    
    Args:
        rgb_path: Path to RGB image
        depth_path: Path to depth image (in meters)
        camera_intrinsics: 3x3 camera intrinsic matrix
        num_points: Number of points to sample
        
    Returns:
        Dictionary of input tensors (NOT COMPLETE - see notes below)
    """
    # Load images
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
    # Normalize RGB to [0, 1]
    rgb = rgb.astype(np.float32) / 255.0
    depth = depth.astype(np.float32) / 1000.0  # Convert to meters if needed
    
    # Prepare RGB tensor [1, 3, H, W]
    rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    
    # Generate point cloud from depth
    h, w = depth.shape
    
    # Create coordinate grids
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Back-project to 3D using camera intrinsics
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    z = depth
    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy
    
    # Stack to get point cloud
    points_3d = np.stack([x_3d, y_3d, z], axis=-1)  # [H, W, 3]
    
    # Filter valid points (where depth > 0)
    valid_mask = z > 0
    valid_points = points_3d[valid_mask]  # [N_valid, 3]
    valid_rgb = rgb[valid_mask]  # [N_valid, 3]
    
    # Sample num_points from valid points
    if len(valid_points) > num_points:
        indices = np.random.choice(len(valid_points), num_points, replace=False)
        sampled_points = valid_points[indices]
        sampled_rgb = valid_rgb[indices]
    else:
        # Pad if not enough points
        sampled_points = np.zeros((num_points, 3))
        sampled_rgb = np.zeros((num_points, 3))
        sampled_points[:len(valid_points)] = valid_points
        sampled_rgb[:len(valid_points)] = valid_rgb
    
    # NOTE: You need to compute normals here
    # For example using normalSpeed library or other methods
    # For now, using zeros as placeholder
    normals = np.zeros_like(sampled_points)  # [N, 3]
    
    # Combine point cloud with RGB and normals [9, N]
    cld_rgb_nrm = np.concatenate([
        sampled_points.T,  # [3, N]
        sampled_rgb.T,     # [3, N]
        normals.T          # [3, N]
    ], axis=0)
    cld_rgb_nrm_tensor = torch.from_numpy(cld_rgb_nrm).unsqueeze(0)
    
    # NOTE: The following tensors need to be properly computed based on
    # your specific sampling strategy and fusion requirements.
    # This is a placeholder showing the structure.
    
    inputs = {
        'rgb': rgb_tensor,
        'cld_rgb_nrm': cld_rgb_nrm_tensor,
        # 'choose': ...,  # Selection indices [1, 1, N]
        # 'cld_xyz0': ..., 'cld_nei_idx0': ..., 'cld_sub_idx0': ...,
        # ... (all other required tensors)
    }
    
    print("\n" + "="*80)
    print("IMPORTANT NOTE:")
    print("="*80)
    print("This function provides a SIMPLIFIED example of data preparation.")
    print("For actual usage, you need to implement:")
    print("  1. Proper depth preprocessing (filling holes, denoising)")
    print("  2. Normal map estimation (using normalSpeed or similar)")
    print("  3. Point cloud downsampling with proper indices")
    print("  4. K-NN computation for local feature aggregation")
    print("  5. RGB-Point fusion index computation")
    print("\nPlease refer to the dataset loaders in:")
    print("  - ffb6d/datasets/ycb/ycb_dataset.py")
    print("  - ffb6d/datasets/linemod/linemod_dataset.py")
    print("\nFor complete implementation details.")
    print("="*80 + "\n")
    
    return inputs


def example_inference():
    """
    Example of model inference (without complete input preparation).
    """
    print("\n" + "="*80)
    print("FFB6D Model Usage Example")
    print("="*80 + "\n")
    
    # 1. Create model for YCB dataset
    print("1. Creating model...")
    model = create_model(dataset='ycb', checkpoint=None)
    model.print_architecture()
    
    # 2. Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n2. Using device: {device}")
    model = model.to(device)
    model.eval()
    
    # 3. Print input specifications
    print("\n3. Model Input Specification:")
    print("-" * 80)
    input_spec = model.get_input_spec()
    for name, shape in list(input_spec.items())[:10]:  # Show first 10
        print(f"  {name}: {shape}")
    print("  ... (and more indexing tensors)")
    
    # 4. Print output specifications
    print("\n4. Model Output Specification:")
    print("-" * 80)
    output_spec = model.get_output_spec()
    for name, shape in output_spec.items():
        print(f"  {name}: {shape}")
    
    print("\n5. Example Forward Pass:")
    print("-" * 80)
    print("To run inference, you need to prepare all required input tensors.")
    print("See the dataset loaders for complete implementation:")
    print("  - YCB: ffb6d/datasets/ycb/ycb_dataset.py")
    print("  - LineMOD: ffb6d/datasets/linemod/linemod_dataset.py")
    
    print("\n6. Loading Pretrained Models:")
    print("-" * 80)
    print("To load a pretrained model:")
    print("  model = FFB6DWrapper.load_pretrained(")
    print("      checkpoint_path='path/to/checkpoint.pth.tar',")
    print("      n_classes=22,  # 22 for YCB, 2 for LineMOD")
    print("      n_pts=12800,")
    print("      n_kps=8")
    print("  )")
    
    print("\n" + "="*80)
    print("For complete training and evaluation scripts, see:")
    print("  - train_ycb.py")
    print("  - train_lm.py")
    print("  - demo.py")
    print("="*80 + "\n")


def example_with_dummy_data():
    """
    Example with dummy data to show the complete forward pass structure.
    """
    print("\n" + "="*80)
    print("Example with Dummy Data (for structure demonstration)")
    print("="*80 + "\n")
    
    # Create model
    model = create_model(dataset='ycb', checkpoint=None)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # Create dummy inputs with correct structure
    batch_size = 1
    n_pts = 12800
    
    print("Creating dummy input tensors...")
    inputs = {
        'rgb': torch.randn(batch_size, 3, 480, 640).to(device),
        'cld_rgb_nrm': torch.randn(batch_size, 9, n_pts).to(device),
        'choose': torch.randint(0, 480*640, (batch_size, 1, n_pts)).to(device),
    }
    
    # Add downsampling stage tensors (simplified)
    for i in range(4):
        n_pts_stage = n_pts // (4 ** (i+1))
        inputs[f'cld_xyz{i}'] = torch.randn(batch_size, n_pts // (4**i), 3).to(device)
        inputs[f'cld_nei_idx{i}'] = torch.randint(0, n_pts, (batch_size, n_pts // (4**i), 16)).to(device)
        inputs[f'cld_sub_idx{i}'] = torch.randint(0, n_pts, (batch_size, n_pts_stage, 16)).to(device)
        inputs[f'p2r_ds_nei_idx{i}'] = torch.randint(0, n_pts, (batch_size, 480*640//(4**(i+1)), 1)).to(device)
        inputs[f'r2p_ds_nei_idx{i}'] = torch.randint(0, 480*640, (batch_size, n_pts // (4**i), 16)).to(device)
    
    # Add upsampling stage tensors
    for i in range(5):
        inputs[f'cld_interp_idx{i}'] = torch.randint(0, n_pts, (batch_size, n_pts // (4**max(0, 3-i)), 1)).to(device)
    
    for i in range(3):
        inputs[f'p2r_up_nei_idx{i}'] = torch.randint(0, n_pts, (batch_size, 480*640//(4**(2-i)), 1)).to(device)
        inputs[f'r2p_up_nei_idx{i}'] = torch.randint(0, 480*640, (batch_size, n_pts // (4**(2-i)), 16)).to(device)
    
    print("Running forward pass...")
    with torch.no_grad():
        try:
            outputs = model(inputs)
            
            print("\nOutput shapes:")
            for key, value in outputs.items():
                print(f"  {key}: {value.shape}")
            
            print("\nPredictions:")
            print(f"  - Semantic segmentation logits: {outputs['pred_rgbd_segs'].shape}")
            print(f"  - Keypoint offsets: {outputs['pred_kp_ofs'].shape}")
            print(f"  - Center offsets: {outputs['pred_ctr_ofs'].shape}")
            
            print("\n✓ Forward pass successful!")
            
        except Exception as e:
            print(f"\n✗ Error during forward pass: {e}")
            print("\nNote: This is expected with dummy data as tensor dimensions")
            print("need to be carefully matched. See dataset loaders for proper")
            print("input tensor preparation.")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    # Run basic example
    example_inference()
    
    # Uncomment to run with dummy data
    # example_with_dummy_data()
    
    print("\nFor complete working examples, see:")
    print("  - Train: python3 train_ycb.py --help")
    print("  - Evaluate: python3 train_ycb.py -eval_net -checkpoint <path> -test")
    print("  - Demo: python3 -m demo -checkpoint <path> -dataset ycb")
    print()
