# FFB6D Model Architecture Documentation

## Overview

FFB6D (Full Flow Bidirectional Fusion Network for 6D Pose Estimation) is a deep learning model that fuses RGB and point cloud features through bidirectional fusion modules for 6D object pose estimation.

## Model Architecture

### 1. Input Requirements

The model expects a dictionary with the following keys:

#### Required Inputs:
- **`rgb`**: `torch.FloatTensor` of shape `[batch_size, 3, height, width]`
  - RGB image normalized to [0, 1] or standardized
  - Typical size: `[bs, 3, 480, 640]`

- **`cld_rgb_nrm`**: `torch.FloatTensor` of shape `[batch_size, 9, num_points]`
  - Point cloud with RGB and normal information
  - 9 channels: [x, y, z, r, g, b, nx, ny, nz]
  - Default num_points: `480 * 640 // 24 = 12,800` points

- **`choose`**: `torch.LongTensor` of shape `[batch_size, 1, num_points]`
  - Indices to select valid points from RGB feature map
  - Maps between 2D image pixels and 3D points

#### Point Cloud Downsampling Information (for each stage i=0,1,2,3):
- **`cld_xyz{i}`**: `torch.FloatTensor` - XYZ coordinates at stage i
- **`cld_nei_idx{i}`**: `torch.LongTensor` - Neighbor indices for local feature aggregation
- **`cld_sub_idx{i}`**: `torch.LongTensor` - Subsampling indices for downsampling

#### Point Cloud Upsampling Information:
- **`cld_interp_idx{i}`**: `torch.LongTensor` - Interpolation indices for upsampling (i=0,1,2,3,4)

#### RGB-Point Fusion Information:
Downsampling stages (i=0,1,2,3):
- **`p2r_ds_nei_idx{i}`**: Point-to-RGB nearest neighbor indices
- **`r2p_ds_nei_idx{i}`**: RGB-to-Point nearest neighbor indices

Upsampling stages (i=0,1,2):
- **`p2r_up_nei_idx{i}`**: Point-to-RGB nearest neighbor indices
- **`r2p_up_nei_idx{i}`**: RGB-to-Point nearest neighbor indices

### 2. Model Structure

#### A. RGB Branch (PSPNet with ResNet34 backbone)
```
Input: [bs, 3, H, W] RGB image
↓ Conv + BN + ReLU + MaxPool
↓ Layer1 (ResNet): [bs, 64, H/4, W/4]
↓ Layer2 (ResNet): [bs, 128, H/8, W/8]
↓ Layer3+4 (ResNet): [bs, 512, H/8, W/8]
↓ PSP Module: [bs, 1024, H/8, W/8]
↓ Upsampling stages
Output: [bs, 64, H, W]
```

#### B. Point Cloud Branch (RandLA-Net)
```
Input: [bs, 9, N] Point cloud with RGB+Normal
↓ FC Layer: Initial feature extraction
↓ 4 Dilated Residual Blocks (with local spatial encoding)
↓ 4 Upsampling Decoder Blocks
Output: [bs, C, N] Point features
```

#### C. Bidirectional Fusion
At each encoding and decoding stage:
1. **RGB → Point**: RGB features are sampled and projected to point cloud
2. **Point → RGB**: Point features are interpolated back to RGB grid
3. Features are concatenated and fused via 1×1 convolutions

### 3. Model Outputs

The model returns a dictionary `end_points` with:

- **`pred_rgbd_segs`**: `torch.FloatTensor` of shape `[batch_size, num_classes, num_points]`
  - Per-point semantic segmentation logits
  - num_classes includes background + object classes

- **`pred_kp_ofs`**: `torch.FloatTensor` of shape `[batch_size, num_keypoints, num_points, 3]`
  - Per-point offsets to each of the object's 3D keypoints
  - Default num_keypoints = 8

- **`pred_ctr_ofs`**: `torch.FloatTensor` of shape `[batch_size, 1, num_points, 3]`
  - Per-point offsets to the object center

### 4. Network Configuration

Default configuration (from `common.py`):

```python
ConfigRandLA:
    k_n = 16              # KNN neighbors
    num_layers = 4        # Number of encoding/decoding layers
    num_points = 12800    # Number of input points (480*640//24)
    num_classes = 22      # Number of classes (YCB: 21 objects + background)
    sub_sampling_ratio = [4, 4, 4, 4]
    d_out = [32, 64, 128, 256]  # Feature dimensions at each layer
```

### 5. Key Design Principles

1. **Full Flow Fusion**: Fusion happens at every encoding and decoding stage
2. **Bidirectional**: Both RGB→Point and Point→RGB fusion at each stage
3. **Multi-scale**: Features are fused at multiple resolutions
4. **Complementary Information**: RGB provides texture, point cloud provides geometry

### 6. Training Targets

For training, additional inputs are required:

- **`labels`**: `torch.LongTensor` - Per-point semantic labels
- **`kp_targ_ofst`**: `torch.FloatTensor` - Ground truth keypoint offsets
- **`ctr_targ_ofst`**: `torch.FloatTensor` - Ground truth center offsets

### 7. Post-processing

The model predictions are used to:
1. Segment object points from background
2. Vote for 3D keypoint locations using predicted offsets (via mean-shift clustering)
3. Estimate 6D pose from predicted keypoints using PnP algorithm

## Usage Example

```python
import torch
from models.ffb6d import FFB6D
from common import ConfigRandLA

# Initialize model
config = ConfigRandLA
model = FFB6D(
    n_classes=22,
    n_pts=config.num_points,
    rndla_cfg=config,
    n_kps=8
)

# Prepare inputs (example shapes)
batch_size = 1
inputs = {
    'rgb': torch.randn(batch_size, 3, 480, 640),
    'cld_rgb_nrm': torch.randn(batch_size, 9, 12800),
    'choose': torch.randint(0, 480*640, (batch_size, 1, 12800)),
    # ... plus all the indexing tensors for fusion
}

# Forward pass
model.eval()
with torch.no_grad():
    end_points = model(inputs)

# Access predictions
seg_logits = end_points['pred_rgbd_segs']  # [bs, 22, 12800]
kp_offsets = end_points['pred_kp_ofs']     # [bs, 8, 12800, 3]
ctr_offsets = end_points['pred_ctr_ofs']   # [bs, 1, 12800, 3]
```

## Model Parameters

- **Total Parameters**: ~33.8M
- **RGB Branch**: ResNet34 backbone + PSPNet decoder
- **Point Branch**: RandLA-Net with 4 encoding/decoding layers
- **Fusion Layers**: 1×1 convolutions at each stage

## Hardware Requirements

- **Minimum GPU Memory**: 8GB (with batch_size=1)
- **Recommended**: 16GB+ for batch_size=3
- **Training**: 8x GPUs recommended for original batch size (24 samples)

## References

1. FFB6D Paper: [CVPR 2021](https://arxiv.org/abs/2103.02242)
2. PVN3D (keypoint prediction): [CVPR 2020](https://arxiv.org/abs/1911.04231)
3. RandLA-Net (point cloud backbone): [CVPR 2020](https://arxiv.org/abs/1911.11236)
