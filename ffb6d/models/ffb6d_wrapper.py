"""
FFB6D Model Wrapper with Clear Input/Output Interface

This wrapper provides a clear interface for the FFB6D model, making it easier
to understand what inputs are needed and how to prepare your own dataset.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from models.ffb6d import FFB6D


class FFB6DWrapper(nn.Module):
    """
    Wrapper for FFB6D model with explicit input/output documentation.
    
    This wrapper helps users understand exactly what inputs the model needs
    and provides helper methods for preparing data from custom datasets.
    """
    
    def __init__(self, n_classes: int = 22, n_pts: int = 12800, n_kps: int = 8, 
                 rndla_cfg=None):
        """
        Initialize FFB6D model.
        
        Args:
            n_classes: Number of object classes (including background)
            n_pts: Number of points in point cloud
            n_kps: Number of keypoints per object
            rndla_cfg: RandLA-Net configuration object
        """
        super(FFB6DWrapper, self).__init__()
        
        # Import default config if not provided
        if rndla_cfg is None:
            from common import ConfigRandLA
            rndla_cfg = ConfigRandLA
        
        self.n_classes = n_classes
        self.n_pts = n_pts
        self.n_kps = n_kps
        
        # Initialize the actual model
        self.model = FFB6D(n_classes=n_classes, n_pts=n_pts, 
                          rndla_cfg=rndla_cfg, n_kps=n_kps)
    
    def forward(self, inputs: Dict[str, torch.Tensor], 
                end_points: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            inputs: Dictionary containing:
                Required keys:
                - 'rgb': [B, 3, H, W] - RGB image
                - 'cld_rgb_nrm': [B, 9, N] - Point cloud with RGB and normals
                - 'choose': [B, 1, N] - Selection indices
                
                Downsampling stage keys (i=0,1,2,3):
                - 'cld_xyz{i}': Point coordinates at stage i
                - 'cld_nei_idx{i}': Neighbor indices
                - 'cld_sub_idx{i}': Subsampling indices
                
                Upsampling stage keys (i=0,1,2,3,4):
                - 'cld_interp_idx{i}': Interpolation indices
                
                Fusion keys for downsampling (i=0,1,2,3):
                - 'p2r_ds_nei_idx{i}': Point-to-RGB indices
                - 'r2p_ds_nei_idx{i}': RGB-to-Point indices
                
                Fusion keys for upsampling (i=0,1,2):
                - 'p2r_up_nei_idx{i}': Point-to-RGB indices  
                - 'r2p_up_nei_idx{i}': RGB-to-Point indices
                
            end_points: Optional dictionary for accumulating outputs
        
        Returns:
            Dictionary containing:
            - 'pred_rgbd_segs': [B, n_classes, N] - Semantic segmentation logits
            - 'pred_kp_ofs': [B, n_kps, N, 3] - Keypoint offset predictions
            - 'pred_ctr_ofs': [B, 1, N, 3] - Center offset predictions
        """
        return self.model(inputs, end_points=end_points)
    
    def get_input_spec(self) -> Dict[str, Tuple]:
        """
        Get specification of input tensors.
        
        Returns:
            Dictionary with input names and their expected shapes
        """
        B = 'batch_size'
        H, W = 480, 640
        N = self.n_pts
        
        spec = {
            # Core inputs
            'rgb': (B, 3, H, W),
            'cld_rgb_nrm': (B, 9, N),
            'choose': (B, 1, N),
        }
        
        # Add all the indexing tensors
        for i in range(4):
            spec[f'cld_xyz{i}'] = (B, f'N{i}', 3)
            spec[f'cld_nei_idx{i}'] = (B, f'N{i}', 'k_n')
            spec[f'cld_sub_idx{i}'] = (B, f'N{i+1}', 'max_num')
            spec[f'p2r_ds_nei_idx{i}'] = (B, f'H{i}*W{i}', 1)
            spec[f'r2p_ds_nei_idx{i}'] = (B, f'N{i}', 'k_n')
        
        for i in range(5):
            spec[f'cld_interp_idx{i}'] = (B, f'N{i}', 1)
        
        for i in range(3):
            spec[f'p2r_up_nei_idx{i}'] = (B, f'H{i}*W{i}', 1)
            spec[f'r2p_up_nei_idx{i}'] = (B, f'N{i}', 'k_n')
        
        return spec
    
    def get_output_spec(self) -> Dict[str, Tuple]:
        """
        Get specification of output tensors.
        
        Returns:
            Dictionary with output names and their shapes
        """
        B = 'batch_size'
        N = self.n_pts
        
        return {
            'pred_rgbd_segs': (B, self.n_classes, N),
            'pred_kp_ofs': (B, self.n_kps, N, 3),
            'pred_ctr_ofs': (B, 1, N, 3),
        }
    
    @staticmethod
    def load_pretrained(checkpoint_path: str, n_classes: int = 22, 
                       n_pts: int = 12800, n_kps: int = 8, 
                       rndla_cfg=None, device: str = 'cuda') -> 'FFB6DWrapper':
        """
        Load a pretrained model from checkpoint.
        
        Args:
            checkpoint_path: Path to .pth.tar checkpoint file
            n_classes: Number of classes
            n_pts: Number of points
            n_kps: Number of keypoints
            rndla_cfg: RandLA configuration
            device: Device to load model on
            
        Returns:
            Loaded model wrapper
        """
        model = FFB6DWrapper(n_classes=n_classes, n_pts=n_pts, 
                            n_kps=n_kps, rndla_cfg=rndla_cfg)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        if 'module.' in list(state_dict.keys())[0]:
            state_dict = {k.replace('module.', ''): v 
                         for k, v in state_dict.items()}
        
        model.model.load_state_dict(state_dict)
        model.to(device)
        
        return model
    
    def print_architecture(self):
        """Print a summary of the model architecture."""
        print("=" * 80)
        print("FFB6D Model Architecture")
        print("=" * 80)
        print(f"Number of classes: {self.n_classes}")
        print(f"Number of points: {self.n_pts}")
        print(f"Number of keypoints: {self.n_kps}")
        print("\nModel Components:")
        print("  1. RGB Branch: ResNet34 + PSPNet")
        print("  2. Point Cloud Branch: RandLA-Net")
        print("  3. Bidirectional Fusion: RGB ↔ Point Cloud")
        print("  4. Prediction Heads:")
        print("     - Semantic Segmentation")
        print("     - Keypoint Offset Prediction")
        print("     - Center Offset Prediction")
        print("\nTotal Parameters:", 
              sum(p.numel() for p in self.parameters()))
        print("=" * 80)


def create_model(dataset: str = 'ycb', checkpoint: Optional[str] = None) -> FFB6DWrapper:
    """
    Factory function to create FFB6D model for specific dataset.
    
    Args:
        dataset: 'ycb' or 'linemod'
        checkpoint: Optional path to pretrained checkpoint
        
    Returns:
        Initialized (and optionally loaded) model
    """
    from common import ConfigRandLA
    
    if dataset == 'ycb':
        n_classes = 22  # 21 objects + background
    elif dataset == 'linemod':
        n_classes = 2   # 1 object + background
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    model = FFB6DWrapper(
        n_classes=n_classes,
        n_pts=ConfigRandLA.num_points,
        n_kps=8,
        rndla_cfg=ConfigRandLA
    )
    
    if checkpoint is not None:
        model = FFB6DWrapper.load_pretrained(
            checkpoint, n_classes=n_classes,
            n_pts=ConfigRandLA.num_points, n_kps=8,
            rndla_cfg=ConfigRandLA
        )
    
    return model


if __name__ == '__main__':
    # Example usage
    print("\n=== FFB6D Model Wrapper Example ===\n")
    
    # Create model
    model = create_model(dataset='ycb')
    model.print_architecture()
    
    # Print input specifications
    print("\n=== Input Specification ===")
    print("Required input tensors:")
    for name, shape in model.get_input_spec().items():
        print(f"  {name}: {shape}")
    
    # Print output specifications
    print("\n=== Output Specification ===")
    print("Output tensors:")
    for name, shape in model.get_output_spec().items():
        print(f"  {name}: {shape}")
    
    print("\n=== Example Forward Pass ===")
    print("See MODEL_ARCHITECTURE.md for detailed usage examples")
