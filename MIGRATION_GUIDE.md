# Migration Guide: PyTorch 1.4 to PyTorch 2.x

This guide helps you migrate from the original FFB6D implementation (PyTorch 1.4.0, CUDA 10.x) to the updated version (PyTorch 2.x, CUDA 12.x).

## Summary of Changes

### Dependencies
- **PyTorch**: 1.4.0 → 2.0.0+
- **TorchVision**: 0.2.0 → 0.15.0+
- **CUDA**: 10.x → 11.8+ or 12.x
- **apex**: Removed (replaced with native PyTorch features)
- **Other libraries**: Updated to compatible versions

### Code Changes

#### 1. Mixed Precision Training

**Old (apex):**
```python
from apex import amp

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

# In training loop
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```

**New (PyTorch 2.x):**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Note**: The current code has mixed precision disabled by default. To enable it, uncomment the relevant sections in `train_ycb.py` or `train_lm.py`.

#### 2. Distributed Training

**Old (apex):**
```python
from apex.parallel import DistributedDataParallel
from apex.parallel import convert_syncbn_model

model = convert_syncbn_model(model)
model = DistributedDataParallel(model)
```

**New (PyTorch 2.x):**
```python
from torch.nn.parallel import DistributedDataParallel
from torch.nn import SyncBatchNorm

model = SyncBatchNorm.convert_sync_batchnorm(model)
model = DistributedDataParallel(
    model, 
    device_ids=[local_rank], 
    output_device=local_rank,
    find_unused_parameters=True
)
```

#### 3. Checkpoint Format

**Old:**
```python
checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "amp": amp.state_dict(),  # apex amp state
    ...
}
```

**New:**
```python
checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scaler": None,  # Optional: scaler.state_dict() if using amp
    ...
}
```

**Compatibility**: Old checkpoints can still be loaded. The code handles missing `amp` state gracefully.

## Installation Changes

### Old Installation (PyTorch 1.4)

```bash
# Install CUDA 10.1/10.2
pip install torch==1.4.0

# Install apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### New Installation (PyTorch 2.x)

```bash
# Install CUDA 12.x (or 11.8+)
pip install torch>=2.0.0 torchvision>=0.15.0

# No apex needed!
pip install -r requirement.txt
```

## Training Scripts

### Command Line Arguments

The following argument is **deprecated**:
- `--opt_level`: No longer used (was for apex amp optimization level)

All other arguments remain the same.

### Training Commands

Commands remain **unchanged**:
```bash
# YCB-Video training
python3 -m torch.distributed.launch --nproc_per_node=8 train_ycb.py --gpus=8

# LineMOD training
python3 -m torch.distributed.launch --nproc_per_node=8 train_lm.py --gpus=8 --cls=ape
```

## Model Architecture

The model architecture is **unchanged**. The core FFB6D network remains identical:
- Same ResNet34 + PSPNet RGB branch
- Same RandLA-Net point cloud branch
- Same bidirectional fusion modules
- Same prediction heads

This means:
- ✅ Old checkpoints are compatible
- ✅ Model outputs are identical
- ✅ Evaluation results should match

## Performance Considerations

### PyTorch 2.x Benefits
1. **Faster training**: PyTorch 2.x includes performance optimizations
2. **Better memory efficiency**: Improved memory management
3. **torch.compile**: Optional JIT compilation for speedup (experimental)
4. **Native AMP**: More stable mixed precision training

### Potential Differences
- Training speed may vary slightly due to different backends
- Memory usage patterns might differ
- Exact loss values may have minor numerical differences

## Enabling Mixed Precision (Optional)

To enable mixed precision training for memory savings:

1. **In train_ycb.py** or **train_lm.py**, add at the beginning:
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

2. **Wrap the forward pass** in the model_fn:
```python
with autocast():
    end_points = model(cu_dt)
    # ... loss computation
```

3. **Update the backward pass**:
```python
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

## Testing Your Migration

### 1. Basic Functionality Test
```bash
cd ffb6d
python3 -c "from models.ffb6d import FFB6D; from common import ConfigRandLA; \
    model = FFB6D(22, ConfigRandLA.num_points, ConfigRandLA); \
    print('Model created successfully!')"
```

### 2. Model Example
```bash
cd ffb6d
python3 example_usage.py
```

### 3. Dataset Loading Test
```bash
cd ffb6d
python3 -m datasets.ycb.ycb_dataset  # If you have YCB dataset
```

### 4. Training Test (if you have data)
```bash
cd ffb6d
python3 train_ycb.py --gpu='0' --gpus=1  # Single GPU test
```

## Troubleshooting

### Issue: "ImportError: cannot import name 'amp' from 'apex'"
**Solution**: Remove apex installation or upgrade to the new code that doesn't require apex.

### Issue: "RuntimeError: CUDA out of memory"
**Solution**: 
1. Reduce batch size in `common.py` (`mini_batch_size`)
2. Enable mixed precision training (see above)
3. Use fewer GPUs with smaller per-GPU batch size

### Issue: Old checkpoint won't load
**Solution**: The code handles old checkpoints. If you still have issues:
```python
checkpoint = torch.load('old_checkpoint.pth.tar')
# Remove 'amp' key if present
if 'amp' in checkpoint:
    checkpoint.pop('amp')
torch.save(checkpoint, 'converted_checkpoint.pth.tar')
```

### Issue: Training is slower than before
**Solution**:
1. Check you're using CUDA 12.x or 11.8+
2. Enable cudnn benchmarking: `torch.backends.cudnn.benchmark = True`
3. Try torch.compile (PyTorch 2.x feature - experimental)

## Backward Compatibility

### Can I still use PyTorch 1.4?
The updated code is designed for PyTorch 2.x. For PyTorch 1.4, use the original repository version (before this update).

### Can I load old checkpoints?
Yes! The code automatically handles:
- Old checkpoint format with `amp` state
- `module.` prefix from DataParallel
- Different state_dict structures

### Can I use the same pretrained models?
Yes! Pretrained model weights are compatible. Only the training infrastructure changed, not the model architecture.

## New Features

### 1. Model Wrapper
Use the new wrapper for cleaner code:
```python
from models.ffb6d_wrapper import create_model

model = create_model(dataset='ycb', checkpoint='path/to/checkpoint.pth.tar')
```

### 2. Documentation
- `MODEL_ARCHITECTURE.md`: Detailed model documentation
- `example_usage.py`: Usage examples
- This migration guide

### 3. Better Input Specifications
```python
model.get_input_spec()  # See what inputs are needed
model.get_output_spec()  # See what outputs are produced
```

## Questions?

If you encounter issues:
1. Check this migration guide
2. Review `MODEL_ARCHITECTURE.md` for model details
3. Look at `example_usage.py` for usage patterns
4. Check the updated `README.md`
5. Open an issue on GitHub with your specific problem

## Summary Checklist

- [ ] Uninstall old apex
- [ ] Install PyTorch 2.x and dependencies
- [ ] Update CUDA to 11.8+ or 12.x
- [ ] Pull updated code
- [ ] Test model creation works
- [ ] Test loading old checkpoints (if applicable)
- [ ] Run training/evaluation with your data

## References

- [PyTorch 2.0 Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch AMP Tutorial](https://pytorch.org/docs/stable/amp.html)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
