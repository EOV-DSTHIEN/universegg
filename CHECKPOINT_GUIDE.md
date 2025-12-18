# Checkpoint Management Guide

## Overview
The training script now supports full checkpoint saving and resuming, including:
- Model weights
- Optimizer state
- Scheduler state
- Training history
- Current epoch number

## Checkpoint Files

### Automatic Checkpoints
- **`latest_checkpoint.pt`** - Always updated after each epoch (for easy resuming)
- **`epoch_XXX_checkpoint.pt`** - Periodic checkpoints at intervals (e.g., epoch 10, 20, 30...)
- **`best_model.pt`** - Best model based on validation Dice score

### File Structure
```
checkpoints/
├── latest_checkpoint.pt           # Latest (use this to resume)
├── epoch_010_checkpoint.pt        # Periodic backup
├── epoch_020_checkpoint.pt
├── epoch_050_checkpoint.pt
└── best_model.pt                  # Best validation score
```

## Usage

### Method 1: Using Python Script (Recommended)

```bash
# Show latest checkpoint info
python scripts1/checkpoint_manager.py --latest

# List all checkpoints
python scripts1/checkpoint_manager.py --list

# Resume from latest checkpoint
python scripts1/checkpoint_manager.py --resume

# Resume from specific checkpoint
python scripts1/checkpoint_manager.py --resume checkpoints/epoch_020_checkpoint.pt

# Clean up old checkpoints (keep only 3 latest)
python scripts1/checkpoint_manager.py --clean --keep 3
```

### Method 2: Manual Modification

Edit `train_universeg_ovatus.py`:

```python
RESUME_FROM_CHECKPOINT = "checkpoints/latest_checkpoint.pt"  # Change this
```

Then run:
```bash
python scripts1/train_universeg_ovatus.py
```

### Method 3: Command Line Override

```bash
cd /thiends/hdd2t/UniverSeg

# Create temporary script with checkpoint path
RESUME_FROM_CHECKPOINT='checkpoints/latest_checkpoint.pt' python scripts1/train_universeg_ovatus.py
```

## What's Saved in Checkpoints

```python
checkpoint = {
    'epoch': 10,                     # Last completed epoch
    'model_state_dict': {...},       # Model weights
    'optimizer_state_dict': {...},   # Optimizer state (momentum, etc.)
    'scheduler_state_dict': {...},   # LR scheduler state
    'history': {                     # Training history
        'train_loss': [...],
        'val_dice': [...],
        'best_val_dice': 0.5234,
        'best_epoch': 8
    }
}
```

## Resume Behavior

When resuming training:
1. ✓ Loads exact optimizer state (momentum, adaptive learning rates, etc.)
2. ✓ Resumes from next epoch after checkpoint
3. ✓ Continues validation metrics tracking
4. ✓ Preserves learning rate schedule
5. ✓ Maintains best model tracking

## Example Workflow

```bash
# Start training
python scripts1/train_universeg_ovatus.py
# ... training runs for 5 epochs then crashes

# Check status
python scripts1/checkpoint_manager.py --latest

# Resume from epoch 5
python scripts1/checkpoint_manager.py --resume

# Training continues from epoch 6...
```

## Checkpoint Size Reference
- `latest_checkpoint.pt`: ~200 MB (includes full training state)
- `best_model.pt`: ~150 MB (model weights only - smaller)

## Tips

1. **Regular Backups**: Keep `epoch_XXX_checkpoint.pt` files for important milestones
2. **Storage**: Clean old checkpoints periodically to save disk space
   ```bash
   python scripts1/checkpoint_manager.py --clean --keep 2
   ```
3. **Interruption**: Can safely interrupt training (Ctrl+C) - latest checkpoint is updated
4. **Debugging**: If training seems stuck, use a previous checkpoint
   ```bash
   python scripts1/checkpoint_manager.py --resume checkpoints/epoch_010_checkpoint.pt
   ```

## Troubleshooting

### Checkpoint Not Found
```
❌ No checkpoints found
```
Solution: Make sure `checkpoints/` directory exists and checkpoint files are saved there.

### Checkpoint Corrupted
```
Error loading checkpoint
```
Solution: Try an earlier checkpoint:
```bash
python scripts1/checkpoint_manager.py --list
python scripts1/checkpoint_manager.py --resume checkpoints/epoch_010_checkpoint.pt
```

### Training Won't Resume
Make sure path is correct:
```bash
python scripts1/checkpoint_manager.py --list  # See available checkpoints
python scripts1/checkpoint_manager.py --resume checkpoints/YOUR_CHECKPOINT.pt
```

## Files Modified

1. **`train_universeg_ovatus.py`**:
   - Added `RESUME_FROM_CHECKPOINT` configuration
   - Added `save_checkpoint()` and `load_checkpoint()` functions
   - Checkpoint saved every epoch + periodic backups

2. **`checkpoint_manager.py`** (NEW):
   - Command-line utility for checkpoint management
   - List, show info, resume, and cleanup

3. **`resume_training.sh`** (NEW):
   - Bash helper script for quick resuming
