# UniverSeg Training - How to Run

## 1. **First Time Training (Start from scratch)**

```bash
cd /thiends/hdd2t/UniverSeg
python scripts1/train_universeg_ovatus.py
```

**This will:**
- Load Ovatus dataset (automatically splits 60/20/20)
- Train for 50 epochs
- Save checkpoints to `checkpoints/` folder
- Save training logs to `logs/` folder
- Save visualizations to `visualizations_training/` folder

---

## 2. **Resume Training from Latest Checkpoint**

If training was interrupted, resume from the latest saved checkpoint:

```bash
cd /thiends/hdd2t/UniverSeg
python scripts1/train_universeg_ovatus.py
```

Then modify line 38 in `train_universeg_ovatus.py`:

**Before:**
```python
RESUME_FROM_CHECKPOINT = None  # Line 38
```

**After:**
```python
RESUME_FROM_CHECKPOINT = "checkpoints/latest_checkpoint.pt"  # Line 38
```

Then run:
```bash
python scripts1/train_universeg_ovatus.py
```

---

## 3. **Resume from Specific Epoch**

If you want to resume from a specific epoch checkpoint:

**Edit line 38:**
```python
RESUME_FROM_CHECKPOINT = "checkpoints/epoch_010_checkpoint.pt"  # Resume from epoch 10
```

Then run:
```bash
python scripts1/train_universeg_ovatus.py
```

---

## 4. **Quick Python Script (Easier Way)**

Create and run this simple Python script:

```python
# File: quick_resume.py
import subprocess
import sys

# Option 1: Start fresh
# subprocess.run([sys.executable, "scripts1/train_universeg_ovatus.py"])

# Option 2: Resume from latest
with open("scripts1/train_universeg_ovatus.py", "r") as f:
    content = f.read()

content = content.replace(
    'RESUME_FROM_CHECKPOINT = None',
    'RESUME_FROM_CHECKPOINT = "checkpoints/latest_checkpoint.pt"'
)

with open("scripts1/train_universeg_ovatus.py", "w") as f:
    f.write(content)

subprocess.run([sys.executable, "scripts1/train_universeg_ovatus.py"])
```

---

## 5. **What Gets Saved**

After each epoch, you'll find:

```
checkpoints/
├── best_model.pt              # Best Dice score so far
├── latest_checkpoint.pt       # Latest epoch (for easy resuming)
├── epoch_010_checkpoint.pt    # Full checkpoint at epoch 10
├── epoch_020_checkpoint.pt    # Full checkpoint at epoch 20
└── ...

logs/
├── training_20251216_143022.json  # Training history

visualizations_training/
├── epoch_000_batch_000.png    # Sample predictions
├── epoch_000_batch_001.png
└── ...
```

---

## 6. **Monitor Training**

Check the JSON log file to track performance:

```python
import json

with open("logs/training_20251216_143022.json") as f:
    history = json.load(f)
    
print(f"Best validation Dice: {history['best_val_dice']:.4f}")
print(f"Best epoch: {history['best_epoch']}")
print(f"Final train loss: {history['train_loss'][-1]:.4f}")
```

---

## 7. **Common Issues & Solutions**

### Issue: "CUDA out of memory"
- Reduce `BATCH_SIZE` from 4 to 2 in line 33
- Or reduce support set sizes in `SUPPORT_CONFIG`

### Issue: "Training is very slow"
- Check GPU usage: `nvidia-smi`
- Make sure CUDA is being used (line 28 should show `cuda`)

### Issue: "Want to change learning rate"
- Edit line 35: `LEARNING_RATE = 1e-4` to `LEARNING_RATE = 5e-5`
- Only affects new runs (not resumed training)

---

## 8. **Expected Training Times**

- **Per epoch**: 2-5 minutes (depending on GPU)
- **Full 50 epochs**: ~2-4 hours on RTX 3090
- **Validation**: Every 5 epochs (adds ~5 min per validation)

---

## 9. **Final Results Location**

After training completes:

1. **Best model weights**: `checkpoints/best_model.pt`
2. **Training history**: Check the JSON file in `logs/`
3. **Test results**: Printed to console at end
4. **Visualizations**: View in `visualizations_training/`

---

## Quick Commands Reference

```bash
# Start training
cd /thiends/hdd2t/UniverSeg && python scripts1/train_universeg_ovatus.py

# List all checkpoints
ls -lh checkpoints/

# Check latest checkpoint info
ls -lh checkpoints/latest_checkpoint.pt

# View training logs
ls -lh logs/

# Count visualizations
ls visualizations_training/ | wc -l
```
