#!/bin/bash
# ================== Quick Reference Card ==================
# Lệnh nhanh để quản lý checkpoint và đào tạo

# Navigation
cd /thiends/hdd2t/UniverSeg

# ========== START TRAINING ==========
# 1. Start NEW training (no resume)
python scripts1/train_universeg_ovatus.py

# ========== RESUME TRAINING ==========
# 1. Quick resume from latest checkpoint
python scripts1/checkpoint_manager.py --resume

# 2. Interactive manager (menu)
python scripts1/training_manager.py

# 3. Resume from specific checkpoint
python scripts1/checkpoint_manager.py --resume checkpoints/epoch_020_checkpoint.pt

# ========== CHECK STATUS ==========
# Show latest checkpoint info
python scripts1/checkpoint_manager.py --latest

# List all checkpoints
python scripts1/checkpoint_manager.py --list

# ========== MANAGE CHECKPOINTS ==========
# Clean old checkpoints (keep 3 latest)
python scripts1/checkpoint_manager.py --clean --keep 3

# ========== VIEW RESULTS ==========
# Show training logs
ls -lh logs/

# View visualizations
ls -lh visualizations_training/

# ========== QUICK COMMANDS ==========
# Resume + run (one-liner)
python scripts1/checkpoint_manager.py --resume && python scripts1/train_universeg_ovatus.py

# Check disk space
du -sh checkpoints/ logs/ visualizations_*

# Remove all old checkpoints except latest and best
rm checkpoints/epoch_*_checkpoint.pt

# Backup checkpoint
cp checkpoints/latest_checkpoint.pt checkpoints/backup_$(date +%Y%m%d_%H%M%S).pt

# ========== TROUBLESHOOTING ==========
# Check for corrupted checkpoints
python -c "import torch; torch.load('checkpoints/latest_checkpoint.pt', map_location='cpu'); print('✓ OK')"

# View checkpoint info
python -c "
import torch
ckpt = torch.load('checkpoints/latest_checkpoint.pt', map_location='cpu')
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'History: {ckpt[\"history\"]}')"

# Find largest files
du -sh checkpoints/* | sort -h | tail -10

echo ""
echo "For interactive menu, run:"
echo "  python scripts1/training_manager.py"
echo ""
echo "For detailed guide, read:"
echo "  cat CHECKPOINT_GUIDE.md"
