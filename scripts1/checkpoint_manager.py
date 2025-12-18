#!/usr/bin/env python3
"""
Checkpoint Management Helper
Dùng để xem, quản lý, và tiếp tục đào tạo từ checkpoint
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

CHECKPOINT_DIR = "checkpoints"
TRAIN_SCRIPT = "scripts1/train_universeg_ovatus.py"


def list_checkpoints():
    """List all available checkpoints"""
    if not os.path.exists(CHECKPOINT_DIR):
        print("❌ No checkpoints directory found")
        return
    
    checkpoints = sorted(Path(CHECKPOINT_DIR).glob("*_checkpoint.pt"))
    
    if not checkpoints:
        print("❌ No checkpoints found")
        return
    
    print("\n" + "="*70)
    print("AVAILABLE CHECKPOINTS")
    print("="*70)
    
    for ckpt in checkpoints:
        size_mb = ckpt.stat().st_size / (1024*1024)
        mtime = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            checkpoint = torch.load(ckpt, map_location='cpu')
            epoch = checkpoint['epoch']
            history = checkpoint.get('history', {})
            best_dice = history.get('best_val_dice', 0.0)
            
            print(f"\n📊 {ckpt.name}")
            print(f"   Size: {size_mb:.1f} MB")
            print(f"   Modified: {mtime}")
            print(f"   Epoch: {epoch + 1}")
            print(f"   Best Val Dice: {best_dice:.4f}")
        except Exception as e:
            print(f"\n⚠️  {ckpt.name}")
            print(f"   Error loading: {e}")


def show_latest_checkpoint():
    """Show info about latest checkpoint"""
    latest = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
    
    if not os.path.exists(latest):
        print("❌ No latest checkpoint found")
        return
    
    try:
        checkpoint = torch.load(latest, map_location='cpu')
        epoch = checkpoint['epoch']
        history = checkpoint.get('history', {})
        
        print("\n" + "="*70)
        print("LATEST CHECKPOINT")
        print("="*70)
        print(f"Path: {latest}")
        print(f"Size: {os.path.getsize(latest) / (1024*1024):.1f} MB")
        print(f"Epoch: {epoch + 1}")
        print(f"Best Val Dice: {history.get('best_val_dice', 0.0):.4f}")
        print(f"Best Epoch: {history.get('best_epoch', 0) + 1}")
        
        if history.get('train_loss'):
            print(f"Train Loss (last): {history['train_loss'][-1]:.4f}")
        if history.get('val_dice'):
            print(f"Val Dice (last): {history['val_dice'][-1]:.4f}")
        
        print("\n✓ Ready to resume training")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")


def resume_training(checkpoint_path=None):
    """Resume training from checkpoint"""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        list_checkpoints()
        return False
    
    print(f"\n{'='*70}")
    print("RESUMING TRAINING")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Modify train script to load checkpoint
    train_script_path = TRAIN_SCRIPT
    
    if not os.path.exists(train_script_path):
        print(f"❌ Train script not found: {train_script_path}")
        return False
    
    # Read the script
    with open(train_script_path, 'r') as f:
        content = f.read()
    
    # Find and update RESUME_FROM_CHECKPOINT
    import re
    pattern = r'RESUME_FROM_CHECKPOINT\s*=\s*[^\n]*'
    replacement = f'RESUME_FROM_CHECKPOINT = "{checkpoint_path}"'
    
    new_content = re.sub(pattern, replacement, content)
    
    # Write back
    with open(train_script_path, 'w') as f:
        f.write(new_content)
    
    print(f"✓ Updated RESUME_FROM_CHECKPOINT = \"{checkpoint_path}\"")
    print(f"\nRun training:")
    print(f"  python {train_script_path}")
    
    return True


def clean_old_checkpoints(keep=3):
    """Keep only N most recent checkpoints"""
    checkpoints = sorted(
        Path(CHECKPOINT_DIR).glob("epoch_*_checkpoint.pt"),
        key=lambda p: p.stat().st_mtime
    )
    
    if len(checkpoints) <= keep:
        print(f"✓ Only {len(checkpoints)} checkpoints (keeping {keep})")
        return
    
    to_remove = checkpoints[:-keep]
    
    print(f"🗑️  Removing {len(to_remove)} old checkpoints (keeping {keep} most recent):")
    for ckpt in to_remove:
        print(f"   Removing: {ckpt.name}")
        ckpt.unlink()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Checkpoint Management")
    parser.add_argument('--list', action='store_true', help='List all checkpoints')
    parser.add_argument('--latest', action='store_true', help='Show latest checkpoint info')
    parser.add_argument('--resume', type=str, nargs='?', const='latest', 
                       help='Resume from checkpoint (default: latest)')
    parser.add_argument('--clean', action='store_true', help='Clean old checkpoints')
    parser.add_argument('--keep', type=int, default=3, help='Number of checkpoints to keep (default: 3)')
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints()
    elif args.latest:
        show_latest_checkpoint()
    elif args.resume:
        if args.resume == 'latest':
            resume_training()
        else:
            resume_training(args.resume)
    elif args.clean:
        clean_old_checkpoints(keep=args.keep)
    else:
        # Default: show latest and list all
        show_latest_checkpoint()
        print()
        list_checkpoints()
        print(f"\n{'='*70}")
        print("USAGE:")
        print(f"{'='*70}")
        print("  python checkpoint_manager.py --latest          # Show latest info")
        print("  python checkpoint_manager.py --list            # List all checkpoints")
        print("  python checkpoint_manager.py --resume          # Resume from latest")
        print("  python checkpoint_manager.py --resume path/to/checkpoint.pt  # Resume from specific")
        print("  python checkpoint_manager.py --clean --keep 3  # Keep only 3 latest")
        print()


if __name__ == "__main__":
    main()
