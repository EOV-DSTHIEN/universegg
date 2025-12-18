#!/usr/bin/env python3
"""
Simple training launcher with automatic checkpoint resuming
Usage:
    python run_training.py              # Start fresh
    python run_training.py --resume     # Resume from latest
    python run_training.py --epoch 10   # Resume from epoch 10
"""

import os
import sys
import argparse
import subprocess

def check_checkpoint(path):
    """Check if checkpoint exists"""
    if not os.path.exists(path):
        print(f"❌ Checkpoint not found: {path}")
        return False
    print(f"✓ Found checkpoint: {path}")
    return True

def get_latest_checkpoint():
    """Get latest checkpoint if it exists"""
    path = "checkpoints/latest_checkpoint.pt"
    if os.path.exists(path):
        return path
    return None

def get_epoch_checkpoint(epoch):
    """Get checkpoint for specific epoch"""
    path = f"checkpoints/epoch_{epoch:03d}_checkpoint.pt"
    if os.path.exists(path):
        return path
    return None

def update_resume_flag(checkpoint_path):
    """Update RESUME_FROM_CHECKPOINT in training script"""
    script_path = "scripts1/train_universeg_ovatus.py"
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Find and replace the RESUME_FROM_CHECKPOINT line
    old_line = 'RESUME_FROM_CHECKPOINT = None'
    new_line = f'RESUME_FROM_CHECKPOINT = "{checkpoint_path}"'
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        with open(script_path, 'w') as f:
            f.write(content)
        print(f"✓ Updated RESUME_FROM_CHECKPOINT to: {checkpoint_path}")
        return True
    else:
        print(f"❌ Could not find RESUME_FROM_CHECKPOINT line in {script_path}")
        return False

def reset_resume_flag():
    """Reset RESUME_FROM_CHECKPOINT to None"""
    script_path = "scripts1/train_universeg_ovatus.py"
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Replace any RESUME_FROM_CHECKPOINT setting back to None
    import re
    content = re.sub(
        r'RESUME_FROM_CHECKPOINT = ".*?"',
        'RESUME_FROM_CHECKPOINT = None',
        content
    )
    
    with open(script_path, 'w') as f:
        f.write(content)
    print("✓ Reset RESUME_FROM_CHECKPOINT to None")

def main():
    parser = argparse.ArgumentParser(description="UniverSeg Training Launcher")
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from latest checkpoint'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        help='Resume from specific epoch checkpoint'
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Start fresh training (ignore checkpoints)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("UniverSeg Training Launcher")
    print("="*70)
    
    # Decide which checkpoint to use
    checkpoint_to_load = None
    
    if args.fresh:
        print("\n→ Starting fresh training")
        reset_resume_flag()
    elif args.epoch is not None:
        print(f"\n→ Resume from epoch {args.epoch}")
        checkpoint_to_load = get_epoch_checkpoint(args.epoch)
        if not checkpoint_to_load or not check_checkpoint(checkpoint_to_load):
            print(f"\n❌ Could not find epoch {args.epoch} checkpoint")
            print("Available checkpoints:")
            if os.path.exists("checkpoints"):
                for f in sorted(os.listdir("checkpoints")):
                    print(f"  - {f}")
            sys.exit(1)
    elif args.resume:
        print("\n→ Resume from latest checkpoint")
        checkpoint_to_load = get_latest_checkpoint()
        if not checkpoint_to_load:
            print("\n⚠️  No latest checkpoint found")
            print("Starting fresh training instead")
        else:
            check_checkpoint(checkpoint_to_load)
    else:
        # Default: check if latest exists, ask user
        latest = get_latest_checkpoint()
        if latest and os.path.exists(latest):
            print("\n⚠️  Found existing checkpoint!")
            print(f"   {latest}")
            response = input("\nResume training? (y/n): ").strip().lower()
            if response == 'y':
                checkpoint_to_load = latest
                check_checkpoint(checkpoint_to_load)
            else:
                print("→ Starting fresh training")
                reset_resume_flag()
        else:
            print("\n→ No checkpoint found, starting fresh training")
    
    # Update resume flag if needed
    if checkpoint_to_load:
        if not update_resume_flag(checkpoint_to_load):
            print("❌ Failed to update checkpoint flag")
            sys.exit(1)
    elif not args.fresh:
        reset_resume_flag()
    
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")
    
    # Run training script
    result = subprocess.run(
        [sys.executable, "scripts1/train_universeg_ovatus.py"],
        cwd=os.getcwd()
    )
    
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
