#!/usr/bin/env python3
"""
Quick Training Management Script
Dùng để bắt đầu, tiếp tục, hoặc quản lý đào tạo
"""

import os
import sys
import subprocess
from pathlib import Path


def print_menu():
    print("\n" + "="*70)
    print("UniverSeg Training Manager")
    print("="*70)
    print("1. Start new training")
    print("2. Resume from latest checkpoint")
    print("3. Resume from specific checkpoint")
    print("4. View checkpoint info")
    print("5. Clean old checkpoints")
    print("6. Show training history")
    print("0. Exit")
    print("="*70)


def start_new_training():
    print("\n▶️  Starting new training...")
    print("RESUME_FROM_CHECKPOINT = None")
    
    # Create a temp config
    train_script = "scripts1/train_universeg_ovatus.py"
    
    if not os.path.exists(train_script):
        print(f"❌ {train_script} not found")
        return
    
    # Read and update
    with open(train_script, 'r') as f:
        content = f.read()
    
    import re
    pattern = r'RESUME_FROM_CHECKPOINT\s*=\s*[^\n]*'
    replacement = 'RESUME_FROM_CHECKPOINT = None'
    content = re.sub(pattern, replacement, content)
    
    with open(train_script, 'w') as f:
        f.write(content)
    
    print(f"✓ Updated {train_script}")
    print(f"\nStart training:")
    print(f"  python {train_script}")


def resume_latest():
    print("\n▶️  Resuming from latest checkpoint...")
    subprocess.run([
        sys.executable, "scripts1/checkpoint_manager.py", "--resume"
    ])


def resume_specific():
    print("\n📂 Available checkpoints:")
    subprocess.run([
        sys.executable, "scripts1/checkpoint_manager.py", "--list"
    ])
    
    checkpoint = input("\nEnter checkpoint path (or press Enter to skip): ").strip()
    
    if checkpoint:
        subprocess.run([
            sys.executable, "scripts1/checkpoint_manager.py", "--resume", checkpoint
        ])


def view_checkpoint_info():
    print("\n📊 Checkpoint Information:")
    subprocess.run([
        sys.executable, "scripts1/checkpoint_manager.py", "--latest"
    ])


def clean_checkpoints():
    keep = input("Keep how many latest checkpoints? (default: 3): ").strip()
    keep = int(keep) if keep else 3
    
    print(f"\n🗑️  Cleaning checkpoints (keeping {keep} latest)...")
    subprocess.run([
        sys.executable, "scripts1/checkpoint_manager.py", "--clean", "--keep", str(keep)
    ])


def show_training_history():
    log_dir = "logs"
    
    if not os.path.exists(log_dir):
        print(f"❌ {log_dir} not found")
        return
    
    logs = sorted(Path(log_dir).glob("training_*.json"))
    
    if not logs:
        print(f"❌ No training logs found in {log_dir}")
        return
    
    print(f"\n📋 Training History Files ({len(logs)} total):")
    
    for i, log in enumerate(logs[-10:], 1):  # Show last 10
        print(f"  {i}. {log.name}")
        
        try:
            import json
            with open(log, 'r') as f:
                history = json.load(f)
                best_dice = history.get('best_val_dice', 0.0)
                print(f"     Best Val Dice: {best_dice:.4f}")
        except:
            pass


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
    
    while True:
        print_menu()
        choice = input("Enter choice (0-6): ").strip()
        
        if choice == "1":
            start_new_training()
        elif choice == "2":
            resume_latest()
        elif choice == "3":
            resume_specific()
        elif choice == "4":
            view_checkpoint_info()
        elif choice == "5":
            clean_checkpoints()
        elif choice == "6":
            show_training_history()
        elif choice == "0":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
