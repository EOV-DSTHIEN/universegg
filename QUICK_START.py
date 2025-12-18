#!/usr/bin/env python3
"""
QUICK START GUIDE - Visual Instructions
"""

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   UniverSeg Training - Checkpoint System                    ║
║                          đào tạo & tiếp tục dễ dàng                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def print_start_new():
    print("""
┌─ START NEW TRAINING ──────────────────────────────────────────────────────┐
│                                                                            │
│  $ python scripts1/train_universeg_ovatus.py                              │
│                                                                            │
│  Training will:                                                            │
│    ✓ Save latest_checkpoint.pt every epoch                                │
│    ✓ Save periodic backups (epoch_010, 020, 030...)                       │
│    ✓ Save best_model.pt when validation improves                          │
│    ✓ Log to logs/training_*.json                                          │
│    ✓ Visualize to visualizations_training/                                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
    """)

def print_resume():
    print("""
┌─ RESUME FROM CHECKPOINT ──────────────────────────────────────────────────┐
│                                                                            │
│  Option 1: Use manager (RECOMMENDED)                                       │
│  $ python scripts1/checkpoint_manager.py --resume                          │
│                                                                            │
│  Option 2: Interactive menu                                               │
│  $ python scripts1/training_manager.py                                     │
│  (Choose option 2)                                                         │
│                                                                            │
│  Option 3: Manual edit                                                    │
│  Edit train_universeg_ovatus.py:                                           │
│    RESUME_FROM_CHECKPOINT = "checkpoints/latest_checkpoint.pt"             │
│  Then run:                                                                 │
│    $ python scripts1/train_universeg_ovatus.py                             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
    """)

def print_check_status():
    print("""
┌─ CHECK STATUS ────────────────────────────────────────────────────────────┐
│                                                                            │
│  View latest checkpoint:                                                   │
│  $ python scripts1/checkpoint_manager.py --latest                          │
│                                                                            │
│  List all checkpoints:                                                     │
│  $ python scripts1/checkpoint_manager.py --list                            │
│                                                                            │
│  Sample output:                                                            │
│    ✓ latest_checkpoint.pt       200 MB   Epoch: 35   Best Dice: 0.7234    │
│    - epoch_010_checkpoint.pt     200 MB   Epoch: 10                        │
│    - epoch_020_checkpoint.pt     200 MB   Epoch: 20                        │
│    * best_model.pt              150 MB   Best validation score             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
    """)

def print_workflow():
    print("""
┌─ TYPICAL WORKFLOW ────────────────────────────────────────────────────────┐
│                                                                            │
│  Day 1:                                                                    │
│    python scripts1/train_universeg_ovatus.py                              │
│    ... runs for 10 epochs then crashes ...                                 │
│                                                                            │
│  Day 2:                                                                    │
│    python scripts1/checkpoint_manager.py --latest                         │
│    ... shows: Epoch: 10, Best Dice: 0.6234 ...                            │
│                                                                            │
│    python scripts1/checkpoint_manager.py --resume                         │
│    ... resuming from epoch 10 ...                                         │
│    ... training continues from epoch 11 ...                               │
│                                                                            │
│  Day 3:                                                                    │
│    Training completed! Check results:                                      │
│      - checkpoints/best_model.pt                                          │
│      - logs/training_20251216_*.json                                      │
│      - visualizations_training/                                           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
    """)

def print_cleanup():
    print("""
┌─ CLEANUP & MANAGEMENT ────────────────────────────────────────────────────┐
│                                                                            │
│  Clean old checkpoints (keep 3 latest):                                    │
│  $ python scripts1/checkpoint_manager.py --clean --keep 3                  │
│                                                                            │
│  Backup important checkpoint:                                              │
│  $ cp checkpoints/best_model.pt checkpoints/backup_20251216.pt             │
│                                                                            │
│  Check disk usage:                                                         │
│  $ du -sh checkpoints/ logs/ visualizations_*                              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
    """)

def print_files_created():
    print("""
┌─ FILES CREATED ───────────────────────────────────────────────────────────┐
│                                                                            │
│  Modified:                                                                 │
│    ✓ scripts1/train_universeg_ovatus.py                                    │
│      - Added checkpoint saving/loading                                     │
│      - Added RESUME_FROM_CHECKPOINT config                                 │
│                                                                            │
│  New Files:                                                                │
│    ✓ scripts1/checkpoint_manager.py                                        │
│      - Checkpoint management CLI                                           │
│                                                                            │
│    ✓ scripts1/training_manager.py                                          │
│      - Interactive training menu                                           │
│                                                                            │
│    ✓ CHECKPOINT_GUIDE.md                                                   │
│      - Detailed comprehensive guide                                        │
│                                                                            │
│    ✓ CHECKPOINT_IMPLEMENTATION.md                                          │
│      - Implementation details & examples                                   │
│                                                                            │
│    ✓ QUICK_COMMANDS.sh                                                     │
│      - Quick reference for common commands                                 │
│                                                                            │
│  Auto-generated during training:                                           │
│    - checkpoints/latest_checkpoint.pt     (every epoch)                    │
│    - checkpoints/epoch_XXX_checkpoint.pt  (periodic)                       │
│    - checkpoints/best_model.pt            (when improves)                  │
│    - logs/training_*.json                 (every validation)               │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
    """)

def print_troubleshoot():
    print("""
┌─ TROUBLESHOOTING ─────────────────────────────────────────────────────────┐
│                                                                            │
│  Problem: "Checkpoint not found"                                           │
│  Solution: $ python scripts1/checkpoint_manager.py --list                  │
│                                                                            │
│  Problem: "Training seems stuck"                                           │
│  Solution: Use --latest to see last saved epoch, verify it's updating      │
│                                                                            │
│  Problem: "Checkpoint file corrupted"                                      │
│  Solution: Use previous epoch checkpoint:                                  │
│    $ python scripts1/checkpoint_manager.py --resume                        │
│              checkpoints/epoch_020_checkpoint.pt                           │
│                                                                            │
│  Problem: "Running out of disk space"                                      │
│  Solution: $ python scripts1/checkpoint_manager.py --clean --keep 1        │
│           (keeps only latest, still safe)                                  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
    """)

def print_tips():
    print("""
┌─ PRO TIPS ────────────────────────────────────────────────────────────────┐
│                                                                            │
│  💾 Backup: Periodically copy checkpoints to external storage              │
│     $ cp -r checkpoints ~/backup/universeg_$(date +%Y%m%d)                │
│                                                                            │
│  📊 Monitor: Check training_*.json files to track progress                 │
│                                                                            │
│  ⚡ Speed: Use --latest to quickly check epoch without loading weights     │
│                                                                            │
│  🔄 Restore: Keep epoch_XXX_checkpoint.pt for important milestones         │
│                                                                            │
│  🧹 Clean: Run --clean regularly to save space (after validation)          │
│                                                                            │
│  🎯 Multi-resume: Can resume and change EPOCHS to extend training          │
│     E.g.: Epoch 50 complete, set EPOCHS=100, then resume                  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
    """)

def main():
    print_banner()
    
    print("\n1️⃣  START NEW TRAINING")
    print_start_new()
    
    print("\n2️⃣  RESUME FROM CHECKPOINT")
    print_resume()
    
    print("\n3️⃣  CHECK STATUS")
    print_check_status()
    
    print("\n4️⃣  TYPICAL WORKFLOW")
    print_workflow()
    
    print("\n5️⃣  FILES CREATED")
    print_files_created()
    
    print("\n6️⃣  CLEANUP & MANAGEMENT")
    print_cleanup()
    
    print("\n7️⃣  TROUBLESHOOTING")
    print_troubleshoot()
    
    print("\n8️⃣  PRO TIPS")
    print_tips()
    
    print("\n" + "="*80)
    print("📚 For detailed guide: cat CHECKPOINT_GUIDE.md")
    print("📚 For examples: cat CHECKPOINT_IMPLEMENTATION.md")
    print("📚 For quick commands: cat QUICK_COMMANDS.sh")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
