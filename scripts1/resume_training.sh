#!/bin/bash
# ================== Resume Training Script ==================
# Sử dụng để tiếp tục đào tạo từ checkpoint cuối cùng

cd "$(dirname "$0")/.."

CHECKPOINT="checkpoints/latest_checkpoint.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    echo "Available checkpoints:"
    ls -lh checkpoints/*checkpoint.pt 2>/dev/null || echo "No checkpoints found"
    exit 1
fi

echo "📂 Resuming from: $CHECKPOINT"
echo "Start training with:"
echo ""
echo "RESUME_FROM_CHECKPOINT='$CHECKPOINT' python scripts1/train_universeg_ovatus.py"
echo ""
echo "Or modify RESUME_FROM_CHECKPOINT in train_universeg_ovatus.py and run:"
echo "python scripts1/train_universeg_ovatus.py"
