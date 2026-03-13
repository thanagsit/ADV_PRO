#!/bin/bash
set -e

PROJECT_DIR="/nfs-share-stgnode/home/663380266-5/ADV_Project/SAdvGAN"
PYTHON="python"

echo "============================================"
echo "  SAdvGAN v22 - GlobalAvgPool target (no periodic gradient)"
echo "  $(date)"
echo "============================================"

# ── v22: Root cause of stripes identified ────────────────────────────────
# SmallCNN_MNIST used AvgPool2d(2) × 2:
#   28→14→7 creates gradient structure with 4px block periodicity
#   G learns to produce periodic perturbations to minimize C&W loss
#   → vertical stripes at 4px frequency
#
# Fix: Replace AvgPool2d(2) with AdaptiveAvgPool2d(1) (GlobalAvgPool)
#   ∂output/∂pixel_ij = 1/(H*W) = 1/784 for ALL pixels
#   Completely uniform gradient → G has no spatial bias → uniform perturbation
#
# Must retrain target classifier (architecture changed: Linear(128*49→256) → Linear(128→256))
# Then retrain G from scratch (target gradient structure fundamentally changed)

CKPT="$PROJECT_DIR/checkpoints/target_mnist_best.pth"
GEN_CKPT="$PROJECT_DIR/output_sadvgan/saved_model/netG_epoch_80.pth"

# Always retrain target (architecture changed)
echo "[Step 1/3] Retraining target (GlobalAvgPool) ..."
rm -f $CKPT
$PYTHON $PROJECT_DIR/pretrain_target.py \
    --dataset mnist --epochs 20 --batch_size 256 \
    --save_dir $PROJECT_DIR/checkpoints/

echo ""
echo "[Step 2/3] Training SAdvGAN v22 (fresh) ..."
$PYTHON $PROJECT_DIR/train_sadvgan.py \
    --dataset     mnist \
    --epochs      80 \
    --batch_size  256 \
    --target_ckpt $CKPT \
    --save_path   $PROJECT_DIR/output_sadvgan/ \
    --c           5.0 \
    --adv_lambda  10.0 \
    --alpha       0.0 \
    --beta        0.5 \
    --rec_lambda  50.0 \
    --tv_lambda   0.0

echo ""
echo "[Step 3/3] Evaluating ..."
$PYTHON $PROJECT_DIR/evaluate_sadvgan.py \
    --dataset     mnist \
    --target_ckpt $CKPT \
    --gen_ckpt    $GEN_CKPT \
    --batch_size  256 \
    --output_dir  $PROJECT_DIR/eval_output/

echo ""
echo "============================================"
echo "  Done!  $(date)"
echo "============================================"