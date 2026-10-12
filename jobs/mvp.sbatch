#!/bin/bash
# One ensemble member: train -> predict val and test at 3 image sizes (TTA) -> score val.
# Submit from cvpdl_hw1/:  sbatch jobs/mvp.sbatch
# Override defaults:       sbatch --export=ALL,NAME=yolo11m_1600,MODEL=weights/yolo11m.pt,EPOCHS=100,IMGSZ=1600,BATCH=2 jobs/mvp.sbatch
#SBATCH --account=MST113101
#SBATCH --job-name=hw1-mvp
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

module purge
module load cuda/12.6
module load miniconda3/26.1.1
# `source activate` silently falls back to base Python when the submitting shell already
# has cvpdl active (inherited CONDA_* vars); the shell hook activates reliably.
eval "$(conda shell.bash hook)"
conda activate cvpdl
set -e
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python=$(which python)"
# Some nodes have GPUs stuck in "GPU requires reset" state: nvidia-smi lists them but CUDA
# cannot initialize. Fail fast with a hint instead of a traceback.
if ! python -c "import torch; torch.zeros(1, device='cuda')" 2>/dev/null; then
    nvidia-smi --query-gpu=name,mig.mode.current --format=csv,noheader
    echo "CUDA init failed on $(hostname) (faulty GPU). Resubmit with --exclude=$(hostname)"
    exit 1
fi

# Run from cvpdl_hw1/ so Ultralytics' cwd-relative downloads stay inside this folder.
cd "$SLURM_SUBMIT_DIR"
NAME=${NAME:-mvp}
MODEL=${MODEL:-weights/yolo11s.pt}
EPOCHS=${EPOCHS:-50}
IMGSZ=${IMGSZ:-640}
BATCH=${BATCH:-16}
RUN=runs/$NAME

python code/train.py --name "$NAME" --model "$MODEL" --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH"
for SIZE in 1920 2240 2880; do  # multi-scale inference: ensemble.py fuses all sizes of all members
    python code/predict.py --weights "$RUN/weights/best.pt" --split val --imgsz "$SIZE" --out "$RUN/val_pred_$SIZE.csv"
    python code/eval_val.py --csv "$RUN/val_pred_$SIZE.csv"
    python code/predict.py --weights "$RUN/weights/best.pt" --split test --imgsz "$SIZE" --out "$RUN/test_pred_$SIZE.csv"
done
