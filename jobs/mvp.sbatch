#!/bin/bash
# Full MVP pipeline: train -> predict val -> score val -> predict test.
# Submit from cvpdl_hw1/:  sbatch jobs/mvp.sbatch
# Override defaults:       sbatch --export=ALL,NAME=smoke,EPOCHS=1 jobs/mvp.sbatch
#                          sbatch --export=ALL,NAME=v2_imgsz1280,IMGSZ=1280,BATCH=4 jobs/mvp.sbatch
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
EPOCHS=${EPOCHS:-50}
IMGSZ=${IMGSZ:-640}
BATCH=${BATCH:-16}
RUN=runs/$NAME

python code/train.py --name "$NAME" --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH"
python code/predict.py --weights "$RUN/weights/best.pt" --split val --imgsz "$IMGSZ" --out "$RUN/val_pred.csv"
python code/eval_val.py --csv "$RUN/val_pred.csv"
python code/predict.py --weights "$RUN/weights/best.pt" --split test --imgsz "$IMGSZ" --out "$RUN/submission_test.csv"
