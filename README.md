# CVPDL HW1: Object Detection (b12901144)

Detect 10 kinds of objects (pedestrian, car, bus, ...) in drone images.

This file explains how to make `submission_test.csv` again.

| Result | mAP50:95 |
|---|---|
| Val (700 images, our own scoring with torchmetrics) | 0.4105 |
| Kaggle public leaderboard | 0.36243 |

---

## 1. Method

We train **4 YOLO models** and combine their boxes.

| Name | Model | Train image size | Batch | Epochs |
|---|---|---|---|---|
| `yolo11m_1280` | YOLO11m | 1280 | 4 | 50 |
| `yolo11m_1600` | YOLO11m | 1600 | 2 | 100 |
| `yolo11l_1280` | YOLO11l | 1280 | 2 | 100 |
| `yolo12m_1280` | YOLO12m | 1280 | 2 | 70 |

- **Pretrained weights:** every model starts from the official COCO weights of Ultralytics
  (`yolo11m.pt`, `yolo11l.pt`, `yolo12m.pt`). No other dataset is used.
- **Training:** Ultralytics default settings (AdamW, mosaic, flips, color changes), seed 0.
  The val set is never used to update weights. It is only used to pick the best epoch
  and to choose the prediction settings below.
- **Prediction:** each model predicts every image 3 times, at image size 1920, 2240 and 2880,
  with test-time augmentation (TTA). This gives 4 × 3 = 12 sets of boxes.
- **Ensemble:** the 12 sets are merged with Weighted Boxes Fusion (WBF). Boxes of the same class
  that overlap (IoU > 0.75) become one box. We keep at most 600 boxes per image.

---

## 2. Files

| Path | What it does |
|---|---|
| `code/train.py` | Train one model |
| `code/predict.py` | Predict one image size with one model, write a csv |
| `code/ensemble.py` | Merge many csvs into one csv (WBF) |
| `code/eval_val.py` | Score a val csv (same metric as Kaggle) |
| `code/visualize.py` | Draw boxes on images (report Q3) |
| `jobs/mvp.sbatch` | Slurm script for our GPU cluster: train + predict one model |
| `weights/final/` | Our 4 trained models (173 MB), used in section 4 |

All csvs use the Kaggle format: `image_id,PredictionString`, and each box is
`class_id confidence x_min y_min width height` in pixels.

---

## 3. Setup

### 3.1 Python packages

```bash
conda create -n cvpdl python=3.11 -y
conda activate cvpdl
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics==8.4.173 torchmetrics==1.9.0 pycocotools==2.0.11 ensemble-boxes==1.0.9
```

Install torch **first**. Otherwise `ultralytics` installs a different torch build.

### 3.2 Dataset

Put the course dataset in `dataset/` like this:

```
dataset/
├── annotations/instances_train.json, instances_val.json
├── images/train/, images/val/, images/test/
├── labels/train/, labels/val/
├── classes.txt
├── dataset.yaml
└── sample_submission.csv
```

> The course zip unpacks to `dataset/dataset/...` and also has `__MACOSX/` and `.DS_Store` files.
> Move the inner folders up one level and delete those extra files.

**Run every command from this folder** (the one with `README.md`).

### 3.3 GPU

One GPU is enough. Every step stays under 12 GB of GPU memory:
training peaks at 10.8 GB, prediction peaks at 10.9 GB (`yolo12m_1280` at size 2880).
`train.py` limits PyTorch to 11 GB, so a run that needs more stops with an error.

---

## 4. Reproduce `submission_test.csv` with our trained weights (fast, about 40 min)

**Step 1. Check that the 4 trained models are there.** They come with this homework:

```bash
ls weights/final   # yolo11l_1280.pt  yolo11m_1280.pt  yolo11m_1600.pt  yolo12m_1280.pt
```

**Step 2. Predict the test images** (4 models × 3 image sizes = 12 csvs):

```bash
for m in yolo11m_1280 yolo11m_1600 yolo11l_1280 yolo12m_1280; do
    for s in 1920 2240 2880; do
        python code/predict.py --weights weights/final/$m.pt --split test --imgsz $s --out runs/$m/test_pred_$s.csv
    done
done
```

**Step 3. Merge the 12 csvs** (CPU, about 6 min):

```bash
python code/ensemble.py --split test --out submission_test.csv \
    --csvs runs/{yolo11m_1280,yolo11m_1600,yolo11l_1280,yolo12m_1280}/test_pred_{1920,2240,2880}.csv
```

Keep the csv order shown above. The result should be the same as our `submission_test.csv`
(a different GPU may change a few low-score boxes).

---

## 5. Train from scratch (slow, about 11 GPU hours)

The COCO weights download by themselves on the first run (internet needed).

**Step 1. Train the 4 models.** Each run saves its best epoch to `runs/<name>/weights/best.pt`.

```bash
python code/train.py --name yolo11m_1280 --model weights/yolo11m.pt --imgsz 1280 --batch 4 --epochs 50   # ~1 h
python code/train.py --name yolo11m_1600 --model weights/yolo11m.pt --imgsz 1600 --batch 2 --epochs 100  # ~3 h
python code/train.py --name yolo11l_1280 --model weights/yolo11l.pt --imgsz 1280 --batch 2 --epochs 100  # ~4 h
python code/train.py --name yolo12m_1280 --model weights/yolo12m.pt --imgsz 1280 --batch 2 --epochs 70   # ~3 h
```

(Times are for one H100 / H200.)

**Step 2.** Do steps 2 and 3 of section 4, but use `--weights runs/$m/weights/best.pt`.

On a Slurm cluster, `jobs/mvp.sbatch` does step 1 and step 2 for one model, for example:

```bash
sbatch --partition=<partition> --time=10:00:00 \
    --export=ALL,NAME=yolo11m_1600,MODEL=weights/yolo11m.pt,EPOCHS=100,IMGSZ=1600,BATCH=2 jobs/mvp.sbatch
```

(The script has our account name and module names. Change them for your cluster.)

---

## 6. Check the score on val

Make val csvs the same way (`--split val`), merge them, then score:

```bash
for m in yolo11m_1280 yolo11m_1600 yolo11l_1280 yolo12m_1280; do
    for s in 1920 2240 2880; do
        python code/predict.py --weights weights/final/$m.pt --split val --imgsz $s --out runs/$m/val_pred_$s.csv
    done
done
python code/ensemble.py --split val --out runs/val_ensemble.csv \
    --csvs runs/{yolo11m_1280,yolo11m_1600,yolo11l_1280,yolo12m_1280}/val_pred_{1920,2240,2880}.csv
python code/eval_val.py --csv runs/val_ensemble.csv   # mAP50:95 = 0.4105
```

Single models on val (size 2240 only): 0.3536, 0.3711, 0.3644 and 0.3628 (same order as the table in section 1).

---

## 7. Draw boxes (report Q3)

```bash
python code/visualize.py --csv submission_test.csv
```

This draws boxes with confidence ≥ 0.3 on 4 test images and saves them to `runs/visualize/`:
the 2 images asked for by the assignment, one crowded street, and one 2000×1500 image.
Use `--ids` to pick other images and `--conf` to change the threshold.
