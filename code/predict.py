"""Run a trained model on val or test images and write a Kaggle-format csv."""

import argparse
import csv
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]  # cvpdl_hw1/
DATASET = ROOT / "dataset"
# The metric caps detections at 100 per image *per class* (pycocotools), not 100 per image,
# so keep more than 100 boxes in total. 600 was the best max_det on val (1000 gained < 0.001).
MAX_DET = 600


def to_prediction_string(boxes):
    """Format detections as `class_id confidence x_min y_min width height ...`."""
    parts = []
    for (x1, y1, x2, y2), conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
        parts.append(f"{int(cls)} {conf:.5f} {x1:.2f} {y1:.2f} {x2 - x1:.2f} {y2 - y1:.2f}")
    return " ".join(parts) if parts else "none"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--out", required=True, help="output csv path")
    parser.add_argument("--imgsz", type=int, default=2240)
    args = parser.parse_args()

    model = YOLO(args.weights)
    image_paths = sorted((DATASET / "images" / args.split).glob("*.jpg"))

    rows = []
    for path in image_paths:
        # Low conf threshold keeps low-score boxes, which raises mAP. augment=True is test-time augmentation
        # (flip + down-scaled passes). iou is the NMS threshold (Ultralytics default).
        result = model.predict(
            path, imgsz=args.imgsz, conf=0.001, iou=0.7, max_det=MAX_DET, augment=True, verbose=False
        )[0]
        rows.append([path.stem, to_prediction_string(result.boxes)])

    if args.split == "test":
        with open(DATASET / "sample_submission.csv") as f:
            expected = {row["image_id"] for row in csv.DictReader(f)}
        assert {r[0] for r in rows} == expected, "image ids do not match sample_submission.csv"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "PredictionString"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"Peak inference VRAM reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
