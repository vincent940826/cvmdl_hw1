"""Fuse the prediction csvs of several models with Weighted Boxes Fusion (WBF)."""

import argparse
import csv
from pathlib import Path

import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]  # cvpdl_hw1/
DATASET = ROOT / "dataset"


def read_predictions(path):
    """Return {image_id: [N, 6] array of (class, conf, x_min, y_min, width, height)} from a Kaggle-format csv."""
    with open(path) as f:
        return {
            row["image_id"]: np.array(
                [] if row["PredictionString"] == "none" else row["PredictionString"].split(), dtype=float
            ).reshape(-1, 6)
            for row in csv.DictReader(f)
        }


def fuse(dets_per_model, width, height, args):
    """Fuse one image's detections from all models. Returns [M, 6] (class, conf, x_min, y_min, width, height)."""
    scale = np.array([width, height, width, height])
    boxes, scores, labels = [], [], []
    for dets in dets_per_model:
        dets = dets[(dets[:, 4] > 0) & (dets[:, 5] > 0)]  # zero-size boxes (clipped at the border) can never match; WBF warns
        xyxy = np.concatenate([dets[:, 2:4], dets[:, 2:4] + dets[:, 4:6]], axis=1)
        boxes.append(np.clip(xyxy / scale, 0, 1))  # WBF expects (x1, y1, x2, y2) normalized to [0, 1]
        scores.append(dets[:, 1])
        labels.append(dets[:, 0])
    boxes, scores, labels = weighted_boxes_fusion(
        boxes, scores, labels, weights=args.weights, iou_thr=args.iou,
        skip_box_thr=args.skip_box_thr, conf_type=args.conf_type,
    )
    # WBF returns boxes sorted by fused confidence, highest first
    boxes, scores, labels = boxes[:args.max_det] * scale, scores[:args.max_det], labels[:args.max_det]
    return np.column_stack([labels, scores, boxes[:, :2], boxes[:, 2:] - boxes[:, :2]])


def to_prediction_string(dets):
    """Format detections as `class_id confidence x_min y_min width height ...` (same as predict.py)."""
    parts = [f"{int(c)} {s:.5f} {x:.2f} {y:.2f} {w:.2f} {h:.2f}" for c, s, x, y, w, h in dets]
    return " ".join(parts) if parts else "none"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", required=True, help="prediction csvs of one split, one per model")
    parser.add_argument("--weights", nargs="+", type=float, help="one weight per csv (default: all equal)")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--out", required=True, help="output csv path")
    parser.add_argument("--iou", type=float, default=0.55, help="boxes of one class overlapping more than this are fused")
    parser.add_argument("--conf-type", default="avg", choices=["avg", "max", "box_and_model_avg", "absent_model_aware_avg"],
                        help="how fused confidences are computed (see ensemble_boxes)")
    parser.add_argument("--skip-box-thr", type=float, default=0.0, help="ignore input boxes below this confidence")
    parser.add_argument("--max-det", type=int, default=600, help="max boxes kept per image after fusion")
    args = parser.parse_args()
    assert args.weights is None or len(args.weights) == len(args.csvs), "need one weight per csv"

    preds = [read_predictions(p) for p in args.csvs]
    image_ids = sorted(preds[0])
    assert all(sorted(p) == image_ids for p in preds), "csvs cover different images"
    if args.split == "test":
        with open(DATASET / "sample_submission.csv") as f:
            expected = {row["image_id"] for row in csv.DictReader(f)}
        assert set(image_ids) == expected, "image ids do not match sample_submission.csv"

    rows = []
    for image_id in image_ids:
        with Image.open(DATASET / "images" / args.split / f"{image_id}.jpg") as im:
            width, height = im.size
        dets = fuse([p[image_id] for p in preds], width, height, args)
        rows.append([image_id, to_prediction_string(dets)])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "PredictionString"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
