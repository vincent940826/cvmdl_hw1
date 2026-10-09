"""Score a prediction csv on the val set with the official torchmetrics mAP."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision

ROOT = Path(__file__).resolve().parents[1]  # cvpdl_hw1/
DATASET = ROOT / "dataset"


def parse_prediction_string(s):
    """Return (boxes [N, 4] xywh, scores [N], labels [N]) from a PredictionString."""
    values = [] if s == "none" else [float(v) for v in s.split()]
    dets = torch.tensor(values).reshape(-1, 6)
    return dets[:, 2:], dets[:, 1], dets[:, 0].long()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="prediction csv produced by predict.py --split val")
    args = parser.parse_args()

    coco = json.loads((DATASET / "annotations" / "instances_val.json").read_text())
    gt = defaultdict(list)
    for ann in coco["annotations"]:
        gt[ann["image_id"]].append(ann)

    with open(args.csv) as f:
        pred = {row["image_id"]: row["PredictionString"] for row in csv.DictReader(f)}

    preds, targets = [], []
    for image in coco["images"]:
        boxes, scores, labels = parse_prediction_string(pred[image["id"]])
        preds.append({"boxes": boxes, "scores": scores, "labels": labels})
        anns = gt[image["id"]]
        targets.append({
            "boxes": torch.tensor([a["bbox"] for a in anns], dtype=torch.float).reshape(-1, 4),
            "labels": torch.tensor([a["category_id"] for a in anns], dtype=torch.long),
        })

    metric = MeanAveragePrecision(box_format="xywh", iou_type="bbox", class_metrics=True)
    metric.update(preds, targets)
    result = metric.compute()

    names = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
    print(f"mAP50:95 = {result['map']:.4f}")
    print(f"mAP50    = {result['map_50']:.4f}")
    print(f"mAP75    = {result['map_75']:.4f}")
    print(f"mAP small/medium/large = {result['map_small']:.4f} / {result['map_medium']:.4f} / {result['map_large']:.4f}")
    for cls_id, ap in zip(result["classes"].tolist(), result["map_per_class"].tolist()):
        print(f"  {cls_id} {names[cls_id]:<16} {ap:.4f}")


if __name__ == "__main__":
    main()
