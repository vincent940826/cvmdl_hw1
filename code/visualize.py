"""Draw predicted boxes and labels on a few images (report Q3)."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors

ROOT = Path(__file__).resolve().parents[1]  # cvpdl_hw1/
DATASET = ROOT / "dataset"
# Q3: the two images required by the assignment, plus a crowded scene and a 2000x1500 image (a size not in val)
DEFAULT_IDS = ["img_1ae8b7b991128bc7", "img_2e9b763b691129bf", "img_3ae049fc1dc4f530", "img_53ac7dd208708d78"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="prediction csv, e.g. submission_test.csv")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--ids", nargs="+", default=DEFAULT_IDS, help="image ids to draw")
    parser.add_argument("--conf", type=float, default=0.3, help="only draw boxes with at least this confidence")
    parser.add_argument("--out-dir", default="runs/visualize")
    args = parser.parse_args()

    names = (DATASET / "classes.txt").read_text().split()
    with open(args.csv) as f:
        preds = {row["image_id"]: row["PredictionString"] for row in csv.DictReader(f) if row["image_id"] in args.ids}

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for image_id in args.ids:
        image = cv2.imread(str(DATASET / "images" / args.split / f"{image_id}.jpg"))  # BGR, as Annotator expects
        dets = np.array([] if preds[image_id] == "none" else preds[image_id].split(), dtype=float).reshape(-1, 6)
        dets = dets[dets[:, 1] >= args.conf]
        annotator = Annotator(image, line_width=2)
        for cls, conf, x, y, w, h in dets[np.argsort(dets[:, 1])]:  # draw the most confident boxes last (on top)
            annotator.box_label([x, y, x + w, y + h], f"{names[int(cls)]} {conf:.2f}", color=colors(int(cls), True))
        out = Path(args.out_dir) / f"{image_id}.jpg"
        cv2.imwrite(str(out), annotator.result())
        print(f"{out}: {len(dets)} boxes with confidence >= {args.conf}")


if __name__ == "__main__":
    main()
