"""Break down val errors of a prediction csv produced by `predict.py --split val`.

Reports per-class AP/AR, recall by object size, AP by image size / brightness / density,
and class confusions. Uses pycocotools, the backend of the official torchmetrics metric.
"""

import argparse
import contextlib
import csv
import io
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]  # cvpdl_hw1/
DATASET = ROOT / "dataset"
SIZE_BINS = [0, 8, 16, 32, 64, 128, float("inf")]  # object size = sqrt(box area), in pixels
CONF = 0.25  # score threshold for the confusion analysis (same as Ultralytics' confusion matrix)


def load_detections(csv_path):
    dets = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["PredictionString"] == "none":
                continue
            for cls, score, x, y, w, h in np.array(row["PredictionString"].split(), dtype=float).reshape(-1, 6):
                dets.append({"image_id": row["image_id"], "category_id": int(cls), "bbox": [x, y, w, h], "score": score})
    return dets


def run_eval(gt, dt, img_ids=None):
    e = COCOeval(gt, dt, "bbox")
    if img_ids is not None:
        e.params.imgIds = sorted(img_ids)
    with contextlib.redirect_stdout(io.StringIO()):  # silence pycocotools' progress prints
        e.evaluate()
        e.accumulate()
        e.summarize()
    return e  # e.stats[0] = AP50:95, [1] = AP50, [2] = AP75, [8] = AR@100


def print_ap_by_group(title, gt, dt, groups):
    print(f"\n## AP50:95 by {title}")
    for name, ids in groups.items():
        n_obj = sum(len(gt.getAnnIds(imgIds=[i])) for i in ids)
        print(f"  {name:<22} images {len(ids):>4}  objects {n_obj:>6}  AP {run_eval(gt, dt, ids).stats[0]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="prediction csv produced by predict.py --split val")
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(DATASET / "annotations" / "instances_val.json"))
        dt = gt.loadRes(load_detections(args.csv))
    names = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}

    # 1. Overall and per class
    e = run_eval(gt, dt)
    print(f"## Overall: AP50:95 {e.stats[0]:.4f}  AP50 {e.stats[1]:.4f}  AP75 {e.stats[2]:.4f}  AR@100 {e.stats[8]:.4f}")
    print("\n## Per class (AP50:95 / AP50 / AP75 / AR@100)")
    precision = e.eval["precision"]  # [IoU thr, recall thr, class, area range, max dets]
    recall = e.eval["recall"]  # [IoU thr, class, area range, max dets]
    for k, cat_id in enumerate(e.params.catIds):
        p = precision[:, :, k, 0, -1]
        ap50, ap75 = (np.mean(p[t][p[t] > -1]) for t in (0, 5))
        print(f"  {names[cat_id]:<16} {np.mean(p[p > -1]):.4f}  {ap50:.4f}  {ap75:.4f}  {np.mean(recall[:, k, 0, -1]):.4f}")

    # 2. Recall by object size: share of GT boxes matched at all (up to 100 dets per class per image)
    print("\n## Recall by object size, sqrt(area) px (IoU 0.5 / IoU 0.75)")
    matched = {0: [], 5: []}
    sizes = []
    for ev in e.evalImgs:
        if ev is None or ev["aRng"] != e.params.areaRng[0]:
            continue
        for g, gt_id in enumerate(ev["gtIds"]):
            sizes.append(gt.anns[gt_id]["area"] ** 0.5)
            for t in matched:
                matched[t].append(ev["gtMatches"][t, g] > 0)
    sizes = np.array(sizes)
    for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:]):
        m = (sizes >= lo) & (sizes < hi)
        r50, r75 = (np.mean(np.array(matched[t])[m]) for t in (0, 5))
        print(f"  [{lo:>4.0f}, {hi:>4.0f})  objects {m.sum():>6} ({m.mean():5.1%})  recall {r50:.3f} / {r75:.3f}")

    # 3. AP by image resolution, brightness and object density
    imgs = gt.imgs.values()
    by_res = {}
    for im in imgs:
        by_res.setdefault(f"{im['width']}x{im['height']}", []).append(im["id"])
    print_ap_by_group("image resolution", gt, dt, by_res)

    brightness = {}
    for im in imgs:
        with Image.open(DATASET / "images" / "val" / im["file_name"]) as pil:
            brightness[im["id"]] = np.asarray(pil.convert("L").reduce(8)).mean()
    by_light = {"dark (<50)": [], "dim [50, 100)": [], "bright (>=100)": []}
    for i, b in brightness.items():
        by_light["dark (<50)" if b < 50 else "dim [50, 100)" if b < 100 else "bright (>=100)"].append(i)
    print_ap_by_group("mean brightness (0-255)", gt, dt, by_light)

    by_density = {"<=50 objects": [], "51-100 objects": [], ">100 objects": []}
    for im in imgs:
        n = len(gt.getAnnIds(imgIds=[im["id"]]))
        by_density["<=50 objects" if n <= 50 else "51-100 objects" if n <= 100 else ">100 objects"].append(im["id"])
    print_ap_by_group("objects per image", gt, dt, by_density)

    # 4. What happens to each GT box, using detections with score >= CONF
    outcome = {c: Counter() for c in names}
    confused_as = {c: Counter() for c in names}
    false_pos = Counter()
    for img_id in gt.getImgIds():
        g = gt.loadAnns(gt.getAnnIds(imgIds=[img_id]))
        d = [x for x in dt.loadAnns(dt.getAnnIds(imgIds=[img_id])) if x["score"] >= CONF]
        if not g or not d:
            for a in g:
                outcome[a["category_id"]]["missed"] += 1
            continue
        iou = np.array(mask_utils.iou([x["bbox"] for x in d], [a["bbox"] for a in g], [0] * len(g)))  # [D, G]
        d_cls = np.array([x["category_id"] for x in d])
        for j, a in enumerate(g):
            hit = iou[:, j] >= 0.5
            if (hit & (d_cls == a["category_id"])).any():
                outcome[a["category_id"]]["correct"] += 1
            elif hit.any():
                outcome[a["category_id"]]["confused"] += 1
                confused_as[a["category_id"]][d_cls[np.argmax(np.where(hit, iou[:, j], 0))]] += 1
            else:
                outcome[a["category_id"]]["missed"] += 1
        for i in np.where(iou.max(axis=1) < 0.5)[0]:
            false_pos[d_cls[i]] += 1
    print(f"\n## GT outcome at score >= {CONF}, IoU >= 0.5 (correct / wrong class / missed) and false positives")
    for c, name in names.items():
        n = sum(outcome[c].values())
        top = ", ".join(f"{names[k]} {v / n:.1%}" for k, v in confused_as[c].most_common(2))
        print(f"  {name:<16} GT {n:>6}  correct {outcome[c]['correct'] / n:5.1%}  wrong-class {outcome[c]['confused'] / n:5.1%}"
              f"  missed {outcome[c]['missed'] / n:5.1%}  FP {false_pos[c]:>5}  | confused as: {top}")


if __name__ == "__main__":
    main()
