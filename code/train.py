"""Fine-tune a COCO-pretrained YOLO11s on the HW1 dataset."""

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]  # cvpdl_hw1/
DATASET = ROOT / "dataset"
VRAM_LIMIT_GB = 11  # assignment rule is 12 GB; keep ~1 GB for the CUDA context, which this cap does not cover


def write_data_yaml(path):
    # The official dataset.yaml has `path: .`, which Ultralytics resolves against the
    # current working directory. Write a copy with an absolute dataset path instead.
    cfg = yaml.safe_load((DATASET / "dataset.yaml").read_text())
    cfg["path"] = str(DATASET)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="mvp", help="run name, outputs go to runs/<name>/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    # Hard cap on PyTorch's GPU memory so a run that would break the rule fails with OOM.
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(VRAM_LIMIT_GB * 1024**3 / total)

    data_yaml = ROOT / "runs" / args.name / "data.yaml"
    write_data_yaml(data_yaml)

    model = YOLO(ROOT / "weights" / "yolo11s.pt")  # COCO-pretrained, downloaded on first use
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        seed=0,
    )

    print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Peak VRAM reserved:  {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
