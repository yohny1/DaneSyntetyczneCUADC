"""
Wizualizacja wyników datasetu YOLO.
Zapisuje obrazy z narysowanymi bounding boxami do katalogu output/viz/.

Użycie:
    python viz_yolo.py --dataset ./uav_dataset_yolo --split train --num 20
"""

import argparse
import random
from pathlib import Path
import cv2
import numpy as np
import yaml 


def load_class_names(dataset_dir):
    yaml_path = Path(dataset_dir) / "data.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return list(data["names"])

def draw_yolo_bboxes(img_path, label_path, class_names):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    if not label_path.exists():
        return img

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, xc, yc, bw, bh = map(float, parts)
            cls_id = int(cls_id)

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            color = [(0, 255, 0), (255, 0, 0)][cls_id % 2]
            name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, name, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./uav_dataset_yolo")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--output", default="./viz_output")
    args = parser.parse_args()

    ds = Path(args.dataset)
    img_dir = ds / "images" / args.split
    lbl_dir = ds / "labels" / args.split
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob("*.jpg"))
    random.shuffle(images)

    class_names = load_class_names(args.dataset)   # zamiast hardkodowanej listy
    saved = 0

    for img_path in images[:args.num]:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        vis = draw_yolo_bboxes(img_path, lbl_path, class_names)
        if vis is not None:
            out_path = out_dir / f"viz_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved += 1
            print(f"  Saved: {out_path}")

    print(f"\nDone. Saved {saved} visualization images to: {out_dir.absolute()}")


if __name__ == "__main__":
    main()
