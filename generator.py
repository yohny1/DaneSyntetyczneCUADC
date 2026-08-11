import argparse, os
import cv2
import numpy as np

DIGITS = [str(d) for d in range(10)]

# ------------------------------------------------------------- klasy / cele
def build_classes(targets_dir):
    stems = [os.path.splitext(f)[0] for f in sorted(os.listdir(targets_dir))
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    digits = [d for d in DIGITS if d in stems]
    others = sorted(s for s in stems if s not in DIGITS)
    return digits + others

def find_target_file(targets_dir, stem):
    for f in os.listdir(targets_dir):
        if os.path.splitext(f)[0] == stem and f.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(targets_dir, f)
    raise FileNotFoundError(stem)

def load_target(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:                      # RGBA -> biale tlo
        a = img[:, :, 3:4] / 255.0
        img = (img[:, :, :3] * a + 255.0 * (1 - a)).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray < 128).astype(np.uint8) * 255                   # czarny symbol
    return img, mask

# ------------------------------------------------------------- tla
def procedural_background(w, h, rng):
    kind = rng.choice(["grass", "dry", "soil"])
    if kind == "grass":  hue, sat, val = rng.uniform(35, 75), rng.uniform(70, 160), rng.uniform(70, 140)
    elif kind == "dry":  hue, sat, val = rng.uniform(20, 38), rng.uniform(60, 140), rng.uniform(90, 160)
    else:                hue, sat, val = rng.uniform(10, 25), rng.uniform(40, 110), rng.uniform(60, 130)
    low = rng.uniform(-1, 1, (h // 16 + 2, w // 16 + 2))
    low = cv2.GaussianBlur(cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), 4)
    high = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)), (0, 0), 1)
    hsv = cv2.merge([np.clip(hue + low * 6 + high * 2, 0, 179),
                     np.clip(sat + low * 30, 0, 255),
                     np.clip(val + low * 35 + high * 12, 0, 255)]).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def get_background(bg_files, w, h, rng):
    if bg_files:
        img = cv2.imread(str(rng.choice(bg_files)))
        if img is not None:
            bh, bw = img.shape[:2]
            s = max(w / bw, h / bh)
            img = cv2.resize(img, (int(bw * s) + 1, int(bh * s) + 1))
            nh, nw = img.shape[:2]
            y0, x0 = rng.integers(0, nh - h + 1), rng.integers(0, nw - w + 1)
            img = (img[y0:y0+h, x0:x0+w].astype(np.float32) * rng.uniform(0.7, 1.2))
            return img.clip(0, 255).astype(np.uint8)
    return procedural_background(w, h, rng)

# ------------------------------------------------------------- nakladanie celu
def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    if inter <= 0: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def place_target(canvas, timg, tmask, rng, max_rot, persp, boxes):
    h, w = canvas.shape[:2]
    th, tw = timg.shape[:2]
    for _ in range(12):                                          # proby bez kolizji
        f   = rng.uniform(0.08, 0.40)                            # wysokosc celu wzgledem kadru
        ang = np.deg2rad(rng.uniform(-max_rot, max_rot))
        R   = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
        dst = ((src - [tw/2, th/2]) * (f * h / th)) @ R.T + \
              [rng.uniform(0.10, 0.90) * w, rng.uniform(0.10, 0.90) * h]
        dst = (dst + rng.uniform(-persp, persp, (4, 2)) * (f * h)).astype(np.float32)

        M   = cv2.getPerspectiveTransform(src, dst)
        sym = cv2.warpPerspective(tmask, M, (w, h), borderValue=0)
        ys, xs = np.nonzero(sym > 50)
        if len(xs) < 20: continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h: continue
        if x1 - x0 < 10 or y1 - y0 < 10: continue
        box = (int(x0), int(y0), int(x1)+1, int(y1)+1)
        if any(iou(box, b) > 0 for b in boxes): continue
        break
    else:
        return None

    if rng.random() < 0.7:                                       # cien
        off = np.array([[rng.uniform(0.01, 0.04)*w, rng.uniform(0.01, 0.05)*h]])
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [(dst + off).astype(np.int32)], 255)
        m = cv2.GaussianBlur(m, (0, 0), 6) / 255.0 * rng.uniform(0.15, 0.35)
        canvas[:] = (canvas * (1 - m[..., None])).astype(np.uint8)

    t = (timg.astype(np.float32) * rng.uniform(0.80, 1.0) + rng.uniform(-12, 4))
    t = t.clip(0, 255).astype(np.uint8)                          # jitter "wydruku"
    if rng.random() < 0.3: t = cv2.GaussianBlur(t, (3, 3), 0)

    warped = cv2.warpPerspective(t, M, (w, h), borderValue=(255, 255, 255))
    panel  = cv2.warpPerspective(np.ones((th, tw), np.float32), M, (w, h), borderValue=0)
    alpha  = np.clip(cv2.GaussianBlur(panel, (3, 3), 0), 0, 1)[..., None]
    canvas[:] = (canvas * (1 - alpha) + warped * alpha).astype(np.uint8)
    return box

def augment_final(img, rng):
    if rng.random() < 0.4: img = cv2.GaussianBlur(img, (3, 3), 0)
    if rng.random() < 0.4:
        img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(2, 10), img.shape), 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 96))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

# ------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-dir", default="targets")
    ap.add_argument("--backgrounds-dir", default=None, help="opcjonalnie: prawdziwe zdjecia trawy/pola")
    ap.add_argument("--out-dir", default="dataset")
    ap.add_argument("--num-images", type=int, default=2000)
    ap.add_argument("--img-w", type=int, default=1280)
    ap.add_argument("--img-h", type=int, default=720)
    ap.add_argument("--max-targets", type=int, default=3)        # wg regulaminu 3 cele na strefe
    ap.add_argument("--max-rot", type=float, default=20, help="rotacja w plaszczyznie kadru [deg]")
    ap.add_argument("--persp", type=float, default=0.15, help="sila znieksztalcen perspektywy")
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    classes = build_classes(args.targets_dir)
    print(f"Klasy ({len(classes)}):")
    for i, c in enumerate(classes): print(f"  {i:2d}  {c}")
    targets = {c: load_target(find_target_file(args.targets_dir, c)) for c in classes}
    digit_classes  = [c for c in classes if c in DIGITS]
    object_classes = [c for c in classes if c not in DIGITS]

    bg_files = []
    if args.backgrounds_dir:
        bg_files = [os.path.join(args.backgrounds_dir, f) for f in os.listdir(args.backgrounds_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for s in ("train", "val"):
        for d in ("images", "labels"):
            os.makedirs(os.path.join(args.out_dir, d, s), exist_ok=True)

    for i in range(args.num_images):
        canvas = get_background(bg_files, args.img_w, args.img_h, rng)
        group = digit_classes if (digit_classes and (not object_classes or rng.random() < 0.5)) else object_classes
        boxes, lines = [], []
        for _ in range(int(rng.integers(1, args.max_targets + 1))):
            cls = str(rng.choice(group))
            box = place_target(canvas, *targets[cls], rng, args.max_rot, args.persp, boxes)
            if box is None: continue
            boxes.append(box)
            x0, y0, x1, y1 = box
            lines.append(f"{classes.index(cls)} {(x0+x1)/2/args.img_w:.6f} {(y0+y1)/2/args.img_h:.6f} "
                         f"{(x1-x0)/args.img_w:.6f} {(y1-y0)/args.img_h:.6f}")
        canvas = augment_final(canvas, rng)
        split = "train" if rng.random() < args.train_ratio else "val"
        cv2.imwrite(os.path.join(args.out_dir, "images", split, f"{i:06d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(args.out_dir, "labels", split, f"{i:06d}.txt"), "w") as fp:
            fp.write("\n".join(lines) + ("\n" if lines else ""))
        if i % 200 == 0: print(f"  {i}/{args.num_images}")

    names = "\n".join(f"  - '{c}'" for c in classes)
    yaml_txt = (f"path: {os.path.abspath(args.out_dir).replace(chr(92), '/')}\n"
                f"train: images/train\nval: images/val\n\nnc: {len(classes)}\nnames:\n{names}\n")
    with open(os.path.join(args.out_dir, "data.yaml"), "w") as fp: fp.write(yaml_txt)
    print("Gotowe. Trenuj:  yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640")

if __name__ == "__main__":
    main()