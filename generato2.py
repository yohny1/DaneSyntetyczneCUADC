import argparse, os
import cv2
import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None

# ------------------------------------------------------------- wymiary [mm] (1 px = 1 mm)
SHAFT_W   = 1000                                  # szerokosc czesci kwadratowej
SHAFT_SQ  = 1000                                  # bok czesci kwadratowej
SHAFT_TIP = int(round(SHAFT_W * (3 ** 0.5) / 2))  # 866 mm - wysokosc trojkata (boki 1 m)
SHAFT_H   = SHAFT_SQ + SHAFT_TIP                  # 1866 mm
IMG_TARGET_MM = 600                               # Task One: 600x600
NUM_CELL_MM   = (300, 600)                        # Task Two: pojedynczy cel 300x600
NUM_GAP_MM    = 20                                # szczelina miedzy dwoma celami
DIGIT_H_MM    = 400                               # wysokosc cyfry ~40 cm
SILH_SCALE    = 0.80                              # wypeknienie sylwetki w Image Target
SHAFT_POLY = np.array([[SHAFT_W // 2, 0], [SHAFT_W - 1, SHAFT_TIP],
                       [SHAFT_W - 1, SHAFT_H - 1], [0, SHAFT_H - 1], [0, SHAFT_TIP]], np.int32)

# ------------------------------------------------------------- klasy / cele
def find_target_file(targets_dir, stem):
    for f in os.listdir(targets_dir):
        if os.path.splitext(f)[0] == stem and f.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(targets_dir, f)
    raise FileNotFoundError(stem)

def load_target(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[:, :, 3:4] / 255.0
        img = (img[:, :, :3] * a + 255.0 * (1 - a)).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, (gray < 128).astype(np.uint8) * 255

def crop_symbol(mask):
    ys, xs = np.nonzero(mask > 0)
    return mask[ys.min():ys.max()+1, xs.min():xs.max()+1]

def fit_size(mh, mw, max_h, max_w):
    s = min(max_h / mh, max_w / mw)
    return max(1, int(round(mh * s))), max(1, int(round(mw * s)))

def area_color(rng):
    """Kolor podstawy: Area A = niebieski, Area B = czerwony."""
    if rng.random() < 0.5:
        return (int(rng.uniform(180, 230)), int(rng.uniform(60, 100)), int(rng.uniform(30, 70)))   # BGR blue
    return (int(rng.uniform(30, 70)), int(rng.uniform(40, 80)), int(rng.uniform(170, 225)))         # BGR red

# ------------------------------------------------------------- render cyfr (bold sans-serif, ~400 mm)
_FONT_CACHE = {}
def _get_font(size):
    if ImageFont is None: return None
    if size in _FONT_CACHE: return _FONT_CACHE[size]
    names = ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial-Bold.ttf", "arialbd.ttf",
             "Helvetica-Bold.ttf", "DejaVuSans.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
             "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
             "C:/Windows/Fonts/arialbd.ttf"]
    f = None
    for n in names:
        try:
            f = ImageFont.truetype(n, size); break
        except Exception:
            continue
    _FONT_CACHE[size] = f
    return f

def render_digit_mask(d, cw, ch, digit_h):
    """Maska cyfry o zadanej wysokosci [px=mm], wycentrowana w panelu cw x ch."""
    f = _get_font(100)
    if f is not None:
        bb = f.getbbox(d); h0 = bb[3] - bb[1]
        f = _get_font(max(8, int(100 * digit_h / h0))) or f
        bb = f.getbbox(d); w, h = bb[2] - bb[0], bb[3] - bb[1]
        img = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(img).text(((cw - w) / 2 - bb[0], (ch - h) / 2 - bb[1]), d, fill=255, font=f)
        return (np.array(img) > 127).astype(np.uint8) * 255
    # fallback cv2 (bez PIL)
    (_, th0), _ = cv2.getTextSize(d, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    scale = digit_h / th0
    thick = max(2, int(round(digit_h * 0.09)))
    (tw, th), _ = cv2.getTextSize(d, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    img = np.zeros((ch, cw), np.uint8)
    cv2.putText(img, d, ((cw - tw) // 2, (ch + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thick, cv2.LINE_AA)
    return img

# ------------------------------------------------------------- plansze "strzałka" (1 px = 1 mm)
def make_shaft(color):
    board = np.zeros((SHAFT_H, SHAFT_W, 3), np.uint8)
    cv2.fillPoly(board, [SHAFT_POLY], color)
    cov = np.zeros((SHAFT_H, SHAFT_W), np.uint8)
    cv2.fillPoly(cov, [SHAFT_POLY], 255)
    return board, cov

def rotate_all(board, cov, symbols, ang):
    """Losowy kierunek strzałki: obrocaly board, maski symboli i polygon."""
    if abs(ang) < 1e-6:
        return board, cov, symbols, SHAFT_POLY
    h, w = board.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    board = cv2.warpAffine(board, M, (nw, nh), borderValue=(255, 255, 255))
    cov   = cv2.warpAffine(cov,   M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    sym   = [(d, cv2.warpAffine(m, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)) for d, m in symbols]
    p = np.hstack([SHAFT_POLY.astype(np.float32), np.ones((len(SHAFT_POLY), 1), np.float32)])
    return board, cov, sym, (M @ p.T).T.astype(np.int32)

def compose_image_board(cls, targets, color, rng, arrow_rot):
    """Task One: strzałka + bialy 600x600 mm + czarna sylwetka."""
    board, cov = make_shaft(color)
    P = IMG_TARGET_MM
    x0 = (SHAFT_W - P) // 2; y0 = SHAFT_TIP + (SHAFT_SQ - P) // 2
    board[y0:y0+P, x0:x0+P] = (255, 255, 255)
    _, tmask = targets[cls]
    sym = crop_symbol(tmask)
    nh, nw = fit_size(*sym.shape, int(SILH_SCALE * P), int(SILH_SCALE * P))
    sym_r = (cv2.resize(sym, (nw, nh), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8) * 255
    m = np.zeros((SHAFT_H, SHAFT_W), np.uint8)
    py, px = y0 + (P - nh) // 2, x0 + (P - nw) // 2
    m[py:py+nh, px:px+nw] = sym_r
    board[m > 127] = (0, 0, 0)
    return rotate_all(board, cov, [(cls, m)], rng.uniform(0, arrow_rot))

def compose_digit_board(digit_classes, color, rng, arrow_rot):
    """Task Two: strzałka + 2 x 300x600 mm, cyfry bold ~400 mm."""
    board, cov = make_shaft(color)
    cw, ch = NUM_CELL_MM
    xs = (SHAFT_W - (2 * cw + NUM_GAP_MM)) // 2
    y0 = SHAFT_TIP + (SHAFT_SQ - ch) // 2
    symbols = []
    for i in range(2):
        d = str(rng.choice(digit_classes))
        px = xs + i * (cw + NUM_GAP_MM)
        board[y0:y0+ch, px:px+cw] = (255, 255, 255)
        m = np.zeros((SHAFT_H, SHAFT_W), np.uint8)
        m[y0:y0+ch, px:px+cw] = render_digit_mask(d, cw, ch, DIGIT_H_MM)
        board[m > 127] = (0, 0, 0)
        symbols.append((d, m))
    return rotate_all(board, cov, symbols, rng.uniform(0, arrow_rot))

def compose_blank_shaft(color, rng, arrow_rot):
    """Hard negative: kołek bez celów (strzałka bez bialych paneli)."""
    board, cov = make_shaft(color)
    return rotate_all(board, cov, [], rng.uniform(0, arrow_rot))

# ------------------------------------------------------------- geometria / blend
def make_quad(tw, th, w, h, rng, f, max_rot, persp):
    ang = np.deg2rad(rng.uniform(-max_rot, max_rot))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
    ctr = np.array([rng.uniform(0.08, 0.92) * w, rng.uniform(0.08, 0.92) * h])
    dst = ((src - [tw / 2, th / 2]) * (f * h / th)) @ R.T + ctr
    dst = (dst + rng.uniform(-persp, persp, (4, 2)) * (f * h)).astype(np.float32)
    return src, dst

def draw_shadow(canvas, poly, rng, strength=None):
    h, w = canvas.shape[:2]
    off = np.array([[rng.uniform(0.005, 0.02) * w, rng.uniform(0.005, 0.03) * h]])
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [(poly + off).astype(np.int32)], 255)
    s = strength if strength is not None else rng.uniform(0.15, 0.35)
    m = (cv2.GaussianBlur(m, (0, 0), 6) / 255.0 * s)[..., None]
    canvas[:] = (canvas * (1 - m)).astype(np.uint8)

def blend(canvas, overlay, mask, blur=3):
    a = (cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32) / 255.0)[..., None]
    canvas[:] = (canvas * (1 - a) + overlay * a).astype(np.uint8)

def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    if inter <= 0: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

# ------------------------------------------------------------- tła
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
        if img is None:
            return procedural_background(w, h, rng)
        if img is not None:
            bh, bw = img.shape[:2]
            s = max(w / bw, h / bh)
            img = cv2.resize(img, (int(bw * s) + 1, int(bh * s) + 1))
            nh, nw = img.shape[:2]
            y0, x0 = rng.integers(0, nh - h + 1), rng.integers(0, nw - w + 1)
            return (img[y0:y0+h, x0:x0+w].astype(np.float32) * rng.uniform(0.7, 1.2)).clip(0, 255).astype(np.uint8)
    return procedural_background(w, h, rng)

# ------------------------------------------------------------- wklejanie planszy (+bboxe symboli)
def place_board(canvas, board, cov, symbols, poly, rng, args, boxes):
    h, w = canvas.shape[:2]
    th, tw = board.shape[:2]
    for _ in range(12):
        f = rng.uniform(args.min_scale, args.max_scale)
        src, dst = make_quad(tw, th, w, h, rng, f, args.max_rot, args.persp)
        M = cv2.getPerspectiveTransform(src, dst)

        panel_w = cv2.warpPerspective(cov, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        pys, pxs = np.nonzero(panel_w > 0)
        if len(pxs) < 30: continue
        px0, px1, py0, py1 = pxs.min(), pxs.max(), pys.min(), pys.max()
        if px0 < 0 or py0 < 0 or px1 >= w or py1 >= h: continue
        panel_rect = (int(px0), int(py0), int(px1) + 1, int(py1) + 1)

        rects, ok = [], True
        for _, m in symbols:
            sw = cv2.warpPerspective(m, M, (w, h), borderValue=0)
            ys, xs = np.nonzero(sw > 50)
            if len(xs) < 15: ok = False; break
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h or x1 - x0 < 6 or y1 - y0 < 6: ok = False; break
            rects.append((int(x0), int(y0), int(x1) + 1, int(y1) + 1))
        if not ok: continue
        if any(iou(panel_rect, b) > 0 for b in boxes): continue
        break
    else:
        return None

    dst_poly = cv2.perspectiveTransform(poly.reshape(1, -1, 2).astype(np.float32), M)[0]
    draw_shadow(canvas, dst_poly, rng)
    b = (board.astype(np.float32) * rng.uniform(0.80, 1.0) + rng.uniform(-12, 4)).clip(0, 255).astype(np.uint8)
    if rng.random() < 0.3: b = cv2.GaussianBlur(b, (3, 3), 0)
    warped = cv2.warpPerspective(b, M, (w, h), borderValue=(255, 255, 255))
    panel  = cv2.warpPerspective(cov, M, (w, h), borderValue=0)
    blend(canvas, warped, panel)
    return panel_rect, [(cls, r) for (cls, _), r in zip(symbols, rects)]

# ------------------------------------------------------------- przeszkadzacze
def add_distractors(canvas, rng, args, d_files):
    h, w = canvas.shape[:2]
    for _ in range(int(rng.integers(0, args.max_distractors + 1))):
        opts = ["stone", "stick", "paper", "blank"] + (["custom"] if d_files else [])
        kind = str(rng.choice(opts))
        if kind == "stone":
            r = max(2, int(rng.uniform(0.008, 0.04) * h))
            cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
            ax, ay = max(2, int(r * rng.uniform(0.8, 1.8))), max(2, int(r * rng.uniform(0.6, 1.3)))
            ang = int(rng.uniform(0, 180))
            m = np.zeros((h, w), np.uint8);  cv2.ellipse(m, (cx, cy), (ax, ay), ang, 0, 360, 255, -1)
            sm = np.zeros((h, w), np.uint8); cv2.ellipse(sm, (cx + r, cy + r // 2), (ax, ay), ang, 0, 360, 255, -1)
            s = (cv2.GaussianBlur(sm, (0, 0), 4) / 255.0 * rng.uniform(0.15, 0.30))[..., None]
            canvas[:] = (canvas * (1 - s)).astype(np.uint8)
            g = rng.uniform(70, 160)
            ov = canvas.copy(); ov[m > 0] = [int(g*rng.uniform(.85, 1)), int(g*rng.uniform(.9, 1.05)), int(g*rng.uniform(.95, 1.2))]
            blend(canvas, ov, m)
        elif kind == "stick":
            L = int(rng.uniform(0.03, 0.12) * h); tk = max(1, int(L * rng.uniform(0.04, 0.09)))
            cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
            a = np.deg2rad(rng.uniform(0, 360))
            p1 = (int(cx - np.cos(a)*L/2), int(cy - np.sin(a)*L/2)); p2 = (int(cx + np.cos(a)*L/2), int(cy + np.sin(a)*L/2))
            m = np.zeros((h, w), np.uint8); cv2.line(m, p1, p2, 255, tk, cv2.LINE_AA)
            ov = canvas.copy(); cv2.line(ov, p1, p2, [int(rng.uniform(25, 60)), int(rng.uniform(35, 75)), int(rng.uniform(55, 105))], tk, cv2.LINE_AA)
            blend(canvas, ov, m)
        elif kind == "paper":
            src, dst = make_quad(100, int(rng.uniform(70, 140)), w, h, rng, rng.uniform(0.02, 0.08), 180, 0.3)
            draw_shadow(canvas, dst, rng)
            g = int(rng.uniform(190, 255))
            m = np.zeros((h, w), np.uint8);  cv2.fillPoly(m, [dst.astype(np.int32)], 255)
            ov = canvas.copy(); cv2.fillPoly(ov, [dst.astype(np.int32)], (g, g, min(255, g + 5)))
            blend(canvas, ov, m)
        elif kind == "blank":                      # pusty kołek (strzałka bez paneli) - hard negative
            board, cov, _, poly = compose_blank_shaft(area_color(rng), rng, args.arrow_rot)
            bh, bw = board.shape[:2]
            src, dst = make_quad(bw, bh, w, h, rng, rng.uniform(args.min_scale, args.max_scale), args.max_rot, args.persp)
            M = cv2.getPerspectiveTransform(src, dst)
            draw_shadow(canvas, cv2.perspectiveTransform(poly.reshape(1, -1, 2).astype(np.float32), M)[0], rng)
            warped = cv2.warpPerspective(board, M, (w, h), borderValue=(255, 255, 255))
            panel  = cv2.warpPerspective(cov, M, (w, h), borderValue=0)
            blend(canvas, warped, panel)
        else:
            img = cv2.imread(str(rng.choice(d_files)), cv2.IMREAD_UNCHANGED)
            if img is None: continue
            if img.ndim == 3 and img.shape[2] == 4:
                m0, img = img[:, :, 3], img[:, :, :3]
            else:
                m0 = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 240).astype(np.uint8) * 255
            src, dst = make_quad(img.shape[1], img.shape[0], w, h, rng,
                                 rng.uniform(args.min_scale, args.max_scale), 180, args.persp)
            M = cv2.getPerspectiveTransform(src, dst)
            draw_shadow(canvas, dst, rng)
            blend(canvas, cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255)),
                  cv2.warpPerspective(m0, M, (w, h), borderValue=0))

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
    ap.add_argument("--backgrounds-dir", default=None)
    ap.add_argument("--distractors-dir", default=None)
    ap.add_argument("--out-dir", default="dataset")
    ap.add_argument("--num-images", type=int, default=2000)
    ap.add_argument("--img-w", type=int, default=1280)
    ap.add_argument("--img-h", type=int, default=720)
    ap.add_argument("--max-targets", type=int, default=3)
    ap.add_argument("--min-scale", type=float, default=0.06)
    ap.add_argument("--max-scale", type=float, default=0.30)
    ap.add_argument("--max-distractors", type=int, default=4)
    ap.add_argument("--arrow-rot", type=float, default=360, help="zakres losowania kierunku strzałki [deg]")
    ap.add_argument("--max-rot", type=float, default=20)
    ap.add_argument("--persp", type=float, default=0.15)
    ap.add_argument("--digits", default="0123456789", help="klasy cyfr (renderowane fontem)")
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["mixed", "digits", "images"], default="mixed")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    digit_classes = list(args.digits)
    object_classes = []
    if os.path.isdir(args.targets_dir):
        stems = [os.path.splitext(f)[0] for f in sorted(os.listdir(args.targets_dir))
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        object_classes = [s for s in stems if s not in digit_classes]
    classes = digit_classes + object_classes + ["panel", "empty"]
    if not classes:
        raise SystemExit("Brak klas: dodaj sylwetki do --targets-dir i/lub --digits.")
    print(f"Klasy ({len(classes)}):")
    for i, c in enumerate(classes): print(f"  {i:2d}  {c}")
    targets = {c: load_target(find_target_file(args.targets_dir, c)) for c in object_classes}

    def list_imgs(d):
        return [os.path.join(d, f) for f in os.listdir(d)] if d and os.path.isdir(d) else []
    bg_files = [f for f in list_imgs(args.backgrounds_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    d_files  = [f for f in list_imgs(args.distractors_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for s in ("train", "val"):
        for d in ("images", "labels"):
            os.makedirs(os.path.join(args.out_dir, d, s), exist_ok=True)

    for i in range(args.num_images):
        canvas = get_background(bg_files, args.img_w, args.img_h, rng)
        add_distractors(canvas, rng, args, d_files)
        if args.mode == "digits":
            scene_digits = True
        elif args.mode == "images":
            scene_digits = False
        else:
            scene_digits = bool(digit_classes) and (not object_classes or rng.random() < 0.5)
        boxes, lines = [], []
        for _ in range(int(rng.integers(1, args.max_targets + 1))):
            color = area_color(rng)                    # A/B per kołek
            if scene_digits:
                board, cov, symbols, poly = compose_digit_board(digit_classes, color, rng, args.arrow_rot)
            else:
                board, cov, symbols, poly = compose_image_board(str(rng.choice(object_classes)), targets, color, rng, args.arrow_rot)
            placed = place_board(canvas, board, cov, symbols, poly, rng, args, boxes)
            if not placed: continue
            panel_rect, sym_rects = placed
            boxes.append(panel_rect)
            px0, py0, px1, py1 = panel_rect
            lines.append(f"{classes.index('panel')} {(px0+px1)/2/args.img_w:.6f} {(py0+py1)/2/args.img_h:.6f} "
                        f"{(px1-px0)/args.img_w:.6f} {(py1-py0)/args.img_h:.6f}")
            if sym_rects:
                for cls, (x0, y0, x1, y1) in sym_rects:
                    boxes.append((x0, y0, x1, y1))
                    lines.append(f"{classes.index(cls)} {(x0+x1)/2/args.img_w:.6f} {(y0+y1)/2/args.img_h:.6f} "
                                f"{(x1-x0)/args.img_w:.6f} {(y1-y0)/args.img_h:.6f}")
            else:
                lines.append(f"{classes.index('empty')} {(px0+px1)/2/args.img_w:.6f} {(py0+py1)/2/args.img_h:.6f} "
                            f"{(px1-px0)/args.img_w:.6f} {(py1-py0)/args.img_h:.6f}")
        canvas = augment_final(canvas, rng)
        split = "train" if rng.random() < args.train_ratio else "val"
        cv2.imwrite(os.path.join(args.out_dir, "images", split, f"{i:06d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(args.out_dir, "labels", split, f"{i:06d}.txt"), "w") as fp:
            fp.write("\n".join(lines) + ("\n" if lines else ""))
        if i % 200 == 0: print(f"  {i}/{args.num_images}")

    names = "\n".join(f"  - '{c}'" for c in classes)
    with open(os.path.join(args.out_dir, "data.yaml"), "w") as fp:
        fp.write(f"path: {os.path.abspath(args.out_dir).replace(chr(92), '/')}\n"
                 f"train: images/train\nval: images/val\nnc: {len(classes)}\nnames:\n{names}\n")
    print("Gotowe. Trenuj:  yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640")

if __name__ == "__main__":
    main()
