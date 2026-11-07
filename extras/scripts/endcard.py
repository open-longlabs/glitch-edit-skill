#!/usr/bin/env python3
"""End card: shoe in the dark, flowing light strands behind it, small centred Swoosh, JUST DO IT. fading in.
endcard.py <shoe frame or clip> <out.png|out.mp4> [frames] [small print line]"""
import sys, subprocess, tempfile
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from rembg import remove, new_session
W, H = 1920, 1080; rng = np.random.default_rng(4)
FUT = "/System/Library/Fonts/Supplemental/Futura.ttc"; MONO = "/System/Library/Fonts/Menlo.ttc"
def font(path, size, index=0):
    """macOS font if present, else DejaVu, else Pillow's default (so it runs on Linux and in sandboxes)."""
    for p, i in ((path, index), ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0)):
        try: return ImageFont.truetype(p, size, index=i)
        except OSError: pass
    return ImageFont.load_default(size)
def fit(img):
    h, w = img.shape[:2]; t = W / H
    if w / h > t: nw = int(h * t); img = img[:, (w - nw) // 2:(w - nw) // 2 + nw]
    else: nh = int(w / t); img = img[(h - nh) // 2:(h - nh) // 2 + nh]
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
def swoosh(size, cx, cy):
    def bez(p0, p1, p2, p3, n=40):
        t = np.linspace(0, 1, n)[:, None]
        return (1 - t) ** 3 * np.array(p0) + 3 * (1 - t) ** 2 * t * np.array(p1) + 3 * (1 - t) * t ** 2 * np.array(p2) + t ** 3 * np.array(p3)
    pts = np.concatenate([bez((1, 0), (.62, .17), (.30, .40), (.12, .37)), bez((.12, .37), (-.02, .35), (-.03, .22), (.07, .11)),
                          bez((.07, .11), (.04, .22), (.14, .29), (.30, .255)), bez((.30, .255), (.55, .20), (.82, .08), (1, 0))])
    pts = (pts - [.5, .2]) * size + [cx, cy]; mk = np.zeros((H, W), np.float32)
    cv2.fillPoly(mk, [pts.astype(np.int32)], 1.0, cv2.LINE_AA); return cv2.GaussianBlur(mk, (0, 0), .8)
# flow-field strands (computed once, animated by drawing a sliding window of each path)
N, STEPS = 2600, 170
def field(x, y, t):
    return np.sin(x / 260 + t) * 1.3 + np.cos(y / 190 - t * .7) * .9 + np.sin((x + y) / 420) * .6
starts = np.stack([rng.uniform(-300, W * .7, N), rng.normal(H * .5, H * .16, N)], 1)
cols = np.array([(230, 90, 170), (255, 120, 90), (240, 200, 120), (80, 200, 255), (200, 80, 255)], np.float32)  # BGR: purple, blue, pale, yellow, pink
pal = cols[rng.integers(0, len(cols), N)]
paths = np.zeros((N, STEPS, 2), np.float32); p = starts.copy()
for s in range(STEPS):
    a = field(p[:, 0], p[:, 1], 0) * .55 - .25; p = p + np.stack([np.cos(a), np.sin(a)], 1) * 9; paths[:, s] = p
def strands(phase):
    c = np.zeros((H, W, 3), np.float32); L = 60; o = int(phase * (STEPS - L))
    for k in range(N):
        seg = paths[k, o:o + L].astype(np.int32); cv2.polylines(c, [seg], False, tuple(float(v) * .22 for v in pal[k]), 1, cv2.LINE_AA)
    return np.clip(c + cv2.GaussianBlur(c, (0, 0), 6) * 1.5, 0, 255)
seg = new_session("u2net")
def frame(shoe, k, n):
    t = k / max(1, n - 1)
    sm = cv2.resize(shoe, (int(W * .62), int(H * .62))); canvas = np.zeros((H, W, 3), np.uint8)
    y0, x0 = int(H * .2), int(W * .19); canvas[y0:y0 + sm.shape[0], x0:x0 + sm.shape[1]] = sm; shoe = canvas
    m = np.array(remove(Image.fromarray(shoe[:, :, ::-1]), session=seg))[:, :, 3].astype(np.float32)[:, :, None] / 255
    rim = np.clip(cv2.GaussianBlur(m[:, :, 0], (0, 0), 4) - cv2.GaussianBlur(m[:, :, 0], (0, 0), 14), 0, 1)[:, :, None]
    s = shoe.astype(np.float32) * .55 + rim * np.array([255, 120, 200]) * 1.2
    bg = strands(.15 + .7 * t) * min(1, t * 3)
    o = bg * (1 - m) + s * m
    a_logo = np.clip((t - .25) / .25, 0, 1); a_txt = np.clip((t - .5) / .25, 0, 1)
    sw = swoosh(240, W // 2, int(H * .47))[:, :, None]; o = o * (1 - sw * a_logo) + 250 * sw * a_logo
    pil = Image.fromarray(np.clip(o, 0, 255).astype(np.uint8)[:, :, ::-1]).convert("RGBA"); lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    f = font(FUT, 72, 4); txt = "JUST DO IT."; tw = d.textlength(txt, font=f)
    d.text(((W - tw) / 2, H * .47 + 60), txt, font=f, fill=(250, 250, 250, int(255 * a_txt)))
    fs = font(MONO, 20); l2 = LINE
    d.text(((W - d.textlength(l2, font=fs)) / 2, H - 70), l2, font=fs, fill=(170, 170, 170, int(255 * a_txt)))
    return np.array(Image.alpha_composite(pil, lay).convert("RGB"))[:, :, ::-1]
src, out = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 72
LINE = sys.argv[4] if len(sys.argv) > 4 else ""   # small print, e.g. "unofficial spec ad, not affiliated with <brand>."
if src.endswith(".png"): shoes = [fit(cv2.imread(src))] * n
else:
    c = cv2.VideoCapture(src); shoes = []
    while True:
        ok, f_ = c.read()
        if not ok: break
        shoes.append(fit(f_))
    shoes = [shoes[min(len(shoes) - 1, int(k * len(shoes) / n))] for k in range(n)]
if out.endswith(".png"): cv2.imwrite(out, frame(shoes[-1], n - 1, n)); print("wrote", out); sys.exit()
tmp = tempfile.mkdtemp()
for k in range(n): cv2.imwrite(f"{tmp}/{k:04d}.png", frame(shoes[k], k, n))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", f"{tmp}/%04d.png", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out], check=True); print("wrote", out)
