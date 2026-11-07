#!/usr/bin/env python3
# Reference implementation from the Nike ACG Mowabb spec ad (Nov 2026). Paths point at that project layout
# (clips/, gen3/, edit/); adapt the clip paths and cue times to your own footage.
"""Cut v15 = storyboard v4: fixed shoe clips (A2/B2), whip eye-to-shoe with shoe flash burst, fewer close-ups, final jump + white Swoosh.
Cut v14 = v13 + data pages flashed on the moving close-ups (shades + shoe facts, one joke line each).
Cut v13 = v12 with the close-ups as generated clips (clips/CU-*.mp4) and a 1.2 s run.
Cut v12 = v11 opening + Nike Lab style close-ups (stills with drift as previs) + transitions + end card.
Cut v11 (opening of storyboard v3) built on v10 looks.
Sources: clips/A4-seedance-orbit-480p.mp4 (room jump + hologram bullet time, ends on J2) and clips/B-forest-480p.mp4 (starts on J2F).
Cut v10, built like the reference glitch edit: every look is a treatment of the CURRENT moving
frame (same pose, motion never stops), 2 to 3 frames per look, no white frames, no stills from other shots.
Arc: living room = raw B&W hook, flash burst from the push-off, match cut at the peak = switch to graded colour,
then short forest scenes with a short burst on each impact frame.

fx_v10.py <test-jump dir>      source: clips/matchcut-15s-480p.mp4 (one generation, 24 fps)
"""
import sys, os, glob, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

D = sys.argv[1]; W, H = 1920, 1080; FPS = 24
tmp = f"{D}/edit/v16frames"; os.makedirs(tmp, exist_ok=True)
for f in glob.glob(f"{tmp}/*.png"): os.remove(f)
rng = np.random.default_rng(7)

def fit(img):
    h, w = img.shape[:2]; t = W / H
    if w / h > t: nw = int(h * t); img = img[:, (w - nw) // 2:(w - nw) // 2 + nw]
    else: nh = int(w / t); img = img[(h - nh) // 2:(h - nh) // 2 + nh]
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
def load(p):
    c = cv2.VideoCapture(p); o = []
    while True:
        ok, f = c.read()
        if not ok: break
        o.append(fit(f))
    return o
A = load(f"{D}/clips/A5-seedance-orbit-fix-480p.mp4" if os.path.exists(f"{D}/clips/A5-seedance-orbit-fix-480p.mp4") else f"{D}/clips/A4-seedance-orbit-480p.mp4"); B = load(f"{D}/clips/B2-forest-fix-360p.mp4") if os.path.exists(f"{D}/clips/B2-forest-fix-360p.mp4") else []
V = A + B; TB = len(A) / FPS          # clip B starts at virtual time TB

def zoomat(img, z, cx, cy):
    M = cv2.getRotationMatrix2D((cx, cy), 0, z); M[0, 2] += W / 2 - cx; M[1, 2] += H / 2 - cy
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
from rembg import remove, new_session
S_HUMAN, S_OBJ = new_session("u2net_human_seg"), new_session("u2net")
from transformers import pipeline
dp = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
cache = {}
def mask(i, obj=False):
    k = ("m", i, obj)
    if k not in cache:
        small = cv2.resize(V[i], (960, 540))
        a = np.array(remove(Image.fromarray(small[:, :, ::-1]), session=S_OBJ if obj else S_HUMAN))[:, :, 3]
        cache[k] = cv2.resize(a, (W, H)) > 128
    return cache[k]
def depth(i):
    if ("d", i) not in cache:
        d = np.array(dp(Image.fromarray(cv2.resize(V[i], (960, 540))[:, :, ::-1]))["depth"].resize((W, H))).astype(np.float32)
        cache[("d", i)] = (d - d.min()) / (d.max() - d.min() + 1e-6)
    return cache[("d", i)]

MONO = "/System/Library/Fonts/Menlo.ttc"; HEL = "/System/Library/Fonts/HelveticaNeue.ttc"
F_S, F_M = ImageFont.truetype(MONO, 22), ImageFont.truetype(MONO, 30)
F_BIG = ImageFont.truetype(HEL, 120, index=1)
gray = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
GRAIN = [rng.normal(0, 1, (H, W)).astype(np.float32) for _ in range(6)]
grain = lambda n: GRAIN[n % 6]
def cont(g, k=1.8, mid=None):
    mid = np.median(g) if mid is None else mid
    return np.clip((g - mid) * k + 128, 0, 255)
def g3(g): return cv2.merge([g, g, g]).astype(np.uint8)
def text(img, lines, color, n):
    """Small print around the body: own generic values, readable only on pause."""
    p = Image.fromarray(img[:, :, ::-1].copy()); d = ImageDraw.Draw(p)
    for (x, y, s, big) in lines: d.text((x, y), s, font=F_M if big else F_S, fill=color)
    return np.array(p)[:, :, ::-1].copy()
RECEIPT_L = ["STORE #0611", "TRAIL SET B", "STAGE 4  STUDIO 2", "", "SAT 11/07/26   07:42:16", "TERM 003  RN 5824", "",
             "1 ITEM          $0.00", "  SUBTOTAL      $0.00", "  TAX           $0.00", "TOTAL           $0.00", "",
             "GRIP **********4732", "APPROVED", "AUTH 3F7A9C"]
RECEIPT_R = ["TERMINAL ID 00348729", "BATCH 000317", "SEQ 0062", "MUD  0.84", "H2O  0.61", "", "THANK YOU", "RUN AGAIN"]
def receipt(img, color, n):
    L = [(40, 40 + 30 * k, s, False) for k, s in enumerate(RECEIPT_L)] + [(W - 360, 40 + 30 * k, s, False) for k, s in enumerate(RECEIPT_R)]
    out = text(img, L, color, n)
    for k in range(70):                                   # barcodes
        w = int(rng.integers(1, 5)); x = 40 + k * 5
        if rng.random() > .35: cv2.rectangle(out, (x, H - 120), (x + w, H - 70), color[::-1], -1)
        y = 300 + k * 5
        if rng.random() > .35: cv2.rectangle(out, (W - 120, y), (W - 80, y + w), color[::-1], -1)
    for (x, y) in [(30, 30), (W - 30, 30), (30, H - 30), (W - 30, H - 30)]:
        cv2.line(out, (x - 12, y), (x + 12, y), color[::-1], 1); cv2.line(out, (x, y - 12), (x, y + 12), color[::-1], 1)
    return out
def hud(img, n, c=(235, 235, 235)):
    for (x0, y0, x1, y1) in [(60, 60, 140, 60), (60, 60, 60, 140), (W - 60, 60, W - 140, 60), (W - 60, 60, W - 60, 140),
                             (60, H - 60, 140, H - 60), (60, H - 60, 60, H - 140), (W - 60, H - 60, W - 140, H - 60), (W - 60, H - 60, W - 60, H - 140)]:
        cv2.line(img, (x0, y0), (x1, y1), c, 2)
    pts = [(80 + k * 9, int(H - 140 - 30 * abs(np.sin(k * .7 + n)) * (k % 3 == 0) - 6 * np.sin(k + n))) for k in range(36)]
    cv2.polylines(img, [np.array(pts, np.int32)], False, c, 1, cv2.LINE_AA)
    cv2.circle(img, (W - 220, 200), 46, c, 1, cv2.LINE_AA); cv2.ellipse(img, (W - 220, 200), (46, 46), 0, -90, -90 + 40 * (n % 9), c, 3)
    return img

# ---------- looks (each takes the frame index, works on the moving frame) ----------
def L_toner_inv(i, n, obj):
    m = mask(i, obj); g = gray(V[i]); hp = g - cv2.GaussianBlur(g, (0, 0), 5)
    body = np.clip(205 + hp * 5 + grain(n) * 30 - cv2.GaussianBlur(grain(n + 1), (0, 0), 3) * 60, 0, 255)
    out = np.zeros((H, W), np.float32); out[m] = body[m]
    return receipt(g3(out), (230, 230, 230), n)
def L_toner_dark(i, n, obj):
    m = mask(i, obj); g = cont(gray(V[i]), 1.6, 150) * .8 + grain(n) * 14
    out = np.full((H, W), 12, np.float32); out[m] = np.clip(g, 0, 255)[m]
    return receipt(g3(out), (230, 230, 230), n)
def L_thermal(i, n, obj):
    m = mask(i, obj); g = gray(V[i]); out = np.full((H, W, 3), 150, np.uint8)
    th = cv2.applyColorMap(cv2.normalize(cv2.GaussianBlur(g, (0, 0), 3), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
    sh = -int(W * .16); Ms = np.float32([[1, 0, sh], [0, 1, 0]])
    ths, ms = cv2.warpAffine(th, Ms, (W, H)), cv2.warpAffine(m.astype(np.uint8), Ms, (W, H)) > 0
    out[ms] = ths[ms]; bw = g3(cont(g, 1.5) * .55); out[m] = bw[m]
    out = hud(out, n); cv2.putText(out, "CORE 38.9", (int(W * .62), int(H * .42)), cv2.FONT_HERSHEY_PLAIN, 1.4, (40, 40, 220), 2)
    return out
def L_bars(i, n, obj):
    m = mask(i, obj); g = gray(V[i]); out = np.full((H, W, 3), 128, np.uint8); bw = g3(cont(g, 1.6) * .5); out[m] = bw[m]
    for k in range(4):
        x = int(rng.integers(W // 2, W - 400)); y = int(rng.integers(60, 360)); w = int(rng.integers(160, 420))
        cv2.rectangle(out, (x, y), (x + w, y + 34), [(30, 30, 200), (20, 110, 240), (40, 60, 230)][k % 3], -1)
    return hud(out, n)
def L_points(i, n, obj):
    m = mask(i, obj); g = gray(V[i]) / 255; out = np.zeros((H, W, 3), np.uint8)
    ys, xs = np.nonzero(m[::4, ::4]); keep = rng.random(len(xs)) < (.25 + .75 * g[ys * 4, xs * 4])
    for x, y in zip(xs[keep] * 4, ys[keep] * 4): out[y:y + 2, x:x + 2] = 235
    d = depth(i)
    for row in range(14):
        yb = int(H * .5 + row * 34); src = cv2.GaussianBlur(d[min(H - 1, int(H * (.3 + row * .04)))].reshape(1, -1), (0, 0), 14).ravel()
        for x in range(0, W, 9):
            y = int(yb - 140 * src[x] - 16 * np.sin(x / 140 + row + n * .4))
            if 0 <= y < H and not m[y, x]: cv2.circle(out, (x, y), 2, (225, 225, 225), -1, cv2.LINE_AA)
    return hud(out, n, (200, 200, 200))
def duo(g, lo, hi):
    t = (np.clip(g, 0, 255) / 255)[:, :, None]; return (np.array(lo) * (1 - t) + np.array(hi) * t).astype(np.uint8)
def L_orange(i, n, obj):
    m = mask(i, obj); g = cont(gray(V[i]), 1.4) + grain(n) * 18; g[m] *= .45; g[~m] = 150 + g[~m] * .4
    return duo(g, (20, 40, 80), (60, 150, 245))
def L_burst(i, n, obj):
    o = L_orange(i, n, obj).astype(np.float32); yy, xx = np.mgrid[0:H, 0:W]
    r = np.exp(-(((xx - W * .45) / (W * .5)) ** 2 + ((yy - H * .4) / (H * .6)) ** 2))[:, :, None]
    return np.clip(o * (1 - .6 * r) + np.array([200, 240, 255]) * .6 * r + 20, 0, 255).astype(np.uint8)
PAPER = np.clip(232 + cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 6) * 40, 180, 255)
def L_paper(i, n, obj, warm=False):
    m = mask(i, obj); g = cont(gray(V[i]), 1.7) + grain(n) * 22
    out = PAPER.copy(); out[m] = (PAPER * np.clip(g, 0, 255) / 255)[m]
    o = duo(out, (30, 40, 60) if warm else (25, 25, 28), (215, 228, 240) if warm else (226, 230, 232))
    return receipt(o, (60, 60, 60), n)
def L_film(i, n, obj):
    o = L_paper(i, n, obj, warm=n % 2 == 0)
    o = text(o, [(70, 50, "ALL CONDITIONS", False), (70, 90, f"00:00:0{3 + n % 6}:{(11 * n) % 60:02d}", True)], (40, 40, 40), n)
    p = Image.fromarray(o[:, :, ::-1]); d = ImageDraw.Draw(p); d.text((W - 640, 40), "SET B", font=F_BIG, fill=(70, 70, 70))
    o = np.array(p)[:, :, ::-1].copy()
    frame = np.zeros((H, W), np.uint8); cv2.rectangle(frame, (40, 30), (W - 40, H - 30), 255, -1)
    frame = cv2.GaussianBlur(cv2.erode(frame, np.ones((61, 61))), (0, 0), 8); frame = cv2.dilate(frame, np.ones((61, 61)))
    return (o * (frame[:, :, None] / 255)).astype(np.uint8)
def L_outline(i, n, obj):
    m = mask(i, obj).astype(np.uint8) * 255; e = cv2.dilate(cv2.Canny(m, 50, 150), np.ones((3, 3)))
    g = cv2.Canny(cv2.GaussianBlur(gray(V[i]).astype(np.uint8), (5, 5), 0), 40, 110) * (m > 0)
    out = np.zeros((H, W), np.uint8); out[e > 0] = 245; out[g > 0] = 170
    return hud(g3(out.astype(np.float32)), n, (180, 180, 180))
LOOK = dict(tinv=L_toner_inv, tdark=L_toner_dark, thermal=L_thermal, bars=L_bars, points=L_points, orange=L_orange,
            burst=L_burst, paper=L_paper, film=L_film, outline=L_outline)

# ---------- footage treatments ----------
def raw_bw(f, n):
    """The hook: grainy, dark, crushed B&W."""
    g = cont(gray(f), 1.5, 140) * .8 + grain(n) * 22
    o = g3(np.clip(g, 0, 255))
    if rng.random() < .3: x = int(rng.integers(0, W)); cv2.line(o, (x, 0), (x + int(rng.integers(-8, 8)), H), (200, 200, 200), 1)
    return o
def grade(f):
    """The reality: deeper contrast, cool greens, warm skin kept."""
    x = f.astype(np.float32) / 255; x = np.clip((x - .5) * 1.12 + .5, 0, 1); x[:, :, 0] *= 1.03; x[:, :, 2] *= .98
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)
def zoom(img, z):
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((W / 2, H / 2), 0, z), (W, H), borderMode=cv2.BORDER_REFLECT)


# ---------- data pages (minimal, accurate, one joke line each) ----------
F_T = ImageFont.truetype(MONO, 26); F_TB = ImageFont.truetype(MONO, 34); F_XS = ImageFont.truetype(MONO, 18)
SHADES = [("SHADES", "SPORT WRAP"), ("LENS", "8-BASE CURVE"), ("UV", "100% UVA/UVB"), ("NOSE PADS", "TPE, WET GRIP"),
          ("LENS W", "70 MM"), ("BRIDGE", "11 MM"), ("TEMPLE", "140 MM"), ("COLOUR", "BLACK/GREY")]
SHOE = [("MODEL", "ACG AIR MOWABB"), ("ACG PATCH", "OUTER COLLAR"), ("SIDE HOLES", "15  (5-6-4)"), ("HEEL DOTS", "9  (3x3)"),
        ("OUTSOLE", "LUGGED RUBBER"), ("MUD", "YES")]
JOKES = ["MAKING NIKE ADS UNTIL THEY CALL", "COLLAB STATUS: PENDING", "FOLLOW FOR MORE",
         "TESTED IN A FOREST BUILT IN A STUDIO", "NIKE HAS NOT ASKED. YET."]
def table(img, rows, x, y, col, ink, title=None, joke=None):
    p = Image.fromarray(img[:, :, ::-1].copy()); d = ImageDraw.Draw(p)
    if title: d.text((x, y), title, font=F_TB, fill=ink); y += 54
    for k, (a, b) in enumerate(rows):
        d.text((x, y + 38 * k), a, font=F_T, fill=ink); d.text((x + col, y + 38 * k), b, font=F_T, fill=ink)
    yy = y + 38 * len(rows) + 14; d.line((x, yy, x + col + 330, yy), fill=ink, width=1)
    if joke: d.text((x, yy + 14), joke, font=F_XS, fill=ink)
    return np.array(p)[:, :, ::-1].copy()
def barcode(img, x, y, w, h, ink):
    for k in range(w // 5):
        if rng.random() > .3: cv2.rectangle(img, (x + 5 * k, y), (x + 5 * k + int(rng.integers(1, 4)), y + h), ink[::-1], -1)
def lens_point(i, m):
    """Darkest blob in the upper part of the body mask = the sunglasses lens."""
    ys, xs = np.nonzero(m)
    if len(ys) == 0: return W // 2, H // 3
    top = ys.min() + (ys.max() - ys.min()) * .45
    g = gray(V[i]); cand = (g < 45) & m; cand[int(top):] = False
    n, lab, st, cen = cv2.connectedComponentsWithStats(cand.astype(np.uint8))
    if n < 2: return int(xs.mean()), int(ys.min() + 80)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]); return int(cen[k][0]), int(cen[k][1])
def P_shades_receipt(i, n, obj):
    o = L_paper(i, n, obj, warm=True) if False else None
    m = mask(i, obj); g = cont(gray(V[i]), 1.7) + grain(n) * 22
    out = PAPER.copy(); out[m] = (PAPER * np.clip(g, 0, 255) / 255)[m]
    o = duo(out, (30, 40, 60), (215, 228, 240))
    o = table(o, SHADES, 60, 70, 210, (55, 50, 45), "ITEM 01  SHADES", JOKES[0] + "   " + JOKES[3])
    barcode(o, 60, H - 140, 420, 60, (55, 50, 45)); return o
def ironbow():
    """Thermal camera palette: black, violet, red, orange, yellow, white (BGR LUT)."""
    stops = [(0, (0, 0, 0)), (.2, (90, 10, 40)), (.4, (140, 20, 150)), (.6, (40, 70, 235)), (.8, (20, 190, 255)), (1, (235, 250, 255))]
    lut = np.zeros((256, 3), np.float32)
    for k in range(256):
        t = k / 255
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t0 <= t <= t1: u = (t - t0) / (t1 - t0); lut[k] = np.array(c0) * (1 - u) + np.array(c1) * u
    return lut.astype(np.uint8)
IRON = ironbow()
def P_shades_hud(i, n, obj):
    """Thermal scan: warm skin, the area behind the lens stays cold (the shades shield it). Values illustrative."""
    m = mask(i, obj); g = cv2.GaussianBlur(gray(V[i]), (0, 0), 6) / 255
    heat = np.full((H, W), .08, np.float32) + cv2.GaussianBlur(np.random.default_rng(n).random((H // 8, W // 8)).astype(np.float32), (0, 0), 2).repeat(8, 0).repeat(8, 1)[:H, :W] * .06
    body = .55 + .35 * g + .1 * cv2.GaussianBlur(m.astype(np.float32), (0, 0), 30)
    heat[m] = body[m]
    # lens = darkest region inside the head band (relative to the face, so a reflective lens still counts)
    ys, xs = np.nonzero(m); lens = np.zeros((H, W), np.uint8); lx, ly = W // 2, H // 3
    if len(ys):
        y0 = ys.min(); y1 = int(y0 + (ys.max() - y0) * .42); band = m.copy(); band[y1:] = False; band[:int(y0 + (y1 - y0) * .25)] = False
        gv = gray(V[i]); thr = np.percentile(gv[band], 18)
        cand = (cv2.morphologyEx(((gv <= thr) & band).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))).astype(np.uint8)
        nl, lab, st, cen = cv2.connectedComponentsWithStats(cand)
        if nl > 1:
            k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]); lens[lab == k] = 1; lx, ly = int(cen[k][0]), int(cen[k][1])
    heat = heat * (1 - lens * .8) + .12 * lens * .8
    o = IRON[np.clip(heat * 255, 0, 255).astype(np.uint8)].copy()
    o[::3] = (o[::3] * .85).astype(np.uint8)
    # scale bar
    for y in range(160, H - 160):
        t = 1 - (y - 160) / (H - 320); o[y, W - 110:W - 80] = IRON[int(t * 255)]
    p = Image.fromarray(o[:, :, ::-1]); d = ImageDraw.Draw(p); ink = (240, 240, 240)
    for k, tc in enumerate(["38.0", "33.0", "28.0", "23.0", "18.0"]):
        d.text((W - 230, 150 + k * (H - 320) / 4), tc + " C", font=F_XS, fill=ink)
    sx, sy = int(lx), int(ly + 150)
    if len(ys) and not m[min(H - 1, sy), min(W - 1, sx)]: sx = int(xs[ys == min(ys.max(), sy)].mean())
    for (px, py, lab_, val) in [(lx, ly, "LENS", "21.4 C"), (sx, sy, "SKIN", "34.8 C")]:
        d.ellipse((px - 30, py - 30, px + 30, py + 30), outline=ink, width=2); d.line((px - 44, py, px + 44, py), fill=ink); d.line((px, py - 44, px, py + 44), fill=ink)
        d.text((px + 50, py - 30), lab_, font=F_XS, fill=ink); d.text((px + 50, py - 8), val, font=F_TB, fill=ink)
    d.text((60, 60), "THERMAL SCAN  /  SHADES ON", font=F_T, fill=ink)
    d.text((60, 100), "8-BASE CURVED LENS   100% UVA/UVB", font=F_XS, fill=ink)
    d.text((60, H - 80), "VALUES ILLUSTRATIVE.", font=F_XS, fill=(210, 210, 210))
    d.text((60, H - 56), "NOTE TO NIKE: PLZ COLLAB.", font=F_XS, fill=(170, 170, 170))
    return np.array(p)[:, :, ::-1].copy()
def P_shoe_spec(i, n, obj):
    m = mask(i, obj); g = cont(gray(V[i]), 1.6) + grain(n) * 20
    out = PAPER.copy(); out[m] = (PAPER * np.clip(g, 0, 255) / 255)[m]
    o = duo(out, (25, 25, 28), (226, 230, 232))
    o = table(o, SHOE, W - 640, 70, 230, (40, 40, 40), "SPEC  FW26", JOKES[4]); barcode(o, W - 640, H - 140, 420, 60, (40, 40, 40)); return o
def P_shoe_dark(i, n, obj):
    m = mask(i, obj); g = cont(gray(V[i]), 1.6, 150) * .8 + grain(n) * 14
    out = np.full((H, W), 12, np.float32); out[m] = np.clip(g, 0, 255)[m]
    o = table(g3(out), SHOE[:4], 60, 70, 230, (230, 230, 230), "ACG / ALL CONDITIONS", JOKES[1]); return o
def P_barcode(i, n, obj):
    o = L_toner_dark(i, n, obj); p = Image.fromarray(o[:, :, ::-1]); d = ImageDraw.Draw(p)
    d.text((W - 620, H - 200), "SKU  YOUR-HANDLE-001", font=F_T, fill=(230, 230, 230))
    d.text((W - 620, H - 160), "REQUEST  NIKE COLLAB #1", font=F_T, fill=(230, 230, 230))
    d.text((W - 620, H - 110), JOKES[5], font=F_XS, fill=(170, 170, 170)); o = np.array(p)[:, :, ::-1].copy()
    barcode(o, W - 620, H - 300, 520, 80, (230, 230, 230)); return o
F_FUT = "/System/Library/Fonts/Supplemental/Futura.ttc"
def P_roto(i, n, obj):
    """Nike-style poster frame: huge condensed italic type BEHIND him (auto roto from his mask)."""
    base = moody(V[i], n); m = mask(i, obj)
    layer = Image.new("L", (W, H), 0); d = ImageDraw.Draw(layer)
    lines = ["JUST", "DO", "IT."]; size = 330
    big = ImageFont.truetype(F_FUT, size, index=4)
    size = int(size * 700 / max(d.textlength(t, font=big) for t in lines)); big = ImageFont.truetype(F_FUT, size, index=4)
    y0 = (H - size * 3) // 2
    for k, t in enumerate(lines):
        tw = d.textlength(t, font=big); d.text((W - 110 - tw, y0 + size * k), t, font=big, fill=255)
    layer = np.array(layer.transform((W, H), Image.AFFINE, (1, 0.2, -60, 0, 1, 0), Image.BICUBIC)).astype(np.float32) / 255
    o = base.astype(np.float32); o = o * (1 - layer[:, :, None]) + 245 * layer[:, :, None]
    mm = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.2)[:, :, None]
    o = o * (1 - mm) + base.astype(np.float32) * mm          # he stands in front of the type
    p = Image.fromarray(np.clip(o, 0, 255).astype(np.uint8)[:, :, ::-1]); dd = ImageDraw.Draw(p)
    dd.text((96, H - 70), "AN UNSOLICITED AD BY @YOURHANDLE.  FOLLOW FOR MORE.", font=F_XS, fill=(200, 200, 200))
    return np.array(p)[:, :, ::-1].copy()
from ultralytics import YOLO
POSE = YOLO("yolov8n-pose.pt")
SKEL = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
def L_track(i, n, obj):
    """Green tracking HUD: dim grey frame, real pose keypoints + skeleton + small labels."""
    g = gray(V[i]) * .55 + grain(n) * 10; o = g3(np.clip(g, 0, 255)); o[::3] = (o[::3] * .7).astype(np.uint8)
    r = POSE(V[i], verbose=False)[0]; G = (60, 255, 90)
    if r.keypoints is not None and len(r.keypoints):
        xy = r.keypoints.xy[0].cpu().numpy(); cf = r.keypoints.conf[0].cpu().numpy()
        for a, b in SKEL:
            if cf[a] > .3 and cf[b] > .3: cv2.line(o, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), G, 2, cv2.LINE_AA)
        for k, (x, y) in enumerate(xy.astype(int)):
            if cf[k] > .3:
                cv2.circle(o, (x, y), 9, G, 1, cv2.LINE_AA); cv2.drawMarker(o, (x, y), G, cv2.MARKER_CROSS, 26, 1)
                cv2.putText(o, f"{k:02d} {cf[k]:.2f}", (x + 12, y - 8), cv2.FONT_HERSHEY_PLAIN, 1.1, G, 1, cv2.LINE_AA)
    return o
def L_xray(i, n, obj):
    """Neon x-ray: inverted luminance through a hot colour map, glowing on black."""
    m = mask(i, obj); g = gray(V[i]); inv = 255 - cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    c = cv2.applyColorMap(inv.astype(np.uint8), cv2.COLORMAP_TWILIGHT_SHIFTED).astype(np.float32)
    c = np.clip(c * 1.3 + cv2.GaussianBlur(c, (0, 0), 12) * .7, 0, 255); o = np.zeros_like(c); o[m] = c[m]
    o += cv2.GaussianBlur(o, (0, 0), 25) * .6; return np.clip(o, 0, 255).astype(np.uint8)
def echo_stack(i, k_back=(4, 8, 12, 16), obj=False):
    """Time-delayed cyan copies of him BEHIND him (afterimages), he stays solid in front."""
    base = grade(V[i]).astype(np.float32); m0 = cv2.GaussianBlur(mask(i, obj).astype(np.float32), (0, 0), 1.5)[:, :, None]
    lo = np.array([90, 60, 10], np.float32); hi = np.array([255, 245, 170], np.float32)
    for j, kb in enumerate(reversed(k_back)):
        jj = i - kb
        if jj < 0: continue
        mj = cv2.GaussianBlur(mask(jj, obj).astype(np.float32), (0, 0), 2)[:, :, None]
        t = (gray(V[jj]) / 255)[:, :, None]; tint = lo * (1 - t) + hi * t
        a = .32 + .14 * j; base = base * (1 - mj * a) + tint * mj * a
    out = base * (1 - m0) + grade(V[i]).astype(np.float32) * m0
    return np.clip(out, 0, 255).astype(np.uint8)
LOOK.update(track=L_track, xray=L_xray, roto=P_roto, srec=P_shades_receipt, shud=P_shades_hud, sspec=P_shoe_spec, sdark=P_shoe_dark, sbar=P_barcode)


def P_sketch(i, n, obj):
    """White technical sketch of the shoe with a few true callouts."""
    m = mask(i, obj); g = gray(V[i]).astype(np.uint8)
    e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 30, 90); e[~cv2.dilate(m.astype(np.uint8), np.ones((9, 9))).astype(bool)] = 0
    o = np.full((H, W, 3), 238, np.uint8); o[cv2.dilate(e, np.ones((2, 2))) > 0] = (40, 40, 40)
    ys, xs = np.nonzero(m)
    if len(xs):
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cv2.rectangle(o, (x0, y0), (x1, y1), (120, 120, 120), 1)
        p = Image.fromarray(o[:, :, ::-1]); d = ImageDraw.Draw(p)
        for k, (t, fx, fy) in enumerate([("ACG PATCH / OUTER COLLAR", .25, .2), ("SIDE HOLES 15", .55, .45), ("LUGGED OUTSOLE", .6, .92)]):
            px, py = int(x0 + (x1 - x0) * fx), int(y0 + (y1 - y0) * fy); tx, ty = min(W - 420, x1 + 40), 120 + 70 * k
            d.line((px, py, tx - 10, ty + 12), fill=(90, 90, 90), width=1); d.ellipse((px - 5, py - 5, px + 5, py + 5), outline=(200, 80, 30), width=2)
            d.text((tx, ty), t, font=F_T, fill=(40, 40, 40))
        d.text((60, H - 70), "DRAWING NO. LI-0001   " + JOKES[2], font=F_XS, fill=(90, 90, 90)); o = np.array(p)[:, :, ::-1].copy()
    return o
def P_glow(i, n, obj):
    """Molten orange render of the shoe on deep blue."""
    m = mask(i, obj); g = cv2.normalize(gray(V[i]), None, 0, 255, cv2.NORM_MINMAX)
    c = cv2.applyColorMap(g.astype(np.uint8), cv2.COLORMAP_HOT).astype(np.float32)
    bg = np.zeros((H, W, 3), np.float32); bg[:] = (70, 30, 10)
    o = bg.copy(); o[m] = c[m]; o += cv2.GaussianBlur(o * (m[:, :, None]), (0, 0), 30) * .8
    return hud(np.clip(o, 0, 255).astype(np.uint8), n, (200, 170, 120))
LOOK.update(sketch=P_sketch, glow=P_glow)
def swoosh(size, cx, cy):
    """Our own drawing of the Swoosh shape from cubic curves, as a filled mask (H x W float). size = width in px."""
    def bez(p0, p1, p2, p3, n=40):
        t = np.linspace(0, 1, n)[:, None]
        return (1 - t) ** 3 * np.array(p0) + 3 * (1 - t) ** 2 * t * np.array(p1) + 3 * (1 - t) * t ** 2 * np.array(p2) + t ** 3 * np.array(p3)
    pts = np.concatenate([
        bez((1.00, 0.00), (0.62, 0.17), (0.30, 0.40), (0.12, 0.37)),   # outer edge, tip down to the belly
        bez((0.12, 0.37), (-0.02, 0.35), (-0.03, 0.22), (0.07, 0.11)), # the hook curling up
        bez((0.07, 0.11), (0.04, 0.22), (0.14, 0.29), (0.30, 0.255)),  # inner edge of the hook
        bez((0.30, 0.255), (0.55, 0.20), (0.82, 0.08), (1.00, 0.00)),  # inner edge back up to the tip
    ])
    pts = (pts - [0.5, 0.2]) * size + [cx, cy]
    mk = np.zeros((H, W), np.float32); cv2.fillPoly(mk, [pts.astype(np.int32)], 1.0, cv2.LINE_AA)
    return cv2.GaussianBlur(mk, (0, 0), 1.0)
def pose_kp(i):
    r = POSE(V[i], verbose=False)[0]
    if r.keypoints is None or not len(r.keypoints) or r.keypoints.conf is None: return None, None
    return r.keypoints.xy[0].cpu().numpy(), r.keypoints.conf[0].cpu().numpy()
def jline(img, p, q, col, w=1, rngj=None, passes=2):
    """Pencil-like line: a few slightly jittered passes."""
    rr = rngj or rng
    for _ in range(passes):
        a = (int(p[0] + rr.normal(0, 1.5)), int(p[1] + rr.normal(0, 1.5))); b = (int(q[0] + rr.normal(0, 1.5)), int(q[1] + rr.normal(0, 1.5)))
        cv2.line(img, a, b, col, w, cv2.LINE_AA)
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
def L_pencil(i, n, obj):
    """Figure-drawing pencil sketch: real contours as light strokes + construction lines from the real pose."""
    g = gray(V[i]).astype(np.uint8); m = mask(i, obj)
    paper = np.clip(236 + grain(n) * 6, 0, 255); o = g3(paper)
    e1 = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 25, 70); e1[~cv2.dilate(m.astype(np.uint8), np.ones((15, 15))).astype(bool)] = 0
    ys, xs = np.nonzero(e1); keep = rng.random(len(xs)) < .8
    for x, y in zip(xs[keep], ys[keep]): o[y, x] = (95, 95, 95)
    o = cv2.GaussianBlur(o, (0, 0), .6)
    xy, cf = pose_kp(i)
    if xy is not None:
        c = (120, 120, 120); ok = lambda k: cf[k] > .3
        if ok(5) and ok(6):
            sw = np.linalg.norm(xy[5] - xy[6]); head = xy[0] if ok(0) else (xy[5] + xy[6]) / 2 - [0, sw * .8]
            r = int(sw * .38); cv2.ellipse(o, tuple(head.astype(int)), (r, int(r * 1.15)), 0, 0, 360, c, 1, cv2.LINE_AA)
            jline(o, head - [0, r], head + [0, r * 1.1], c); jline(o, head - [r, 0], head + [r, 0], c)
            neck = (xy[5] + xy[6]) / 2
            if ok(11) and ok(12):
                hip = (xy[11] + xy[12]) / 2; jline(o, neck, hip, c, 1)
                mid = (neck + hip) / 2; cv2.ellipse(o, tuple(mid.astype(int)), (int(sw * .55), int(np.linalg.norm(hip - neck) * .35)), 0, 0, 360, c, 1, cv2.LINE_AA)
                jline(o, xy[11], xy[12], c)
            jline(o, xy[5], xy[6], c)
        for a, b in LIMBS:
            if ok(a) and ok(b): jline(o, xy[a], xy[b], c, 1, passes=3); cv2.circle(o, tuple(xy[b].astype(int)), 9, c, 1, cv2.LINE_AA)
    p_ = Image.fromarray(o[:, :, ::-1]); d = ImageDraw.Draw(p_); d.text((60, H - 70), "STUDY NO. 7  /  LANDING", font=F_XS, fill=(110, 110, 110))
    return np.array(p_)[:, :, ::-1].copy()
def L_xray2(i, n, obj):
    """Blueprint x-ray: navy ground, translucent pale-blue body, glowing bones placed on the real pose."""
    m = mask(i, obj); g = gray(V[i]); hp = cv2.GaussianBlur(g - cv2.GaussianBlur(g, (0, 0), 8), (0, 0), 1)
    base = np.zeros((H, W, 3), np.float32); base[:] = (95, 45, 15)
    flesh = np.clip(.35 + np.abs(hp) / 30, 0, 1); body = np.zeros((H, W), np.float32); body[m] = flesh[m] * .55
    edge = cv2.GaussianBlur(cv2.Canny(m.astype(np.uint8) * 255, 50, 150).astype(np.float32) / 255, (0, 0), 3) * 2.5
    bones = np.zeros((H, W), np.float32); xy, cf = pose_kp(i)
    if xy is not None:
        ok = lambda k: cf[k] > .3
        sw = np.linalg.norm(xy[5] - xy[6]) if ok(5) and ok(6) else 120
        for a, b in LIMBS:
            if ok(a) and ok(b):
                cv2.line(bones, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), 1.0, max(3, int(sw * .07)), cv2.LINE_AA)
                for q in (xy[a], xy[b]): cv2.circle(bones, tuple(q.astype(int)), max(4, int(sw * .06)), 1.0, -1, cv2.LINE_AA)
        if ok(5) and ok(6) and ok(11) and ok(12):
            neck = (xy[5] + xy[6]) / 2; hip = (xy[11] + xy[12]) / 2; L = np.linalg.norm(hip - neck)
            cv2.line(bones, tuple(neck.astype(int)), tuple(hip.astype(int)), .9, max(3, int(sw * .05)), cv2.LINE_AA)
            for k in range(8):   # ribs
                c = neck + (hip - neck) * (.12 + .07 * k); cv2.ellipse(bones, tuple(c.astype(int)), (int(sw * (.42 - .02 * abs(k - 3))), int(L * .05)), 0, 0, 360, .7, 2, cv2.LINE_AA)
            cv2.ellipse(bones, tuple(hip.astype(int)), (int(sw * .45), int(L * .12)), 0, 0, 360, .9, 3, cv2.LINE_AA)
            head = xy[0] if ok(0) else neck - [0, sw * .8]
            cv2.ellipse(bones, tuple(head.astype(int)), (int(sw * .3), int(sw * .36)), 0, 0, 360, 1.0, 3, cv2.LINE_AA)
        bones *= m
    glow = body + edge * m + cv2.GaussianBlur(bones, (0, 0), 2) * .9 + cv2.GaussianBlur(bones, (0, 0), 12) * .6
    o = base + glow[:, :, None] * np.array([255, 235, 205])
    o += cv2.GaussianBlur(o, (0, 0), 20) * .15
    return np.clip(o, 0, 255).astype(np.uint8)
LOOK.update(track=L_pencil, xray=L_xray2)

frames = []; put = frames.append
def scene(t0, t1, burst_at=None, looks=(), obj=False, bw=False, step=1, punch=1.0, drift=.0015, cam=None):
    """Footage t0..t1. From burst_at on, each look takes the next 2 or 3 MOVING frames, then the footage continues."""
    a, b = int(t0 * FPS), int(t1 * FPS); idx = list(range(a, b, step))
    plan = {}
    if burst_at is not None:
        k = idx.index(min(idx, key=lambda j: abs(j - burst_at * FPS)))
        for n, name in enumerate(looks):
            for _ in range(1 if name == 'burst' else (2 if n % 3 else 3)):
                if k < len(idx): plan[idx[k]] = (name, n); k += 1
    for c, i in enumerate(idx):
        if i in plan: name, n = plan[i]; img = LOOK[name](i, n, obj)
        else: img = raw_bw(V[i], c) if bw else grade(V[i])
        if cam: z0, z1, cx, cy = cam; put(zoomat(img, z0 + (z1 - z0) * c / max(1, len(idx) - 1), cx, cy))
        else: put(zoom(img, punch + drift * c))



def zoomat2(img, z, cx, cy):
    M = cv2.getRotationMatrix2D((cx, cy), 0, z); M[0, 2] += W / 2 - cx; M[1, 2] += H / 2 - cy
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
def moody(img, n):
    """Close-up grade: crushed blacks, a touch of grain, keeps the purple."""
    x = img.astype(np.float32); x = np.clip((x - 18) * 1.08, 0, 255) + grain(n)[:, :, None] * 7
    return np.clip(x, 0, 255).astype(np.uint8)
def closeup(name, n=17, z0=1.0, z1=1.07, dx=0.0, dy=0.0):
    img = fit(cv2.imread(f"{D}/gen3/{name}/v1.png"))
    for k in range(n):
        t = k / (n - 1); put(moody(zoomat2(img, z0 + (z1 - z0) * t, W / 2 + dx * W * t, H / 2 + dy * H * t), k))
def tr_rgb(a, b):
    """RGB halftone glitch, 3 frames."""
    for k, m in enumerate([.3, .6, .9]):
        x = cv2.addWeighted(a, 1 - m, b, m, 0); bb, gg, rr = cv2.split(x); s = 24 - 6 * k
        x = cv2.merge([np.roll(bb, -s, 1), gg, np.roll(rr, s, 1)])
        for y in range(0, H, 60):
            if rng.random() < .4: x[y:y + 30] = np.roll(x[y:y + 30], int(rng.integers(-120, 120)), 1)
        x[::4] = (x[::4] * .55).astype(np.uint8); put(x)
def tr_leak(a, b):
    """Warm white light leak, 3 frames."""
    for k, m in enumerate([.45, 1.0, .5]):
        base = a if k < 1 else b; o = base.astype(np.float32) * (1 - .8 * m) + np.array([200, 235, 255]) * .8 * m
        put(np.clip(o, 0, 255).astype(np.uint8))
def tr_streak():
    """Black with thin rainbow streak lines, 3 frames."""
    for k in range(3):
        o = np.zeros((H, W, 3), np.uint8)
        for _ in range(26):
            y = int(rng.integers(0, H)); x0 = int(rng.integers(-200, W)); L = int(rng.integers(200, 900))
            c = [(255, 80, 80), (80, 255, 120), (80, 120, 255), (230, 230, 230)][int(rng.integers(0, 4))]
            cv2.line(o, (x0, y), (x0 + L, y + int(rng.integers(-3, 4))), c, 1, cv2.LINE_AA)
        put(cv2.GaussianBlur(o, (0, 0), .8))
first = lambda name: moody(fit(cv2.imread(f"{D}/gen3/{name}/v1.png")), 0)

CUES = {}
def cue(n): CUES[n] = round(len(frames) / FPS, 3)
# 1. the room, raw B&W hook: stance, push-off, burst on the rise (looks on the moving frames)
scene(0.0, 1.25, burst_at=0.45, bw=True, looks=["tinv", "tdark", "thermal", "bars", "orange", "paper", "film", "outline"])
# 2. time freezes, hologram bullet time (first colour = the purple shoes)
cue("freeze"); scene(1.25, 2.2, bw=True, drift=.001)
cue("orbit"); scene(2.2, 5.0, drift=.0008)
# 3. room returns (B&W), 1:1 MATCH CUT into the forest, landing, first strides
cue("room_back"); scene(5.0, 5.45, bw=True, drift=0); cue("matchcut")
# landing + first strides at real speed, CV hits (pose tracking, x-ray) on the impacts
a, b = int((TB + 0.15) * FPS), int((TB + 1.2) * FPS)
hits = {a + 9: "track", a + 10: "track", a + 15: "xray", a + 16: "xray"}
for c, i in enumerate(range(a, b)):
    if i in hits: put(LOOK[hits[i]](i, c, False)); continue
    put(grade(V[i]))                                   # echo stack removed (cut: it belonged to a different look)
# 4. close-ups as generated clips, about 1 s each, data pages flashed on the moving frames
def cu(name, t0=0.7, t1=1.7, burst=None, looks=(), obj=True):
    p = f"{D}/clips/CU-{name}.mp4"
    if not os.path.exists(p): return
    F = load(p); base = len(V); V.extend(F)
    a, b = int(t0 * FPS), min(len(F), int(t1 * FPS)); idx = list(range(base + a, base + b))
    plan = {}
    if burst is not None:
        k = int(burst * FPS) - a
        for n, nm in enumerate(looks):
            for _ in range(6 if nm == "roto" else (3 if nm.startswith("s") else 2)):
                if 0 <= k < len(idx): plan[idx[k]] = (nm, n); k += 1
    for c, i in enumerate(idx):
        img = LOOK[plan[i][0]](i, plan[i][1], obj) if i in plan else moody(V[i], c)
        put(zoom(img, 1 + .03 * c / max(1, len(idx) - 1)))
def head(name):
    p = f"{D}/clips/CU-{name}.mp4"; F = load(p) if os.path.exists(p) else None
    return moody(F[int(.7 * FPS)], 0) if F else np.zeros((H, W, 3), np.uint8)

# 4. WHIP eye to shoe (light speed), flash burst on the shoe, hero hold (ref 6-8 s)
pw = f"{D}/clips/W-whip-360p.mp4"
if os.path.exists(pw):
    Fw = load(pw); base = len(V); V.extend(Fw); stop = int(1.3 * FPS)
    tr_rgb(frames[-1], moody(Fw[0], 0))
    cue("whip")
    for c in range(0, stop): put(moody(V[base + c], c) if c != stop // 2 else LOOK["tinv"](base + c, 0, False))
    cue("whip_stop")
    burst = ["sketch", "glow", "tinv", "sspec", "points", "thermal", "paper", "outline", "bars", "film", "xray", "sdark", "orange", "sbar", "glow", "sketch"]
    for k, nm in enumerate(burst):
        j = min(len(Fw) - 1, stop + k); put(LOOK[nm](base + j, k, True))
        if k % 3 == 0: put(LOOK[nm](base + j, k, True))
    for c in range(stop + len(burst), len(Fw)): put(moody(V[base + c], c))
# 5. a few close-ups with the data pages (poster PLZ COLLAB NIKE. somewhere in the middle)
tr_streak()
cue("torso"); cu("C3-lowtorso", .5, 1.7, 0.9, ["srec", "shud"], obj=False)
cue("poster"); pc = f"{D}/clips/CU-C7-profile.mp4"
if os.path.exists(pc):
    Fc = load(pc); base = len(V); V.extend(Fc); tr_rgb(frames[-1], moody(Fc[int(.5 * FPS)], 0))
    for c, j in enumerate(range(base + int(.5 * FPS), base + int(1.9 * FPS))): put(zoom(P_roto(j, c, False), 1 + .02 * c / 33))
cue("heel"); cu("C6b-heel", .6, 1.5, 1.0, ["sspec"])
# 6. FINAL: bullet-time callback (his face), water hits the lens (4th wall), end card: shoe turntable + strands + Swoosh + JUST DO IT.
DROPS = []
drng = np.random.default_rng(21)
for _ in range(70):
    DROPS.append((drng.uniform(0, W), drng.uniform(0, H), drng.uniform(30, 140), drng.uniform(.6, 1.0), drng.integers(0, 10)))
def lens_splash(img, k):
    """Water drops landing on the lens: each drop refracts a magnified, flipped patch of the frame behind it."""
    o = img.astype(np.float32).copy()
    for (x, y, r, sq, t0) in DROPS:
        if k < t0: continue
        rr = r * min(1, (k - t0 + 1) / 2); x0, x1 = int(max(0, x - rr)), int(min(W, x + rr)); y0, y1 = int(max(0, y - rr * sq)), int(min(H, y + rr * sq))
        if x1 - x0 < 4 or y1 - y0 < 4: continue
        yy, xx = np.mgrid[y0:y1, x0:x1]; d = ((xx - x) / rr) ** 2 + ((yy - y) / (rr * sq)) ** 2; inside = d < 1
        sx = np.clip(x - (xx - x) * .6, 0, W - 1).astype(int); sy = np.clip(y - (yy - y) * .6, 0, H - 1).astype(int)
        patch = img[sy, sx].astype(np.float32) * 1.1
        hl = np.exp(-(((xx - x + rr * .35) / (rr * .25)) ** 2 + ((yy - y + rr * .35 * sq) / (rr * .2)) ** 2))[..., None] * 160
        edge = np.clip(1 - np.abs(d - .92) * 12, 0, 1)[..., None] * -60
        reg = o[y0:y1, x0:x1]; reg[inside] = (patch + hl + edge)[inside]
    return np.clip(o, 0, 255).astype(np.uint8)
# final: run through the puddle, splash, then the end card
DROPS = [(W * .3, H * .35, 120, .9, 0), (W * .66, H * .25, 95, .85, 2), (W * .52, H * .62, 75, .9, 4)]
def drops_run(img, k):
    o = img
    for (x, y, r, sq, t0) in DROPS:
        if k < t0: continue
        dy = max(0, k - t0 - 4) ** 1.6 * 6                      # hangs, then slides down faster
        o = one_drop(o, x, y + dy, r * min(1, (k - t0 + 1) / 2), sq, trail=dy)
    return o
def one_drop(img, x, y, rr, sq, trail=0):
    o = img.astype(np.float32).copy()
    if trail > 0:                                                  # wet streak above the drop
        x0 = int(max(0, x - rr * .25)); x1 = int(min(W, x + rr * .25)); ya = int(max(0, y - trail)); yb = int(min(H, y))
        if yb > ya: o[ya:yb, x0:x1] = cv2.GaussianBlur(img[ya:yb, x0:x1], (0, 0), 3).astype(np.float32) * 1.08
    x0, x1, y0, y1 = int(max(0, x - rr)), int(min(W, x + rr)), int(max(0, y - rr * sq)), int(min(H, y + rr * sq))
    if x1 - x0 < 4 or y1 - y0 < 4: return np.clip(o, 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[y0:y1, x0:x1]; d = ((xx - x) / rr) ** 2 + ((yy - y) / (rr * sq)) ** 2; inside = d < 1
    sx = np.clip(x - (xx - x) * .6, 0, W - 1).astype(int); sy = np.clip(y - (yy - y) * .6, 0, H - 1).astype(int)
    patch = img[sy, sx].astype(np.float32) * 1.1
    hl = np.exp(-(((xx - x + rr * .35) / (rr * .22)) ** 2 + ((yy - y + rr * .35 * sq) / (rr * .18)) ** 2))[..., None] * 170
    edge = np.clip(1 - np.abs(d - .92) * 12, 0, 1)[..., None] * -70
    reg = o[y0:y1, x0:x1]; reg[inside] = (patch + hl + edge)[inside]
    return np.clip(o, 0, 255).astype(np.uint8)
lens_splash = lambda img, k: drops_run(img, k)
ps = f"{D}/clips/matchcut-15s-480p.mp4"
Fs = load(ps); a0, hit, a1 = int(4.75 * FPS), int(5.3 * FPS), int(6.0 * FPS)
tr_streak()
for i in range(a0, hit): put(grade(Fs[i]))
base = len(V); V.extend(Fs[hit:a1])                                 # splash impact: flash burst on the moving frames
cue("splash"); put(grade(Fs[hit]))
for k, nm in enumerate(["tinv", "glow", "xray", "sketch", "thermal", "sspec", "orange", "outline", "tdark", "burst"]):
    j = base + 1 + k
    if j < len(V): put(LOOK[nm](j, k, True))
pe = f"{D}/edit/endcard.mp4"
if os.path.exists(pe):
    E = load(pe); cue("endcard")
    for k in range(8): put(cv2.addWeighted(frames[-1], 1 - (k + 1) / 8, E[0], (k + 1) / 8, 0))   # through the wet lens
    for f in E: put(f)
    for _ in range(18): put(E[-1])
import json; json.dump(CUES, open(f"{D}/edit/cues-v16.json", "w"), indent=1); print(CUES)
for n_, f in enumerate(frames): cv2.imwrite(f"{tmp}/{n_:04d}.png", f)
out = f"{D}/edit/cut-v16-1080p.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", f"{tmp}/%04d.png", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out], check=True)
print("wrote", out, len(frames), "frames", round(len(frames) / FPS, 2), "s")
