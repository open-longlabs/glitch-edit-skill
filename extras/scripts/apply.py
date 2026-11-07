#!/usr/bin/env python3
"""Apply the glitch edit to any clip.

    python3 apply.py <in.mp4> <out.mp4> [--bursts 3] [--looks 7] [--music track.mp3]

What it does (the core of the skill, no project paths):
1. finds impact moments (motion peaks: foot strikes, landings, cuts in action) and spreads up to N bursts over them
2. raw grainy black and white until the first burst, full colour after (the payoff)
3. on each burst, a different designed look on each of the next 2 to 3 MOVING frames (toner, dark receipt,
   thermal echo, orange duotone, outline, blueprint x-ray, point field), no white frames, motion never stops
4. writes a cue list (cues.json next to the output) for sound design, muxes the original audio or --music
Needs: opencv-python, numpy, pillow, rembg, ffmpeg.
"""
import os, json, argparse, subprocess, tempfile
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from rembg import remove, new_session

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--bursts", type=int, default=3); ap.add_argument("--looks", type=int, default=7); ap.add_argument("--music")
A = ap.parse_args()
cap = cv2.VideoCapture(A.src); FPS = cap.get(cv2.CAP_PROP_FPS) or 24; V = []
while True:
    ok, f = cap.read()
    if not ok: break
    V.append(f)
H0, W0 = V[0].shape[:2]; W, H = (1920, 1080) if W0 >= H0 else (1080, 1920)
V = [cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC) for f in V]; N = len(V)
rng = np.random.default_rng(3); SEG = new_session("u2net_human_seg"); cache = {}
def mask(i):
    if i not in cache:
        sm = cv2.resize(V[i], (W // 2, H // 2)); a = np.array(remove(Image.fromarray(sm[:, :, ::-1]), session=SEG))[:, :, 3]
        cache[i] = cv2.resize(a, (W, H)) > 128
    return cache[i]
gray = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
g3 = lambda g: cv2.merge([g, g, g]).astype(np.uint8)
GR = [rng.normal(0, 1, (H, W)).astype(np.float32) for _ in range(4)]
def cont(g, k=1.6, mid=None): return np.clip((g - (np.median(g) if mid is None else mid)) * k + 128, 0, 255)
MONO = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 22) if os.path.exists("/System/Library/Fonts/Menlo.ttc") else ImageFont.load_default()
def small_print(img, lines, col):
    p = Image.fromarray(img[:, :, ::-1].copy()); d = ImageDraw.Draw(p)
    for k, s in enumerate(lines): d.text((40, 40 + 30 * k), s, font=MONO, fill=col)
    for k in range(60):
        if rng.random() > .35: d.rectangle((40 + 5 * k, H - 110, 41 + 5 * k + int(rng.integers(0, 3)), H - 70), fill=col)
    return np.array(p)[:, :, ::-1].copy()

# ---------- looks: each takes a frame index and works on that moving frame ----------
def toner(i, n):
    m = mask(i); g = gray(V[i]); hp = g - cv2.GaussianBlur(g, (0, 0), 5)
    body = np.clip(205 + hp * 5 + GR[n % 4] * 30, 0, 255); o = np.zeros((H, W), np.float32); o[m] = body[m]
    return small_print(g3(o), ["ITEM 01", "FRAME %04d" % i, "TOTAL  $0.00", "THANK YOU"], (230, 230, 230))
def receipt_dark(i, n):
    m = mask(i); g = cont(gray(V[i]), 1.6, 150) * .8 + GR[n % 4] * 14; o = np.full((H, W), 12, np.float32); o[m] = np.clip(g, 0, 255)[m]
    return small_print(g3(o), ["TERMINAL 003", "SEQ %04d" % i], (220, 220, 220))
def thermal(i, n):
    m = mask(i); g = gray(V[i]); o = np.full((H, W, 3), 150, np.uint8)
    th = cv2.applyColorMap(cv2.normalize(cv2.GaussianBlur(g, (0, 0), 3), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
    M = np.float32([[1, 0, -int(W * .12)], [0, 1, 0]]); ths = cv2.warpAffine(th, M, (W, H)); ms = cv2.warpAffine(m.astype(np.uint8), M, (W, H)) > 0
    o[ms] = ths[ms]; bw = g3(cont(g, 1.5) * .55); o[m] = bw[m]; return o
def duotone(i, n):
    m = mask(i); g = cont(gray(V[i]), 1.4) + GR[n % 4] * 18; g[m] *= .45; g[~m] = 150 + g[~m] * .4
    t = (np.clip(g, 0, 255) / 255)[:, :, None]; return (np.array([20, 40, 80]) * (1 - t) + np.array([60, 150, 245]) * t).astype(np.uint8)
def outline(i, n):
    m = mask(i).astype(np.uint8) * 255; e = cv2.dilate(cv2.Canny(m, 50, 150), np.ones((3, 3)))
    g = cv2.Canny(cv2.GaussianBlur(gray(V[i]).astype(np.uint8), (5, 5), 0), 40, 110) * (m > 0)
    o = np.zeros((H, W), np.uint8); o[e > 0] = 245; o[g > 0] = 160; return g3(o.astype(np.float32))
def xray(i, n):
    m = mask(i); g = gray(V[i]); hp = np.abs(cv2.GaussianBlur(g - cv2.GaussianBlur(g, (0, 0), 8), (0, 0), 1))
    base = np.zeros((H, W, 3), np.float32); base[:] = (95, 45, 15); body = np.zeros((H, W), np.float32); body[m] = np.clip(.35 + hp / 30, 0, 1)[m] * .7
    edge = cv2.GaussianBlur(cv2.Canny(m.astype(np.uint8) * 255, 50, 150).astype(np.float32) / 255, (0, 0), 3) * 2.5
    o = base + (body + edge * m)[:, :, None] * np.array([255, 235, 205]); return np.clip(o + cv2.GaussianBlur(o, (0, 0), 20) * .15, 0, 255).astype(np.uint8)
def points(i, n):
    m = mask(i); g = gray(V[i]) / 255; o = np.zeros((H, W, 3), np.uint8)
    ys, xs = np.nonzero(m[::4, ::4]); keep = rng.random(len(xs)) < (.25 + .75 * g[ys * 4, xs * 4])
    for x, y in zip(xs[keep] * 4, ys[keep] * 4): o[y:y + 2, x:x + 2] = 235
    return o
LOOKS = [toner, receipt_dark, thermal, duotone, outline, xray, points]

# ---------- impacts: local maxima of frame difference, spread over the clip ----------
diff = np.array([0] + [np.abs(cv2.resize(V[i], (160, 90)).astype(int) - cv2.resize(V[i - 1], (160, 90)).astype(int)).mean() for i in range(1, N)])
sm = np.convolve(diff, np.ones(5) / 5, "same"); cand = [i for i in range(int(FPS * .6), N - 30) if sm[i] == sm[max(0, i - 12):i + 13].max()]
cand.sort(key=lambda i: -sm[i]); bursts = []
for i in cand:
    if all(abs(i - j) > FPS * 1.5 for j in bursts): bursts.append(i)
    if len(bursts) == A.bursts: break
bursts.sort(); print("bursts at", [round(b / FPS, 2) for b in bursts])

# ---------- render ----------
plan = {}
for b, start in enumerate(bursts):
    k = start; order = [LOOKS[(b * 3 + j) % len(LOOKS)] for j in range(A.looks)]
    for j, lk in enumerate(order):
        for _ in range(3 if j % 3 == 0 else 2):
            if k < N: plan[k] = (lk, j); k += 1
first = bursts[0] if bursts else N
tmp = tempfile.mkdtemp()
for i in range(N):
    if i in plan: img = plan[i][0](i, plan[i][1])
    elif i < first: img = g3(np.clip(cont(gray(V[i]), 1.5, 140) * .8 + GR[i % 4] * 22, 0, 255))
    else: img = V[i]
    cv2.imwrite(f"{tmp}/{i:05d}.png", img)
silent = f"{tmp}/v.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", f"{tmp}/%05d.png", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", silent], check=True)
aud = A.music or A.src
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", aud, "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest", A.out], check=True)
json.dump({"fps": FPS, "bursts_s": [round(b / FPS, 3) for b in bursts], "colour_from_s": round(first / FPS, 3)}, open(os.path.splitext(A.out)[0] + "-cues.json", "w"), indent=1)
print("wrote", A.out)
