#!/usr/bin/env python3
"""Bullet-time dot orbit: the frozen apex becomes a 3D point cloud in black space, the camera swings around him
and lands back on the exact front view (so the next frame can be the 1:1 match cut).

orbit_dots.py <apex.png> <out.mp4> [frames=36] [max_angle=70]
Single view: depth gives the front half of the body, so the swing stays within +-max_angle (beyond that the back is empty).
"""
import sys, subprocess, tempfile
import numpy as np, cv2
from PIL import Image
from transformers import pipeline
from rembg import remove, new_session

src, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 36
AMAX = float(sys.argv[4]) if len(sys.argv) > 4 else 70
W, H = 1920, 1080
rng = np.random.default_rng(3)
img = cv2.imread(src); h0, w0 = img.shape[:2]
d = np.array(pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")(Image.fromarray(img[:, :, ::-1]))["depth"].resize((w0, h0))).astype(np.float32)
d = (d - d.min()) / (d.max() - d.min() + 1e-6)
m = np.array(remove(Image.fromarray(img[:, :, ::-1]), session=new_session("u2net_human_seg")))[:, :, 3] > 128

ys, xs = np.nonzero(m); pick = rng.random(len(xs)) < 0.22; xs, ys = xs[pick], ys[pick]
lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)[ys, xs] / 255
shoe = (img[ys, xs, 2].astype(int) - img[ys, xs, 1] > 25) & (img[ys, xs, 0].astype(int) - img[ys, xs, 1] > 25)  # purple = the shoes
z = 1 - d[ys, xs]
cx, cy = xs.mean(), ys.mean()
X = (xs - cx) / w0; Y = (ys - cy) / w0; Z = (z - z.mean()) * 0.55
# a thin back shell so a side view is not paper flat: mirror the depth behind the body, sparse
back = rng.random(len(X)) < 0.35
X = np.concatenate([X, X[back]]); Y = np.concatenate([Y, Y[back]]); Z = np.concatenate([Z, -Z[back] + 0.06])
lum = np.concatenate([lum, lum[back] * .5]); shoe = np.concatenate([shoe, shoe[back]])
# dot terrain flowing through the void below him
gx, gz = np.meshgrid(np.linspace(-1.2, 1.2, 140), np.linspace(-0.8, 0.8, 50))
gx, gz = gx.ravel(), gz.ravel()

def ease(t): return t * t * (3 - 2 * t)
tmp = tempfile.mkdtemp()
f = 1.25 * w0                       # focal so that angle 0 reproduces the still's scale
for k in range(N):
    t = k / (N - 1)
    a = np.radians(AMAX * np.sin(2 * np.pi * ease(t)))      # out to one side, through the front, out the other, back to 0
    def proj(x, y, zz):
        xr = x * np.cos(a) + zz * np.sin(a); zr = -x * np.sin(a) + zz * np.cos(a) + 1.25
        return xr / zr * f + W / 2, y / zr * f + H / 2 + (cy - h0 / 2) * W / w0 * .0, zr
    canvas = np.zeros((H, W, 3), np.float32)
    gy = 0.42 + 0.03 * np.sin(gx * 6 + k * .35) * np.cos(gz * 5)
    u, v, zr = proj(gx, gy, gz)
    for uu, vv, zz in zip(u.astype(int), v.astype(int), zr):
        if 0 <= uu < W and 0 <= vv < H: cv2.circle(canvas, (uu, vv), 1, (150, 150, 150), -1, cv2.LINE_AA)
    u, v, zr = proj(X, Y, Z); o = np.argsort(-zr)
    for i in o:
        uu, vv = int(u[i]), int(v[i])
        if 0 <= uu < W and 0 <= vv < H:
            c = (230, 120, 170) if shoe[i] else (60 + 195 * lum[i],) * 3
            canvas[vv:vv + 2, uu:uu + 2] = c
    canvas = cv2.add(canvas, cv2.GaussianBlur(canvas, (0, 0), 6) * .5)
    cv2.imwrite(f"{tmp}/{k:04d}.png", np.clip(canvas, 0, 255).astype(np.uint8))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", f"{tmp}/%04d.png", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out], check=True)
print("wrote", out, N, "frames")
