#!/usr/bin/env python3
"""Moving watermark overlay (DVD-logo bounce): low opacity, drifts around the frame, bounces off inset edges,
x and y speeds chosen so it practically never meets a corner.  watermark.py <in.mp4> <out.mp4> [text]"""
import sys, subprocess, tempfile, glob
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
src, out = sys.argv[1], sys.argv[2]; text = sys.argv[3] if len(sys.argv) > 3 else "@yourhandle"
cap = cv2.VideoCapture(src); W, H = int(cap.get(3)), int(cap.get(4)); fps = cap.get(5) or 24
def font(path, size, index=0):
    """macOS font if present, else DejaVu, else Pillow's default (so it runs on Linux and in sandboxes)."""
    for p, i in ((path, index), ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0)):
        try: return ImageFont.truetype(p, size, index=i)
        except OSError: pass
    return ImageFont.load_default(size)
f = font("/System/Library/Fonts/Menlo.ttc", int(H * .024))
tw = int(ImageDraw.Draw(Image.new("L", (1, 1))).textlength(text, font=f)) + 4; th = int(H * .034)
mark = Image.new("L", (tw, th), 0); ImageDraw.Draw(mark).text((2, 2), text, font=f, fill=255); mark = np.array(mark).astype(np.float32) / 255
mx, my = int(W * .04), int(H * .05); rx, ry = W - 2 * mx - tw, H - 2 * my - th
def tri(t, span): p = t % (2 * span); return p if p < span else 2 * span - p
tmp = tempfile.mkdtemp(); n = 0; vx, vy = 97.0, 61.0                 # px/s, ratio irrational-ish so corners are avoided
while True:
    ok, fr = cap.read()
    if not ok: break
    t = n / fps; x = mx + int(tri(t * vx + rx * .3, rx)); y = my + int(tri(t * vy + ry * .7, ry))
    reg = fr[y:y + th, x:x + tw].astype(np.float32); a = (mark * .22)[:, :, None]
    fr[y:y + th, x:x + tw] = (reg * (1 - a) + 255 * a).astype(np.uint8)
    cv2.imwrite(f"{tmp}/{n:05d}.png", fr); n += 1
cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", f"{tmp}/%05d.png", "-i", src, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", out]
subprocess.run(cmd, check=True); print("wrote", out, n, "frames")
