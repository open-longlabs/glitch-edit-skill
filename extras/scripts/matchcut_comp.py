#!/usr/bin/env python3
"""1:1 match cut frame: cut the man out of the living room apex still and put the forest set behind him.
His pixels, position and size stay identical; only the background changes. The shades swap is done by the
forest clip, never here (the cut must be pixel-identical on the body).

matchcut_comp.py <apex.png> <forest_plate.png> <out.png>
"""
import sys
import numpy as np, cv2
from PIL import Image
from rembg import remove, new_session

src, plate, out = sys.argv[1:4]
a = cv2.imread(src); h, w = a.shape[:2]
b = cv2.imread(plate)
s = max(w / b.shape[1], h / b.shape[0]); b = cv2.resize(b, (int(b.shape[1] * s + .5), int(b.shape[0] * s + .5)))
y0, x0 = (b.shape[0] - h) // 2, (b.shape[1] - w) // 2; b = b[y0:y0 + h, x0:x0 + w]
m = np.array(remove(Image.fromarray(a[:, :, ::-1]), session=new_session("u2net_human_seg")))[:, :, 3].astype(np.float32) / 255
m = cv2.erode(m, np.ones((5, 5)))                       # drop the light halo of the old room
m = cv2.GaussianBlur(m, (0, 0), 1.0)[:, :, None]
# relight him for the forest (position untouched): darker, colder, more contrast, cool rim from the light shafts
fg = a.astype(np.float32) / 255
fg = np.clip((fg - .5) * 1.15 + .5, 0, 1) ** 1.35 * .78
fg = fg * np.array([1.12, 1.0, .86])                     # BGR: cooler
rim = np.clip(cv2.GaussianBlur(m[:, :, 0], (0, 0), 3) - cv2.GaussianBlur(m[:, :, 0], (0, 0), 9), 0, 1)
dx = cv2.Sobel(cv2.GaussianBlur(m[:, :, 0], (0, 0), 4), cv2.CV_32F, 1, 0)
fg = fg + (rim * (dx < 0) * .35)[:, :, None] * np.array([1.0, .85, .7])  # light from the upper left
fg = np.clip(fg, 0, 1) * 255
# contact shadow: a soft dark ellipse on the ground below him
bb = b.astype(np.float32); ys, xs = np.nonzero(m[:, :, 0] > .5)
cx, by = xs.mean(), ys.max(); sh = np.zeros((h, w), np.float32)
cv2.ellipse(sh, (int(cx), int(min(h - 40, by + h * .22))), (int(w * .14), int(h * .03)), 0, 0, 360, 1, -1)
bb *= 1 - .45 * cv2.GaussianBlur(sh, (0, 0), 25)[:, :, None]
c = fg * m + bb * (1 - m)
cv2.imwrite(out, np.clip(c, 0, 255).astype(np.uint8))
cv2.imwrite(out.replace(".png", "-mask.png"), (m[:, :, 0] * 255).astype(np.uint8))
print("wrote", out)
