---
name: glitch-edit
description: Turn any clip into a glitch edit - designed looks flashed 2 to 3 frames on the moving frames, raw black and white hook, colour payoff, cue list for sound. Also the recipe for a full spec ad (bullet time, 1:1 match cut, whip to the product, end card). Use when asked for a glitch edit, a flash cut, a hype edit, a spec ad, "make it look like those viral edits", or to apply the glitch-edit skill to a clip.
---

# Glitch edit

**This file is the whole skill.** Give it to Claude, Codex or any agent that can run Python, together with a clip, and say:
"apply the glitch edit to my_clip.mp4". Claude Code: save it as `~/.claude/skills/glitch-edit/SKILL.md`. Codex: `~/.codex/skills/glitch-edit/SKILL.md`. Claude app: upload it as a skill. Any other chat: paste it in.

## Agent: do this when someone hands you a clip
1. Save the script at the end of this file as `glitch_edit.py` next to the clip.
2. Check the tools: `python3 -c "import cv2, numpy, PIL, rembg"` and `ffmpeg -version`. If something is missing: `pip install opencv-python numpy pillow "rembg[cpu]"` and install ffmpeg (`brew install ffmpeg` / `apt install ffmpeg` / winget). The first run downloads a person-segmentation model (about 170 MB).
3. Run `python3 glitch_edit.py my_clip.mp4 my_clip_glitch.mp4 --bursts 3` (`--music track.mp3` replaces the audio, `--looks 7` sets looks per burst).
4. Show the result and `my_clip_glitch-cues.json` (burst times, where colour starts). Ask what to change: more or fewer bursts, other moments, where the colour payoff lands. Re-run with the change.
5. Clip without a person in it: in the script, change `u2net_human_seg` to `u2net` (objects).

## What makes it look right
- One continuous moving shot carries it. Every look is a treatment of the CURRENT frame (same pose, motion never stops), 2 to 3 frames per look, about 7 to 10 looks in under a second. Never a still from another shot, never white frames in between (that looks cheap).
- Arc: raw grainy black and white hook, the first burst starts on an impact frame (push-off, foot strike, landing), then a hard switch to full colour = the payoff.
- Spread short 1 to 2 frame hits through the whole video (landing, mid-move, the end), not only at the start.
- Raw computer-vision output (skeleton, mask, depth map) is never shown as is. Turn it into design: pencil construction sketch, blueprint x-ray, toner print, thermal scan, point field.
- Text pages flashed in (receipt, spec label) carry only true facts plus one joke line for people who pause.
- Sound: put an SFX on every burst and cut from the cue list, duck the music under them, master to -14 LUFS for Instagram/YouTube, -16 for LinkedIn.
- 9:16: keep text and key action inside x 60-940, y 200-1480 of 1080x1920 (the app UI covers the rest). The profile grid crops covers to 3:4, the middle 1080x1440.

## Full spec ad (generated shots), in this order
1. **Storyboard stills first** for every key frame (start and end frames). A human approves them. No video before that.
2. **Product truth:** check every still with a real product against real photos (logo position, counts, which side faces the camera, nothing invented). Videos inherit every still error.
3. **Bullet time:** video model with BOTH a start image (loaded stance) and an end image (exact apex pose). Spell out the timeline in seconds and the camera path step by step, or the camera will not move. Prompt pattern: "0-1.2 s: fixed wide shot, he explodes into a forward leap. 1.2 s: time freezes, he hangs still. 1.2-5 s: the camera travels a full 360 degree circle around him at chest height: to his front, behind his back, past his other side, back to the front; the room dissolves into black space, his body becomes a glowing point cloud of white dots, only the shoes keep their colour. 5-6 s: the camera arrives at the original front view, the room returns, last frame identical to the end frame. He never looks at the camera." Test at the lowest resolution.
4. **1:1 match cut:** make the apex as a still, cut the person out (rembg) and paste the same pixels onto the new world, relit with a rim light and a contact shadow. Clip A ends on the apex (end image), clip B starts on the composite (start image). Only the world changes. If your model treats the start image loosely, use one that holds the exact first frame.
5. **Whip to the product:** start frame = extreme macro of an eye or sunglasses, end frame = product macro, 3 s. Timeline: 0-0.4 s light streak across the lens; 0.4-1.0 s violent downward whip tilt with motion blur; 1.0-1.3 s hard stop on the product with a tiny shake; 1.3-3 s hero hold, slow push-in. In post: one look mid-whip, a 16-look burst on the product right after the stop.
6. **The cut:** run the script on each shot, or build one timeline with the same looks.
7. **End card:** product alone, darkened with a coloured rim, light strands behind it, a small logo you draw yourself from curves (never paste brand files), one line. Spec work: "unofficial spec ad, not affiliated with <brand>."
8. **Resolution:** upscaling is fine for close-ups; wide shots with people need a native 1080p generation. People never look at the camera; check the face in every generated wide shot, it drifts to a different person.

Made by @long.integer (Instagram, new skills there). Glitch-edit style inspired by @nomadatoast. More recipes, my scripts and the full reference cut: the `extras/` folder in https://github.com/open-longlabs/glitch-edit-skill

## The script (save as glitch_edit.py)
```python
#!/usr/bin/env python3
"""Apply the glitch edit to any clip.

    python3 glitch_edit.py <in.mp4> <out.mp4> [--bursts 3] [--looks 7] [--music track.mp3]

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
```
