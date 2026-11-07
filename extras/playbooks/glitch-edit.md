# Glitch edit (looks on the moving frames)

Ref: a 3.5 s, 30 fps reference glitch edit, analysed frame by frame.
- One continuous shot carries it. Each look = treatment of the CURRENT frame, 2 to 3 frames each, about 10 looks in 0.8 s, no white frames between.
- Arc: raw B&W grainy hook, burst starts on an impact frame (push-off, foot strike), peak = 1-frame light burst, then paper/film looks, then hard switch to graded colour (the payoff).
- Shoe-only bursts (nomadatoast ref 6-8 s): 16 looks at 1 to 2 frames on the product while it holds still.
- Spread single 1-2 frame hits through the whole film (landing, mid-whip, final), not only at the start.
Script: `../reference/spec-ad-cut.py` (LOOK dict: tinv, tdark, thermal, bars, points, orange, burst, paper, film, outline, sketch, glow, sspec, sdark, sbar, srec, shud, track=pencil, xray=blueprint, roto). Masks: rembg u2net_human_seg (people), u2net (objects); depth: Depth-Anything-V2-Small; pose: yolov8n-pose.
Failed: v9 dropped static looks from another clip in for 0.4 s with white frames (it looked cheap).
