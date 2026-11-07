#!/usr/bin/env python3
"""Music + SFX mix on the cue list. mix_v16.py <test-jump dir> <music.mp3> <offset s> <out_prefix>"""
import sys, json, subprocess
D, MUS, OFF, OUT = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
C = json.load(open(f"{D}/edit/cues-v16.json")); DUR = 19.25
# (file, time s, gain dB)
S = [("room", 0.0, -14), ("takeoff", 0.40, -4), ("glitch", 0.50, -12), ("freeze", C["freeze"], -2), ("orbit", C["orbit"], -8),
     ("reverse", C["room_back"] - 0.6, -6), ("matchcut", C["matchcut"], -3), ("landing", C["matchcut"] + 0.35, -2),
     ("whip", C["whip"] + 0.35, -3), ("thock", C["whip_stop"] - 0.05, -2), ("glitch", C["whip_stop"] + 0.05, -10), ("hum", C["whip_stop"] + 0.7, -10),
     ("breath", C["torso"] + 0.2, -15), ("breath2", C["poster"] + 0.1, -8), ("glitch", C["poster"], -12), ("splash", C["splash"] - 0.1, 0), ("glitch", C["splash"] + 0.05, -9),
     ("endhit", C["endcard"] + 0.85, -2)]
inp = ["-ss", str(OFF), "-t", str(DUR), "-i", MUS]
for f, _, _ in S: inp += ["-i", f"{D}/sfx/{f}.mp3"]
fc = [f"[0]lowpass=f=700:enable='between(t,{C['freeze']},{C['matchcut']})',volume=0.45:enable='between(t,{C['freeze']},{C['matchcut']})',afade=t=in:d=0.15,afade=t=out:st={DUR - 1.4}:d=1.4,aformat=sample_rates=48000:channel_layouts=stereo[mus]"]
for k, (f, t, g) in enumerate(S):
    fc.append(f"[{k + 1}]aformat=sample_rates=48000:channel_layouts=stereo,volume={g}dB,adelay={int(t * 1000)}:all=1[s{k}]")
fc.append("".join(f"[s{k}]" for k in range(len(S))) + f"amix=inputs={len(S)}:normalize=0,apad,atrim=0:{DUR}[sfx]")
fc.append("[sfx]asplit=2[sfxa][sfxb]")
fc.append("[mus][sfxa]sidechaincompress=threshold=0.05:ratio=4:attack=5:release=250[duck]")
fc.append("[duck][sfxb]amix=inputs=2:normalize=0[mix]")
for name, I, tp in [("IG-YT", -14, -1), ("LinkedIn", -16, -1.5)]:
    lim = 0.89 if name == "IG-YT" else 0.84
    g = fc + [f"[mix]loudnorm=I={I}:TP={tp}:LRA=11,volume=2.8dB,alimiter=limit={lim}:attack=3:release=60:level=0[out]"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inp, "-filter_complex", ";".join(g), "-map", "[out]", "-ar", "48000", f"/tmp/mix-{name}.wav"], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{D}/edit/cut-v16-1080p-wm.mp4", "-i", f"/tmp/mix-{name}.wav", "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", f"{OUT}-{name}.mp4"], check=True)
    print("wrote", f"{OUT}-{name}.mp4")
