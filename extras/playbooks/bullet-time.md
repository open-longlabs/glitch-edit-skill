# Bullet time from the generator

- Start frame = stance, end frame = apex. Prompt the timeline in seconds and the camera path step by step: "the camera physically travels a full 360 degree circle around the frozen man at chest height: front, behind his back with the soles visible, his other side, back to the front view". Without the explicit path the camera does not move.
- Transformation in the void: "the room dissolves into black space, his body becomes a glowing 3D point cloud of fine white dots, only the purple shoes keep their colour, waves of dots below". Seedance gave a hologram/silver outline look, Gemini stays photoreal with ghosting on return.
- The end_image pins the last frame for the next cut. The model may arrive early and hold the end frame: cut when the pose is reached.
- Reuse: the orbit section works again as an ending callback (his real face, no identity drift).
- Local depth point-cloud orbit (`../scripts/orbit_dots.py`) is previs only: single view goes paper-flat past 40 degrees.
