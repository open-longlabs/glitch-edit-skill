# 1:1 match cut

- Make the apex as a still first (Codex). Then `../scripts/matchcut_comp.py apex.png world_plate.png out.png`: cuts the person out, relights darker/colder with a rim, contact shadow, pastes onto the new world. Same pixels, same spot.
- Clip A ends on the apex (end_image), clip B starts on the composite (start_image). The cut lands on identical frames.
- Only the world (and a planned prop, e.g. glasses) changes. Fix product errors on the apex BEFORE compositing (both clips inherit it).
- Seedance omni_reference treats start_image as a loose ref without an end_image (clip B opened in a different pose). For an exact first frame use Gemini Omni Flash image-to-video.
