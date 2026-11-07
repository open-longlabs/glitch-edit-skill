# Which generator for what (Higgsfield, Nov 2026)

| Need | Model | Notes |
|---|---|---|
| Exact first frame (continue from a composite) | Gemini Omni Flash 1.2 image-to-video | 360p about 0.75 cr/s; 3-15 s single frame, 3-10 s with first/last |
| First + last frame, camera moves, strongest stylised transformation | Seedance 2.5 omni_reference with start+end | 480p about 3 cr/s; ignores start_image without end_image |
| Many refs (up to 9) in one multi-shot | Seedance 2.5 omni_reference | identity drifts in small/wide shots |
| Stills, product truth | any image model that takes reference images | one job at a time, max 5 refs, always say LANDSCAPE 1536x1024 |
| Real 3D mesh from one image | generate_3d image_to_3d | 20 cr (30 textured), not used yet |
Upload: media_upload -> curl PUT (retry on 520) -> media_confirm.
- Topaz upscale 360/480p -> 1080p: fine for close-ups/macros, NOT for wide shots with people (blurry, fragmented). Wide shots need native 1080p generation.
