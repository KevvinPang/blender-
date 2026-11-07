# Inference Sources

These selected files were adapted from Comfy-Org/ComfyUI, commit
`ba994879b88d0ebd9950b1e5c1e923d928f216ba`:
https://github.com/Comfy-Org/ComfyUI/tree/ba994879b88d0ebd9950b1e5c1e923d928f216ba

Original paths: `comfy/ldm/sam3d_body/{model,mhr}/*.py`,
`comfy/ldm/sam3d_body/utils.py`, and `comfy/image_encoders/dino3.py`.
Upstream license is preserved as COMFY_LICENSE (GPL-3.0).
Original Meta/MHR attributions remain in the source files.

Local modifications flatten imports, replace framework-specific helpers with
plain PyTorch equivalents, and omit the separate MediaPipe face task. No
ComfyUI application, node system, or server is installed or imported.

Model weights retain their separate SAM license:
https://huggingface.co/Comfy-Org/sam-3d-body
This adapter does not confer access to any gated upstream repository.
