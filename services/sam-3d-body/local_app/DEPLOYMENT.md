# SAM 3D Body Local Evaluation

This installation uses the downloaded upstream source and a dedicated `.venv`.
The optional Blender bridge creates a separate Ellen SAM preview scene; the
original XR scene and XR Animator installation are preserved.

## Scope

- Single-person image or selected video-frame inference.
- Full inference: body decoder plus both hand refinements.
- Full-image person crop and upstream default camera FOV; no Detectron2,
  segmentation, or MoGe camera calibration in this first evaluation.
- 2D overlay, rotatable 3D keypoint skeleton, and original numeric NPZ output.
- The Blender video bridge provides offline video extraction, light quaternion
  smoothing, and rotation-only Ellen upper-body retargeting. Legs and placement
  remain fixed. It does not provide semantic hand-contact correction, facial
  animation, real-time capture, or VMD/FBX export.
- Videos and images stay local. The public BF16 backend runs offline after
  installation. Only the optional official download action contacts Hugging Face.
- The web app only listens on 127.0.0.1 and does not create a public share link.

## Model Access

### Active Local Public Backend (2026-10-15)

The public Comfy-Org repack is now supported without installing ComfyUI.
Source: https://huggingface.co/Comfy-Org/sam-3d-body
Weight location (keep this file in place):
`C:/Users/24995/Desktop/ssam/sam_3d_body_dinov3_bf16.safetensors`

- SHA256: `59fa45200c504c5b56625004d7d3385daf48c616613e88099e43bf83b3e249cf`
- 1206 body/hand/rig tensors load strictly; no missing or unexpected keys.
- 421 separate MediaPipe face-task tensors and 2 unused classifier tensors are
  deliberately omitted. Facial motion capture is not provided by this adapter.
- BF16 backbone, float32 pose heads and rig. This is not a claim of bit-identical
  predictions to the original Meta checkpoint/runtime.
- The INT8 file also matched its published SHA256, but is not used by this loader:
  `ba8d4c34a81bc072b3f53db5241074d370b26e3b72ae793630be5344efd28934`.
- No Hugging Face token or access to the gated Meta repository is used.
- License remains the SAM license linked from the public repository's README;
  public downloading does not remove its usage conditions.
- Selected adapted inference source, upstream commit and license: `vendor/NOTICE.md`.
- Low VRAM automatically enables CPU/GPU backbone-block streaming. To force
  this slower mode, set `SAM3D_LOW_MEMORY=1` before starting the app.

Validated with the included sample and frame 60 (2 seconds) of the user's
`35e4838c2662f4528dfe5be813dcc875.mp4`: full body and hand refinement returned
finite 70-keypoint, 127-joint and 18439-vertex predictions. These checks establish
working inference, not motion-capture accuracy or Blender retargeting quality.

### Optional Original Meta Backend

The following instructions are only needed for the original checkpoint. They
are not required for the currently installed public BF16 backend.

1. Sign in at https://huggingface.co/facebook/sam-3d-body-dinov3
   and request access under the official model license.
2. After approval, create a read token at https://huggingface.co/settings/tokens
   with access to this gated repository.
3. Run the desktop model login shortcut. Paste the token into its hidden prompt.
   Do not paste tokens into chat. The Hugging Face client stores login locally.
4. This downloads the model to `checkpoints/sam-3d-body-dinov3`.
   The required files are `model.ckpt`, `model_config.yaml`, and
   `assets/mhr_model.pt`. Downloading a source ZIP alone does not include these.
5. Alternatively, put these authorized files into that directory manually.

## Start

Use the SAM 3D Body desktop shortcut, or run `local_app/launch.ps1`.
Default URL: http://127.0.0.1:7872 (the launcher selects another port if occupied).

In the first tab, select a video and a timestamp, extract the frame, and run
inference. An image can also be uploaded directly. Keep only one person in frame.
The model is loaded on demand; launching the UI alone does not occupy its GPU VRAM.

Use the environment tab to verify local weights, refresh status, or unload the model.
Close other GPU-heavy applications if inference runs out of memory.

## Diagnostics

Run `.venv/Scripts/python.exe local_app/app.py --check`.
This checks imports and a CUDA computation; it does NOT substitute for an actual
model inference. Missing weights are reported separately from package errors.

Model licensing: https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE
Original documentation: ../README.md and ../INSTALL.md
