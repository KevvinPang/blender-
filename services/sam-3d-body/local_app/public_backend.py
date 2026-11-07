"""Standalone body/hand inference for the public Comfy-Org BF16 checkpoint."""
from project_paths import SAM3D_WEIGHTS

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch import nn

WEIGHTS = SAM3D_WEIGHTS
WEIGHTS_SIZE = 2830737652
WEIGHTS_SHA256 = "59fa45200c504c5b56625004d7d3385daf48c616613e88099e43bf83b3e249cf"
REPO_ID = "Comfy-Org/sam-3d-body"


def weights_available():
    return WEIGHTS.is_file() and WEIGHTS.stat().st_size == WEIGHTS_SIZE


def verify_weights():
    digest = hashlib.sha256()
    with WEIGHTS.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHTS_SHA256:
        raise RuntimeError("BF16 checkpoint checksum mismatch; download the complete file again.")


def checkpoint_state():
    state = {}
    omitted = []
    with safe_open(str(WEIGHTS), framework="pt", device="cpu") as checkpoint:
        for original in checkpoint.keys():
            # The body/hand graph never calls the separate face task or classifier.
            if original.startswith("face_landmarker.") or original in (
                "hand_cls_embed.weight", "hand_cls_embed.bias"
            ):
                omitted.append(original)
                continue
            key = original.replace(".layers.0.0.", ".layers.0.")
            if key in state:
                raise RuntimeError(f"Duplicate normalized weight key: {key}")
            tensor = checkpoint.get_tensor(original)
            if tensor.is_floating_point() and not key.startswith("backbone."):
                tensor = tensor.float()
            state[key] = tensor
    return state, omitted


class _NoInit:
    def reset_parameters(self):
        pass


class _Ops:
    Linear = type("Linear", (_NoInit, nn.Linear), {})
    Conv2d = type("Conv2d", (_NoInit, nn.Conv2d), {})
    Embedding = type("Embedding", (_NoInit, nn.Embedding), {})
    LayerNorm = type("LayerNorm", (_NoInit, nn.LayerNorm), {})


def load_model():
    from vendor.model import SAM3DBody

    verify_weights()
    model = SAM3DBody(device="cpu", dtype=torch.bfloat16, operations=_Ops)
    state, omitted = checkpoint_state()
    # No random/absent body or hand parameters are allowed.
    model.load_state_dict(state, strict=True, assign=True)
    model.requires_grad_(False).eval()
    model.backbone_dtype = torch.bfloat16
    del state

    # Float32 heads/rig for stable geometry; stream BF16 backbone blocks when
    # other applications leave insufficient VRAM. This changes speed, not weights.
    free, _ = torch.cuda.mem_get_info()
    low_memory = os.environ.get("SAM3D_LOW_MEMORY") == "1" or free < 5 * 1024**3
    for name, child in model.named_children():
        if name != "backbone":
            child.to("cuda")
    if low_memory:
        backbone = model.backbone
        backbone.embeddings.to("cuda")
        backbone.rope_embeddings.to("cuda")
        backbone.norm.to("cuda")
        def before(module, inputs):
            module.to("cuda")
        def after(module, inputs, output):
            module.to("cpu")
        for block in backbone.layer:
            block.register_forward_pre_hook(before)
            block.register_forward_hook(after, always_call=True)
    else:
        model.backbone.to("cuda")
    model.load_report = {
        "checkpoint": str(WEIGHTS),
        "sha256": WEIGHTS_SHA256,
        "loaded_tensors": len(model.state_dict()),
        "omitted_face_task_tensors": len([k for k in omitted if k.startswith("face_landmarker.")]),
        "omitted_unused_classifier_tensors": len(omitted) - len([k for k in omitted if k.startswith("face_landmarker.")]),
        "strict_load": True,
        "backbone_dtype": "bfloat16",
        "heads_and_rig_dtype": "float32",
        "low_memory_block_streaming": low_memory,
    }
    return model


class PublicEstimator:
    def __init__(self):
        from sam_3d_body.data.transforms import Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper
        from torchvision.transforms import ToTensor

        self.model = load_model()
        self.transform = Compose([
            GetBBoxCenterScale(),
            TopdownAffine(input_size=(512, 512), use_udp=False),
            VisionTransformWrapper(ToTensor()),
        ])
        self.load_report = self.model.load_report

    @torch.inference_mode()
    def skeleton_reference(self):
        zeros = lambda n: torch.zeros(1, n, device="cuda")
        result = self.model.head_pose.mhr_forward(
            zeros(3), zeros(3), zeros(130), None, zeros(28), zeros(45), zeros(72),
            return_joint_coords=True, return_joint_rotations=True,
        )
        return {
            "positions_raw": result[1][0].cpu().numpy(),
            "rotations_raw": result[2][0].cpu().numpy(),
            "parents": self.model.mhr.skel_joint_parents.cpu().numpy(),
        }

    @torch.inference_mode()
    def process_one_image(self, image, inference_type="full"):
        from sam_3d_body.data.utils.prepare_batch import prepare_batch
        from sam_3d_body.utils import recursive_to

        height, width = image.shape[:2]
        boxes = np.array([[0, 0, width, height]], dtype=np.float32)
        batch = recursive_to(prepare_batch(image, self.transform, boxes), "cuda")
        result = self.model.run_inference(torch.as_tensor(image, device="cuda"), batch, inference_type=inference_type)
        output = result[0] if inference_type == "full" else result
        out = output["mhr"]
        names = {
            "global_rot": "global_rot", "body_pose_params": "body_pose",
            "hand_pose_params": "hand", "scale_params": "scale",
            "shape_params": "shape", "expr_params": "face",
            "pred_global_rots": "joint_global_rots",
        }
        for key in ("focal_length", "pred_keypoints_3d", "pred_keypoints_2d",
                    "pred_vertices", "pred_cam_t", "pred_pose_raw", "pred_joint_coords", "mhr_model_params"):
            names[key] = key
        person = {name: out[key][0].detach().float().cpu().numpy() for name, key in names.items()}
        person["bbox"] = boxes[0]
        if not all(np.isfinite(value).all() for value in person.values()):
            raise RuntimeError("Model prediction contains nonfinite values.")
        return [person]
