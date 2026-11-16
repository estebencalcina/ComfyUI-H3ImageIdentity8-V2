"""Experimental native-reference cache for MiniMax H3-Image ComfyUI.

NOT compatible with MiniMax H3 VIDEO refmods. This stores BOTH a VAE latent and
Qwen vision pixels, so that conditioning uses precisely the two streams expected
by the H3-Image pipeline. This is a full latent, not a learned identity embedding.
"""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.utils
import folder_paths
import node_helpers
from safetensors.torch import load_file, save_file
from safetensors import safe_open

_FORMAT = "h3-image-native-multiref-v2"
_DIR = Path(folder_paths.models_dir) / "h3_image_native_refmods"
_DIR.mkdir(parents=True, exist_ok=True)


def _options():
    files = sorted(x.name for x in _DIR.glob("*.safetensors") if x.is_file())
    return files or ["NO_IDENTITIES_YET.safetensors"]


def _resize(image, width, height):
    """Match the official H3-Image resize operation, RGB BHWC."""
    image = image[..., :3]
    if image.shape[2] == width and image.shape[1] == height:
        return image
    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height,
                                      "lanczos", "disabled").movedim(1, -1)


def _short_edge(width, height, target):
    scale = target / min(width, height)
    return (max(32, round(width * scale / 32) * 32),
            max(32, round(height * scale / 32) * 32))


def _safe_name(name):
    name = "".join(c if c.isascii() and (c.isalnum() or c in "-_") else "_" for c in name).strip("_")
    if not name:
        raise ValueError("Identity name is empty after sanitization")
    return name[:80]


class H3Img8V2Create:
    """Build a reusable multi-view identity from 1-8 image references.

    Each reference is retained separately; this is NOT a learned/merged embedding.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image_1": ("IMAGE",), "vae": ("VAE",),
            "identity_name": ("STRING", {"default": "Personaje_01"}),
            "reference_short_edge": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
            "overwrite_existing": ("BOOLEAN", {"default": False}),
        }, "optional": {f"image_{i}": ("IMAGE",) for i in range(2, 9)}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_file",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3-Image / Native RefMod (experimental)"

    def save(self, image_1, vae, identity_name, reference_short_edge=768, overwrite_existing=False, **other_images):
        images = [image_1] + [other_images.get(f"image_{i}") for i in range(2, 9)]
        images = [x for x in images if x is not None]
        name = _safe_name(identity_name)
        dest = _DIR / f"{name}.safetensors"
        if dest.exists() and not overwrite_existing:
            raise FileExistsError(f"Identity already exists: {dest}. Original preserved. Choose another name or enable overwrite_existing explicitly.")
        tensors, per_view = {}, []
        for j, image in enumerate(images):
            image = image[:1, :, :, :3]
            h, w = image.shape[1:3]
            # Save both native image VAE latents and a Qwen-readable pixel image.
            # Smaller references reduce disk/token cost but may reduce identity fidelity.
            wh = _short_edge(w, h, int(reference_short_edge))
            resized = _resize(image, *wh).clamp(0, 1)
            latent = vae.encode(resized)
            if latent.ndim == 4:
                latent = latent.unsqueeze(2)
            if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != 24 or latent.shape[2] != 1:
                raise ValueError(f"Unexpected H3-Image VAE shape on image {j+1}: {tuple(latent.shape)}")
            tensors[f"latent_{j}"] = latent.detach().cpu().contiguous()
            tensors[f"vision_{j}"] = (resized * 255).round().to(torch.uint8).cpu().contiguous()
            per_view.append({"latent_shape": list(latent.shape), "image_shape": list(resized.shape)})
        meta = {"format": _FORMAT, "vae": "MiniMax-H3-Image only", "count": str(len(images)),
                "views": json.dumps(per_view), "identity_name": name}
        dest = _DIR / f"{name}.safetensors"
        fd, tmp = tempfile.mkstemp(dir=str(_DIR), prefix=".h3img_", suffix=".tmp")
        os.close(fd)
        try:
            save_file(tensors, tmp, metadata=meta)
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return (str(dest),)


class H3Img8V2Load:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"identity_file": (_options(),),
                              "manual_filename": ("STRING", {"default": "", "tooltip": "Optional: e.g. Personaje_01pks.safetensors. Overrides dropdown."})}}
    RETURN_TYPES = ("H3IMAGE_NATIVE_IDENTITY", "IMAGE")
    RETURN_NAMES = ("identity", "first_reference_preview")
    FUNCTION = "load"
    CATEGORY = "MiniMax H3-Image / Native RefMod (experimental)"

    @classmethod
    def IS_CHANGED(cls, identity_file, manual_filename=""):
        path = _DIR / (manual_filename.strip() or identity_file)
        return path.stat().st_mtime_ns if path.is_file() else float("nan")

    def load(self, identity_file, manual_filename=""):
        identity_file = manual_filename.strip() or identity_file
        if identity_file == "NO_IDENTITIES_YET.safetensors":
            raise ValueError("No identity selected. Click REFRESH or enter filename in manual_filename")
        path = (_DIR / identity_file).resolve()
        if path.parent != _DIR.resolve():
            raise ValueError("Invalid path")
        if not path.is_file():
            raise FileNotFoundError(f"Identity not found: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            if meta.get("format") != _FORMAT:
                raise ValueError("Wrong format: regenerate with 8-image creator; video RefMods/older prototype are not compatible")
            count = int(meta["count"])
        if not 1 <= count <= 8:
            raise ValueError(f"Invalid number of reference views: {count}")
        tensors = load_file(str(path), device="cpu")
        latents, visions = [], []
        for i in range(count):
            latent = tensors[f"latent_{i}"]
            vision = tensors[f"vision_{i}"]
            if latent.ndim != 5 or latent.shape[:3] != (1,24,1):
                raise ValueError(f"Invalid H3-Image latent #{i+1}: {tuple(latent.shape)}")
            if vision.ndim != 4 or vision.shape[0] != 1 or vision.shape[-1] != 3:
                raise ValueError(f"Invalid image #{i+1}")
            latents.append(latent)
            visions.append(vision.to(torch.float32).div(255.0))
        identity = {"latents": latents, "visions": visions, "format": _FORMAT, "name": meta.get("identity_name", "")}
        return (identity, visions[0])


def _soften_reference_latent(z, strength):
    """Retention <=1: preserve the latent or mix with a spatially blurred copy.

    Does not amplify latent numerical magnitude. Unlike naive z*1.5, this
    keeps the reference within a more plausible distribution. This is a
    *retention* dial, not an attention coefficient or identity guarantee.
    """
    strength = max(0.0, min(1.0, float(strength)))
    if strength >= 1.0:
        return z
    if strength <= 0.0:
        return None
    # H3 image refs are [1,24,1,h,w]. Blur ONLY spatial axes.
    b, c, t, h, w = z.shape
    z2 = z.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
    softened = F.avg_pool2d(z2.float(), kernel_size=9, stride=1, padding=4)
    softened = F.avg_pool2d(softened, kernel_size=9, stride=1, padding=4)
    softened = softened.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return z * strength + softened.to(z.dtype) * (1.0 - strength)


class H3Img8V2Condition:
    """Eight native H3-Image references with explicit retention and token copies.

    Original H3 RefMod's 'strength' is a retention between 0 and 1, so at 1.0
    we already have maximum unchanged reference. For a stronger influence, use
    duplicated native reference blocks (token copies), optionally prioritising
    one source view. This is experimental; copies consume more VRAM and cannot
    ensure exact facial matching. Saved .safetensors are read-only and unchanged.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "identity": ("H3IMAGE_NATIVE_IDENTITY",),
            "prompt": ("STRING", {"multiline": True,
                      "default": "A natural portrait of the exact same person in all supplied reference pictures."}),
            "width": ("INT", {"default": 1024, "min": 32, "max": 16384, "step": 32}),
            "height": ("INT", {"default": 1024, "min": 32, "max": 16384, "step": 32}),
            "reference_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "0=no references; 1=original references intact (strongest retention). Below 1 softens them, not a true attention gain."}),
            "reference_copies": ("INT", {"default": 1, "min": 1, "max": 3, "step": 1,
                "tooltip": "1=normal; 2=duplicate all 8 reference blocks in DiT; 3=triple. More influence may be possible but costs VRAM/tokens."}),
            "face_priority_view": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1,
                "tooltip": "0=none; 1..8=reference photo to prioritize, ideally clearest frontal facial image."}),
            "face_priority_copies": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1,
                "tooltip": "Extra DiT copies of prioritized view. 0=none. Try 1 before 2 or 3."}),
            "auto_identity_hint": ("BOOLEAN", {"default": True,
                "tooltip": "Prepend short instruction explaining that all <Picture N> show same person."}),
        }}
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "condition"
    CATEGORY = "MiniMax H3-Image / Native RefMod (experimental)"

    def condition(self, clip, identity, prompt, width, height,
                  reference_strength=1.0, reference_copies=1,
                  face_priority_view=0, face_priority_copies=0,
                  auto_identity_hint=True):
        if identity.get("format") != _FORMAT:
            raise ValueError("Expected native 8-view H3-Image identity")
        visions, latents = identity["visions"], identity["latents"]
        n = len(visions)
        if n != len(latents) or not 1 <= n <= 8:
            raise ValueError("Identity reference count mismatch")

        strength = float(reference_strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("reference_strength must be between 0 and 1")
        copies = max(1, min(3, int(reference_copies)))
        selected = int(face_priority_view)
        extra = max(0, min(3, int(face_priority_copies)))
        if selected > n:
            raise ValueError(f"face_priority_view={selected} but identity has only {n} pictures")

        if strength > 0:
            tags = ", ".join(f"<Picture {i}>" for i in range(1, n + 1))
            if auto_identity_hint:
                prompt = (f"Identity instruction: {tags} all portray the SAME single adult individual. "
                          "Preserve the distinctive facial geometry, natural asymmetry and age of this person "
                          "as shown in the reference photos. Do not average the face or invent a different person. "
                          "Scene request: " + prompt)
            # Crucial: follow ComfyUI's official H3 Image picture labeling.
            tokens = clip.tokenize(prompt, images=visions, picture_label="Picture %d: ")
        else:
            # Strength zero disables both Qwen visual images and DiT ref latents.
            tokens = clip.tokenize(prompt)
        cond = clip.encode_from_tokens_scheduled(tokens)

        if strength > 0:
            blocks = []
            for z in latents:
                weighted = _soften_reference_latent(z, strength)
                blocks.append({"kind": "image", "latent_h": int(weighted.shape[-2]),
                               "latent_w": int(weighted.shape[-1]), "latent": weighted})
            # Independent repeated block dictionaries, preserving original tensors.
            all_blocks = [dict(block) for _ in range(copies) for block in blocks]
            if selected and extra:
                all_blocks.extend(dict(blocks[selected - 1]) for _ in range(extra))
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": all_blocks})
            print(f"[H3Img8V2] identity={identity.get('name', '?')!r}, retention={strength:.2f}, "
                  f"{n} Qwen pictures, {len(all_blocks)} DiT reference blocks "
                  f"(copies={copies}, priority={selected}, extras={extra}). "
                  "More copies increase token/VRAM cost, not identity certainty.")
        else:
            print("[H3Img8V2] reference_strength=0: Qwen and DiT refs disabled")
        canvas = torch.zeros([1, 24, 1, height // 16, width // 16],
                             device=comfy.model_management.intermediate_device())
        return (cond, {"samples": canvas})


NODE_CLASS_MAPPINGS = {
    "H3Img8V2Create": H3Img8V2Create,
    "H3Img8V2Load": H3Img8V2Load,
    "H3Img8V2Condition": H3Img8V2Condition,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3Img8V2Create": "H3 IMAGE IDENTITY V2 · CREATE 8 PHOTOS",
    "H3Img8V2Load": "H3 IMAGE IDENTITY V2 · LOAD / REFRESH",
    "H3Img8V2Condition": "H3 IMAGE IDENTITY V2 · STRONG CONDITIONING",
}
