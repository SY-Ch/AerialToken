from mmseg.models.builder import BACKBONES, MODELS
from .calibration import TokenCalibration
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train

import sys
from pathlib import Path

current_dir = Path(__file__).parent

depth_anything_dir = current_dir / "third_party" / "Depth-Anything-V2"

sys.path.insert(0, str(depth_anything_dir))

from depth_anything_v2.dpt import DepthAnythingV2

# from .depth_anything_v2.dpt import DepthAnythingV2
import torch
import torch.nn.functional as F

import types

def forward_features_extra(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        x = self.prepare_tokens_with_masks(x, masks)
        out = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            out.append(x)
        return out


@BACKBONES.register_module()
class AerialTokenDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        calibration_config=None,
        geometry_source="intermediate",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.calibration: TokenCalibration = MODELS.build(calibration_config)
        self.geometry_source = geometry_source

        DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }

        self.depth_anything = DepthAnythingV2(**model_configs["vitl"])
        self.depth_anything = self.depth_anything.to(DEVICE).eval()
        self.depth_anything.pretrained.forward_features_extra = types.MethodType(forward_features_extra, self.depth_anything.pretrained)
        if self.geometry_source == "depth_map":
            self.depth_map_projector = torch.nn.Linear(1, self.embed_dim)

        # Checkpoints produced before the calibration module was renamed store
        # its parameters under a "depthforge." prefix. Remap them on load so
        # those checkpoints keep working instead of silently falling back to
        # randomly initialised calibration weights.
        self._register_load_state_dict_pre_hook(self._remap_legacy_keys)

    @staticmethod
    def _remap_legacy_keys(state_dict, prefix, *args, **kwargs):
        legacy = prefix + "depthforge."
        current = prefix + "calibration."
        for key in [k for k in state_dict if k.startswith(legacy)]:
            state_dict[current + key[len(legacy):]] = state_dict.pop(key)

    def _depth_map_tokens(self, x, height, width):
        depth_input = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        depth = self.depth_anything(depth_input).unsqueeze(1)
        depth = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
        depth = depth - depth.mean(dim=(-2, -1), keepdim=True)
        depth = depth / depth.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        depth_tokens = depth.flatten(2).transpose(1, 2)
        depth_tokens = self.depth_map_projector(depth_tokens)
        cls_token = depth_tokens.new_zeros(depth_tokens.shape[0], 1, depth_tokens.shape[-1])
        return torch.cat([cls_token, depth_tokens], dim=1)

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape

        if self.geometry_source == "intermediate":
            depth_features = self.depth_anything.pretrained.forward_features_extra(x)
        else:
            depth_features = None

        H, W = h // self.patch_size, w // self.patch_size
        depth_map_tokens = None
        if self.geometry_source == "depth_map":
            depth_map_tokens = self._depth_map_tokens(x, H, W)
        x = self.prepare_tokens_with_masks(x, masks)

        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if self.geometry_source == "none":
                geometry_features = x
            elif self.geometry_source == "depth_map":
                geometry_features = depth_map_tokens
            else:
                geometry_features = depth_features[idx]
            x = self.calibration.forward(
                x,
                geometry_features,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return outs, self.calibration.return_auto()

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        trainable = ["calibration"]
        if self.geometry_source == "depth_map":
            trainable.append("depth_map_projector")
        set_requires_grad(self, trainable)
        set_train(self, trainable)

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keep = ["calibration"]
        if self.geometry_source == "depth_map":
            keep.append("depth_map_projector")
        keys = [k for k in state.keys() if not any(item in k for item in keep)]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state
