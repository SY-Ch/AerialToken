from mmseg.models.builder import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor
from typing import Optional, Sequence, Tuple


def layernorm_lastdim(x: torch.Tensor, eps: float = 1e-6):
    """
    Stat-only LayerNorm over the last dimension (no learnable gamma/beta).
    Works for tensors shaped (..., C).
    """
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mu) / (var + eps).sqrt()

def orthogonal_component(d: torch.Tensor, v: torch.Tensor, eps: float = 1e-6):
    """
    Return the component of d that is orthogonal to v, computed per-vector.
    Shapes: d, v -> (..., C)
    """
    denom = (v * v).sum(dim=-1, keepdim=True) + eps
    proj = ((d * v).sum(dim=-1, keepdim=True) / denom) * v
    return d - proj

def _rel_clip(delta: torch.Tensor, base: torch.Tensor, r: float = 0.1, eps: float = 1e-6):

    d = torch.linalg.norm(delta, dim=-1, keepdim=True)           # (N,B,1)
    b = torch.linalg.norm(base,  dim=-1, keepdim=True).clamp_min(eps)
    scale = (r * b / d).clamp(max=1.0)
    return delta * scale

def _fast_energy_topk_mask(depth_nbc: torch.Tensor, spatial_hw, topk_ratio: float = 0.25):

    N, B, _ = depth_nbc.shape
    if spatial_hw is not None:
        H, W = spatial_hw
        assert N == H * W, f"N={N} mismatch with H*W={H*W}"
    k = max(1, int(min(1.0, topk_ratio) * N))
    energy = depth_nbc.detach().abs().mean(dim=-1)            
    idx = energy.transpose(0,1).topk(k=k, dim=1).indices      # (B,k)
    mask = torch.zeros(B, N, 1, device=depth_nbc.device, dtype=depth_nbc.dtype)
    mask.scatter_(dim=1, index=idx.unsqueeze(-1), src=torch.ones_like(mask[:,:k,:]))
    return mask.transpose(0,1)  # (N,B,1)


@MODELS.register_module()
class TokenCalibration(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        query_aggregation: str = "pool",
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
        topk_ratio: float = 0.25,
        rel_clip: float = 0.10,
        alpha_init: float = -1.0,
        alpha_override: Optional[float] = None,
        active_layers: Optional[Sequence[int]] = None,
        affinity_mode: str = "dual",
        query_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.query_aggregation = query_aggregation
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.topk_ratio = topk_ratio
        self.rel_clip = rel_clip
        self.alpha_init = alpha_init
        self.alpha_override = alpha_override
        self.active_layers = set(active_layers) if active_layers is not None else None
        self.affinity_mode = affinity_mode
        # Number of decoder queries built from the token bank. Defaults to the
        # full bank; setting it smaller keeps all M tokens available for
        # calibration while capping the query set, which separates token-bank
        # capacity from decoder query over-parameterisation.
        self.query_length = query_length
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)
        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)
        
        self.log_tau_v = nn.Parameter(torch.log(torch.tensor(1.0)))  # semantic temp
        self.log_tau_d = nn.Parameter(torch.log(torch.tensor(2.0)))
        self.alpha_logit = nn.Parameter(torch.tensor(float(self.alpha_init)))
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))
        self.scale_v = nn.Parameter(torch.tensor(1.0))
        self.scale_d = nn.Parameter(torch.tensor(0.25)) 

        self.log_tau_v.data = torch.log(torch.tensor(0.7))
        self.log_tau_d.data = torch.log(torch.tensor(0.7))
        self.alpha_logit.data = torch.tensor(float(self.alpha_init))

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def return_auto(self):
        if self.link_token_to_query:
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)
            if self.query_aggregation == "last":
                tokens = tokens[:, :, -1:].repeat(1, 1, 3)
            elif self.query_aggregation == "mean":
                tokens = F.avg_pool1d(tokens, kernel_size=self.num_layers).repeat(1, 1, 3)
            else:
                tokens = torch.cat(
                    [
                        F.max_pool1d(tokens, kernel_size=self.num_layers),
                        F.avg_pool1d(tokens, kernel_size=self.num_layers),
                        tokens[:, :, -1].unsqueeze(-1),
                    ],
                    dim=-1,
                )
            querys = self.merge(tokens.flatten(-2, -1))
            if self.query_length is not None:
                querys = querys[: self.query_length]
            return querys
        else:
            return 

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]
        
    @staticmethod
    def _softmax_zero_mean(z: torch.Tensor, dim: int = -1):
        """
        Apply mean-centering per sample before softmax for better calibration robustness.
        """
        z = z - z.mean(dim=dim, keepdim=True)
        return F.softmax(z, dim=dim)
        
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(
        self,
        feats: Tensor,
        depth_features: Tensor,
        layer: int,
        # num_register_tokens: int,
        batch_first=False,
        has_cls_token=True,
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
            depth_features = depth_features.permute(1, 0, 2)

        if has_cls_token:
            _, depth_features = torch.tensor_split(depth_features, [1], dim=0)
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)

        if self.active_layers is not None and layer not in self.active_layers:
            if has_cls_token:
                feats = torch.cat([cls_token, feats], dim=0)
            if batch_first:
                feats = feats.permute(1, 0, 2)
            return feats

        tokens = self.get_tokens(layer)

        delta_feat = self.forward_delta_feat(
            feats,
            tokens,
            depth_features,
            layer,
            topk_ratio=self.topk_ratio,
            rel_clip=self.rel_clip,
        )
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat

        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)

        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward_delta_feat(
        self,
        feats: torch.Tensor,              # (N, B, Cv)
        tokens: torch.Tensor,             # (M, Ct)  Ct == embed_dims
        depth_features: torch.Tensor,     # (N, B, Cd)
        layers: int,                      
        spatial_hw: Optional[Tuple[int,int]] = None,
        topk_ratio: float = 0.25,
        rel_clip: float = 0.10,
    ) -> torch.Tensor:

        N, B, Cv = feats.shape
        M, Ct = tokens.shape
        assert Ct == self.embed_dims

        feats_nbc  = layernorm_lastdim(feats)           # (N,B,Cv)
        depth_nbc  = layernorm_lastdim(depth_features)  # (N,B,Cd)
        tokens_mc  = layernorm_lastdim(tokens)          # (M,Ct)

        tau_v = F.softplus(self.log_tau_v) + 1e-6
        tau_d = F.softplus(self.log_tau_d) + 1e-6

        NB = N * B
        fb = feats_nbc.reshape(NB, Cv)                 # (NB,Cv)
        db = depth_nbc.reshape(NB, Cv)                 # (NB,Cv)
        tb = tokens_mc.transpose(0,1)                  # (Ct,M)

        z_v = (fb @ tb).reshape(N, B, M) / (math.sqrt(self.embed_dims) * tau_v)
        z_d = (db @ tb).reshape(N, B, M) / (math.sqrt(self.embed_dims) * tau_d)

        if self.affinity_mode == "visual":
            z = z_v
        else:
            mask_d = _fast_energy_topk_mask(depth_nbc, spatial_hw, topk_ratio)  # (N,B,1)
            if self.affinity_mode == "geometry":
                z = z_d * mask_d
            else:
                if self.alpha_override is None:
                    alpha = torch.sigmoid(self.alpha_logit)        # (0,1)
                else:
                    alpha = z_v.new_tensor(float(self.alpha_override))
                z = (1.0 - alpha) * z_v + alpha * (z_d * mask_d)

        z = z - z.mean(dim=-1, keepdim=True)
        attn = F.softmax(z, dim=-1)                    # (N,B,M)

        token_vals = self.mlp_token2feat(tokens_mc[1:, :])      # (M-1,Cv)

        att = attn[:,:,1:].reshape(NB, M-1)                     # (NB,M-1)
        delta = (att @ token_vals).reshape(N, B, Cv)            # (N,B,Cv)

        delta_f = self.mlp_delta_f(delta + feats)               # (N,B,Cv)

        if rel_clip is not None and rel_clip > 0:
            delta_f = _rel_clip(delta_f, feats, r=rel_clip)

        return delta_f

@MODELS.register_module()
class LayerwiseDualRouteCalibration(TokenCalibration):
    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

    def create_model(self):
        super().create_model()
        del self.learnable_tokens
        self.learnable_tokens_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.embed_dims * self.lora_dim) ** 0.5
            )
        )
        nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b.data, -val, val)

    def get_tokens(self, layer):
        if layer == -1:
            return self.learnable_tokens_a @ self.learnable_tokens_b
        else:
            return self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]
