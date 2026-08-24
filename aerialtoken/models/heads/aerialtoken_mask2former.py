from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.registry import MODELS
from mmseg.utils import SampleList
from torch import Tensor
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@MODELS.register_module()
class AerialTokenMask2FormerHead(Mask2FormerHead):
    def __init__(
        self,
        replace_query_feat=False,
        use_external_query=True,
        strip_kernel=11,
        attn_relax=0.0,
        attn_thr=0.5,
        enhance_last_n_levels=0,
        structure_init_scale=0.0,
        boundary_init_scale=0.0,
        enable_structure_enhancer=True,
        enable_boundary_refiner=True,
        focus_class_ids=(),
        focus_ce_loss_weight=0.0,
        focus_dice_loss_weight=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        feat_channels = kwargs["feat_channels"]
        self.use_external_query = use_external_query
        if self.use_external_query:
            del self.query_embed
        self.vpt_transforms = nn.ModuleList()
        self.replace_query_feat = replace_query_feat
        if replace_query_feat:
            del self.query_feat
            self.querys2feat = nn.Linear(feat_channels, feat_channels)

        self.shared_memory_enhancer = StructureAwareContextEnhancerLite(
            channels=feat_channels,
            strip_kernel=strip_kernel,
            init_scale=structure_init_scale,
        )
        self.shared_mask_enhancer = StructureAwareContextEnhancerLite(
            channels=feat_channels,
            strip_kernel=strip_kernel,
            init_scale=structure_init_scale,
        )
        self.boundary_refiner = BoundaryAwareMaskRefinerLite(
            channels=feat_channels,
            init_scale=boundary_init_scale,
        )
        self.attn_relax = attn_relax
        self.attn_thr = attn_thr
        self.enable_structure_enhancer = enable_structure_enhancer
        self.enable_boundary_refiner = enable_boundary_refiner
        self.enhance_last_n_levels = max(
            0, min(enhance_last_n_levels, self.num_transformer_feat_level)
        )
        self.focus_class_ids = tuple(
            class_id
            for class_id in focus_class_ids
            if 0 <= class_id < self.num_classes
        )
        self.focus_ce_loss_weight = float(focus_ce_loss_weight)
        self.focus_dice_loss_weight = float(focus_dice_loss_weight)

    def _forward_head(
        self,
        decoder_out: Tensor,
        mask_feature: Tensor,
        attn_mask_target_size: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        cls_pred = self.cls_embed(decoder_out)
        mask_embed = self.mask_embed(decoder_out)
        mask_pred = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feature)

        attn_prob = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        if self.attn_relax > 0:
            uncertainty = 4.0 * attn_prob * (1.0 - attn_prob)
            attn_prob = torch.clamp(attn_prob + self.attn_relax * uncertainty, max=1.0)
        attn_mask = attn_prob.flatten(2)
        attn_mask = attn_mask.unsqueeze(1).repeat((1, self.num_heads, 1, 1))
        attn_mask = attn_mask.flatten(0, 1)
        attn_mask = (attn_mask < self.attn_thr).detach()

        return cls_pred, mask_pred, attn_mask

    def forward(
        self, x: Tuple[List[Tensor], List[Tensor]], batch_data_samples: SampleList
    ) -> Tuple[List[Tensor]]:
        if isinstance(x, tuple):
            x, query_embed = x
        else:
            query_embed = None
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        batch_size = len(batch_img_metas)
        if query_embed is None:
            query_embed = self.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))
        elif query_embed.ndim == 2:
            query_embed = query_embed.expand(batch_size, -1, -1)

        mask_features, multi_scale_memorys = self.pixel_decoder(x)

        refined_multi_scale_memorys = list(multi_scale_memorys)
        if self.enable_structure_enhancer:
            start_idx = self.num_transformer_feat_level - self.enhance_last_n_levels
            for i in range(start_idx, self.num_transformer_feat_level):
                refined_multi_scale_memorys[i] = self.shared_memory_enhancer(
                    refined_multi_scale_memorys[i]
                )

            mask_features = self.shared_mask_enhancer(mask_features)
        if self.enable_boundary_refiner:
            mask_features = self.boundary_refiner(mask_features)

        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](refined_multi_scale_memorys[i])
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            mask = decoder_input.new_zeros(
                (batch_size,) + refined_multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool,
            )
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(
                0, 2, 1
            )
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)

        if self.replace_query_feat:
            query_feat = self.querys2feat(query_embed)
        else:
            query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, refined_multi_scale_memorys[0].shape[-2:]
        )
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                key_padding_mask=None,
            )
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat,
                mask_features,
                refined_multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level
                ].shape[-2:],
            )
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list

    def loss(
        self, x: Tuple[Tensor], batch_data_samples: SampleList, train_cfg
    ) -> dict:
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples
        )

        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        losses = self.loss_by_feat(
            all_cls_scores, all_mask_preds, batch_gt_instances, batch_img_metas
        )

        focus_losses = self._loss_semantic_focus(
            all_cls_scores[-1], all_mask_preds[-1], batch_data_samples
        )
        losses.update(focus_losses)
        return losses

    def _loss_semantic_focus(
        self,
        mask_cls_results: Tensor,
        mask_pred_results: Tensor,
        batch_data_samples: SampleList,
    ) -> dict:
        if not self.focus_class_ids:
            zero = mask_cls_results.sum() * 0.0
            return dict(loss_focus_ce=zero, loss_focus_dice=zero)

        gt_sem_seg = torch.stack(
            [data_sample.gt_sem_seg.data.squeeze(0) for data_sample in batch_data_samples],
            dim=0,
        ).long()
        target_size = gt_sem_seg.shape[-2:]

        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_score = torch.einsum("bqc,bqhw->bchw", cls_score, mask_pred)
        seg_prob = seg_score / seg_score.sum(dim=1, keepdim=True).clamp_min(1e-6)

        valid_mask = gt_sem_seg != self.ignore_index
        focus_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        for class_id in self.focus_class_ids:
            focus_mask |= gt_sem_seg == class_id

        if self.focus_ce_loss_weight > 0 and focus_mask.any():
            log_seg_prob = torch.log(seg_prob.clamp_min(1e-6))
            focus_ce = F.nll_loss(
                log_seg_prob,
                gt_sem_seg,
                reduction="none",
                ignore_index=self.ignore_index,
            )
            focus_ce = focus_ce[focus_mask].mean() * self.focus_ce_loss_weight
        else:
            focus_ce = seg_prob.sum() * 0.0

        if self.focus_dice_loss_weight > 0:
            valid_mask_float = valid_mask.float()
            focus_dice_losses = []
            for class_id in self.focus_class_ids:
                pred = seg_prob[:, class_id] * valid_mask_float
                target = (gt_sem_seg == class_id).float() * valid_mask_float
                intersection = (pred * target).sum(dim=(-2, -1))
                denominator = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
                dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
                focus_dice_losses.append(dice_loss.mean())
            focus_dice = (
                torch.stack(focus_dice_losses).mean() * self.focus_dice_loss_weight
            )
        else:
            focus_dice = seg_prob.sum() * 0.0

        return dict(loss_focus_ce=focus_ce, loss_focus_dice=focus_dice)


class StructureAwareContextEnhancerLite(nn.Module):
    def __init__(self, channels, strip_kernel=11, init_scale=0.0):
        super().__init__()
        self.pre = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.horizontal_branch = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, strip_kernel),
            padding=(0, strip_kernel // 2),
            groups=channels,
            bias=False,
        )
        self.vertical_branch = nn.Conv2d(
            channels,
            channels,
            kernel_size=(strip_kernel, 1),
            padding=(strip_kernel // 2, 0),
            groups=channels,
            bias=False,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.GELU(),
        )
        self.res_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        x_proj = self.pre(x)
        horizontal_feat = self.horizontal_branch(x_proj)
        vertical_feat = self.vertical_branch(x_proj)
        fused_feat = self.fuse(torch.cat([horizontal_feat, vertical_feat], dim=1))
        return x + self.res_scale * fused_feat


class FixedGradientExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, x):
        x = x.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(x, self.kernel_x, padding=1)
        grad_y = F.conv2d(x, self.kernel_y, padding=1)
        return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)


class BoundaryAwareMaskRefinerLite(nn.Module):
    def __init__(self, channels, init_scale=0.0):
        super().__init__()
        self.gradient_extractor = FixedGradientExtractor()
        self.boundary_proj = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        boundary_prior = self.gradient_extractor(x)
        boundary_prior = boundary_prior / boundary_prior.amax(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-6)
        boundary_gate = 2.0 * self.boundary_proj(boundary_prior) - 1.0
        return x + self.res_scale * x * boundary_gate
