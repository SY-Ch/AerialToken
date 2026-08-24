# AerialToken: frozen DINOv2 ViT-L visual encoder, frozen Depth Anything V2
# geometry encoder (itself a DINOv2 ViT-L), Layerwise Dual Route Token
# Calibration, and a Mask2Former decoder with structure and boundary branches.
#
# Both encoders use a 16x16 patch embedding, so a 512x512 input yields a 32x32
# token grid on both sides and the two block sequences are paired one to one at
# all 24 blocks. No interpolation or channel projection is needed to align them.
crop_size = (512, 512)
num_classes = 7

model = dict(
    type="EncoderDecoder",
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        size=crop_size,
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
    ),
    backbone=dict(
        type="AerialTokenDinoVisionTransformer",
        # "intermediate": geometry tokens are read from the frozen depth
        # backbone before its depth decoder. This is the reported setting.
        geometry_source="intermediate",
        calibration_config=dict(
            type="LayerwiseDualRouteCalibration",
            num_layers=24,
            embed_dims=1024,
            patch_size=16,
            token_length=100,      # M, token bank size
            lora_dim=16,           # rank of the factorised token bank
            affinity_mode="dual",  # visual and geometry affinity routes
            topk_ratio=0.25,       # rho, kept fraction of geometry positions
            rel_clip=0.10,         # r, relative bound on the feature update
            alpha_init=-1.0,       # logit of the route mixing coefficient
            query_aggregation="pool",
            link_token_to_query=True,
        ),
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        img_size=512,
        ffn_layer="mlp",
        init_values=1e-05,
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/dinov2_converted_depth.pth",
        ),
    ),
    decode_head=dict(
        type="AerialTokenMask2FormerHead",
        # decoder queries come from the cross-layer token bank
        use_external_query=True,
        replace_query_feat=True,
        # structure and boundary branches
        strip_kernel=11,
        enhance_last_n_levels=2,
        structure_init_scale=0.10,
        boundary_init_scale=0.05,
        attn_relax=0.0,
        in_channels=[1024, 1024, 1024, 1024],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type="mmdet.MSDeformAttnPixelDecoder",
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
            act_cfg=dict(type="ReLU"),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None,
                    ),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                ),
                init_cfg=None,
            ),
            positional_encoding=dict(num_feats=128, normalize=True),
            init_cfg=None,
        ),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True,
                ),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True,
                ),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type="ReLU", inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True,
                ),
            ),
            init_cfg=None,
        ),
        loss_cls=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            reduction="mean",
            class_weight=[1.0] * num_classes + [0.1],
        ),
        loss_mask=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=True,
            reduction="mean",
            loss_weight=5.0,
        ),
        loss_dice=dict(
            type="mmdet.DiceLoss",
            use_sigmoid=True,
            activate=True,
            reduction="mean",
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0,
        ),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type="mmdet.HungarianAssigner",
                match_costs=[
                    dict(type="mmdet.ClassificationCost", weight=2.0),
                    dict(
                        type="mmdet.CrossEntropyLossCost", weight=5.0, use_sigmoid=True
                    ),
                    dict(type="mmdet.DiceCost", weight=5.0, pred_act=True, eps=1.0),
                ],
            ),
            sampler=dict(type="mmdet.MaskPseudoSampler"),
        ),
    ),
    train_cfg=dict(),
    # sliding-window inference, shared by every method compared in the paper
    test_cfg=dict(
        mode="slide",
        crop_size=(512, 512),
        stride=(341, 341),
    ),
)
