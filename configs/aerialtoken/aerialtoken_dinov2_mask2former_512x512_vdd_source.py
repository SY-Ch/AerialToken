# AerialToken, VDD-source protocol (Table III and Table IV of the paper).
#
# Train:    python tools/train.py configs/aerialtoken/aerialtoken_dinov2_mask2former_512x512_vdd_source.py
# Evaluate: python tools/test.py  configs/aerialtoken/aerialtoken_dinov2_mask2former_512x512_vdd_source.py \
#               work_dirs/aerialtoken_dinov2_mask2former_512x512_vdd_source/iter_40000.pth \
#               --backbone checkpoints/dinov2_converted_depth.pth
_base_ = [
    "../_base_/datasets/vdd_source.py",
    "../_base_/models/aerialtoken_dinov2_mask2former.py",
    "../_base_/schedules/schedule_40k.py",
    "../_base_/default_runtime.py",
]

# class-focused supervision on the two unified indices that are hardest under
# this protocol (road and water)
model = dict(
    decode_head=dict(
        focus_class_ids=(2, 6),
        focus_ce_loss_weight=0.5,
        focus_dice_loss_weight=0.5,
    )
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomChoiceResize",
        scales=[int(512 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=2048,
    ),
    dict(type="RandomCrop", crop_size={{_base_.crop_size}}, cat_max_ratio=0.75),
    dict(type="RandomFlip", direction="horizontal", prob=0.5),
    dict(type="RandomFlip", direction="vertical", prob=0.5),
    dict(
        type="RandomRotate",
        prob=0.5,
        degree=(0, 270),
        pad_val=0,
        seg_pad_val=255,
        auto_bound=False,
    ),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

optim_wrapper = dict(optimizer=dict(lr=6e-5))
