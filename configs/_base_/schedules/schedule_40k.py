# 40k-iteration schedule shared by both protocols.
#
# Only the token bank, the lightweight projections, the decoder and the
# prediction heads are updated; both ViT-L encoders stay frozen, which is why
# the optimizer is built by PEFTOptimWrapperConstructor.
#
# The learning rate differs per protocol and is set in the protocol config.
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)

optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW", lr=1e-4, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
            "query_embed": embed_multi,
            "level_embed": embed_multi,
            "learnable_tokens": embed_multi,
            "calibration.scale": embed_multi,
        },
        norm_decay_mult=0.0,
    ),
)

param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=40000, by_epoch=False)
]

# The paper reports the final checkpoint of the 40k schedule, so validation runs
# once at the end and no checkpoint is selected on target-domain scores.
train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=40000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=10000, max_keep_ckpts=1
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
