# VDD-source protocol.
#
# Source (training) domain : VDD
# Evaluated domains        : VDD (source val), UDD7, UAVid, SkyScapes, AeroScapes
#
# All masks share the seven-class VDD taxonomy. Unlike the LoveDA-source
# protocol no class index is excluded, so the metric averages all seven classes
# that are present in a given domain.
_base_ = [
    "./VDD.py",
    "./UDD7.py",
    "./Uavid7.py",
    "./SkyScapes.py",
    "./AeroScapes.py",
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset={{_base_.vdd_train_dataloader}},
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.vdd_val_dataloader}},
            {{_base_.udd7_val_dataloader}},
            {{_base_.uavid7_val_dataloader}},
            {{_base_.skyscapes_val_dataloader}},
            {{_base_.aeroscapes_val_dataloader}},
        ],
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="DGIoUMetric",
    iou_metrics=["mIoU"],
    dataset_keys=["VDD", "UDD7", "Uavid7", "SkyScapes", "AeroScapes"],
    mean_used_keys=["VDD", "UDD7", "Uavid7", "SkyScapes", "AeroScapes"],
)
test_evaluator = val_evaluator
