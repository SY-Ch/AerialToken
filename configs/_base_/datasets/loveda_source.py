# LoveDA-source protocol.
#
# Source (training) domain : LoveDA (Urban + Rural train split)
# Unseen target domains    : FLAIR, FloodNet, Potsdam, Vaihingen, OpenEarthMap
#
# All masks share the unified seven-class label space of Table X in the paper.
# Index 0 collects background and ignored categories and is therefore excluded
# from the class-wise IoU and from every mean reported in the paper.
_base_ = [
    "./LoveDA.py",
    "./FLAIR.py",
    "./FloodNet.py",
    "./Potsdam.py",
    "./Vaihingen.py",
    "./OpenEarthMap.py",
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset={{_base_.loveda_train_dataloader}},
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.flair_val_dataloader}},
            {{_base_.floodnet_val_dataloader}},
            {{_base_.potsdam_val_dataloader}},
            {{_base_.vaihingen_val_dataloader}},
            {{_base_.oem_val_dataloader}},
        ],
    ),
)
test_dataloader = val_dataloader

# DGIoUMetric groups predictions by matching these keys against the mask path,
# then averages the per-domain mIoU over ``mean_used_keys``.
val_evaluator = dict(
    type="DGIoUMetric",
    iou_metrics=["mIoU"],
    excluded_class_indices=[0],
    dataset_keys=["FLAIR", "FloodNet", "Potsdam", "Vaihingen", "OEM"],
    mean_used_keys=["FLAIR", "FloodNet", "Potsdam", "Vaihingen", "OEM"],
)
test_evaluator = val_evaluator
