uavid7_type = 'UAVid7_Dataset'
uavid7_root = './data/DomainUAV/Uavid7'
uavid7_crop_size = (512, 512)
uavid7_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=uavid7_crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
uavid7_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1080)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

uavid7_train_dataloader = dict(
    type=uavid7_type,
    data_root=uavid7_root,
    data_prefix=dict(
        img_path='train/Images', seg_map_path='train/Labels7'),
    pipeline=uavid7_train_pipeline)

uavid7_val_dataloader = dict(
    type=uavid7_type,
    data_root=uavid7_root,
    data_prefix=dict(img_path='val/Images', seg_map_path='val/Labels7'),
    pipeline=uavid7_test_pipeline)
