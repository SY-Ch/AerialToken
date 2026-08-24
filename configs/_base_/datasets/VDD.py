vdd_type = 'VDD_Dataset'
vdd_root = './data/DomainUAV/VDD'
vdd_crop_size = (512, 512)
vdd_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=vdd_crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
vdd_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2560, 1920), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

vdd_train_dataloader = dict(
    type=vdd_type,
    data_root=vdd_root,
    data_prefix=dict(
        img_path='train/src', seg_map_path='train/gt'),
    pipeline=vdd_train_pipeline)

vdd_val_dataloader = dict(
    type=vdd_type,
    data_root=vdd_root,
    data_prefix=dict(img_path='val/src', seg_map_path='val/gt'),
    pipeline=vdd_test_pipeline)