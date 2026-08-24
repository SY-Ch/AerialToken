loveda_type = 'LoveDADataset'
loveda_root = './data/DomainGS'
loveda_crop_size = (512, 512)
loveda_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=loveda_crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
loveda_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

loveda_train_dataloader = dict(
    type=loveda_type,
    data_root=loveda_root,
    data_prefix=dict(
        img_path='Train/images_png', seg_map_path='Train/masks_png'),
    pipeline=loveda_train_pipeline)

urban_val_dataloader = dict(
    type=loveda_type,
    data_root=loveda_root,
    data_prefix=dict(img_path='Val/Urban/images_png', seg_map_path='Val/Urban/masks_png'),
    pipeline=loveda_test_pipeline)

rural_val_dataloader = dict(
    type=loveda_type,
    data_root=loveda_root,
    data_prefix=dict(img_path='Val/Rural/images_png', seg_map_path='Val/Rural/masks_png'),
    pipeline=loveda_test_pipeline)
