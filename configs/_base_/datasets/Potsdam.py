potsdam_type = 'LoveDADataset'
potsdam_root = './data/DomainGS'

potsdam_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

potsdam_val_dataloader = dict(
    type=potsdam_type,
    data_root=potsdam_root,
    data_prefix=dict(img_path='Val/Potsdam/images_png', seg_map_path='Val/Potsdam/masks_png'),
    pipeline=potsdam_test_pipeline)