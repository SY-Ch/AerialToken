vaihingen_type = 'LoveDADataset'
vaihingen_root = './data/DomainGS'

vaihingen_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

vaihingen_val_dataloader = dict(
    type=vaihingen_type,
    data_root=vaihingen_root,
    data_prefix=dict(img_path='Val/Vaihingen/images_png', seg_map_path='Val/Vaihingen/masks_png'),
    pipeline=vaihingen_test_pipeline)