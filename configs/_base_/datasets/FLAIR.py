flair_type = 'LoveDADataset'
flair_root = './data/DomainGS'

flair_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

flair_val_dataloader = dict(
    type=flair_type,
    data_root=flair_root,
    data_prefix=dict(img_path='Val/FLAIR/images_png', seg_map_path='Val/FLAIR/masks_png'),
    pipeline=flair_test_pipeline)
