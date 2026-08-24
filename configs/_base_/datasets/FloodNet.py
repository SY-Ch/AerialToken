floodnet_type = 'LoveDADataset'
floodnet_root = './data/DomainGS'

floodnet_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

floodnet_val_dataloader = dict(
    type=floodnet_type,
    data_root=floodnet_root,
    data_prefix=dict(img_path='Val/FloodNet/images_png', seg_map_path='Val/FloodNet/masks_png'),
    pipeline=floodnet_test_pipeline)