udd7_type = 'UDD7_Dataset'
udd7_root = './data/DomainUAV/UDD7'
udd7_crop_size = (512, 512)
udd7_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
udd7_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

udd7_train_dataloader = dict(
    type=udd7_type,
    data_root=udd7_root,
    data_prefix=dict(
        img_path='img_dir/train', seg_map_path='ann_dir/train'),
    pipeline=udd7_train_pipeline)

udd7_val_dataloader = dict(
    type=udd7_type,
    data_root=udd7_root,
    data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
    pipeline=udd7_test_pipeline)