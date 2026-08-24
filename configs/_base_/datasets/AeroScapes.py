# AeroScapes, evaluated as an unseen target of the VDD-source protocol. Masks are
# already remapped to the seven-class VDD taxonomy, so the VDD reader is reused.
# AeroScapes is natively 1280x720; the pipeline resizes to the UAVid test
# resolution without preserving the aspect ratio, matching the UAVid entry.
aeroscapes_type = 'VDD_Dataset'
aeroscapes_root = './data/DomainUAV/AeroScapes'

aeroscapes_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1080), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

aeroscapes_val_dataloader = dict(
    type=aeroscapes_type,
    data_root=aeroscapes_root,
    img_suffix='.jpg',
    seg_map_suffix='.png',
    data_prefix=dict(img_path='val/src', seg_map_path='val/gt'),
    pipeline=aeroscapes_test_pipeline)
