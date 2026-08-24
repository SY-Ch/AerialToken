# SkyScapes, evaluated as an unseen target of the VDD-source protocol. Masks are
# already remapped to the seven-class VDD taxonomy, so the VDD reader is reused.
# The Resize matches the VDD test pipeline exactly (long edge 2560, keep_ratio),
# which is fixed by protocol consistency rather than chosen on target scores.
skyscapes_type = 'VDD_Dataset'
skyscapes_root = './data/DomainUAV/SkyScapes'

skyscapes_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2560, 1920), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

skyscapes_val_dataloader = dict(
    type=skyscapes_type,
    data_root=skyscapes_root,
    img_suffix='.jpg',
    seg_map_suffix='.png',
    data_prefix=dict(img_path='val/src', seg_map_path='val/gt'),
    pipeline=skyscapes_test_pipeline)
