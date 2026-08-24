# OpenEarthMap, evaluated as an additional unseen target of the LoveDA-source
# protocol. Images and masks are already remapped to the unified label space,
# so the reader is the LoveDA one and no reduce_zero_label is applied here.
oem_type = 'LoveDADataset'
oem_root = './data/DomainGS'

oem_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024)),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

oem_val_dataloader = dict(
    type=oem_type,
    data_root=oem_root,
    img_suffix='.png',
    seg_map_suffix='.png',
    data_prefix=dict(
        img_path='Val/OEM/images_png', seg_map_path='Val/OEM/masks_png'),
    pipeline=oem_test_pipeline)
