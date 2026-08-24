from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class UDD7_Dataset(BaseSegDataset):
    METAINFO = dict(
    classes=('other', 'wall', 'road', 'vegetation', 'vehicle', 'roof','water'),
    palette=[[0, 0, 0], [102, 102, 156], [128, 64, 128], [107, 142, 35],
                 [0, 0, 142], [70, 70, 70], [0, 200, 200]])

    # classes=('wall', 'road', 'vegetation', 'vehicle', 'roof','water'),
    # palette=[[102, 102, 156], [128, 64, 128], [107, 142, 35],
    #              [0, 0, 142], [70, 70, 70], [0, 200, 200]])
    

    def __init__(self,
                 img_suffix='.JPG',
                 seg_map_suffix='.png',
                #  reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, 
            # reduce_zero_label=reduce_zero_label, 
            **kwargs)