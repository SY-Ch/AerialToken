# AerialToken: Geometry-Guided Domain Generalization for Remote Sensing Semantic Segmentation

## Installation & Environment Setup

Clone the repository:

```
git clone https://github.com/SY-Ch/AerialToken.git
```

Follow these steps to set up your environment:

```
conda create -n aerialtoken python=3.11 -y
conda activate aerialtoken
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 #2.6.0
pip install -U openmim
mim install mmengine

#install mmcv
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -r requirements/optional.txt
pip install -e . -v

pip install mmsegmentation
pip install mmdet
pip install xformers=='0.0.30' # optional for DINOv2
pip install -r requirements.txt
pip install future tensorboard
```

Depth Anything V2 is vendored under `aerialtoken/models/backbones/third_party`, so no submodule initialization is needed.

## Dataset Preparation

Two protocols are used, each with its own unified label space. Every mask is expected to be already remapped, following Tables X and XI of the paper:

- **LoveDA-source**, seven unified classes. Index 0 collects background and ignored categories and is excluded from the class-wise IoU and from every reported mean.
- **VDD-source**, the seven-class VDD taxonomy. No class index is excluded.

Images are `.png` under `DomainGS` and `.jpg` under `SkyScapes` and `AeroScapes`; masks are `.png` throughout. The final folder structure should look like this:

```
AerialToken
├── ...
├── checkpoints
│   ├── dinov2_vitl14_pretrain.pth
│   ├── depth_anything_v2_vitl.pth
│   ├── dinov2_converted_depth.pth
├── data
│   ├── DomainGS
│   │   ├── Train
│   │   │   ├── images_png
│   │   │   │   ├── Rural
│   │   │   │   ├── Urban
│   │   │   ├── masks_png
│   │   │   │   ├── Rural
│   │   │   │   ├── Urban
│   │   ├── Val
│   │   │   ├── FLAIR
│   │   │   │   ├── images_png
│   │   │   │   ├── masks_png
│   │   │   ├── FloodNet
│   │   │   ├── Potsdam
│   │   │   ├── Vaihingen
│   │   │   ├── OEM
│   ├── DomainUAV
│   │   ├── VDD
│   │   │   ├── train
│   │   │   │   ├── src
│   │   │   │   ├── gt
│   │   │   ├── val
│   │   │   │   ├── src
│   │   │   │   ├── gt
│   │   ├── UDD7
│   │   │   ├── img_dir
│   │   │   │   ├── train
│   │   │   │   ├── val
│   │   │   ├── ann_dir
│   │   │   │   ├── train
│   │   │   │   ├── val
│   │   ├── Uavid7
│   │   │   ├── train
│   │   │   │   ├── Images
│   │   │   │   ├── Labels7
│   │   │   ├── val
│   │   │   │   ├── Images
│   │   │   │   ├── Labels7
│   │   ├── SkyScapes
│   │   │   ├── val
│   │   │   │   ├── src
│   │   │   │   ├── gt
│   │   ├── AeroScapes
│   │   │   ├── val
│   │   │   │   ├── src
│   │   │   │   ├── gt
├── ...

```

Each dataset has one config under `configs/_base_/datasets`, and the two protocol files `loveda_source.py` and `vdd_source.py` assemble them into the source domain and the evaluated domains.

## Pre-trained Weights & Dataset Downloads

**Download:**

Download the pre-trained weights for testing from [facebookresearch](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth). Ensure the file name remains unchanged and place it in the project directory. You can also download the DepthAnything weights from [DepthAnything GitHub](https://github.com/DepthAnything/Depth-Anything-V2).

**Convert:**

Convert the pre-trained weights for training or evaluation by running:

```bash
python tools/convert_models/convert_dinov2_depth.py checkpoints/dinov2_vitl14_pretrain.pth checkpoints/depth_anything_v2_vitl.pth checkpoints/dinov2_converted_depth.pth
```

This step is what makes the two encoders comparable block by block. Both DINOv2
ViT-L and the Depth Anything V2 encoder are released with a 14x14 patch
embedding and a 37x37 position-embedding grid. For each branch the script
resamples the patch-embedding kernel from 14x14 to 16x16 bicubically, and the
position-embedding grid from 37x37 to 32x32 bicubically, keeping the class
token unchanged. A 512x512 input then divides evenly (512 / 16 = 32), so both
branches produce a 32x32 token grid with 1024 channels over 24 blocks, and the
geometry token of block k pairs with the visual token of block k without any
interpolation or channel projection at run time. Both resampled encoders are
written into a single file, the visual weights under their original names and
the geometry weights under a `depth_anything.` prefix.

## Training

Both protocols train for 40,000 iterations at batch size 4 with a 512x512 crop, AdamW and polynomial decay (power 0.9). Only the token bank, the lightweight projections, the decoder and the prediction heads receive gradients. If you need to resume training from a checkpoint, simply append `--resume` to the command.

*Tips: If resuming training appears to hang or shows no response for a long time, please refer to [this issue](https://github.com/open-mmlab/mmsegmentation/issues/3671) for potential solutions.*

- **LoveDA → FLAIR + FloodNet + Potsdam + Vaihingen + OpenEarthMap:**

  ```
  python tools/train.py configs/aerialtoken/aerialtoken_dinov2_mask2former_512x512_loveda_source.py
  # To resume training, use:
  # python tools/train.py configs/aerialtoken/aerialtoken_dinov2_mask2former_512x512_loveda_source.py --resume
  ```

- **VDD → UDD7 + UAVid + SkyScapes + AeroScapes:**

  ```
  python tools/train.py configs/aerialtoken/aerialtoken_dinov2_mask2former_512x512_vdd_source.py
  # To resume training, use:
  # python tools/train.py configs/aerialtoken/aerialtoken_dinov2_mask2former_512x512_vdd_source.py --resume
  ```

For multi-GPU training, use:

```
bash tools/dist_train.sh configs/aerialtoken/aerialtoken_dinov2_mask2former_512x512_loveda_source.py 4
```

Checkpoints land in `work_dirs/<config name>/`. The paper reports the **final** checkpoint of the schedule, `iter_40000.pth`; nothing is selected on target-domain scores. Validation therefore runs once, at the end of training, and prints exactly the numbers of the tables.

## Evaluation

To evaluate a trained model, replace `<AerialToken model>.pth` with your model file and run the corresponding command. Training saves only the trainable parameters, so the backbone checkpoint `checkpoints/dinov2_converted_depth.pth` is supplied separately in all evaluations:

- **Evaluation with LoveDA-source Configuration:**

  ```
  python tools/test.py configs/aerialtoken/aerialtoken_dinov2_mask2former_512x512_loveda_source.py <AerialToken model>.pth --backbone checkpoints/dinov2_converted_depth.pth
  ```

- **Evaluation with VDD-source Configuration:**

  ```
  python tools/test.py configs/aerialtoken/aerialtoken_dinov2_mask2former_512x512_vdd_source.py <AerialToken model>.pth --backbone checkpoints/dinov2_converted_depth.pth
  ```

- **Evaluation of a single domain:**

  `configs/test` holds one config per reported column, which avoids evaluating all five domains when only one number is in question:

  ```
  python tools/test.py configs/test/skyscapes.py <AerialToken model>.pth --backbone checkpoints/dinov2_converted_depth.pth
  ```

Inference is sliding-window with a 512x512 window and a stride of 341, shared by every method compared in the paper. `tools/merge_backbone.py` folds the backbone into the checkpoint if you would rather distribute a single self-contained file; `--backbone` is then unnecessary.

**Expected results.** LoveDA-source, mIoU (%), unified seven-class space with index 0 excluded:

| FLAIR | FloodNet | Potsdam | Vaihingen | OpenEarthMap | Mean |
|------:|---------:|--------:|----------:|-------------:|-----:|
| 46.41 |    45.44 |   35.57 |     42.67 |        54.61 | 44.94 |

VDD-source, mIoU (%), seven-class VDD taxonomy:

| VDD (source) | UDD7 | UAVid | SkyScapes | AeroScapes | Mean |
|-------------:|-----:|------:|----------:|-----------:|-----:|
|        83.27 | 64.66 | 70.54 |     50.45 |      41.77 | 62.14 |

These are single runs with a fixed seed (4307) and no repeated seeds, so expect agreement to about the second decimal rather than exactly. `DGIoUMetric` groups predictions by matching the keys in `dataset_keys` against the mask path, computes mIoU per domain over the classes that are present, and averages the per-domain values listed in `mean_used_keys`.

## Acknowledgment

Our implementation is mainly based on following repositories. Thanks for their authors.

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [Rein](https://github.com/w1oves/Rein)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
