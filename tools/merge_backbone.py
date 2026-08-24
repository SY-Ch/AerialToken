"""Merge the converted frozen backbone into an AerialToken checkpoint.

Training saves only the trainable part of the model (the calibration module,
the decoder and the prediction heads), so evaluation normally needs the frozen
backbone supplied separately via ``tools/test.py --backbone``. This script
folds the two into one self-contained checkpoint, which is convenient for
distribution.
"""

import argparse

import torch


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    backbone = torch.load(args.backbone, map_location="cpu")
    backbone = {f"backbone.{k}": v for k, v in backbone.items()}

    target = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    target.update(backbone)

    torch.save(checkpoint, args.out)
    print(f"wrote {args.out} ({len(target)} tensors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="AerialToken checkpoint from training")
    parser.add_argument(
        "backbone", help="converted backbone, e.g. checkpoints/dinov2_converted_depth.pth"
    )
    parser.add_argument("out", help="path to write the merged checkpoint to")
    main(parser.parse_args())
