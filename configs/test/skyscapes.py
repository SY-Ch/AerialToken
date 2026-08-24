# Single-domain evaluation: Table III / IV, SkyScapes.
#
# Same model, pipeline and metric as the full protocol run, with the evaluated
# set narrowed to one domain. Useful for checking a single reported number
# without running the other four.
_base_ = ["../aerialtoken/aerialtoken_dinov2_mask2former_512x512_vdd_source.py"]

val_dataloader = dict(dataset=dict(datasets=[{{_base_.skyscapes_val_dataloader}}]))
test_dataloader = val_dataloader

val_evaluator = dict(dataset_keys=["SkyScapes"], mean_used_keys=["SkyScapes"])
test_evaluator = val_evaluator
