# Single-domain evaluation: Table III / IV, AeroScapes.
#
# Same model, pipeline and metric as the full protocol run, with the evaluated
# set narrowed to one domain. Useful for checking a single reported number
# without running the other four.
_base_ = ["../aerialtoken/aerialtoken_dinov2_mask2former_512x512_vdd_source.py"]

val_dataloader = dict(dataset=dict(datasets=[{{_base_.aeroscapes_val_dataloader}}]))
test_dataloader = val_dataloader

val_evaluator = dict(dataset_keys=["AeroScapes"], mean_used_keys=["AeroScapes"])
test_evaluator = val_evaluator
