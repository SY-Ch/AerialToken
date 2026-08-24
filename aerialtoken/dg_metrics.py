import os.path as osp
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from mmengine.logging import MMLogger, print_log
from PIL import Image
from prettytable import PrettyTable

from mmseg.registry import METRICS
from mmseg.evaluation.metrics.iou_metric import IoUMetric
from collections import OrderedDict, defaultdict


@METRICS.register_module()
class DGIoUMetric(IoUMetric):
    def __init__(
        self,
        dataset_keys=[],
        mean_used_keys=[],
        img_root_prefix=None,
        excluded_class_indices=(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_keys = dataset_keys
        self.img_root_prefix = img_root_prefix
        self.excluded_class_indices = tuple(excluded_class_indices)
        if mean_used_keys:
            self.mean_used_keys = mean_used_keys
        else:
            self.mean_used_keys = dataset_keys

    def _get_output_png_filename(self, img_path: str) -> str:
        img_path = Path(img_path).resolve()
        output_dir = Path(self.output_dir).resolve()
        cwd = Path.cwd().resolve()

        if self.img_root_prefix is not None:
            img_root_prefix = Path(self.img_root_prefix).resolve()
            if img_path.is_relative_to(img_root_prefix):
                relative_path = img_path.relative_to(img_root_prefix)
            else:
                relative_path = None
        else:
            relative_path = None

        if relative_path is None and img_path.is_relative_to(cwd):
            relative_path = img_path.relative_to(cwd)
        elif relative_path is None and img_path.parent.name:
            if img_path.parent.name in {"Images", "src", "images_png"} and img_path.parent.parent != img_path.parent:
                relative_path = Path(img_path.parent.parent.name) / img_path.parent.name / img_path.name
            else:
                relative_path = Path(img_path.name)
        elif relative_path is None:
            relative_path = Path(img_path.name)

        png_filename = (output_dir / relative_path).with_suffix(".png")
        png_filename.parent.mkdir(parents=True, exist_ok=True)
        return str(png_filename)

    @staticmethod
    def _safe_nanmean(metric_values: np.ndarray, valid_mask: np.ndarray = None) -> float:
        metric_values = np.asarray(metric_values, dtype=float)
        if valid_mask is not None:
            metric_values = metric_values[valid_mask]
        metric_values = metric_values.reshape(-1)
        metric_values = metric_values[np.isfinite(metric_values)]
        if metric_values.size == 0:
            return np.nan
        return float(metric_values.mean())

    def _compute_group_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            None,
            self.beta,
        )

        class_valid_mask = None
        if "IoU" in ret_metrics:
            class_valid_mask = np.ones_like(ret_metrics["IoU"], dtype=bool)
            for class_index in self.excluded_class_indices:
                if 0 <= class_index < class_valid_mask.shape[0]:
                    class_valid_mask[class_index] = False
        if "Acc" in ret_metrics:
            acc_finite_mask = np.isfinite(ret_metrics["Acc"])
            class_valid_mask = (
                acc_finite_mask
                if class_valid_mask is None
                else class_valid_mask & acc_finite_mask
            )

        ret_metrics_summary = OrderedDict()
        for ret_metric, ret_metric_value in ret_metrics.items():
            valid_mask = class_valid_mask if ret_metric in {"IoU", "Acc"} else None
            ret_metrics_summary[ret_metric] = np.round(
                self._safe_nanmean(ret_metric_value, valid_mask) * 100, 2
            )

        metrics = {}
        for key, val in ret_metrics_summary.items():
            if key == "aAcc":
                metrics[key] = val
            else:
                metrics[f"m{key}"] = val

        ret_metrics_class = ret_metrics
        if self.nan_to_num is not None:
            ret_metrics_class = OrderedDict(
                {
                    metric: np.nan_to_num(metric_value, nan=self.nan_to_num)
                    for metric, metric_value in ret_metrics.items()
                }
            )
        ret_metrics_class.pop("aAcc", None)
        ret_metrics_class = OrderedDict(
            {
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics_class.items()
            }
        )
        ret_metrics_class.update({"Class": self.dataset_meta["classes"]})
        ret_metrics_class.move_to_end("Class", last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log("per class results:", logger)
        print_log("\n" + class_table_data.get_string(), logger=logger)
        return metrics

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta["classes"])
        for data_sample in data_samples:
            pred_label = data_sample["pred_sem_seg"]["data"].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample["gt_sem_seg"]["data"].squeeze().to(pred_label)
                res1, res2, res3, res4 = self.intersect_and_union(
                    pred_label, label, num_classes, self.ignore_index
                )
                dataset_key = "unknown"
                for key in self.dataset_keys:
                    if key in data_sample["seg_map_path"]:
                        dataset_key = key
                        break
                self.results.append([dataset_key, res1, res2, res3, res4])
            # format_result
            if self.output_dir is not None:
                png_filename = self._get_output_png_filename(data_sample["img_path"])
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get("reduce_zero_label", False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f"results are saved to {osp.dirname(self.output_dir)}")
            return OrderedDict()

        dataset_results = defaultdict(list)
        metrics = {}
        for result in results:
            dataset_results[result[0]].append(result[1:])
        metrics_type2mean = defaultdict(list)
        for key, key_result in dataset_results.items():
            print_log(f"----------metrics for {key}------------", logger)
            key_metrics = self._compute_group_metrics(key_result)
            print_log(f"number of samples for {key}: {len(key_result)}")
            for k, v in key_metrics.items():
                metrics[f"{key}_{k}"] = v
                if key in self.mean_used_keys:
                    metrics_type2mean[k].append(v)
        for k, v in metrics_type2mean.items():
            metrics[f"mean_{k}"] = sum(v) / len(v)
        return metrics
