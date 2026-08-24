import os
import os.path as osp
import warnings
from pathlib import Path
from typing import Optional, Sequence

import mmcv
from mmengine import fileio
from mmengine.visualization import Visualizer
from mmseg.engine.hooks import SegVisualizationHook
from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample


@HOOKS.register_module()
class PathAwareSegVisualizationHook(SegVisualizationHook):
    """Save visualization results with dataset-relative paths.

    This keeps the directory structure for concat datasets, e.g.
    ``FLAIR/images_png/xxx.png`` instead of flattening everything into a single
    folder.
    """

    def __init__(
        self,
        draw: bool = False,
        interval: int = 50,
        show: bool = False,
        wait_time: float = 0.0,
        backend_args: Optional[dict] = None,
        output_dir: Optional[str] = None,
        draw_gt: bool = False,
        draw_pred: bool = True,
    ) -> None:
        super().__init__(
            draw=draw,
            interval=interval,
            show=show,
            wait_time=wait_time,
            backend_args=backend_args,
        )
        self.output_dir = output_dir
        self.draw_gt = draw_gt
        self.draw_pred = draw_pred
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self._resolved_output_dir: Optional[Path] = None
        self._common_img_root: Optional[Path] = None

    def _collect_img_roots(self, dataset) -> list[Path]:
        if hasattr(dataset, "datasets"):
            roots = []
            for sub_dataset in dataset.datasets:
                roots.extend(self._collect_img_roots(sub_dataset))
            return roots

        data_prefix = getattr(dataset, "data_prefix", None) or {}
        img_root = data_prefix.get("img_path")
        if img_root is None:
            return []
        return [Path(img_root).resolve()]

    def _get_common_img_root(self, runner, mode: str) -> Optional[Path]:
        if self._common_img_root is not None:
            return self._common_img_root

        loop = getattr(runner, f"{mode}_loop", None)
        dataloader = getattr(loop, "dataloader", None)
        dataset = getattr(dataloader, "dataset", None)
        if dataset is None:
            return None

        img_roots = self._collect_img_roots(dataset)
        if not img_roots:
            return None

        try:
            self._common_img_root = Path(
                os.path.commonpath([str(path) for path in img_roots])
            )
        except ValueError:
            self._common_img_root = None
        return self._common_img_root

    def _get_output_dir(self, runner) -> Optional[Path]:
        if self.output_dir is None:
            return None

        if self._resolved_output_dir is not None:
            return self._resolved_output_dir

        output_dir = Path(self.output_dir)
        if output_dir.is_absolute():
            resolved_output_dir = output_dir
        else:
            timestamp = getattr(runner, "timestamp", None)
            if timestamp:
                resolved_output_dir = Path(runner.work_dir) / timestamp / output_dir
            else:
                resolved_output_dir = Path(runner.work_dir) / output_dir

        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        self._resolved_output_dir = resolved_output_dir
        return self._resolved_output_dir

    def _get_output_path(self, runner, img_path: str, mode: str) -> Optional[str]:
        output_dir = self._get_output_dir(runner)
        if output_dir is None:
            return None

        img_path = Path(img_path).resolve()
        common_img_root = self._get_common_img_root(runner, mode)

        relative_path: Optional[Path] = None
        if common_img_root is not None and img_path.is_relative_to(common_img_root):
            relative_path = img_path.relative_to(common_img_root)
        else:
            cwd = Path.cwd().resolve()
            if img_path.is_relative_to(cwd):
                relative_path = img_path.relative_to(cwd)

        if relative_path is None:
            warnings.warn(
                f"Failed to infer a dataset-relative path for {img_path}. "
                "Falling back to basename."
            )
            relative_path = Path(img_path.name)

        out_file = (output_dir / relative_path).with_suffix(".png")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        return str(out_file)

    def _after_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: dict,
        outputs: Sequence[SegDataSample],
        mode: str = "val",
    ) -> None:
        if self.draw is False or mode == "train":
            return

        if not self.every_n_inner_iters(batch_idx, self.interval):
            return

        for output in outputs:
            img_path = output.img_path
            img_bytes = fileio.get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order="rgb")
            window_name = f"{mode}_{osp.basename(img_path)}"
            out_file = self._get_output_path(runner, img_path, mode)

            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=output,
                draw_gt=self.draw_gt,
                draw_pred=self.draw_pred,
                show=self.show,
                wait_time=self.wait_time,
                out_file=out_file,
                step=runner.iter,
            )
