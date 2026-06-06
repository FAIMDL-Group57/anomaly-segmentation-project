# ---------------------------------------------------------------
# Fine-tuning utility: freeze/unfreeze model parameters by name pattern.
# Added for the COCO -> Cityscapes semantic fine-tuning experiments.
# ---------------------------------------------------------------


import logging
from typing import List

import lightning
from lightning.fabric.utilities import rank_zero_info


class PartialFreeze(lightning.pytorch.Callback):
    """Freeze parameters whose name contains any of ``freeze_patterns``, then
    re-enable any whose name contains an ``unfreeze_patterns`` entry (unfreeze
    wins on conflict).

    Frozen parameters keep ``requires_grad=False``, so they receive no gradient
    and AdamW skips them (no update, no weight decay). The optimizer/scheduler
    construction in ``LightningModule.configure_optimizers`` is unaffected.

    Runs in ``setup(stage="fit")``, before ``configure_optimizers``.

    Example (head-only): ``freeze_patterns=["network.encoder.backbone"]`` freezes
    the whole ViT and trains the queries, class head, mask head and upscale.
    """

    def __init__(
        self,
        freeze_patterns: List[str],
        unfreeze_patterns: List[str] = [],
    ):
        super().__init__()
        self.freeze_patterns = freeze_patterns
        self.unfreeze_patterns = unfreeze_patterns
        self._applied = False

    def _matches(self, name: str, patterns: List[str]) -> bool:
        return any(p in name for p in patterns)

    def setup(self, trainer, pl_module, stage=None):
        if stage != "fit" or self._applied:
            return

        trainable, frozen = 0, 0
        trainable_names = []
        for name, param in pl_module.named_parameters():
            should_freeze = self._matches(name, self.freeze_patterns) and not self._matches(
                name, self.unfreeze_patterns
            )
            param.requires_grad = not should_freeze
            if param.requires_grad:
                trainable += param.numel()
                trainable_names.append(name)
            else:
                frozen += param.numel()

        self._applied = True

        total = trainable + frozen
        rank_zero_info(
            f"[PartialFreeze] trainable {trainable:,} / {total:,} params "
            f"({100.0 * trainable / max(total, 1):.2f}%); frozen {frozen:,}"
        )
        logging.info(
            "[PartialFreeze] trainable parameter groups: "
            + ", ".join(self._summarize(trainable_names))
        )

    @staticmethod
    def _summarize(names: List[str]) -> List[str]:
        """Collapse the trainable parameter names into coarse module prefixes for logging."""
        prefixes = {}
        for name in names:
            parts = name.split(".")
            # keep up to the first 4 components, e.g. "network.encoder.backbone.blocks"
            prefix = ".".join(parts[: min(4, len(parts))])
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        return [f"{k} (x{v})" for k, v in sorted(prefixes.items())]
