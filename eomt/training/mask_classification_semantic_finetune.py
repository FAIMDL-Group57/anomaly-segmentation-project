# ---------------------------------------------------------------
# Fine-tuning module for COCO-trained EoMT on Cityscapes.
# Extends MaskClassificationSemantic with backbone freezing and
# optional gradual unfreezing of the last N ViT blocks.
# ---------------------------------------------------------------

from typing import List, Optional
from training.mask_classification_semantic import MaskClassificationSemantic


class MaskClassificationSemanticFinetune(MaskClassificationSemantic):
    def __init__(
        self,
        *args,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        unfreeze_at_step: int = 2000,
        **kwargs,
    ):
        """
        Args:
            freeze_backbone: If True, freezes all ViT backbone weights at training start.
            unfreeze_last_n_blocks: After unfreeze_at_step steps, unfreeze the last N
                                    ViT blocks. 0 means never unfreeze (head-only training).
            unfreeze_at_step: Global step at which to unfreeze the last N blocks.
        """
        super().__init__(*args, **kwargs)
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.unfreeze_at_step = unfreeze_at_step
        self._backbone_unfrozen = False

    def on_train_start(self):
        if self.freeze_backbone:
            frozen = 0
            for param in self.network.encoder.backbone.parameters():
                param.requires_grad = False
                frozen += param.numel()
            print(f"\n[Finetune] Froze {frozen:,} backbone parameters. Training head only.")

    def on_train_batch_start(self, batch, batch_idx):
        if (
            self.freeze_backbone
            and not self._backbone_unfrozen
            and self.unfreeze_last_n_blocks > 0
            and self.global_step >= self.unfreeze_at_step
        ):
            blocks = self.network.encoder.backbone.blocks
            last_n = blocks[-self.unfreeze_last_n_blocks:]
            unfrozen = 0
            for param in last_n.parameters():
                param.requires_grad = True
                unfrozen += param.numel()
            self._backbone_unfrozen = True
            print(
                f"\n[Finetune] Step {self.global_step}: Unfroze last "
                f"{self.unfreeze_last_n_blocks} ViT blocks ({unfrozen:,} params)."
            )
