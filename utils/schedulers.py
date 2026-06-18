"""Custom learning rate schedulers."""

import torch


class WarmupReduceLROnPlateau:
    """Combined warmup + ReduceLROnPlateau scheduler.

    During the warmup phase, linearly increases the LR of designated
    warmup param groups from 0 to their target LR.  After warmup,
    delegates entirely to ReduceLROnPlateau.

    This avoids the conflict where ReduceLROnPlateau would overwrite
    the manually-set warmup LR mid-ramp.

    Usage:
        scheduler = WarmupReduceLROnPlateau(
            optimizer, warmup_epochs=3, warmup_group_indices=[0],
            mode="min", factor=0.5, patience=2,
        )
        for epoch in range(total_epochs):
            train(...)
            val_loss = validate(...)
            scheduler.step(val_loss, epoch=epoch)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        warmup_group_indices: list[int] | None = None,
        **plateau_kwargs,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        # Which param_groups get warmup (default: group 0 = backbone)
        self.warmup_group_indices = warmup_group_indices or [0]

        # Snapshot the target LR for each warmup group *after* optimizer creation
        self._target_lrs = {
            idx: optimizer.param_groups[idx]["lr"]
            for idx in self.warmup_group_indices
        }
        # Set initial LR to 0 for warmup groups
        for idx in self.warmup_group_indices:
            optimizer.param_groups[idx]["lr"] = 0.0

        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **plateau_kwargs,
        )

    def step(self, metrics, epoch: int):
        """Call once per epoch after validation.

        During warmup: linearly ramp warmup-group LRs, skip plateau.
        After warmup: delegate to ReduceLROnPlateau.
        """
        if epoch < self.warmup_epochs:
            # Linear warmup — directly set LR, do NOT call plateau.step()
            factor = (epoch + 1) / self.warmup_epochs
            for idx in self.warmup_group_indices:
                self.optimizer.param_groups[idx]["lr"] = (
                    self._target_lrs[idx] * factor
                )
        else:
            # Post-warmup: let ReduceLROnPlateau manage all LRs freely.
            # At epoch == warmup_epochs the LR is already at target from the
            # last warmup step; plateau will decay from there if needed.
            self.plateau.step(metrics)

    def state_dict(self):
        return {
            "warmup_epochs": self.warmup_epochs,
            "warmup_group_indices": self.warmup_group_indices,
            "target_lrs": self._target_lrs,
            "plateau": self.plateau.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.warmup_epochs = state_dict["warmup_epochs"]
        self.warmup_group_indices = state_dict["warmup_group_indices"]
        self._target_lrs = state_dict["target_lrs"]
        self.plateau.load_state_dict(state_dict["plateau"])
