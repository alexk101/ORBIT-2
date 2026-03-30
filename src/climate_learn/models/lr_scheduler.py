import math
import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between
    warmup_start_lr and base_lr followed by a cosine annealing schedule between base_lr and
    eta_min."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch
                * (base_lr - self.warmup_start_lr)
                / max(1, self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            for base_lr in self.base_lrs
        ]


class LinearWarmupCosineAnnealingLRSteps(_LRScheduler):
    """Linear warmup then cosine decay to ``eta_min``, indexed by **optimizer step** (not epoch).

    ``last_epoch`` follows PyTorch convention: it is incremented on each ``step()`` call
    after ``optimizer.step()`` (initial value ``-1`` before any step).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if warmup_steps > max_steps:
            raise ValueError("warmup_steps must be <= max_steps")
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch < 0:
            return [self.warmup_start_lr] * len(self.base_lrs)

        # Past the planned schedule: hold eta_min (e.g. resumed past max_steps).
        if self.last_epoch >= self.max_steps:
            return [self.eta_min for _ in self.base_lrs]

        if self.warmup_steps > 0 and self.last_epoch < self.warmup_steps:
            denom = max(1, self.warmup_steps - 1)
            return [
                self.warmup_start_lr
                + self.last_epoch
                * (base_lr - self.warmup_start_lr)
                / denom
                for base_lr in self.base_lrs
            ]

        t = self.last_epoch - self.warmup_steps
        t_cosine = max(1, self.max_steps - self.warmup_steps)
        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * t / t_cosine))
            for base_lr in self.base_lrs
        ]


class LinearWarmupLinearSqrtCooldownLRSteps(_LRScheduler):
    """Warmup -> linear decay -> (1 - sqrt) cooldown to zero, per optimizer step."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        cooldown_start_step: int,
        cooldown_start_factor: float = 0.1,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if warmup_steps > max_steps:
            raise ValueError("warmup_steps must be <= max_steps")
        if cooldown_start_step < warmup_steps:
            raise ValueError("cooldown_start_step must be >= warmup_steps")
        if cooldown_start_step >= max_steps:
            raise ValueError("cooldown_start_step must be < max_steps")
        if cooldown_start_factor < 0.0 or cooldown_start_factor > 1.0:
            raise ValueError("cooldown_start_factor must be in [0, 1]")

        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.cooldown_start_step = cooldown_start_step
        self.cooldown_start_factor = cooldown_start_factor
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch < 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch >= self.max_steps:
            return [0.0 for _ in self.base_lrs]

        if self.warmup_steps > 0 and self.last_epoch < self.warmup_steps:
            denom = max(1, self.warmup_steps - 1)
            return [
                self.warmup_start_lr
                + self.last_epoch * (base_lr - self.warmup_start_lr) / denom
                for base_lr in self.base_lrs
            ]

        if self.last_epoch < self.cooldown_start_step:
            linear_steps = max(1, self.cooldown_start_step - self.warmup_steps)
            p = (self.last_epoch - self.warmup_steps) / linear_steps
            linear_scale = 1.0 - p * (1.0 - self.cooldown_start_factor)
            return [base_lr * linear_scale for base_lr in self.base_lrs]

        cooldown_steps = max(1, self.max_steps - self.cooldown_start_step)
        cooldown_step = self.last_epoch - self.cooldown_start_step
        cooldown_progress = min(1.0, max(0.0, cooldown_step / cooldown_steps))
        cooldown_scale = self.cooldown_start_factor * (
            1.0 - math.sqrt(cooldown_progress)
        )
        return [base_lr * cooldown_scale for base_lr in self.base_lrs]
