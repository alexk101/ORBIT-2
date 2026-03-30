import torch.distributed as dist


def dist_print(*args, **kwargs) -> None:
    """Print only on global rank 0 when distributed is initialized; otherwise print always.

    Use this instead of bare ``print`` for training status so multi-GPU jobs do not
    duplicate lines..
    """
    kwargs.setdefault("flush", True)
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return
    print(*args, **kwargs)
