from enum import Enum
from typing import Union
"""
This module defines the `FusedAttn` enumeration
Classes:
    FusedAttn (Enum): An enumeration representing the modes of fused attention.
        - CK: Represents the "CK" mode using ROCm Composable Kernels.
        - DEFAULT: Represents the "DEFAULT" mode using PyTorch/Triton.
        - NONE: Represents no fused attention.

"""


class FusedAttn(Enum):
    CK = "CK"
    DEFAULT = "DEFAULT"
    NONE = "NONE"


def parse_fused_attn(name: Union[str, FusedAttn]) -> FusedAttn:
    """Resolve a config string (or pass-through enum) to `FusedAttn`.

    Accepts case-insensitive labels: ck, default, none, plus aliases sdpa/pytorch and manual.
    """
    if isinstance(name, FusedAttn):
        return name
    key = name.strip().lower().replace("-", "_")
    mapping = {
        "ck": FusedAttn.CK,
        "default": FusedAttn.DEFAULT,
        "none": FusedAttn.NONE,
        "manual": FusedAttn.NONE,
    }
    if key not in mapping:
        raise ValueError(
            f"Unknown fused_attn {name!r}; expected one of {sorted(mapping.keys())}"
        )
    return mapping[key]

