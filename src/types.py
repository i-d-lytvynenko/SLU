from pathlib import Path
from typing import ClassVar, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch


File = Union[Path, str]
Directory = Union[Path, str]
Device = Union[str, torch.device]


__all__ = (
    'File',
    'Directory',
    'Device',
    'Iterable',
    'List',
    'Optional',
    'Tuple',
    'Union',
    'ClassVar',
)
