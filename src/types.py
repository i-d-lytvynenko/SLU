from pathlib import Path
from typing import Callable, ClassVar, Iterable, List, Optional, Tuple, Type, Union

from torch import device


File = Path
Directory = Path
Device = Union[str, device]


__all__ = (
    'File',
    'Directory',
    'Device',
    'Callable',
    'ClassVar',
    'Iterable',
    'List',
    'Optional',
    'Tuple',
    'Type',
    'Union',
)
