"""
Platypus で同梱している libraw_enhanced をトップレベル import できるようにする。

このリポジトリ内の libraw_enhanced は
`libraw_enhanced/libraw_enhanced/` に実体があるため、
作業ディレクトリがリポジトリルートだと `import libraw_enhanced` が
namespace package として解決され、API(imread 等)が見えなくなることがある。
"""

import sys as _sys

from . import libraw_enhanced as _inner
from .libraw_enhanced import *  # re-export packaged API

__version__ = getattr(_inner, "__version__", None)
_CORE_AVAILABLE = getattr(_inner, "_CORE_AVAILABLE", False)
_core = getattr(_inner, "_core", None)

if _core is not None:
    _sys.modules.setdefault(__name__ + "._core", _core)

del _inner, _sys
