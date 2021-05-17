import sys

__all__ = ["cached_property"]


if sys.version_info < (3, 8):
    from .cached_property import cached_property
else:
    from functools import cached_property
