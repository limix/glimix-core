import sys

__all__ = ["cached_property"]


if sys.version_info < (3, 8):
    from .cached_property import cached_property
else:

    def cached_property(func):
        from functools import lru_cache

        return property(lru_cache(maxsize=None)(func))
