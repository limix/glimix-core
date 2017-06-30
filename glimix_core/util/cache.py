def create_cache():
    from cachetools import LRUCache

    return LRUCache(maxsize=1)


def cached(method):
    from operator import attrgetter
    from cachetools import cachedmethod

    cache_name = '_cache_%s' % method.__name__
    return cachedmethod(attrgetter(cache_name))(method)
