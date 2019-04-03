from numpy import finfo as _finfo, log as _log

logmax = _log(_finfo(float).max)


def safe_log(x):
    from numpy import log, clip, inf
    from numpy_sugar import epsilon

    return log(clip(x, epsilon.small, inf))
