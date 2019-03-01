def vec(x):
    # from numpy import ravel
    from numpy import reshape

    # return ravel(x, order="F")
    return reshape(x, (-1,) + x.shape[2:], order="F")


def unvec(x, shape):
    from numpy import reshape

    return reshape(x, shape, order="F")
