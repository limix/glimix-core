def vec(x):
    from numpy import ravel

    return ravel(x, order="F")


def unvec(x, shape):
    from numpy import reshape

    return reshape(x, shape, order="F")
