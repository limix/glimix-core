from numpy import reshape


def vec(x):
    return reshape(x, (-1,) + x.shape[2:], order="F")


def unvec(x, shape):
    return reshape(x, shape, order="F")
