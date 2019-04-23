from numpy import add
from optimix import Function


class SumMean(Function):
    """
    Sum mean function, 𝐟₀ + 𝐟₁ + ….

    The mathematical representation is

        𝐦 = 𝐟₀ + 𝐟₁ + …

    In other words, it is a sum of mean vectors.

    Example
    -------

    .. doctest::

        >>> from glimix_core.mean import OffsetMean, LinearMean, SumMean
        >>>
        >>> X = [[5.1, 1.0],
        ...      [2.1, -0.2]]
        >>>
        >>> mean0 = LinearMean(X)
        >>> mean0.effsizes = [-1.0, 0.5]
        >>>
        >>> mean1 = OffsetMean(2)
        >>> mean1.offset = 2.0
        >>>
        >>> mean = SumMean([mean0, mean1])
        >>>
        >>> print(mean.value())
        [-2.6 -0.2]
        >>> g = mean.gradient()
        >>> print(g["SumMean[0].effsizes"])
        [[ 5.1  1. ]
         [ 2.1 -0.2]]
        >>> print(g["SumMean[1].offset"])
        [1. 1.]
        >>> mean0.name = "A"
        >>> mean1.name = "B"
        >>> mean.name = "A+B"
        >>> print(mean)
        SumMean(means=...): A+B
          LinearMean(m=2): A
            effsizes: [-1.   0.5]
          OffsetMean(): B
            offset: 2.0
    """

    def __init__(self, means):
        """
        Constructor.

        Parameters
        ----------
        means : list
            List of mean functions.
        """
        self._means = [c for c in means]
        Function.__init__(self, "SumMean", composite=self._means)

    def value(self):
        """
        Sum of mean vectors, 𝐟₀ + 𝐟₁ + ….

        Returns
        -------
        𝐦 : ndarray
            𝐟₀ + 𝐟₁ + ….
        """
        return add.reduce([mean.value() for mean in self._means])

    def gradient(self):
        """
        Sum of mean function derivatives.

        Returns
        -------
        ∂𝐦 : dict
            ∂𝐟₀ + ∂𝐟₁ + ….
        """
        grad = {}
        for i, f in enumerate(self._means):
            for varname, g in f.gradient().items():
                grad[f"{self._name}[{i}].{varname}"] = g
        return grad

    def __str__(self):
        tname = type(self).__name__
        msg = "{}(means=...)".format(tname)
        if self.name is not None:
            msg += ": {}".format(self.name)
        for m in self._means:
            spl = str(m).split("\n")
            msg = msg + "\n" + "\n".join(["  " + s for s in spl])
        return msg
