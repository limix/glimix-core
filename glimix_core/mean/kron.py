from numpy import asarray, kron, zeros

from glimix_core._util import unvec, vec
from optimix import Function, Vector


class KronMean(Function):
    """
    Kronecker mean function, (A⊗F)vec(B).

    Let

    - n be the number of samples;
    - p the number of traits; and
    - c the number of covariates.

    The mathematical representation is

        𝐦 = (A⊗F)vec(B)

    where A is a p×p trait design matrix of fixed effects and F is a n×c sample design
    matrix of fixed effects. B is a c×p matrix of fixed-effect sizes.
    """

    def __init__(self, A, F):
        """
        Constructor.

        Parameters
        ----------
        A : array_like
            p×p array.
        F : array_like
            n×c array.
        """
        self._A = asarray(A, float)
        self._F = asarray(F, float)
        vecB = zeros((F.shape[1], A.shape[0])).ravel()
        self._vecB = Vector(vecB)
        Function.__init__(self, "KronMean", vecB=self._vecB)

    @property
    def A(self):
        """
        Matrix A.
        """
        return self._A

    @property
    def F(self):
        """
        Matrix F.
        """
        return self._F

    @property
    def AF(self):
        """
        A ⊗ F.
        """
        return kron(self.A, self.F)

    def value(self):
        """
        Kronecker mean function.

        Returns
        -------
        𝐦 : ndarray
            (A⊗F)vec(B).
        """
        return self.AF @ self._vecB.value

    def gradient(self):
        """
        Gradient of the linear mean function.

        Returns
        -------
        vecB : ndarray
            Derivative of M over vec(B).
        """
        return {"vecB": self.AF}

    @property
    def B(self):
        """
        Effect-sizes parameter, B.
        """
        return unvec(self._vecB.value, (self.F.shape[1], self.A.shape[0]))

    @B.setter
    def B(self, v):
        self._vecB.value = vec(asarray(v, float))

    def __str__(self):
        tname = type(self).__name__
        msg = "{}(A=..., B=...)".format(tname)
        if self.name is not None:
            msg += ": {}".format(self.name)
        msg += "\n"
        mat = format(self.B)
        msg += "  B: " + "\n     ".join(mat.split("\n"))
        return msg
