from numpy import asarray, kron, zeros
from optimix import Function, Vector

from glimix_core._util import unvec, vec


class KronMean(Function):
    """
    Kronecker mean function, (A⊗X)vec(B).

    Let

    - n be the number of samples;
    - p the number of traits; and
    - c the number of covariates.

    The mathematical representation is

        𝐦 = (A⊗X)vec(B)

    where A is a p×p trait design matrix of fixed effects and X is a n×c sample design
    matrix of fixed effects. B is a c×p matrix of fixed-effect sizes.
    """

    def __init__(self, A, X):
        """
        Constructor.

        Parameters
        ----------
        A : array_like
            p×p array.
        X : array_like
            n×c array.
        """
        self._A = asarray(A, float)
        self._X = asarray(X, float)
        vecB = zeros((X.shape[1], A.shape[0])).ravel()
        self._vecB = Vector(vecB)
        self._nparams = vecB.size
        Function.__init__(self, "KronMean", vecB=self._vecB)

    @property
    def nparams(self):
        """
        Number of parameters.
        """
        return self._nparams

    @property
    def A(self):
        """
        Matrix A.
        """
        return self._A

    @property
    def X(self):
        """
        Matrix X.
        """
        return self._X

    @property
    def AX(self):
        """
        A ⊗ X.
        """
        return kron(self.A, self.X)

    def value(self):
        """
        Kronecker mean function.

        Returns
        -------
        𝐦 : ndarray
            (A⊗X)vec(B).
        """
        return self.AX @ self._vecB.value

    def gradient(self):
        """
        Gradient of the linear mean function.

        Returns
        -------
        vecB : ndarray
            Derivative of M over vec(B).
        """
        return {"vecB": self.AX}

    @property
    def B(self):
        """
        Effect-sizes parameter, B.
        """
        return unvec(self._vecB.value, (self.X.shape[1], self.A.shape[0]))

    @B.setter
    def B(self, v):
        self._vecB.value = vec(asarray(v, float))

    def __str__(self):
        tname = type(self).__name__
        msg = "{}(A=..., X=...)".format(tname)
        if self.name is not None:
            msg += ": {}".format(self.name)
        msg += "\n"
        mat = format(self.B)
        msg += "  B: " + "\n     ".join(mat.split("\n"))
        return msg
