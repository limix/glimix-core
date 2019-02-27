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

    def __init__(self, c, p):
        """
        Constructor.

        Parameters
        ----------
        c : int
            Number of covariates.
        p : int
            Matrix dimension of A.
        """
        vecB = zeros((c, p)).ravel()
        self._c = c
        self._p = p
        self._A = None
        self._F = None
        self._vecB = Vector(vecB)
        Function.__init__(self, "KronMean", vecB=self._vecB)

    @property
    def A(self):
        """
        Matrix A.
        """
        return self._A

    @A.setter
    def A(self, A):
        self._A = A

    @property
    def F(self):
        """
        Matrix F.
        """
        return self._F

    @F.setter
    def F(self, F):
        self._F = F

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
        return unvec(self._vecB.value, (self._c, self._p))

    @B.setter
    def B(self, v):
        self._vecB.value = vec(asarray(v, float))

    def __str__(self):
        tname = type(self).__name__
        p = self._p
        c = self._c
        msg = "{}(c={},p={})".format(tname, c, p)
        if self.name is not None:
            msg += ": {}".format(self.name)
        msg += "\n"
        mat = format(self.B)
        msg += "  B: " + "\n     ".join(mat.split("\n"))
        return msg
