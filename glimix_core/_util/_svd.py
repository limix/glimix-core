class SVD:
    """
    Represents a SVD decomposition with some useful methods.

    Let 𝙰 be a given 𝑚×𝑛 real matrix. The SVD decomposition of 𝙰 is defined by ::

        𝙰 = 𝚄𝚂𝚅ᵀ

    for which 𝚂 is a diagonal matrix.

    Parameters
    ----------
    A
        Matrix 𝙰. Optional if the parameter `USVt` is given.
    USVt
        SVD decomposition of 𝙰. Optional if the parameter `A` is given.
    """

    def __init__(self, A=None, USVt=None):
        from numpy_sugar.linalg import ddot, economic_svd

        if A is None and USVt is None:
            raise ValueError("Both `A` and `USVt` cannot be `None`.")

        if A is None:
            self._US = ddot(USVt[0], USVt[1])
            self._Vt = USVt[2]
            self._A = self._US @ self._Vt
        else:
            USVt = economic_svd(A)
            self._US = ddot(USVt[0], USVt[1])
            self._Vt = USVt[2]
            self._A = A

        self._rank = len(USVt[1])

    @property
    def A(self):
        """
        Get 𝙰.
        """
        return self._A

    @property
    def US(self):
        """
        Get 𝚄𝚂.
        """
        return self._US

    @property
    def Vt(self):
        """
        Get 𝚅ᵀ.
        """
        return self._Vt

    @property
    def rank(self) -> int:
        """
        Get the diagonal size of matrix 𝚂.
        """
        return self._rank
