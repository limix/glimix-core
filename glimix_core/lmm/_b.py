from numpy import asarray


class B:
    """
    Facilitate solving (𝑣₀𝙺 + 𝑣₁𝙸)𝐱 = 𝐲 for 𝐱, where 𝙺 is a symmetric matrix.

    Let 𝚀𝚂𝚀ᵀ = 𝙺, where 𝚀𝚂𝚀ᵀ is the eigendecomposition of 𝙺. Let 𝙳 = (𝑣₀𝚂 + 𝑣₁𝙸) and
    𝙳₀ = (𝑣₀𝚂₀ + 𝑣₁𝙸₀), where 𝚂₀ is the part of 𝚂 with positive values. Let us define:

        𝙱 = 𝚀₀𝙳₀⁻¹𝚀₀ᵀ                    if 𝑣₁=0, and
        𝙱 = 𝚀₀𝙳₀⁻¹𝚀₀ᵀ + 𝑣₁⁻¹(𝙸 - 𝚀₀𝚀₀ᵀ)  if 𝑣₁>0.

    We have 𝐱 = 𝙱𝐲.

    Parameters
    ----------
    Q0 : array_like
        𝚀₀.
    D0 : array_like
        𝙳₀.
    v0
        𝑣₀.
    v1
        𝑣₁.
    """

    def __init__(self, Q0, S0, v0: float, v1: float):

        if v0 < 0.0:
            raise ValueError("Variance `v0` must be non-negative.")

        if v1 < 0.0:
            raise ValueError("Variance `v1` must be non-negative.")

        Q0 = asarray(Q0, float)
        S0 = asarray(S0, float)
        if S0.ndim != 1:
            raise ValueError("`D0` must be an unidimensional array.")

        self._Q0 = Q0
        self._S0 = S0
        self._v0 = v0
        self._v1 = v1
        self._v0S0 = self._v0 * self._S0
        D0 = self._v0S0 + self._v1
        self._Q0D0i = self._Q0 / D0
        self._update_v0 = False
        self._update_v1 = False

    @property
    def v0(self) -> float:
        return self._v0

    @property
    def v1(self) -> float:
        return self._v1

    def set_variances(self, v0: float, v1: float):
        if v0 != self._v0:
            self._v0 = v0
            self._v1 = v1
            self._update_v0 = True
            self._update_v1 = True
        elif v1 != self._v1:
            self._v1 = v1
            self._update_v1 = True

    def dot(self, y):
        """
        Compute 𝙱𝐲.
        """
        from numpy_sugar import epsilon

        if self._update_v0:
            self._v0S0[:] = self._v0 * self._S0
            D0 = self._v0S0 + self._v1
            self._Q0D0i[:] = self._Q0 / D0
            self._update_v0 = self._update_v1 = False

        elif self._update_v1:
            D0 = self._v0S0 + self._v1
            self._Q0D0i[:] = self._Q0 / D0
            self._update_v1 = False

        Q0ty = self._Q0.T @ y
        x = self._Q0D0i @ Q0ty
        # TODO: I should check whether self._v0 is also too small
        if self._v1 > epsilon.small:
            x += (y - self._Q0 @ Q0ty) / self._v1
        return x
