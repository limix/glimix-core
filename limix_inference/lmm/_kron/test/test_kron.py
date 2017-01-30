from numpy import nan
from numpy import eye
from numpy import dot
from numpy import copyto
from numpy.random import RandomState

from limix_inference.mean import KronSumMean

def _gower_normalization(K, out=None):
    """Perform Gower normalizion on covariance matrix K.

    The rescaled covariance matrix has sample variance of 1.
    """
    c = (K.shape[0] - 1) / (K.trace() - K.mean(0).sum())
    if out is None:
        return c * K

    copyto(out, K)
    out *= c

def test_kron_two():
    random = RandomState(1)

    # define phenotype
    N = 10
    P = 3
    Y = random.randn(N,P)

    # pheno with missing data
    Ym = Y.copy()
    Im = random.rand(N, P) < 0.2
    Ym[Im] = nan

    # # define fixed effects
    F = 1.*(random.rand(N,2) < 0.5)
    A = eye(P)

    # mean = MeanKronSum(Y, F=F, A=A)
    mean = KronSumMean(F, A)

    # mean_m = MeanKronSum(Ym, F=F, A=A)

    # define row covariance
    f = 10
    X = 1.*(random.rand(N, f)<0.2)
    R = _gower_normalization(dot(X,X.T))
    R+= 1e-4 * eye(N)

    # define col covariances
    # Cg = FreeFormCov(P)
    # Cn = FreeFormCov(P)
    # Cg.setRandomParams()
    # Cn.setRandomParams()
    #
    # # define covariance matrices
    # covar1 = KronCov(Cg, R)
    # covar2 = KronCov(Cn, eye(N))
    # covar  = SumCov(covar1,covar2)
    #
    # # define covariance matrice with missing data
    # Iok = (~Im).reshape(N * P, order='F')
    # covar1_m = KronCov(copy.copy(Cg), R, Iok=Iok)
    # covar2_m = KronCov(copy.copy(Cn), eye(N), Iok=Iok)
    # covar_m  = SumCov(covar1_m,covar2_m)
    #
    # # define gp
    # self._gp = GP(covar=covar, mean=mean)
    # self._gpm = GP(covar=covar_m, mean=mean_m)
    # self._gp2ks = GP2KronSum(Y=Y, F=F, A=A, Cg=Cg, Cn=Cn, R=R)
