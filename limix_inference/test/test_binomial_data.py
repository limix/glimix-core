import os
import lim
import numpy as np
from lim.tool.normalize import stdnorm
from lim.genetics.qtl import scan
from lim.genetics.phenotype import (NormalPhenotype,
                                    BinomialPhenotype)

dir_path = os.path.dirname(os.path.realpath(__file__))
data = np.load(os.path.join(dir_path, 'data', 'binomial.npz'))
G = data['G']
K = data['K']
ntri = data['ntri']
nsuc = data['nsuc']
covariates = data['covariates']
print(nsuc)
print(ntri)

pheno = BinomialPhenotype(np.asarray(nsuc), np.asarray(ntri))
s = scan(pheno, G, K=K, covariates=covariates, progress=False)
print(s.pvalues())
