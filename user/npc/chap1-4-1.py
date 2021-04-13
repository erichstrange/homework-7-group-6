import permute.data as data
from permute.core import one_sample, two_sample
from permute.ksample import k_sample, bivariate_k_sample
from permute.utils import get_prng
import numpy as np

prng = get_prng(2019426)
ipat = data.ipat()
ipat_res = one_sample(ipat.ya, ipat.yb, stat='mean', alternative='greater', seed=prng)
print("P-value:", round(ipat_res[0], 4))
# P-value: 0.0002
