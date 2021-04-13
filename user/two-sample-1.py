from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from permute.data import macnell2014
ratings = macnell2014()
prof1 = ratings[ratings.tagender==0]
maleid = prof1.overall[prof1.taidgender==1]
femaleid = prof1.overall[prof1.taidgender==0]
df = len(maleid) + len(femaleid) - 2
t, p = stats.ttest_ind(maleid, femaleid)
print('Test statistic:', np.round(t, 5))
# Test statistic: 1.32905
print('P-value (two-sided):', np.round(p, 5))
# P-value (two-sided): 0.20043
