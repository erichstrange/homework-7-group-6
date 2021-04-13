import permute.data as data
from permute.core import two_sample
from permute.npc import fwer_minp
import numpy as np

ww = data.mult()
p1 = two_sample(ww.y1[ww.x == 0], ww.y1[ww.x == 1], alternative="two-sided", seed=40929102, keep_dist=True, reps=5000)
p2 = two_sample(ww.y2[ww.x == 0], ww.y2[ww.x == 1], alternative="two-sided", seed=40929102, keep_dist=True, reps=5000)
p3 = two_sample(ww.y3[ww.x == 0], ww.y3[ww.x == 1], alternative="two-sided", seed=40929102, keep_dist=True, reps=5000)

pvalues = np.array([p1[0], p2[0], p3[0]])
distr = np.vstack([p1[2], p2[2], p3[2]]).T
pvalues_adj_fisher = fwer_minp(pvalues, distr, alternatives="two-sided", combine="fisher")
pvalues_adj_liptak = fwer_minp(pvalues, distr, alternatives="two-sided", combine="liptak")
pvalues_adj_tippett = fwer_minp(pvalues, distr, alternatives="two-sided", combine="tippett")
