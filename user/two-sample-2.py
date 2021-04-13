from permute.core import two_sample
p, t = two_sample(maleid, femaleid, stat='t', alternative='two-sided', seed=20)
print('Test statistic:', np.round(t, 5))
# Test statistic: 1.32905
print('P-value (two-sided):', np.round(p, 5))
# P-value (two-sided): 0.27918

p, t = two_sample(maleid, femaleid, reps=100, stat='t', alternative='two-sided', seed=20)
print('P-value (two-sided):', np.round(p, 5))
# P-value (two-sided): 0.41584
