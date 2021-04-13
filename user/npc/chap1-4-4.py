testosterone = data.testosterone()
x = np.hstack(testosterone.tolist())
group1 = np.hstack([[i]*5 for i in range(len(testosterone))])
group2 = np.array(list(range(5))*len(testosterone))
print(len(group1), len(group2), len(x))
# 55 55 55
res = bivariate_k_sample(x, group1, group2, reps=5000, seed=prng)
print("ANOVA p-value:", round(res[0], 4))
# ANOVA p-value: 0.0002
