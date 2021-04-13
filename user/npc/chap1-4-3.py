worms = data.worms()
res = k_sample(worms.x, worms.y, stat='one-way anova', seed=prng)
print("ANOVA p-value:", round(res[0], 4))
# ANOVA p-value: 0.0107
