from permute.npc import npc
mb = data.massaro_blair()
sam1 = mb.y[mb.group == 1]
sam2 = mb.y[mb.group == 2]
first_moment = two_sample(sam1, sam2, alternative='two-sided', reps=5000, keep_dist=True, seed=42)
second_moment = two_sample(sam1**2, sam2**2, alternative='two-sided', reps=5000, keep_dist=True, seed=423)
partial_pvalues = np.array([first_moment[0], second_moment[0]])
print("Partial p-values:", round(first_moment[0], 3), round(second_moment[0], 3))
# Partial p-values: 0.017 0.009

npc_distr = np.vstack([first_moment[2], second_moment[2]]).T
global_p = npc(partial_pvalues, npc_distr, alternatives='two-sided')
print("Global p-value:", round(global_p, 4))
# Global p-value: 0.0018
