job = data.job()
job_res = two_sample(job.x[job.y == 1], job.x[job.y == 2], stat='mean', reps = 10**5, alternative='greater', seed=prng)
print("P-value:", round(job_res[0], 4))
# P-value: 0.0003
