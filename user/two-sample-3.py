p, t, distr = two_sample(maleid, femaleid, stat='t', reps=10000,
                         alternative='greater', keep_dist=True, seed=55)
n, bins, patches = plt.hist(distr, 25, histtype='bar', normed=True)
plt.title('Permutation Null Distribution')
# <matplotlib.text.Text object at ...>
plt.axvline(x=t, color='red')
# <matplotlib.lines.Line2D object at ...>
x = np.linspace(stats.t.ppf(0.0001, df),
      stats.t.ppf(0.9999, df), 100)
plt.plot(x, stats.t.pdf(x, df), lw=2, alpha=0.6)
# [<matplotlib.lines.Line2D object at ...>]
plt.show()
