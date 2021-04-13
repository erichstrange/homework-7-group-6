from permute.stratified import sim_corr
p, rho, sim = sim_corr(x=ratings.overall, y=ratings.taidgender, group=ratings.tagender, seed = 25)
n, bins, patches = plt.hist(sim, 40, histtype='bar')
plt.axvline(x=rho, color='red')
# <matplotlib.lines.Line2D object at ...>
plt.show()
