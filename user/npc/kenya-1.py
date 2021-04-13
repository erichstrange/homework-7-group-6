from __future__ import print_function
from matplotlib import mlab
from permute.data import kenya
d = kenya()
print(mlab.rec2txt(d))
# classes   ol_molo   kamba
# 1        12       0
# 2         1       0
# 3         8       6
# 4         2       1
# 5         0       0
# 6         1       0
# 7         1       0
# 8         0       0
# 9         0       0
# 10         2       0
# 11         8      15
# 12         6       0
# 13         0       0
# 14         0       0
# 15         3       1
# 16         1       0

import matplotlib.pyplot as plt
plt.plot(d['ol_molo'])
# [<matplotlib.lines.Line2D object at ...>]
plt.plot(d['kamba'])
# [<matplotlib.lines.Line2D object at ...>]
plt.legend(['ol_molo', 'kamba'])
# <matplotlib.legend.Legend object at ...>
plt.show()
