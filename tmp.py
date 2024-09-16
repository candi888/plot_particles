import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.grid()

dia = 2

# dummy_circle = patches.Circle((0, 0), radius=radius)
# radius_cir = dummy_circle.get_radius()

point = dia * 1000 / (25.4 * 72)
s = point**2
scat = ax.scatter(0, 4, s=s, linewidths=0)
print(scat.get_pickradius())
print(scat.get_sizes())


fig.savefig("./tmp.png")
