import numpy as np

array = np.arange(100)
mask = np.array([idx % 4 for idx in range(array.shape[0])])
print(mask)


new = array[mask == 1]

print(new)
