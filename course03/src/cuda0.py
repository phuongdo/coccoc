import numpy as np
from numba import vectorize
@vectorize(['float64(float64, float64)'], target='cuda')
def sum(a, b):
   return a + b

N = 10000
xx = np.random.random(N)
yy = np.random.random(N)
print(sum(xx,yy))