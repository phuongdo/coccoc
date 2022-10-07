from __future__ import division
from numba import cuda
import numpy
import math

# CUDA kernel
@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:])')
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp
        
A = numpy.full((24, 12), 3, numpy.float) # matrix containing all 3's
B = numpy.full((12, 22), 4, numpy.float) # matrix containing all 4's
C = numpy.full((24, 22), 0, numpy.float) # matrix containing all 4's

griddim = 1, 2
blockdim = 3, 4
matmul[griddim,blockdim](A,B,C)
print(C)

