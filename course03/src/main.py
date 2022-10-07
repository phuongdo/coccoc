from fastapi import FastAPI
import scipy.stats # for definition of gaussian distribution
import numpy as np
app = FastAPI()
from numba import vectorize

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

import math  # Note that for the CUDA target, we need to use the scalar functions from the math module, not NumPy

SQRT_2PI = np.float32((2*math.pi)**0.5)  # Precompute this constant as a float32.  Numba will inline it at compile time.

@vectorize(['float32(float32, float32, float32)'], target='cuda')
def gaussian_pdf(x, mean, sigma):
    '''Compute the value of a Gaussian probability density function at x with given mean and sigma.'''
    return math.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/test1/compute/gpu")
async def compute_gpu(size: int = 1000):
    
    # Evaluate the Gaussian distribution PDF a million times!
    x = np.random.uniform(-3, 3, size=size).astype(np.float32)
    mean = np.float32(0.0)
    sigma = np.float32(1.0)
    gaussian_pdf(x, mean, sigma)
    return {"message": "ok"} 
    # res = {"status": "ok", "value": value}
    # return JSONResponse(content=jsonable_encoder(res))
 
@app.get("/test1/compute/cpu")
async def compute_cpu(size: int = 1000):
    
    # Evaluate the Gaussian distribution PDF a million times!
    x = np.random.uniform(-3, 3, size=size).astype(np.float32)
    mean = np.float32(0.0)
    sigma = np.float32(1.0)
    norm_pdf = scipy.stats.norm
    value = norm_pdf.pdf(x, loc=mean, scale=sigma)
    return {"message": "ok"} 
    # res = {"status": "ok", "value": value}
    # return JSONResponse(content=jsonable_encoder(res))
    
    
import torch
import time

def torch_cos(a,b):
    d = torch.mul(a, b)# Calculate the multiplication of the corresponding elements 
    cos = torch.sum(d, dim=1)
    return cos

def torch_cos_new(a,b):
    cos = torch.cosine_similarity(a,b,dim=1)
    return cos
    

REFER_SIZE = 10000
reference_gpu = torch.randn(REFER_SIZE, 768).cuda()
reference_cpu = np.random.randn(REFER_SIZE, 768)

@app.get("/test2/compute/gpu")
async def test2_compute_gpu(size: int = 10000):
    t = time.time()
    # torch.manual_seed(1234)
    query = torch.randn(1,768).cuda()
    # reference = torch.randn(size, 768).cuda()
    query = torch.div(query,torch.norm(query, dim=1).reshape(-1,1))
    reference = torch.div(reference_gpu,torch.norm(reference_gpu, dim=1).reshape(-1,1))
    cos_torch_new = torch_cos_new(query, reference)
    cos_torch_new = torch.topk(cos_torch_new, 5, dim=0).values.tolist()

    print('fp32 torch_cos time is',time.time()-t)
    print('fp32 result is ',cos_torch_new[0:5])

    del query
    del reference
    # del cos_torch_new
    torch.cuda.empty_cache()
    return {"message": cos_torch_new[0:5]} 

def numpy_cos(a,b):
    dot = a*b # Corresponding to the original multiplication dot.sum(axis=1) Get inner product 
    a_len = np.linalg.norm(a,axis=1)# Vector module length 
    b_len = np.linalg.norm(b,axis=1)
    cos = dot.sum(axis=1)/(a_len*b_len)
    return cos

@app.get("/test2/compute/cpu")
async def test2_compute_cpu(size: int = 1000000):
    # np.random.seed(1234)
    t = time.time()
    query = np.random.randn(1, 768)
    # reference = np.random.randn(size, 768)
    reference = reference_cpu
    query = query/np.linalg.norm(query,axis=1)# Get the unit vector 
    reference = reference/np.linalg.norm(reference,axis=1).reshape(-1,1)# Get the unit vector 
    # for i in range(5):
    cos_numpy = numpy_cos(query,reference)
    np.sort(cos_numpy)[::-1]
    print('numpy_cos average time is ',(time.time()-t))
    return {"message": cos_numpy[0:5].tolist()} 
