# High-Performance Computing - Boost python with your GPU 


### 

https://docs.google.com/presentation/d/1kqT1HDt3CH9VVUb6jxn30XbQNCIn0nqCDNoZXjTm86o/edit#slide=id.p

###  How to create env


```
conda create -n gpu python=3.9 cupy cudatoolkit=11.4 numba pytorch scipy fastapi uvicorn[standard]
conda activate gpu

```

### Benchark CPU, GPU

```
cd src
uvicorn --workers 1 --host 0.0.0.0 --port 1234 main:app

```

```
autocannon 'http://localhost:1234/test2/compute/cpu' -d 20 -c 7 -w 3  -t 10


autocannon 'http://localhost:1234/test2/compute/gpu' -d 20 -c 7 -w 3  -t 10
```

#### Pass Hacker Demo..

This demo attempts to use brute force to discover the password ( in SHA-1 hash format).
It takes the initial part of the session key (ip:port:salt), appends
a possible secret key, takes the SHA-1 hash, and compares the result to a hash
that is known to be good.

This is a lot of computation, so hasspass offloads the secret key generation,
SHA-1 calculation, and verification to the GPU using PyCUDA.  On my system, with
a GeForce RTX 2060, it's capable of checking about 4M passwords per second.

The password supporse  is 9 characters of lowercase alpha and digits. Thus, a
brute-force crack needs to try 36^9 keys, which take a lot of time to compute ( ~ 20 days).
You can set to 6 for faster compuation.

Run with:

```
cd src
python src/hashpass.py

```
    
