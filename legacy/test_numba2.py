from numba import cuda

from numba.cuda.random import (create_xoroshiro128p_states,

                               xoroshiro128p_uniform_float32)

import numpy as np


@cuda.jit
def random_3d(arr, rng_states):

    # Per-dimension thread indices and strides

    startx, starty, startz = cuda.grid(3)

    stridex, stridey, stridez = cuda.gridsize(3)


    # Linearized thread index

    tid = (startz * stridey * stridex) + (starty * stridex) + startx


    # Use strided loops over the array to assign a random value to each entry

    for i in range(startz, arr.shape[0], stridez):

        for j in range(starty, arr.shape[1], stridey):

            for k in range(startx, arr.shape[2], stridex):

                arr[i, j, k] = xoroshiro128p_uniform_float32(rng_states, tid)


# Array dimensions

X, Y, Z = 1000, 1000, 1000


# Block and grid dimensions

bx, by, bz = 8, 8, 8

gx, gy, gz = 16, 16, 16


# Total number of threads

nthreads = bx * by * bz * gx * gy * gz


# Initialize a state for each thread

rng_states = create_xoroshiro128p_states(nthreads, seed=1)


# Generate random numbers

arr = cuda.device_array((X, Y, Z), dtype=np.float32)

random_3d[(gx, gy, gz), (bx, by, bz)](arr, rng_states)
