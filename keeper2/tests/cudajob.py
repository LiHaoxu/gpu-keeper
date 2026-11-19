#!/usr/bin/env python3
"""A minimal GPU job for check_real.py: create a CUDA context, allocate MIB in one call, keep it for SECS seconds.
Needs only python3 and the NVIDIA driver. Prints OK or FAILED; exit code 0, 3 (no context) or 4 (no memory).

    python3 cudajob.py MIB [SECS]        the GPU is the first one in CUDA_VISIBLE_DEVICES
"""
import ctypes
import sys
import time

mib = int(sys.argv[1])
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
cuda = ctypes.CDLL('libcuda.so.1')
if cuda.cuInit(0) != 0:
    print('FAILED: cuInit')
    sys.exit(3)
dev, ctx, ptr = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_ulonglong()
cuda.cuDeviceGet(ctypes.byref(dev), 0)
rc = cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
if rc != 0:
    print('FAILED: no CUDA context (error %d%s)' % (rc, ', out of memory' if rc == 2 else ''))
    sys.exit(3)
cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t]
rc = cuda.cuMemAlloc_v2(ctypes.byref(ptr), mib << 20)
if rc != 0:
    print('FAILED: could not allocate %d MiB (error %d%s)' % (mib, rc, ', out of memory' if rc == 2 else ''))
    sys.exit(4)
print('OK: %d MiB' % mib, flush=True)
time.sleep(secs)
