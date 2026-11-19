#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""keeper2 - keep GPUs on a shared machine without being squeezed out between your own runs.

gpu_keeper.sh (v1) parks a fixed block of memory on each held GPU and leaves RESERVE_GB free for your jobs. While none
of your jobs runs, that free part is open to anybody. keeper2 closes the gap: its holder fills the whole GPU while you
are not using it, and steps aside for your processes only.

    sealed    nothing of yours on the GPU: the holder keeps everything except a small landing strip (256 MiB), which
              is less than a CUDA context needs, so a process of another user cannot even start there
    open      one of your processes shows up (same uid, or a container you declared as yours): the holder sees it
              within a few milliseconds, while the process is still creating its CUDA context, and frees the memory
    leased    `keeper2 run -m 12 -- cmd` asks for 12 GiB explicitly; the holder frees exactly that, waits for nothing
    tracking  with --headroom H the holder closes in again once your job has settled and keeps only H GiB free
    sealed    your process ends: the GPU is full again a few milliseconds later

The holder only reserves memory. It never computes on the GPU and never touches a process of another user.

    keeper2.py up -n 2                    start the keeper, hold 2 GPUs (waits for idle ones)
    keeper2.py status                     what is held, who is on the GPUs, what was blocked
    keeper2.py run -- python bench.py     run a command on a held GPU (sets CUDA_VISIBLE_DEVICES, fills {gpu})
    keeper2.py down                       give everything back

Needs python3 (3.6+) and the NVIDIA driver; no torch, no pip packages. Holders run in docker containers when docker is
usable (as in v1) and as plain processes otherwise. See README.md.
"""
from __future__ import print_function

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import pwd
import re
import signal
import socket
import subprocess
import sys
import time

__version__ = '2.0.0'

MIB = 1 << 20
GIB = 1 << 30
GRAIN = 16 * MIB                 # the holder ignores differences below this
GROW_MIN = 64 * MIB              # ... and does not bother to take back less than this
STATUS_FRESH = 6.0               # seconds after which a holder status counts as stale
SLACK = 16 * MIB                 # per process of yours: memory the driver does not list as that process's own
                                 # (about 8 MiB on a 5090); without it a few jobs of yours would look like an intruder

DEFAULTS = {
    'target': 1,                 # number of usable GPUs to keep
    'candidates': None,          # GPU indices the keeper may claim (None: any)
    'spare': 1,                  # extra GPUs it may claim while held ones are intruded
    'idle_mib': 100,             # a GPU is idle with less than this many MiB in use and no compute process ...
    'idle_secs': 30.0,           # ... for this long; only then it is claimed
    'idle_util': 100,            # ... and a utilisation of at most this (100: do not look at utilisation)
    'landing_mib': None,         # free memory left on a sealed GPU: room for one of your processes to show up.
                                 # None: half a CUDA context (at most 256), so nobody else's context can fit
    'headroom_gib': None,        # None: stay out of the way while your processes run; number: keep only this much free
    'settle_secs': 10.0,         # with headroom: a new process of yours has this long without growth before tracking
    'own_containers': None,      # regex on docker container names that count as yours
    'own_mounts': None,          # host paths; a container that bind-mounts something below one of them is yours
    'intruder_mib': 100,         # another user's memory on a held GPU above this ...
    'intruder_secs': 20.0,       # ... for this long marks the GPU as intruded
    'mode': 'auto',              # docker | host | auto
    'image': 'nvcr.io/nvidia/pytorch:25.06-py3',     # any image with python3; torch is not needed
    'pid_host': True,            # docker mode: run holders with --pid=host so that they can tell whose process it is
    'poll': 1.0,                 # seconds between keeper rounds
    'tick_ms': 10,               # milliseconds between holder rounds
    'chunk_mib': 4096,           # size of the blocks the holder allocates
}
HOLDER_KEYS = ('landing_mib', 'headroom_gib', 'settle_secs', 'own_containers', 'own_mounts', 'intruder_mib',
               'intruder_secs', 'tick_ms', 'chunk_mib')


# ------------------------------------------------------------------------------------------------------------ helpers

class GpuError(Exception):
    pass


class StateError(Exception):
    pass


def user_name():
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return os.environ.get('USER') or str(os.getuid())


def uid_name(uid):
    if uid is None:
        return '?'
    try:
        return pwd.getpwuid(uid).pw_name
    except (KeyError, TypeError):
        return 'uid%s' % uid


def atomic_write(path, text):
    tmp = '%s.%d.tmp' % (path, os.getpid())
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def try_lock(path, exclusive=True):
    """Take a flock on path (created if missing). Returns the fd, or None when another process holds it."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        return fd
    except OSError as e:
        os.close(fd)
        if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
            return None
        raise


def lock_held(path):
    """True while some process holds an exclusive flock on path. This is how liveness is checked everywhere: the
    kernel drops the lock the moment its owner dies, and it works across containers and pid namespaces."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def pid_namespace():
    try:
        return os.readlink('/proc/self/ns/pid')
    except OSError:
        return None


def proc_start(pid):
    """Start time of a running process (clock ticks after boot); None when it is gone or a zombie. Together with the
    pid it names one process, even after the pid has been reused."""
    try:
        with open('/proc/%d/stat' % pid, 'rb') as f:
            data = f.read()
    except (OSError, TypeError, ValueError):
        return None
    rest = data[data.rfind(b')') + 2:].split()
    if not rest or rest[0] in (b'Z', b'X'):
        return None
    try:
        return int(rest[19])
    except (IndexError, ValueError):
        return None


def fmt_mib(mib, dash=True):
    if mib is None or (dash and mib < 1):
        return '-'
    if mib >= 1000:
        return '%.1fG' % (mib / 1024.0)
    return '%dM' % mib


def fmt_age(seconds):
    seconds = int(max(0, seconds))
    if seconds < 3600:
        return '%d:%02d' % (seconds // 60, seconds % 60)
    return '%dh%02d' % (seconds // 3600, seconds % 3600 // 60)


def parse_mem(text):
    """'12' or '12g' -> GiB, '500m' -> MiB, 'all' -> None. Returns MiB."""
    t = str(text).strip().lower()
    if t in ('all', 'max', ''):
        return None
    m = re.match(r'^([0-9]*\.?[0-9]+)\s*(g|gb|gib|m|mb|mib)?$', t)
    if not m:
        raise ValueError('cannot read the memory size %r (examples: 12, 12g, 500m, all)' % text)
    v = float(m.group(1))
    return int(v if (m.group(2) or 'g')[0] == 'm' else v * 1024)


def parse_ids(text):
    """'0,2,4-6' -> [0, 2, 4, 5, 6]"""
    out = []
    for part in str(text).replace(' ', ',').split(','):
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def stamp(t=None):
    """Local time with milliseconds. A holder in a container gets the host's UTC offset from the keeper, so that
    all logs can be read side by side."""
    t = time.time() if t is None else t
    off = os.environ.get('KEEPER2_TZ_OFFSET')
    tm = time.gmtime(t + int(off)) if off else time.localtime(t)
    return '%s.%03d' % (time.strftime('%Y-%m-%d %H:%M:%S', tm), int(t * 1000) % 1000)


class Log(object):
    """Event log: one line per event, rotated at max_bytes with a single backup."""

    def __init__(self, path, echo=False, max_bytes=2 * MIB):
        self.path, self.echo, self.max_bytes = path, echo, max_bytes

    def __call__(self, msg):
        line = '%s %s\n' % (stamp(), msg)
        try:
            if os.path.getsize(self.path) > self.max_bytes:
                os.replace(self.path, self.path + '.1')
        except OSError:
            pass
        try:
            with open(self.path, 'a', encoding='utf-8', errors='replace') as f:
                f.write(line)
        except OSError:
            pass
        if self.echo:
            try:
                sys.stderr.write(line)
                sys.stderr.flush()
            except (OSError, ValueError):
                pass


def default_state():
    """Runtime files live in RAM by default: status files are rewritten all the time, and a full disk must never keep
    a job from asking the holder for memory."""
    return os.environ.get('KEEPER2_STATE') or builtin_state()


def builtin_state():
    if os.path.isdir('/dev/shm') and os.access('/dev/shm', os.W_OK | os.X_OK):
        return '/dev/shm/gpu-keeper2-' + user_name()
    return os.path.expanduser('~/.gpu_keeper2')


class State(object):
    """Layout of the state directory.
        config.json  keeper.log  keeper.pid  daemon.lock  held  usable
        gpu<N>/status.json  holder.log  holder.lock  holder.pid  run.lock  leases/<id>.json  acks/<id>
    """

    def __init__(self, root):
        self.root = os.path.abspath(root)

    def ensure(self, index=None):
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        info = os.stat(self.root)
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise StateError('the state directory %s must belong to you and must not be writable by others '
                             '(chmod go-w it, or point KEEPER2_STATE somewhere else)' % self.root)
        if index is not None:
            for d in (self.leases(index), self.acks(index)):
                os.makedirs(d, mode=0o700, exist_ok=True)

    def p(self, name):
        return os.path.join(self.root, name)

    def gdir(self, i):
        return os.path.join(self.root, 'gpu%d' % i)

    def g(self, i, name):
        return os.path.join(self.root, 'gpu%d' % i, name)

    def leases(self, i):
        return self.g(i, 'leases')

    def acks(self, i):
        return self.g(i, 'acks')

    def gpu_dirs(self):
        out = []
        try:
            for n in os.listdir(self.root):
                m = re.match(r'^gpu(\d+)$', n)
                if m:
                    out.append(int(m.group(1)))
        except OSError:
            pass
        return sorted(out)

    def holder_alive(self, i):
        return lock_held(self.g(i, 'holder.lock'))

    def daemon_alive(self):
        return lock_held(self.p('daemon.lock'))

    def status(self, i, fresh=True):
        st = read_json(self.g(i, 'status.json'))
        if not st or (fresh and time.time() - st.get('ts', 0) > STATUS_FRESH):
            return None
        return st


def load_cfg(st):
    cfg = dict(DEFAULTS)
    cfg.update(read_json(st.p('config.json'), {}) or {})
    return cfg


# ------------------------------------------------------------------------------------- the GPU, through driver libraries

class _Mem1(ctypes.Structure):
    _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]


class _Mem2(ctypes.Structure):
    _fields_ = [('version', ctypes.c_uint), ('total', ctypes.c_ulonglong), ('reserved', ctypes.c_ulonglong),
                ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]


class _Proc2(ctypes.Structure):       # nvmlProcessInfo_t of the _v2/_v3 calls
    _fields_ = [('pid', ctypes.c_uint), ('used', ctypes.c_ulonglong), ('gi', ctypes.c_uint), ('ci', ctypes.c_uint)]


class _Proc1(ctypes.Structure):       # nvmlProcessInfo_t of old drivers
    _fields_ = [('pid', ctypes.c_uint), ('used', ctypes.c_ulonglong)]


class _Util(ctypes.Structure):
    _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]


class Nvml(object):
    """The few NVML calls keeper2 needs, straight from libnvidia-ml.so.1 (ships with the driver)."""
    NA = 0xFFFFFFFFFFFFFFFF
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        try:
            self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        except OSError as e:
            raise GpuError('cannot load libnvidia-ml.so.1 (is the NVIDIA driver installed?): %s' % e)
        self._c('nvmlInit_v2')
        self._pfn = None
        for name, typ in (('nvmlDeviceGetComputeRunningProcesses_v3', _Proc2),
                          ('nvmlDeviceGetComputeRunningProcesses_v2', _Proc2),
                          ('nvmlDeviceGetComputeRunningProcesses', _Proc1)):
            if hasattr(self.lib, name):
                self._pfn, self._ptype = getattr(self.lib, name), typ
                self._pfn.restype = ctypes.c_int
                break
        self._cap = 64
        self._parr = (self._ptype * self._cap)() if self._pfn else None
        self._mem2 = hasattr(self.lib, 'nvmlDeviceGetMemoryInfo_v2')

    def _c(self, name, *args):
        fn = getattr(self.lib, name)
        fn.restype = ctypes.c_int
        rc = fn(*args)
        if rc != 0:
            raise GpuError('%s failed (NVML error %d)' % (name, rc))

    def count(self):
        n = ctypes.c_uint()
        self._c('nvmlDeviceGetCount_v2', ctypes.byref(n))
        return n.value

    def handle(self, index=None, uuid=None):
        h = ctypes.c_void_p()
        if uuid:
            try:
                self._c('nvmlDeviceGetHandleByUUID', ctypes.c_char_p(uuid.encode()), ctypes.byref(h))
                return h
            except GpuError:
                if index is None:
                    raise
        self._c('nvmlDeviceGetHandleByIndex_v2', ctypes.c_uint(index), ctypes.byref(h))
        return h

    def text(self, name, h, size=96):
        buf = ctypes.create_string_buffer(size)
        self._c(name, h, buf, ctypes.c_uint(size))
        return buf.value.decode('utf-8', 'replace')

    def mem(self, h):
        """(total, used, free) in bytes. `used` leaves out what the driver reserves for itself, as nvidia-smi does."""
        if self._mem2:
            m = _Mem2()
            m.version = ctypes.sizeof(_Mem2) | (2 << 24)
            try:
                self._c('nvmlDeviceGetMemoryInfo_v2', h, ctypes.byref(m))
                return m.total, m.used, m.free
            except GpuError:
                self._mem2 = False
        m = _Mem1()
        self._c('nvmlDeviceGetMemoryInfo', h, ctypes.byref(m))
        return m.total, m.used, m.free

    def raw_used(self, h):
        m = _Mem1()
        self._c('nvmlDeviceGetMemoryInfo', h, ctypes.byref(m))
        return m.used

    def procs(self, h):
        """{pid: bytes or None} of the compute processes on the GPU."""
        if self._pfn is None:
            raise GpuError('this driver has no process list call')
        for _ in range(4):
            n = ctypes.c_uint(self._cap)
            rc = self._pfn(h, ctypes.byref(n), self._parr)
            if rc == 7:                                  # NVML_ERROR_INSUFFICIENT_SIZE
                self._cap = max(self._cap * 2, n.value + 16)
                self._parr = (self._ptype * self._cap)()
                continue
            if rc != 0:
                raise GpuError('process list failed (NVML error %d)' % rc)
            arr = self._parr
            return {arr[i].pid: (None if arr[i].used == self.NA else arr[i].used) for i in range(n.value)}
        raise GpuError('process list keeps growing')

    def util(self, h):
        u = _Util()
        try:
            self._c('nvmlDeviceGetUtilizationRates', h, ctypes.byref(u))
            return u.gpu
        except GpuError:
            return None

    def compute_mode(self, h):
        m = ctypes.c_int()
        try:
            self._c('nvmlDeviceGetComputeMode', h, ctypes.byref(m))
            return m.value
        except GpuError:
            return 0


class CudaDevice(object):
    """One CUDA context on one GPU, used to reserve and release device memory. libcuda.so.1 ships with the driver, and
    the nvidia container runtime puts it into every container that gets a GPU, so no toolkit and no torch is needed."""
    OOM = 2

    def __init__(self, index, uuid):
        self.nv = Nvml.get()
        self.h = self.nv.handle(index, uuid)
        before = self.nv.raw_used(self.h)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid or str(index)
        try:
            lib = ctypes.CDLL('libcuda.so.1')
        except OSError as e:
            raise GpuError('cannot load libcuda.so.1: %s' % e)
        self.lib = lib
        self._alloc = lib.cuMemAlloc_v2
        self._alloc.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t]
        self._alloc.restype = ctypes.c_int
        self._free = lib.cuMemFree_v2
        self._free.argtypes = [ctypes.c_ulonglong]
        self._free.restype = ctypes.c_int
        self._info = lib.cuMemGetInfo_v2
        self._info.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self._info.restype = ctypes.c_int
        self._f, self._t = ctypes.c_size_t(), ctypes.c_size_t()
        self._cu('cuInit', ctypes.c_uint(0))
        dev = ctypes.c_int()
        self._cu('cuDeviceGet', ctypes.byref(dev), ctypes.c_int(0))
        self.cuctx = ctypes.c_void_p()
        self._cu('cuCtxCreate_v2', ctypes.byref(self.cuctx), ctypes.c_uint(0), dev)
        self.free()
        self.total = self._t.value                       # what CUDA can hand out on this GPU
        ctx = self.nv.raw_used(self.h) - before          # memory the context itself takes (about 0.5 GiB on a 5090)
        try:
            listed = self.nv.procs(self.h).get(os.getpid())
        except GpuError:
            listed = None
        if listed and 32 * MIB <= listed <= 2 * GIB and not listed <= ctx <= listed + 32 * MIB:
            ctx = listed                                 # somebody else allocated or freed while we measured
        self.ctx = ctx if 32 * MIB <= ctx <= 2 * GIB else self.total - self._f.value
        if not 32 * MIB <= self.ctx <= 2 * GIB:
            self.ctx = 512 * MIB

    def _cu(self, name, *args):
        fn = getattr(self.lib, name)
        fn.restype = ctypes.c_int
        rc = fn(*args)
        if rc != 0:
            raise GpuError('%s failed (CUDA error %d%s)' % (name, rc, ': out of memory' if rc == self.OOM else ''))

    def free(self):
        rc = self._info(ctypes.byref(self._f), ctypes.byref(self._t))
        if rc != 0:
            raise GpuError('cuMemGetInfo failed (CUDA error %d)' % rc)
        return self._f.value

    def alloc(self, nbytes):
        """Reserve nbytes. Returns a handle, or None when the GPU does not have that much left."""
        p = ctypes.c_ulonglong()
        rc = self._alloc(ctypes.byref(p), nbytes)
        if rc == 0:
            return p.value
        if rc == self.OOM:
            return None
        raise GpuError('cuMemAlloc failed (CUDA error %d)' % rc)

    def release(self, handle):
        rc = self._free(handle)
        if rc != 0:
            raise GpuError('cuMemFree failed (CUDA error %d)' % rc)

    def procs(self):
        return self.nv.procs(self.h)

    def close(self):
        try:
            self.lib.cuCtxDestroy_v2(self.cuctx)
        except Exception:
            pass


class RealBackend(object):
    """Everything keeper2 asks of the machine. tests/simgpu.py provides the same interface without hardware."""

    def list_gpus(self):
        nv = Nvml.get()
        out = []
        for i in range(nv.count()):
            h = nv.handle(i)
            total, used, free = nv.mem(h)
            try:
                procs = nv.procs(h)
            except GpuError:
                procs = {}
            out.append({'index': i, 'uuid': nv.text('nvmlDeviceGetUUID', h), 'name': nv.text('nvmlDeviceGetName', h),
                        'total_mib': total // MIB, 'used_mib': used // MIB, 'free_mib': free // MIB,
                        'util': nv.util(h), 'mode': nv.compute_mode(h),
                        'procs': {pid: (None if b is None else b // MIB) for pid, b in procs.items()}})
        return out

    def open_device(self, index, uuid):
        return CudaDevice(index, uuid)

    def proc_info(self, pid):
        """Who runs this pid: uid, docker container id (if any) and command line. None when it is gone."""
        try:
            uid = os.stat('/proc/%d' % pid).st_uid
        except OSError:
            return None
        cid, cmd = None, ''
        try:
            with open('/proc/%d/cgroup' % pid) as f:
                m = re.search(r'[0-9a-f]{64}', f.read())
            cid = m.group(0) if m else None
        except OSError:
            pass
        try:
            with open('/proc/%d/cmdline' % pid, 'rb') as f:
                cmd = ' '.join(f.read(400).replace(b'\0', b' ').decode('utf-8', 'replace').split())
        except OSError:
            pass
        return {'uid': uid, 'cid': cid, 'cmd': cmd}

    def container_info(self, cid):
        """Name and bind-mount sources of a docker container, asked from the docker socket (takes about a
        millisecond). None when docker cannot be asked."""
        path = os.environ.get('DOCKER_HOST', 'unix:///var/run/docker.sock')
        if not path.startswith('unix://'):
            return None
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(path[len('unix://'):])
            s.sendall(('GET /containers/%s/json HTTP/1.0\r\nHost: docker\r\n\r\n' % cid).encode())
            data = b''
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
            s.close()
            head, _, body = data.partition(b'\r\n\r\n')
            if b' 200 ' not in head.split(b'\r\n', 1)[0]:
                return None
            j = json.loads(body.decode('utf-8', 'replace'))
            return {'name': (j.get('Name') or '').lstrip('/'),
                    'mounts': [m.get('Source') for m in (j.get('Mounts') or []) if m.get('Source')]}
        except (OSError, ValueError):
            return None


_backend = None


def backend():
    """The machine interface. KEEPER2_BACKEND=<file.py> swaps in another one (the tests use a simulated GPU)."""
    global _backend
    if _backend is None:
        path = os.environ.get('KEEPER2_BACKEND')
        if path:
            import importlib.util
            spec = importlib.util.spec_from_file_location('keeper2_backend', path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _backend = mod.Backend()
        else:
            _backend = RealBackend()
    return _backend


# ------------------------------------------------------------------------------------------------------------- holder

class Lease(object):
    __slots__ = ('name', 'path', 'nbytes', 'ttl', 'created', 'cmd', 'fd', 'acked', 'mib', 'pid', 'start')


PIDNS = pid_namespace()


class Holder(object):
    """One per held GPU. Every tick (10 ms) it looks at the free memory, the leases and the processes on the GPU,
    decides how much room your jobs are entitled to, and resizes its reservation to everything else.

    room      what the holder must leave alone: `landing` when sealed, the leased amount (+landing) with leases,
              the whole GPU while your processes run, or their usage + headroom when tracking
    max_hold  = total - own context - room. The holder shrinks to this at once: that is the only reason it ever
              gives memory back, so memory taken by another user's process never makes it retreat.
    Growing is opportunistic: it takes whatever is free beyond the part of `room` that is still unused."""

    def __init__(self, st, index, dev, procs_on, opts, log):
        self.st, self.index, self.dev, self.log = st, index, dev, log
        self.procs_on = procs_on
        self.uid = os.getuid()
        self.opts = dict(DEFAULTS)
        self.opts.update(opts or {})
        self.blocks, self.held = [], 0
        self.leases = {}
        self.cls = {}                    # pid -> what we know about the process
        self.cids = {}                   # container id -> (mine?, name)
        self.own, self.foreign = {}, {}  # pid -> bytes, as of the last look
        self.blocked = 0                 # processes of other users that came and left without getting anywhere
        self.stop = False
        self.state = 'starting'
        self.started = time.time()
        self.peak_own = 0
        self._cfg_next = 0.0
        self._cfg_mtime = None
        self._lmtime, self._lnext = None, 0.0
        self._pfree, self._pnext, self._chase_until = None, 0.0, 0.0
        self._burst, self._ogrow, self._opeak = False, 0.0, 0
        self._fsince, self._intruded, self._intr_logged = None, False, False
        self._skey, self._snext, self._smin = None, 0.0, 0.0
        self._warned = set()
        self._perr, self._plist_ok, self._unknown = None, False, False
        self.lock_fd = None              # set by holder_main
        self.apply_opts()

    # -- settings: start values come on the command line, config.json overrides them while the keeper runs
    def apply_opts(self):
        o = self.opts
        if o['landing_mib'] is None:
            self.landing = max(32 * MIB, min(256 * MIB, self.dev.ctx // 2))
        else:
            self.landing = max(0, int(o['landing_mib'])) * MIB
        self.headroom = None if o['headroom_gib'] is None else int(float(o['headroom_gib']) * GIB)
        self.settle = float(o['settle_secs'])
        self.tick = max(1, int(o['tick_ms'])) / 1000.0
        self.chunk = max(GRAIN, int(o['chunk_mib']) * MIB)
        self.intruder_bytes = int(o['intruder_mib']) * MIB
        self.intruder_secs = float(o['intruder_secs'])
        try:
            self.own_re = re.compile(o['own_containers']) if o['own_containers'] else None
        except re.error:
            self.own_re = None
        self.own_mounts = [os.path.normpath(p) for p in (o['own_mounts'] or []) if p]

    def load_cfg(self):
        try:
            m = os.stat(self.st.p('config.json')).st_mtime_ns
        except OSError:
            return
        if m == self._cfg_mtime:
            return
        self._cfg_mtime = m
        cfg = read_json(self.st.p('config.json'))
        if not isinstance(cfg, dict):
            return
        new = dict(self.opts)
        new.update({k: cfg[k] for k in HOLDER_KEYS if k in cfg})
        if new != self.opts:
            self.opts = new
            self.apply_opts()
            self.cids.clear()
            for pid in [p for p, c in self.cls.items() if not c['own']]:
                del self.cls[pid]                        # look at them again with the new rules ...
            self.foreign, self._pfree = {}, None         # ... in this very round
            self.log('settings changed: landing %d MiB, headroom %s' % (
                self.landing // MIB, 'off' if self.headroom is None else '%.1f GiB' % (self.headroom / GIB)))

    # -- leases
    def scan_leases(self, mono):
        d = self.st.leases(self.index)
        try:
            m = os.stat(d).st_mtime_ns
        except OSError:
            try:
                self.st.ensure(self.index)
            except (OSError, StateError):
                pass
            m = None
        if m != self._lmtime or mono >= self._lnext:
            self._lmtime, self._lnext = m, mono + 0.5
            try:
                names = set(n for n in os.listdir(d) if n.endswith('.json'))
            except OSError:
                names = set()
            for n in names - set(self.leases):
                self.add_lease(d, n)
            for n in set(self.leases) - names:
                self.drop_lease(n, 'released')
        wall = None
        for n, l in list(self.leases.items()):
            if l.ttl > 0:
                wall = wall or time.time()
                if wall >= l.created + l.ttl:
                    self.drop_lease(n, 'expired')
            elif l.pid is not None:                      # the command keeps its pid through exec
                if proc_start(l.pid) != l.start:
                    self.drop_lease(n, 'ended')
            elif l.fd is not None:
                try:
                    fcntl.flock(l.fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except OSError:
                    continue                             # still locked: the command that owns the lease is alive
                self.drop_lease(n, 'ended')

    def add_lease(self, d, name):
        path = os.path.join(d, name)
        j = read_json(path)
        if not isinstance(j, dict):
            return
        l = Lease()
        l.name, l.path = name, path
        l.mib = j.get('mib')
        l.nbytes = None if l.mib is None else int(l.mib) * MIB
        l.ttl = float(j.get('ttl') or 0)
        l.created = float(j.get('created') or time.time())
        l.cmd = j.get('cmd') or ''
        l.acked = False
        l.fd, l.pid, l.start = None, None, None
        if l.ttl <= 0:
            # Where the holder sees the command's process, the lease lives exactly as long as that process: a
            # background child that inherited the lock does not keep it, and a command that closes its file
            # descriptors (sudo) does not end it. Elsewhere the lock on the lease file decides.
            pid, start = j.get('pid'), j.get('start')
            if PIDNS and j.get('pidns') == PIDNS and isinstance(pid, int) and start is not None \
                    and proc_start(pid) == start:
                l.pid, l.start = pid, start
            try:
                l.fd = os.open(path, os.O_RDONLY)
            except OSError:
                return
        self.leases[name] = l
        self.log('lease %s: %s for %s' % (name[:-5], 'the whole GPU' if l.nbytes is None else '%d MiB' % l.mib,
                                          l.cmd[:80] or '?'))

    def drop_lease(self, name, why):
        l = self.leases.pop(name)
        if l.fd is not None:
            try:
                os.close(l.fd)
            except OSError:
                pass
        for p in (l.path, os.path.join(self.st.acks(self.index), name[:-5])):
            try:
                os.unlink(p)
            except OSError:
                pass
        self.log('lease %s %s' % (name[:-5], why))

    def ack_leases(self):
        for l in self.leases.values():
            if not l.acked:
                try:
                    with open(os.path.join(self.st.acks(self.index), l.name[:-5]), 'w'):
                        pass
                except OSError:
                    pass
                l.acked = True

    # -- whose process is it
    def classify(self, pid, mono):
        """What we know about a process on the GPU. `retry` is set while the answer is not certain."""
        c = {'first': time.time(), 'peak': 0, 'leases': set(self.leases), 'retry': None}
        info = backend().proc_info(pid)
        if info is None:                 # gone already, or /proc does not show it: somebody else's until we know
            c.update(own=False, why='', uid=None, cmd='', cid=None, retry=mono + 1.0)
            return c
        own, why, sure = info['uid'] == self.uid, 'your uid', True
        if not own and info.get('cid') and (self.own_re or self.own_mounts):
            own, why, sure = self.own_container(info['cid'])
        c.update(own=own, why=why, uid=info['uid'], cmd=info.get('cmd') or '', cid=info.get('cid'),
                 retry=None if sure else mono + 1.0)
        return c

    def own_container(self, cid):
        """(yours?, why, certain?) for a docker container."""
        if cid in self.cids:
            return self.cids[cid] + (True,)
        info = backend().container_info(cid)
        if info is None:                 # docker did not answer: ask again later, and do not remember this answer
            if 'docker' not in self._warned:
                self._warned.add('docker')
                self.log('cannot ask docker about container %s: its processes count as another user\'s until docker '
                         'answers' % cid[:12])
            return False, '', False
        res = (False, '')
        if self.own_re and self.own_re.search(info['name']):
            res = (True, 'container %s' % info['name'])
        else:
            for src in info['mounts']:
                src = os.path.normpath(src)
                if any(src == p or src.startswith(p.rstrip('/') + '/') for p in self.own_mounts):
                    res = (True, 'container %s mounts %s' % (info['name'], src))
                    break
        self.cids[cid] = res
        return res + (True,)

    def covered(self, c):
        """True when the process came while a lease that is still open was there: it counts as that lease's job,
        whoever runs it (a root container started through `run` looks like another user)."""
        return any(n in self.leases for n in c['leases'])

    def announce(self, pid, c, mono):
        if c['own']:
            self._burst, self._ogrow, self._opeak = True, mono, 0
            self.log('yours: pid %d (%s) %s' % (pid, c['why'], c['cmd'][:90]))
        elif self.covered(c):
            self.log('leased: pid %d (%s) %s' % (pid, uid_name(c['uid']), c['cmd'][:90]))
        else:
            self.log('other user: pid %d (%s) %s' % (pid, uid_name(c['uid']), c['cmd'][:90] or '(cannot see it)'))

    def scan_procs(self, mono, free):
        if not self.procs_on:
            return
        if free == self._pfree and mono >= self._chase_until and mono < self._pnext:
            return
        if free != self._pfree:
            # the free memory moved: someone is allocating. A new process is listed some 50 ms after it takes its
            # first few MiB, so keep looking at every tick for a while
            self._chase_until = mono + 1.0
        self._pfree, self._pnext = free, mono + 0.25
        try:
            plist = self.dev.procs()
        except GpuError as e:
            if self._perr is None:
                self._perr = mono
                self.log('process list failed (%s): keeping the last one' % e)
            if not self._plist_ok or mono - self._perr >= 60:
                self.procs_on = False
                self.own, self.foreign, self._unknown = {}, {}, False
                self.log('process list unavailable: only leases open the GPU from now on')
            return
        if self._perr is not None:
            self.log('process list works again')
        self._perr, self._plist_ok = None, True
        me = os.getpid()
        own, foreign, unknown = {}, {}, False
        for pid, used in plist.items():
            if pid == me:
                continue
            c = self.cls.get(pid)
            if c is None:
                c = self.cls[pid] = self.classify(pid, mono)
                self.announce(pid, c, mono)
            elif c['retry'] is not None and mono >= c['retry']:
                old, c = c, self.classify(pid, mono)
                c['first'], c['peak'], c['leases'] = old['first'], old['peak'], old['leases']
                self.cls[pid] = c
                if c['own']:
                    self.announce(pid, c, mono)
            if used is None:                             # the driver does not say how much it uses
                unknown = unknown or c['own'] or self.covered(c)
                used = 0
            if used > c['peak']:
                c['peak'] = used
            (own if c['own'] else foreign)[pid] = used
        for pid in [p for p in self.cls if p not in plist]:
            c = self.cls.pop(pid)
            lived = time.time() - c['first']
            if c['own'] or c['leases']:
                self.log('%s: pid %d gone after %s, peak %d MiB' % ('yours' if c['own'] else 'leased', pid,
                                                                    fmt_age(lived), c['peak'] // MIB))
            elif c['uid'] is not None and lived < 30 and c['peak'] < 700 * MIB:
                self.blocked += 1
                self.log('blocked: pid %d (%s) left after %.1f s with %d MiB at most: %s' % (
                    pid, uid_name(c['uid']), lived, c['peak'] // MIB, c['cmd'][:90]))
            else:
                self.log('other user: pid %d (%s) gone after %s, peak %d MiB' % (
                    pid, uid_name(c['uid']), fmt_age(lived), c['peak'] // MIB))
        self.own, self.foreign, self._unknown = own, foreign, unknown

    # -- the decision
    def room(self, mono, total, mine):
        """Bytes the holder must leave alone, and the name of the state. `mine` is what your jobs use now."""
        need = 0
        for l in self.leases.values():
            need += total if l.nbytes is None else l.nbytes
        loose = [p for p in self.own if not self.covered(self.cls[p])]     # yours, but not started under a lease
        if self.leases and not loose:
            return min(total, need + self.landing), 'leased'
        if loose:
            if self.headroom is None:
                return total, 'open'
            if self._burst:
                if mine > self._opeak + 64 * MIB:
                    self._opeak, self._ogrow = mine, mono
                elif mono - self._ogrow >= self.settle:
                    self._burst = False
            if self._burst:
                return total, 'open'
            return min(total, mine + self.headroom + need), 'tracking'
        self._burst = False
        return self.landing, 'sealed'

    def resize(self, target):
        dev = self.dev
        while self.held > target and self.blocks:
            handle, n = self.blocks.pop()
            dev.release(handle)
            self.held -= n
        need = target - self.held
        chunk = self.chunk
        while need >= GRAIN:
            n = min(chunk, need) // GRAIN * GRAIN
            handle = dev.alloc(n)
            if handle is None:                           # somebody was faster: try smaller pieces, then give up
                if n <= GRAIN:
                    break
                chunk = max(GRAIN, n // 2)
                continue
            self.blocks.append((handle, n))
            self.held += n
            need -= n

    def step(self, mono):
        dev = self.dev
        total = dev.total
        if mono >= self._cfg_next:
            self._cfg_next = mono + 1.0
            self.load_cfg()
            if self.lock_fd is not None:
                self.keep_lock()
        free = dev.free()
        self.scan_leases(mono)
        self.scan_procs(mono, free)
        others = max(0, total - free - self.held - dev.ctx)      # everything on the GPU that is not this holder
        if self.procs_on:
            leased = [b for p, b in self.foreign.items() if self.covered(self.cls[p])]
            mine = sum(self.own.values()) + sum(leased)
            if self._unknown:            # the driver does not say how much one of yours uses: the rest is yours
                fgn = sum(self.foreign.values()) - sum(leased)
                mine = max(mine, others - fgn)
            else:                        # all that is not yours is somebody else's, listed or not
                fgn = max(0, others - mine - SLACK * (len(self.own) + len(leased)))
        else:                            # blind: outside a lease, whatever is there is not yours
            fgn = 0 if self.leases or others < GRAIN else others
            mine = others - fgn
        room, state = self.room(mono, total, mine)
        max_hold = max(0, total - dev.ctx - room)
        # Growing stops short of the part of `room` your jobs have not used yet. What another user holds does not
        # count against it: their memory must neither make the holder retreat nor make it take your headroom.
        want_free = max(0, room - mine)
        target = min(max_hold, self.held + max(0, free - want_free))
        target -= target % GRAIN
        if self.held - target >= GRAIN or target - self.held >= GROW_MIN:
            self.resize(target)
            free = dev.free()
        self.ack_leases()
        own_used = mine if mine >= GRAIN else 0          # below that it is accounting noise
        if own_used > self.peak_own:
            self.peak_own = own_used
        if fgn >= self.intruder_bytes:                   # another user's process that stays on the GPU
            if self._fsince is None:
                self._fsince = mono
            self._intruded = mono - self._fsince >= self.intruder_secs
        else:
            self._fsince, self._intruded, self._intr_logged = None, False, False
        if self._intruded and not self._intr_logged:
            self._intr_logged = True
            who = ', '.join('pid %d (%s) %d MiB' % (p, uid_name(self.cls[p]['uid']), b // MIB)
                            for p, b in self.foreign.items() if not self.covered(self.cls[p]))
            self.log('INTRUDED: another user is on this GPU: %s' % (who or '%d MiB that no listed process '
                                                                     'accounts for' % (fgn // MIB)))
        if state != self.state:
            self.log('%s -> %s (reserved %d MiB, free %d MiB)' % (self.state, state, self.held // MIB, free // MIB))
            self.state = state
            self._snext = 0.0
        self.publish(mono, free, room, own_used, fgn if fgn >= GRAIN else 0)

    def publish(self, mono, free, room, own_used, fgn, force=False):
        key = (self.state, self.held // (64 * MIB), free // (64 * MIB), own_used // (64 * MIB), fgn // (64 * MIB),
               tuple(sorted(self.leases)), tuple(sorted(self.own)), tuple(sorted(self.foreign)),
               self._intruded, self.blocked)
        if not force and (mono < self._smin or (key == self._skey and mono < self._snext)):
            return
        self._skey, self._snext, self._smin = key, mono + 1.0, mono + 0.05
        wall = time.time()
        procs = []
        for pid, b in list(self.own.items()) + list(self.foreign.items()):
            c = self.cls.get(pid)
            if c:
                procs.append({'pid': pid, 'mib': b // MIB, 'own': c['own'], 'uid': c['uid'], 'cmd': c['cmd'][:120],
                              'since': c['first'], 'lease': self.covered(c)})
        data = {
            'version': __version__, 'ts': wall, 'pid': os.getpid(), 'gpu': self.index, 'started': self.started,
            'state': self.state, 'total_mib': self.dev.total // MIB, 'ctx_mib': self.dev.ctx // MIB,
            'held_mib': self.held // MIB, 'free_mib': free // MIB, 'room_mib': room // MIB,
            'own_mib': own_used // MIB, 'foreign_mib': fgn // MIB, 'peak_own_mib': self.peak_own // MIB,
            'intruded': self._intruded, 'blocked': self.blocked, 'procs_visible': self.procs_on,
            'landing_mib': self.landing // MIB,
            'headroom_gib': None if self.headroom is None else self.headroom / float(GIB),
            'leases': [{'id': l.name[:-5], 'mib': l.mib, 'cmd': l.cmd[:120], 'created': l.created, 'ttl': l.ttl}
                       for l in self.leases.values()],
            'procs': procs,
        }
        try:
            atomic_write(self.st.g(self.index, 'status.json'), json.dumps(data))
        except OSError:
            pass

    def keep_lock(self):
        """The lock file is how everybody else knows this holder is alive. If the state directory was removed and
        made again, lock the new file, or the holder would run on unseen."""
        path = self.st.g(self.index, 'holder.lock')
        try:
            if os.stat(path).st_ino == os.fstat(self.lock_fd).st_ino:
                return
        except OSError:
            pass
        try:
            self.st.ensure(self.index)
            fd = try_lock(path)
        except (OSError, StateError):
            return
        if fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = fd
            self._snext = 0.0
            try:
                atomic_write(self.st.g(self.index, 'holder.pid'), str(os.getpid()))
            except OSError:
                pass
            self.log('state directory was replaced: lock taken again')

    def loop(self):
        d = self.dev
        self.log('holder up on GPU %d (pid %d): %d MiB usable, context %d MiB, landing %d MiB, headroom %s, %s' % (
            self.index, os.getpid(), d.total // MIB, d.ctx // MIB, self.landing // MIB,
            'off' if self.headroom is None else '%.1f GiB' % (self.headroom / GIB),
            'sees processes' if self.procs_on else 'cannot see processes (leases only)'))
        while not self.stop:
            self.step(time.monotonic())
            time.sleep(self.tick)

    def shutdown(self):
        try:
            self.resize(0)
        except GpuError:
            pass
        self.state = 'stopped'
        try:
            self.publish(time.monotonic(), 0, 0, 0, 0, force=True)
        except Exception:
            pass
        self.log('holder down on GPU %d' % self.index)


def holder_main(args):
    st = State(args.state)
    st.ensure(args.gpu)
    log = Log(st.g(args.gpu, 'holder.log'), echo=True)
    lock = None
    for _ in range(40):                                  # `status` and the keeper peek at the lock now and then
        lock = try_lock(st.g(args.gpu, 'holder.lock'))
        if lock is not None:
            break
        time.sleep(0.05)
    if lock is None:
        log('another holder already runs for GPU %d' % args.gpu)
        time.sleep(5)
        return 1
    try:
        atomic_write(st.g(args.gpu, 'holder.pid'), str(os.getpid()))
    except OSError:
        pass
    try:
        dev = backend().open_device(args.gpu, args.uuid)
    except GpuError as e:
        log('cannot take GPU %d: %s' % (args.gpu, e))
        try:                                             # tells the keeper at once, so that it does not wait for us
            atomic_write(st.g(args.gpu, 'status.json'), json.dumps({
                'ts': time.time(), 'pid': os.getpid(), 'gpu': args.gpu, 'state': 'failed', 'error': str(e)}))
        except OSError:
            pass
        time.sleep(5)                                    # no hot restart loop
        return 1
    try:
        opts = json.loads(args.opts) if args.opts else {}
    except ValueError:
        opts = {}
    h = Holder(st, args.gpu, dev, args.procs == 'on', opts, log)
    h.lock_fd = lock

    def on_signal(signum, frame):
        h.stop = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    code = 0
    try:
        h.loop()
    except GpuError as e:
        log('GPU error: %s' % e)
        code = 1
    except Exception as e:                               # never die silently
        import traceback
        log('holder crashed: %r\n%s' % (e, traceback.format_exc()))
        code = 1
    finally:
        h.shutdown()
        dev.close()
    return code


# ------------------------------------------------------------------------------------------------------------- keeper

def run_cmd(cmd, timeout=60):
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True)
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        return 124, '', 'timed out: %s' % ' '.join(cmd[:4])
    except OSError as e:
        return 127, '', str(e)


def docker_version(timeout=90):
    """(major, minor) of the docker server, or None when docker cannot be used here. A daemon that does not answer
    is an error, not "no docker": guessing wrong would start holders in the wrong mode."""
    rc, out, err = run_cmd(['docker', 'info', '--format', '{{.ServerVersion}}'], timeout=timeout)
    if rc == 124:
        raise SystemExit('keeper2: docker did not answer within %d s. Try again, or pass --mode host' % timeout)
    if rc != 0:
        return None
    m = re.match(r'(\d+)\.(\d+)', out.strip())
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def container_name(st, index):
    """keeper2-<user>-g<N>; a keeper with a state directory of its own gets its own names."""
    tag = '' if st.root == os.path.abspath(builtin_state()) else '-' + hashlib.sha1(st.root.encode()).hexdigest()[:6]
    return 'keeper2-%s%s-g%d' % (re.sub(r'[^a-zA-Z0-9_.-]', '_', user_name()), tag, index)


def holder_containers(st):
    """(names of the holder containers that belong to this state directory, what went wrong asking docker)."""
    rc, out, err = run_cmd(['docker', 'ps', '-a', '--filter', 'label=keeper2.state=%s' % st.root,
                            '--format', '{{.Names}}'], timeout=120)
    if rc == 0:
        return out.split(), None
    return [], 'timeout' if rc == 124 else 'no docker' if rc == 127 else (err.strip()[-200:] or 'exit code %d' % rc)


def holder_opts(cfg):
    return json.dumps({k: cfg[k] for k in HOLDER_KEYS}, sort_keys=True)


_children = []                   # host-mode holders started by this process, reaped by the keeper


def holder_ok(s):
    """A holder status that stands for a working holder."""
    return bool(s) and s.get('state') not in ('failed', 'stopped')


def start_holder(st, cfg, gpu, log, abort=None):
    """Start the holder of one GPU and wait until it is up. Returns True on success."""
    i = gpu['index']
    st.ensure(i)
    for name in ('status.json', 'holder.pid'):           # what a predecessor left must not pass for this holder
        try:
            os.unlink(st.g(i, name))
        except OSError:
            pass
    script = os.path.realpath(__file__)                  # a symlink would point nowhere inside the container
    base = ['_holder', '--state', st.root, '--gpu', str(i), '--uuid', gpu['uuid'], '--opts', holder_opts(cfg)]
    if cfg['mode'] == 'docker':
        name = container_name(st, i)
        run_cmd(['docker', 'rm', '-f', name])
        # the GPU is named by its UUID: a container that restarts after a reboot must get the same card again
        cmd = ['docker', 'run', '-d', '--name', name, '--label', 'keeper2.state=%s' % st.root,
               '--restart', 'unless-stopped', '--gpus', 'device=%s' % (gpu['uuid'] or i),
               '--network', 'none', '--log-driver', 'json-file', '--log-opt', 'max-size=1m', '--log-opt', 'max-file=1',
               '-u', '%d:%d' % (os.getuid(), os.getgid()), '-e', 'HOME=/tmp', '-e', 'PYTHONUNBUFFERED=1',
               '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'KEEPER2_TZ_OFFSET=%d' % time.localtime().tm_gmtoff,
               '-v', '%s:/opt/keeper2:ro' % os.path.dirname(script)]
        shm = st.root.startswith('/dev/shm/')
        cmd += ['-v', '/dev/shm:/dev/shm' if shm else '%s:%s' % (st.root, st.root)]
        if cfg['pid_host']:
            cmd += ['--pid=host']
            if cfg.get('cgroupns'):                      # so that /proc/<pid>/cgroup names the container of a process
                cmd += ['--cgroupns=host']
        sock = '/var/run/docker.sock'                    # lets the holder ask which container a process belongs to
        if cfg['pid_host'] and os.access(sock, os.R_OK | os.W_OK):
            cmd += ['-v', '%s:%s' % (sock, sock), '--group-add', str(os.stat(sock).st_gid)]
        cmd += ['--entrypoint', 'python3', cfg['image'], '/opt/keeper2/' + os.path.basename(script)] + base
        cmd += ['--procs', 'on' if cfg['pid_host'] else 'off']
        rc, out, err = run_cmd(cmd, timeout=120)
        if rc != 0:
            log('GPU %d: docker could not start the holder: %s' % (i, (err or out).strip()[-300:]))
            run_cmd(['docker', 'rm', '-f', name])
            return False
        child = None
    else:
        try:
            out = open(st.g(i, 'holder.out'), 'a')
            child = subprocess.Popen([sys.executable, script] + base + ['--procs', 'on'], stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=out, start_new_session=True)
            out.close()
            _children.append(child)
        except OSError as e:
            log('GPU %d: could not start the holder: %s' % (i, e))
            return False
    why = 'it did not report within 90 s'
    deadline = time.time() + 90                          # docker can be slow on a busy machine
    look = time.time() + 2
    while time.time() < deadline:
        s = st.status(i)
        if s and s.get('state') == 'failed':
            why = s.get('error') or 'it failed'
            break
        if holder_ok(s) and st.holder_alive(i):
            return True
        if abort and abort():
            why = 'the keeper is stopping'
            break
        if child is not None and child.poll() is not None:
            why = 'it exited with code %s' % child.returncode
            break
        if child is None and time.time() >= look:        # a container that keeps failing is restarted by docker
            look = time.time() + 2
            rc, out, _ = run_cmd(['docker', 'inspect', '-f', '{{.State.Status}} {{.RestartCount}}',
                                  container_name(st, i)], timeout=30)
            words = out.split()
            if rc == 0 and len(words) == 2 and (words[0] in ('exited', 'dead') or words[1] != '0'):
                why = 'its container is %s after %s restart(s)' % (words[0], words[1])
                break
        time.sleep(0.1)
    log('GPU %d: the holder did not come up: %s (see %s)' % (i, why, st.g(i, 'holder.log')))
    stop_holder(st, cfg, i)
    return False


def stop_holder(st, cfg, i):
    if cfg['mode'] == 'docker':
        run_cmd(['docker', 'rm', '-f', container_name(st, i)], timeout=120)
    pid = None
    try:
        with open(st.g(i, 'holder.pid')) as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        pass
    deadline = time.time() + 5
    sent = False
    while st.holder_alive(i) and time.time() < deadline:
        if pid and not sent and cfg['mode'] != 'docker':
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
            sent = True
        time.sleep(0.05)
    if st.holder_alive(i) and pid and cfg['mode'] != 'docker':
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        time.sleep(0.2)
    for name in ('status.json', 'holder.pid'):
        try:
            os.unlink(st.g(i, name))
        except OSError:
            pass


class Daemon(object):
    """The keeper: claims idle GPUs until `target` usable ones are held, restarts holders that died, and replaces a
    GPU that another user got into (at most `spare` extra ones), giving the surplus back afterwards."""

    def __init__(self, st, log):
        self.st, self.log = st, log
        self.stop = False
        self.held = []                   # GPU indices in the order they were claimed
        self.idle_since = {}
        self.fail = {}                   # index -> (count, retry not before)
        self.dead_since = {}             # index -> when its holder was first seen not running
        self.last_lists = None
        self.last_cfg = None
        self.noted = {}
        self.cfg = None                  # the settings as last read; written back if config.json disappears
        self.lock_fd = None              # set by daemon_main
        # Nothing is taken over from an earlier keeper's `held` list: only holders that are running are adopted.
        # A GPU whose holder died while no keeper watched it may belong to somebody else by now.

    def note(self, key, msg):
        """Log a condition once, and again only after it changed."""
        if self.noted.get(key) != msg:
            self.noted[key] = msg
            if msg:
                self.log(msg)

    def activity(self, i):
        """True while something of yours is on GPU i: such a GPU is never given back."""
        s = self.st.status(i)
        return bool(s and s.get('state') in ('open', 'tracking', 'leased'))

    def release(self, cfg, i, why):
        """Give GPU i back unless something of yours is on it. The run lock keeps a `run` from starting there
        in the meantime."""
        try:
            self.st.ensure(i)
            fd = try_lock(self.st.g(i, 'run.lock'))
        except (OSError, StateError):
            fd = None
        if fd is None:
            return False
        try:
            if self.activity(i):
                return False
            stop_holder(self.st, cfg, i)
        finally:
            os.close(fd)
        self.held.remove(i)
        self.log('GPU %d: released (%s)' % (i, why))
        return True

    def settings(self):
        raw = read_json(self.st.p('config.json'))
        if isinstance(raw, dict):
            cfg = dict(DEFAULTS)
            cfg.update(raw)
        elif self.cfg is not None:                       # the state directory was wiped: keep going as before
            cfg = self.cfg
            try:
                self.st.ensure()
                atomic_write(self.st.p('config.json'), json.dumps(cfg, indent=1, sort_keys=True))
                self.log('config.json was missing: written again from the settings in use')
            except (OSError, StateError):
                pass
        else:
            cfg = dict(DEFAULTS)
        self.cfg = cfg
        return cfg

    def keep_lock(self):
        """Like Holder.keep_lock: stay visible to `status` and `down` after the state directory was replaced."""
        path = self.st.p('daemon.lock')
        try:
            if os.stat(path).st_ino == os.fstat(self.lock_fd).st_ino:
                return
        except OSError:
            pass
        try:
            self.st.ensure()
            fd = try_lock(path)
        except (OSError, StateError):
            return
        if fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = fd
            self.last_lists = None
            try:
                atomic_write(self.st.p('keeper.pid'), str(os.getpid()))
            except OSError:
                pass
            self.log('state directory was replaced: lock taken again')

    def still_idle(self, i, cfg):
        """Look at GPU i once more right before claiming it: claiming others may have taken a while."""
        for g in backend().list_gpus():
            if g['index'] == i:
                return g['used_mib'] < cfg['idle_mib'] and not g['procs'] and g
        return None

    def backoff(self, i):
        count = self.fail.get(i, (0, 0))[0]
        self.fail[i] = (count + 1, time.time() + min(60, 2 ** count))

    def write_lists(self, usable):
        lists = (tuple(self.held), tuple(usable))
        if lists != self.last_lists:
            self.last_lists = lists
            atomic_write(self.st.p('held'), ''.join('%d\n' % i for i in self.held))
            atomic_write(self.st.p('usable'), ''.join('%d\n' % i for i in usable))

    def step(self):
        st, log = self.st, self.log
        cfg = self.settings()
        if self.lock_fd is not None:
            self.keep_lock()
        _children[:] = [c for c in _children if c.poll() is None]
        shown = {k: cfg[k] for k in ('target', 'candidates', 'spare', 'headroom_gib', 'landing_mib')}
        if shown != self.last_cfg:
            if self.last_cfg is not None:
                log('settings: %s' % json.dumps(shown, sort_keys=True))
            self.last_cfg = shown
        gpus = {g['index']: g for g in backend().list_gpus()}
        now = time.time()
        cand = [i for i in sorted(gpus) if cfg['candidates'] is None or i in cfg['candidates']]
        for i in sorted(gpus):                           # holders that run already (keeper restart, reboot)
            if i not in self.held and st.holder_alive(i):
                self.held.append(i)
                log('GPU %d: found a running holder, keeping it' % i)
        self.held = [i for i in self.held if i in gpus]
        for i, g in gpus.items():
            calm = cfg['idle_util'] >= 100 or (g.get('util') or 0) <= cfg['idle_util']
            if i not in self.held and g['used_mib'] < cfg['idle_mib'] and not g['procs'] and calm:
                self.idle_since.setdefault(i, now)
            else:
                self.idle_since.pop(i, None)
        for i in list(self.held):                        # a holder that died comes back
            if st.holder_alive(i):
                s = st.status(i)
                if s and s.get('state') == 'failed':     # e.g. restarted by docker after a reboot, and failing
                    stop_holder(st, cfg, i)
                    self.held.remove(i)
                    self.backoff(i)
                    log('GPU %d: the holder fails (%s); GPU given back' % (i, s.get('error')))
                else:
                    self.fail.pop(i, None)
                    self.dead_since.pop(i, None)
                continue
            # a holder re-takes its lock within a second when the state directory was replaced: give it the time
            if now - self.dead_since.setdefault(i, now) < 2.0:
                continue
            if self.stop or time.time() < self.fail.get(i, (0, 0))[1]:
                continue
            if start_holder(st, cfg, gpus[i], log, abort=lambda: self.stop):
                log('GPU %d: holder was not running, restarted' % i)
                self.fail.pop(i, None)
                self.dead_since.pop(i, None)
            else:
                self.backoff(i)
        stats = {i: st.status(i) for i in self.held}
        intruded = [i for i in self.held if stats[i] and stats[i].get('intruded')]
        for i in self.held:
            s = stats[i]
            who = ''
            if i in intruded:
                who = ', '.join(sorted(set(uid_name(p['uid']) for p in s.get('procs', [])
                                           if not p['own'] and not p.get('lease')))) or '?'
            self.note(('intr', i), 'GPU %d: intruded by %s; not usable until they leave' % (i, who) if who else '')
        usable = [i for i in self.held if holder_ok(stats[i]) and i not in intruded]
        # a held GPU that is no longer a candidate goes back, once nothing of yours runs there
        for i in [i for i in self.held if i not in cand]:
            if self.release(cfg, i, 'not a candidate any more') and i in usable:
                usable.remove(i)
        # claim
        want = cfg['target'] - len(usable)
        cap = cfg['target'] + max(0, cfg['spare']) - len(self.held)
        if want > 0 and cap > 0:
            ready = [i for i in cand if i not in self.held and gpus[i].get('mode', 0) == 0 and i in self.idle_since
                     and now - self.idle_since[i] >= cfg['idle_secs'] and now >= self.fail.get(i, (0, 0))[1]]
            ready.sort(key=lambda i: ((gpus[i].get('util') or 0) > 50, i))      # quiet GPUs first
            for i in ready:
                if self.stop:
                    break
                g = self.still_idle(i, cfg)
                if not g:
                    self.idle_since.pop(i, None)
                    continue
                if start_holder(st, cfg, g, log, abort=lambda: self.stop):
                    self.held.append(i)
                    usable.append(i)
                    self.idle_since.pop(i, None)
                    self.fail.pop(i, None)
                    log('GPU %d: claimed (%d of %d)' % (i, len(usable), cfg['target']))
                    want -= 1
                    cap -= 1
                else:
                    self.backoff(i)
                if want <= 0 or cap <= 0:
                    break
        self.note('wait', 'waiting for %d more idle GPU(s) among %s' % (want, cand) if want > 0 and cap > 0 else '')
        # surplus: more usable GPUs than asked for (target lowered, or an intruder left)
        extra = len(usable) - cfg['target']
        for i in reversed(list(self.held)):
            if extra <= 0:
                break
            if i in usable and self.release(cfg, i, 'surplus'):
                usable.remove(i)
                extra -= 1
        over = len(self.held) - cfg['target'] - max(0, cfg['spare'])
        for i in reversed(list(self.held)):
            if over <= 0:
                break
            if self.release(cfg, i, 'over the limit'):
                if i in usable:
                    usable.remove(i)
                over -= 1
        self.write_lists(sorted(usable))
        return cfg

    def loop(self):
        while not self.stop:
            poll = 1.0
            try:
                poll = max(0.05, float(self.step()['poll']))
            except Exception as e:                       # a docker or driver hiccup must not end the keeper
                self.note('error', 'error: %r' % (e,))
            else:
                self.note('error', '')
            end = time.monotonic() + poll
            while not self.stop and time.monotonic() < end:
                time.sleep(min(0.2, poll))


def daemon_main(args):
    st = State(args.state)
    st.ensure()
    lock = try_lock(st.p('daemon.lock'))
    if lock is None:
        print('keeper2 already runs with this state directory', file=sys.stderr)
        return 1
    log = Log(st.p('keeper.log'), echo=args.foreground)
    atomic_write(st.p('keeper.pid'), str(os.getpid()))
    d = Daemon(st, log)
    d.lock_fd = lock

    def on_signal(signum, frame):
        d.stop = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    cfg = load_cfg(st)
    log('keeper2 %s up (pid %d): target %d, candidates %s, %s mode, state %s' % (
        __version__, os.getpid(), cfg['target'], cfg['candidates'] or 'all', cfg['mode'], st.root))
    try:
        d.loop()
    finally:
        log('keeper stopped (holders keep running; `keeper2.py down` frees the GPUs)')
        try:
            os.unlink(st.p('keeper.pid'))
        except OSError:
            pass
    return 0


# ----------------------------------------------------------------------------------------------------------- commands

def cmd_up(args):
    st = State(args.state)
    st.ensure()
    running = st.daemon_alive()
    live = running or any(st.holder_alive(i) for i in st.gpu_dirs())
    cfg = load_cfg(st) if live else dict(DEFAULTS)      # a fresh start does not inherit old settings
    if args.target is not None:
        cfg['target'] = args.target
    if args.candidates is not None:
        cfg['candidates'] = None if args.candidates.lower() in ('all', 'any', '') else parse_ids(args.candidates)
    if args.headroom is not None:
        if args.headroom.lower() in ('off', 'none'):
            cfg['headroom_gib'] = None
        else:
            mib = parse_mem(args.headroom)
            if mib is None:
                raise SystemExit('keeper2: --headroom takes a size such as 4 or 512m, or "off"')
            cfg['headroom_gib'] = mib / 1024.0
    if args.own_containers is not None:
        if args.own_containers:
            re.compile(args.own_containers)
        cfg['own_containers'] = args.own_containers or None
    if args.own_mounts is not None:
        cfg['own_mounts'] = [os.path.abspath(p) for p in args.own_mounts.split(',') if p] or None
    for key in ('spare', 'idle_secs', 'idle_mib', 'idle_util', 'landing_mib', 'settle_secs', 'intruder_mib',
                'intruder_secs', 'image', 'poll', 'tick_ms'):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v
    if args.no_pid_host:
        cfg['pid_host'] = False
    if args.mode is not None or cfg['mode'] == 'auto':
        want = args.mode or 'auto'
        ver = None if want == 'host' else docker_version()
        if want == 'docker' and ver is None:
            raise SystemExit('keeper2: docker cannot be used here (`docker info` fails)')
        mode = 'host' if ver is None else 'docker'
        if live and cfg['mode'] in ('docker', 'host') and mode != cfg['mode']:
            raise SystemExit('keeper2: holders run in %s mode; run `down` before changing the mode' % cfg['mode'])
        cfg['mode'] = mode
        cfg['cgroupns'] = bool(ver and ver >= (20, 10))  # docker 20.10 knows --cgroupns
    if cfg['target'] < 0:
        raise SystemExit('keeper2: --target must not be negative')
    if cfg['mode'] == 'docker' and run_cmd(['docker', 'image', 'inspect', cfg['image']], timeout=60)[0] != 0:
        raise SystemExit('keeper2: the docker image %s is not on this machine. `docker pull` it first, or pass '
                         '--image with an image that has python3' % cfg['image'])
    try:
        gpus = backend().list_gpus()
    except GpuError as e:
        raise SystemExit('keeper2: %s' % e)
    bad = [i for i in (cfg['candidates'] or []) if i not in [g['index'] for g in gpus]]
    if bad:
        raise SystemExit('keeper2: no such GPU: %s (this machine has %d)' % (bad, len(gpus)))
    atomic_write(st.p('config.json'), json.dumps(cfg, indent=1, sort_keys=True))
    if running:
        print('keeper2 already runs; settings updated.')
    else:
        argv = [sys.executable, os.path.abspath(__file__), '_daemon', '--state', st.root]
        if args.foreground:
            os.execv(sys.executable, argv + ['--foreground'])
        out = open(st.p('keeper.out'), 'a')
        subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=out, start_new_session=True)
        out.close()
        deadline = time.time() + 10
        while not st.daemon_alive() and time.time() < deadline:
            time.sleep(0.05)
        if not st.daemon_alive():
            raise SystemExit('keeper2: the keeper did not start, see %s' % st.p('keeper.out'))
        print('keeper2 started: target %d GPU(s), %s mode, state in %s' % (cfg['target'], cfg['mode'], st.root))
    time.sleep(min(3.0, 1.0 + float(cfg['poll'])))
    print_status(st)
    return 0


def cmd_down(args):
    st = State(args.state)
    cfg = load_cfg(st)
    if st.daemon_alive():
        try:
            with open(st.p('keeper.pid')) as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
        except (OSError, ValueError):
            pid = None
        deadline = time.time() + 10
        while st.daemon_alive() and time.time() < deadline:
            time.sleep(0.05)
        if st.daemon_alive() and pid:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            time.sleep(0.3)
        print('keeper stopped')
    released = []
    # Containers first, whatever the settings say (they may be gone after a reboot): a holder container that is
    # merely killed would be started again by docker.
    names, problem = holder_containers(st)
    if problem and (cfg['mode'] == 'docker' or problem == 'timeout'):
        print('keeper2: could not ask docker for the holder containers (%s). If they still run, try `down` again.'
              % problem, file=sys.stderr)
    for name in names:
        run_cmd(['docker', 'rm', '-f', name], timeout=120)
        m = re.search(r'-g(\d+)$', name)
        if m:
            released.append(int(m.group(1)))
    for i in st.gpu_dirs():
        if st.holder_alive(i) and i not in released:
            released.append(i)
        stop_holder(st, dict(cfg, mode='host'), i)
        for d in (st.leases(i), st.acks(i)):
            try:
                for n in os.listdir(d):
                    os.unlink(os.path.join(d, n))
            except OSError:
                pass
    for name in ('held', 'usable'):
        try:
            atomic_write(st.p(name), '')
        except OSError:
            pass
    print('released GPU(s): %s' % (', '.join(map(str, sorted(released))) or 'none were held'))
    return 0


def gather(st):
    """Everything `status` shows, as a dict."""
    cfg = load_cfg(st)
    try:
        gpus = backend().list_gpus()
        err = None
    except GpuError as e:
        gpus, err = [], str(e)
    held = set(i for i in st.gpu_dirs() if st.holder_alive(i))
    rows = []
    for g in gpus:
        i = g['index']
        row = {'gpu': i, 'held': i in held, 'total_mib': g['total_mib'], 'used_mib': g['used_mib'],
               'free_mib': g['free_mib'], 'util': g.get('util'), 'holder': None, 'others': []}
        if i in held:
            row['holder'] = st.status(i)
        else:
            for pid, mib in sorted(g['procs'].items()):
                info = backend().proc_info(pid) or {}
                row['others'].append({'pid': pid, 'mib': mib, 'user': uid_name(info.get('uid')) if info else '?',
                                      'cmd': info.get('cmd', '')[:60]})
        rows.append(row)
    usable = [r['gpu'] for r in rows if r['held'] and holder_ok(r['holder']) and not r['holder'].get('intruded')]
    return {'version': __version__, 'state_dir': st.root, 'daemon': st.daemon_alive(), 'config': cfg, 'error': err,
            'held': sorted(held), 'usable': usable, 'gpus': rows}


def print_status(st):
    info = gather(st)
    cfg = info['config']
    print('keeper2 %s   keeper: %s   target %d (+%d spare)   %s mode   state %s' % (
        info['version'], 'running' if info['daemon'] else 'NOT running', cfg['target'], cfg['spare'], cfg['mode'],
        info['state_dir']))
    if info['error']:
        print('  cannot read the GPUs: %s' % info['error'])
    if info['held'] and not info['daemon']:
        print('  holders run without the keeper: GPUs stay held, but none is claimed or replaced. Start it with `up`.')
    missing = cfg['target'] - len(info['usable'])
    if info['daemon'] and missing > 0:
        print('  waiting for %d more idle GPU(s)%s: a GPU is claimed after %g s with less than %d MiB in use' % (
            missing, '' if cfg['candidates'] is None else ' among %s' % ','.join(map(str, cfg['candidates'])),
            cfg['idle_secs'], cfg['idle_mib']))
    fmt = '%-4s %-9s %8s %8s %8s %8s  %s'
    print(fmt % ('GPU', 'STATE', 'RESERVED', 'YOURS', 'OTHERS', 'FREE', 'NOTES'))
    now = time.time()
    for r in info['gpus']:
        s = r['holder']
        if not r['held']:
            who = ', '.join('%s %s' % (o['user'], fmt_mib(o['mib'])) for o in r['others'])
            print(fmt % (r['gpu'], '-', '-', '-', fmt_mib(r['used_mib'] if r['used_mib'] >= 16 else 0),
                         fmt_mib(r['free_mib'], False), who))
            continue
        if s is None:
            print(fmt % (r['gpu'], 'starting', '-', '-', '-', fmt_mib(r['free_mib'], False), 'holder has not reported yet'))
            continue
        notes = []
        for l in s.get('leases', []):
            notes.append('lease %s %s (%s)' % ('all' if l['mib'] is None else fmt_mib(l['mib']),
                                               fmt_age(now - l['created']), l['cmd'][:40]))
        for p in s.get('procs', []):
            whose = 'yours' if p['own'] else 'leased' if p.get('lease') else uid_name(p['uid'])
            notes.append('%s pid %d %s %s' % (whose, p['pid'], fmt_mib(p['mib'], False), p['cmd'][:40]))
        if s.get('blocked'):
            notes.append('blocked %d' % s['blocked'])
        if not s.get('procs_visible'):
            notes.append('leases only')
        print(fmt % (r['gpu'], 'INTRUDED' if s.get('intruded') else s['state'], fmt_mib(s['held_mib']),
                     fmt_mib(s['own_mib']), fmt_mib(s['foreign_mib']), fmt_mib(s['free_mib'], False),
                     '; '.join(notes)))
    return info


def cmd_status(args):
    st = State(args.state)
    if args.json:
        print(json.dumps(gather(st), indent=1, sort_keys=True))
    else:
        print_status(st)
    return 0


def cmd_gpus(args):
    info = gather(State(args.state))
    ids = info['held'] if args.all else info['usable']
    print(','.join(map(str, ids)))
    return 0 if ids else 1


def cmd_log(args):
    st = State(args.state)
    lines = []
    for path, tag in [(st.p('keeper.log'), 'keeper')] + [(st.g(i, 'holder.log'), 'gpu%d' % i) for i in st.gpu_dirs()]:
        for p in (path + '.1', path):
            try:
                with open(p, encoding='utf-8', errors='replace') as f:
                    for ln in f:
                        if len(ln) > 24 and ln[4] == '-' and ln[19] == '.':
                            lines.append((ln[:23], tag, ln[24:].rstrip('\n')))
                        elif lines:
                            lines.append((lines[-1][0], tag, ln.rstrip('\n')))
            except OSError:
                pass
    lines.sort(key=lambda x: x[0])
    for ts, tag, msg in lines[-args.lines:]:
        print('%s %-6s %s' % (ts, tag, msg))
    return 0


def live_holders(st):
    """{gpu index: status} of the GPUs whose holder is alive and reporting."""
    out = {}
    for i in st.gpu_dirs():
        if st.holder_alive(i):
            s = st.status(i)
            if holder_ok(s):
                out[i] = s
    return out


def make_lease(st, i, mib, cmd, ttl=0.0):
    """Write a lease file for GPU i. Without a ttl the lease lives as long as the returned fd stays open (the holder
    watches the flock on it), so it ends with the command, however the command ends."""
    st.ensure(i)
    lid = '%d-%d-%s' % (int(time.time()), os.getpid(), os.urandom(3).hex())
    d = st.leases(i)
    tmp = os.path.join(d, '.%s.tmp' % lid)
    fd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, json.dumps({'id': lid, 'mib': mib, 'ttl': ttl, 'created': time.time(), 'pid': os.getpid(),
                             'start': proc_start(os.getpid()), 'pidns': PIDNS, 'cmd': cmd}).encode())
    if ttl <= 0:
        fcntl.flock(fd, fcntl.LOCK_EX)
    os.rename(tmp, os.path.join(d, lid + '.json'))
    if ttl > 0:
        os.close(fd)
        fd = None
    return lid, fd


def wait_ack(st, pairs, timeout):
    """Wait until the holders have made room for the leases in pairs [(gpu, lease id)]."""
    deadline = time.time() + timeout
    left = list(pairs)
    while left and time.time() < deadline:
        left = [(i, lid) for i, lid in left if not os.path.exists(os.path.join(st.acks(i), lid))]
        if left:
            time.sleep(0.002)
    return left


def pick_gpus(st, want, num, exclusive, wait):
    """Choose GPUs for `run` and lock them. Returns ([(index, lock fd)], live status dict) or exits."""
    deadline = None if wait is None else (time.time() + wait if wait > 0 else float('inf'))
    told = False
    while True:
        live = live_holders(st)
        if want:
            pool = [i for i in want]
        else:
            pool = sorted((i for i, s in live.items() if not s.get('intruded')),
                          key=lambda i: (len(live[i].get('leases', [])), live[i].get('own_mib', 0), i))
        got = []
        if len(pool) >= num:
            for i in pool:
                st.ensure(i)
                fd = try_lock(st.g(i, 'run.lock'), exclusive=exclusive)
                if fd is not None:
                    got.append((i, fd))
                    if len(got) == (len(pool) if want else num):
                        return got, live
            for _, fd in got:
                os.close(fd)
            reason = 'the held GPUs are busy with an exclusive run'
            waitable = True
        else:
            reason = ('no GPU is held (is the keeper running? `keeper2.py status`)' if not live else
                      'only %d usable GPU(s) held, %d wanted' % (len(pool), num))
            waitable = deadline is not None
        if not waitable or (deadline is not None and time.time() >= deadline):
            raise SystemExit('keeper2: %s' % reason)
        if not told:
            told = True
            print('keeper2: %s; waiting' % reason, file=sys.stderr)
        time.sleep(0.2)


def cmd_run(args):
    cmd = list(args.cmd)
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit('keeper2: nothing to run (usage: keeper2.py run [options] -- command ...)')
    if args.num < 1:
        raise SystemExit('keeper2: -n must be at least 1')
    st = State(args.state)
    mib = parse_mem(args.mem)
    want = parse_ids(args.gpu) if args.gpu else None
    got, live = pick_gpus(st, want, args.num, args.exclusive, args.wait)
    ids = [i for i, _ in got]
    shown = ' '.join(cmd)[:200]
    pairs, keep = [], [fd for _, fd in got]
    for i in ids:
        if i in live:
            lid, fd = make_lease(st, i, mib, shown)
            pairs.append((i, lid))
            keep.append(fd)
        else:
            print('keeper2: GPU %d is not held; running there without a lease' % i, file=sys.stderr)
    late = wait_ack(st, pairs, args.ack_timeout)
    if late:
        print('keeper2: no answer from the holder of GPU %s; running anyway' % ','.join(str(i) for i, _ in late),
              file=sys.stderr)
    env = dict(os.environ)
    gpu_list = ','.join(map(str, ids))
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    env['CUDA_VISIBLE_DEVICES'] = gpu_list
    env['KEEPER2_GPU'] = gpu_list
    argv = [a.replace('{gpu}', gpu_list) for a in cmd]
    if args.verbose:
        print('keeper2: GPU %s, %s: %s' % (gpu_list, 'whole GPU' if mib is None else '%d MiB' % mib, ' '.join(argv)),
              file=sys.stderr)
    for fd in keep:                                      # the locks travel with the command and end with it,
        high = fcntl.fcntl(fd, fcntl.F_DUPFD, 100)       # on numbers that scripts do not use (exec 3>...)
        os.set_inheritable(high, True)
        os.close(fd)
    for name in ('SIGPIPE', 'SIGXFSZ'):                  # Python ignores these; the command must not inherit that
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_DFL)
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.execvpe(argv[0], argv, env)
    except OSError as e:
        for i, lid in pairs:
            try:
                os.unlink(os.path.join(st.leases(i), lid + '.json'))
            except OSError:
                pass
        print('keeper2: cannot run %s: %s' % (argv[0], e.strerror), file=sys.stderr)
        return 127 if e.errno == errno.ENOENT else 126


def cmd_lease(args):
    st = State(args.state)
    live = live_holders(st)
    ids = parse_ids(args.gpu) if args.gpu else sorted(live)
    ids = [i for i in ids if i in live]
    if not ids:
        raise SystemExit('keeper2: no held GPU to lease')
    mib = parse_mem(args.mem)
    pairs = [(i, make_lease(st, i, mib, args.note or 'manual lease', ttl=float(args.ttl))[0]) for i in ids]
    wait_ack(st, pairs, 5.0)
    for i, lid in pairs:
        print('%s  GPU %d  %s  %ds' % (lid, i, 'all' if mib is None else '%dM' % mib, args.ttl))
    return 0


def cmd_release(args):
    st = State(args.state)
    n = 0
    for i in st.gpu_dirs():
        try:
            names = os.listdir(st.leases(i))
        except OSError:
            continue
        for name in names:
            if name.endswith('.json') and (args.all or name[:-5] in args.ids):
                try:
                    os.unlink(os.path.join(st.leases(i), name))
                    n += 1
                except OSError:
                    pass
    print('released %d lease(s)' % n)
    return 0 if n or args.all else 1


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--state', default=default_state(), metavar='DIR',
                        help='state directory (default: %(default)s; env KEEPER2_STATE)')
    ap = argparse.ArgumentParser(prog='keeper2.py', description='Hold GPUs on a shared machine and step aside only '
                                 'for your own jobs.')
    ap.add_argument('--version', action='version', version='keeper2 ' + __version__)
    sub = ap.add_subparsers(dest='command', metavar='{up,down,status,run,lease,release,gpus,log}')

    p = sub.add_parser('up', parents=[common], help='start the keeper, or change its settings while it runs')
    p.add_argument('-n', '--target', type=int, help='GPUs to keep (default 1)')
    p.add_argument('-c', '--candidates', metavar='IDS', help='only claim among these GPUs, e.g. 0,2,4-7 (default all)')
    p.add_argument('--headroom', metavar='GIB', help='once your job has settled, keep only this much free above its '
                   'usage instead of the whole GPU ("off": default)')
    p.add_argument('--own-containers', metavar='REGEX', help='docker containers with a matching name are yours')
    p.add_argument('--own-mounts', metavar='PATHS', help='docker containers that bind-mount something below one of '
                   'these comma-separated host paths are yours')
    p.add_argument('--spare', type=int, help='extra GPUs to claim while held ones are intruded (default 1)')
    p.add_argument('--idle-secs', type=float, dest='idle_secs', help='claim a GPU after it was idle this long (30)')
    p.add_argument('--idle-mib', type=int, dest='idle_mib', help='idle means less than this many MiB in use (100)')
    p.add_argument('--idle-util', type=int, dest='idle_util', help='... and at most this utilisation in %% (100: any)')
    p.add_argument('--landing-mib', type=int, dest='landing_mib', help='free memory left on a sealed GPU (default: '
                   'half of what a CUDA context takes, at most 256)')
    p.add_argument('--settle-secs', type=float, dest='settle_secs', help='with --headroom: seconds without growth '
                   'before a new job of yours is tracked (10)')
    p.add_argument('--intruder-mib', type=int, dest='intruder_mib', help='another user\'s memory on a held GPU above '
                   'this many MiB ... (100)')
    p.add_argument('--intruder-secs', type=float, dest='intruder_secs', help='... for this long marks the GPU as '
                   'intruded (20)')
    p.add_argument('--mode', choices=['auto', 'docker', 'host'], help='run holders in docker containers or as plain '
                   'processes (default: docker when it is usable)')
    p.add_argument('--image', help='docker image for the holders; needs python3 only (%s)' % DEFAULTS['image'])
    p.add_argument('--no-pid-host', action='store_true', help='docker mode without --pid=host: the holder cannot see '
                   'whose process it is, so only `run`/`lease` open the GPU')
    p.add_argument('--poll', type=float, help=argparse.SUPPRESS)
    p.add_argument('--tick-ms', type=int, dest='tick_ms', help=argparse.SUPPRESS)
    p.add_argument('--foreground', action='store_true', help='stay in the foreground and log to the terminal')
    p.set_defaults(fn=cmd_up)

    p = sub.add_parser('down', parents=[common], help='stop the keeper and give all GPUs back')
    p.set_defaults(fn=cmd_down)

    p = sub.add_parser('status', parents=[common], help='held GPUs, leases, and who is on each GPU')
    p.add_argument('--json', action='store_true')
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser('run', parents=[common], help='run a command on a held GPU',
                       description='Runs the command on a held GPU: asks the holder for memory first, sets '
                       'CUDA_VISIBLE_DEVICES and replaces {gpu} in the arguments by the GPU index. The lease ends '
                       'when the command ends.')
    p.add_argument('-m', '--mem', default='all', metavar='GIB', help='memory the command needs, e.g. 12 or 500m '
                   '(default: the whole GPU). The rest of the GPU stays reserved.')
    p.add_argument('-g', '--gpu', metavar='IDS', help='use these GPUs (default: the least busy held one)')
    p.add_argument('-n', '--num', type=int, default=1, help='number of GPUs (default 1)')
    p.add_argument('-x', '--exclusive', action='store_true', help='wait until no other `run` uses the GPU and keep '
                   'others out meanwhile (for timing measurements)')
    p.add_argument('-w', '--wait', type=float, nargs='?', const=0.0, default=None, metavar='SECS',
                   help='wait until a GPU is held instead of failing (no value: forever)')
    p.add_argument('--ack-timeout', type=float, default=10.0, help=argparse.SUPPRESS)
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('cmd', nargs=argparse.REMAINDER, help='the command')
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser('lease', parents=[common], help='open room on held GPUs for a fixed time (for jobs you cannot '
                       'start through `run`)')
    p.add_argument('-m', '--mem', default='all', metavar='GIB')
    p.add_argument('-g', '--gpu', metavar='IDS', help='default: all held GPUs')
    p.add_argument('-t', '--ttl', type=float, default=3600, metavar='SECS', help='lifetime in seconds (3600)')
    p.add_argument('--note', help='shown in status')
    p.set_defaults(fn=cmd_lease)

    p = sub.add_parser('release', parents=[common], help='end leases before they run out')
    p.add_argument('ids', nargs='*')
    p.add_argument('-a', '--all', action='store_true')
    p.set_defaults(fn=cmd_release)

    p = sub.add_parser('gpus', parents=[common], help='print the usable held GPU indices, comma-separated')
    p.add_argument('-a', '--all', action='store_true', help='include intruded ones')
    p.set_defaults(fn=cmd_gpus)

    p = sub.add_parser('log', parents=[common], help='recent events of the keeper and the holders')
    p.add_argument('-n', '--lines', type=int, default=40)
    p.set_defaults(fn=cmd_log)

    p = sub.add_parser('_holder', parents=[common])
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--uuid', default='')
    p.add_argument('--procs', choices=['on', 'off'], default='on')
    p.add_argument('--opts', default='')
    p.set_defaults(fn=holder_main)

    p = sub.add_parser('_daemon', parents=[common])
    p.add_argument('--foreground', action='store_true')
    p.set_defaults(fn=daemon_main)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, 'fn', None):
        ap.print_help()
        return 2
    try:
        return args.fn(args) or 0
    except (ValueError, StateError) as e:
        print('keeper2: %s' % e, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
