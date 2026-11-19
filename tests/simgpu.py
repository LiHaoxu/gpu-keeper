#!/usr/bin/env python3
"""Simulated GPUs, so that keeper2 can be tested on any machine: no NVIDIA hardware, no docker.

keeper2 loads this file instead of the driver libraries when KEEPER2_BACKEND points at it; KEEPER2_SIM_DIR names the
directory that holds the fake machine (one JSON file, guarded by a lock). The numbers copy what was measured on an
RTX 5090: 32110 MiB usable, a CUDA context of about 500 MiB, and a new process that holds 12 MiB long before its
context is complete.

    python simgpu.py create DIR [--gpus 4] [--mib 32110]
    python simgpu.py job --mib 8000 [--secs 2] [--uid 0] [--cid ID] ...     a fake job; exit 0 = it got its memory
    python simgpu.py show DIR
"""
import argparse
import fcntl
import json
import os
import sys
import time

MIB = 1 << 20
CTX = 500 * MIB              # a complete CUDA context
FIRST = 12 * MIB             # what a process holds as soon as it is listed

_main = sys.modules.get('__main__')
GpuError = getattr(_main, 'GpuError', None) or type('GpuError', (Exception,), {})


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


class Machine(object):
    """The fake machine: `with Machine(dir) as m:` gives exclusive access to its state, m.dirty = True saves it."""

    def __init__(self, root=None):
        self.root = root or os.environ['KEEPER2_SIM_DIR']

    def __enter__(self):
        self.fd = os.open(os.path.join(self.root, 'lock'), os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        with open(os.path.join(self.root, 'sim.json')) as f:
            self.s = json.load(f)
        self.dirty = False
        for g in self.s['gpus'].values():                # the driver frees the memory of a process that died
            for pid in [p for p, e in g['procs'].items() if not e.get('fake') and not _alive(int(p))]:
                del g['procs'][pid]
                self.dirty = True
        return self

    def __exit__(self, *exc):
        if self.dirty and exc[0] is None:
            tmp = os.path.join(self.root, 'sim.json.tmp')
            with open(tmp, 'w') as f:
                json.dump(self.s, f)
            os.replace(tmp, os.path.join(self.root, 'sim.json'))
        os.close(self.fd)

    def gpu(self, index):
        return self.s['gpus'][str(index)]

    @staticmethod
    def free(g):
        return g['total'] - sum(e['used'] for e in g['procs'].values())


def create(root, gpus=4, mib=32110):
    os.makedirs(root, exist_ok=True)
    s = {'gpus': {str(i): {'uuid': 'GPU-sim-%d' % i, 'total': mib * MIB, 'util': 0, 'mode': 0, 'procs': {}}
                  for i in range(gpus)}, 'containers': {}}
    with open(os.path.join(root, 'sim.json'), 'w') as f:
        json.dump(s, f)


def take(index, nbytes, pid=None, root=None, **entry):
    """Allocate nbytes on a GPU for pid. False when the GPU does not have that much left."""
    pid = str(pid or os.getpid())
    with Machine(root) as m:
        g = m.gpu(index)
        if nbytes > 0 and Machine.free(g) < nbytes:
            return False
        e = g['procs'].setdefault(pid, {'used': 0, 'uid': os.getuid(), 'cid': None, 'cmd': ' '.join(sys.argv)[:200]})
        e.update(entry)
        e['used'] += nbytes
        if e['used'] <= 0 and nbytes < 0:
            del g['procs'][pid]
        m.dirty = True
        return True


def leave(index, pid=None, root=None):
    with Machine(root) as m:
        if m.gpu(index)['procs'].pop(str(pid or os.getpid()), None) is not None:
            m.dirty = True


def add_container(cid, name, mounts=(), root=None):
    with Machine(root) as m:
        m.s['containers'][cid] = {'name': name, 'mounts': list(mounts)}
        m.dirty = True


def snapshot(root=None):
    with Machine(root) as m:
        return m.s


class SimDevice(object):
    """What keeper2's holder gets from open_device()."""

    def __init__(self, index):
        self.index = index
        with Machine() as m:
            g = m.gpu(index)
            if g.get('broken'):
                raise GpuError('cuCtxCreate_v2 failed (CUDA error 999: simulated broken GPU)')
            if Machine.free(g) < CTX:
                raise GpuError('cuCtxCreate_v2 failed (CUDA error 2: out of memory)')
            g['procs'][str(os.getpid())] = {'used': CTX, 'uid': os.getuid(), 'cid': None, 'cmd': 'keeper2 holder'}
            m.dirty = True
            self.total = g['total']
        self.ctx = CTX

    def free(self):
        with Machine() as m:
            return Machine.free(m.gpu(self.index))

    def alloc(self, nbytes):
        with Machine() as m:
            g = m.gpu(self.index)
            if Machine.free(g) < nbytes:
                return None
            g['procs'][str(os.getpid())]['used'] += nbytes
            m.dirty = True
            return nbytes

    def release(self, handle):
        with Machine() as m:
            m.gpu(self.index)['procs'][str(os.getpid())]['used'] -= handle
            m.dirty = True

    def procs(self):
        with Machine() as m:
            return {int(p): (None if e.get('na') else e['used']) for p, e in m.gpu(self.index)['procs'].items()}

    def close(self):
        leave(self.index)


class Backend(object):
    def list_gpus(self):
        with Machine() as m:
            out = []
            for i, g in sorted(m.s['gpus'].items(), key=lambda kv: int(kv[0])):
                used = sum(e['used'] for e in g['procs'].values())
                out.append({'index': int(i), 'uuid': g['uuid'], 'name': 'Simulated GPU', 'total_mib': g['total'] // MIB,
                            'used_mib': used // MIB, 'free_mib': (g['total'] - used) // MIB, 'util': g.get('util', 0),
                            'mode': g.get('mode', 0),
                            'procs': {int(p): e['used'] // MIB for p, e in g['procs'].items()}})
            return out

    def open_device(self, index, uuid):
        return SimDevice(index)

    def proc_info(self, pid):
        with Machine() as m:
            for g in m.s['gpus'].values():
                e = g['procs'].get(str(pid))
                if e and e.get('hidden'):                # like hidepid=2, or another pid namespace
                    return None
                if e:
                    return {'uid': e['uid'], 'cid': e.get('cid'), 'cmd': e.get('cmd', '')}
        return None

    def container_info(self, cid):
        with Machine() as m:
            return m.s['containers'].get(cid)


def job(a):
    """A fake GPU job with the start-up behaviour of a real CUDA process."""
    index = a.gpu if a.gpu is not None else int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
    entry = {'uid': os.getuid() if a.uid is None else a.uid, 'cid': a.cid, 'cmd': a.cmd or 'python job.py'}

    def fail(code, what):
        leave(index)
        print('CUDA out of memory (%s)' % what, flush=True)
        sys.exit(code)
    if not take(index, FIRST, **entry):
        fail(3, 'no room to start')
    time.sleep(a.ctx_ms / 1000.0)                        # the driver is still building the context
    if not take(index, CTX - FIRST):
        fail(3, 'context')
    steps = max(1, a.steps)
    for _ in range(steps):
        if not take(index, a.mib * MIB // steps):
            fail(4, 'allocation of %d MiB' % (a.mib // steps))
        if steps > 1:
            time.sleep(a.step_ms / 1000.0)
    print('READY', flush=True)
    end = time.time() + a.secs
    late_at = time.time() + a.late_after if a.late_mib else None
    while time.time() < end:
        if late_at and time.time() >= late_at:
            late_at = None
            if not take(index, a.late_mib * MIB):
                fail(5, 'late allocation of %d MiB' % a.late_mib)
            print('LATE-OK', flush=True)
        time.sleep(0.02)
    leave(index)
    print('OK', flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='what')
    p = sub.add_parser('create')
    p.add_argument('dir')
    p.add_argument('--gpus', type=int, default=4)
    p.add_argument('--mib', type=int, default=32110)
    p = sub.add_parser('show')
    p.add_argument('dir')
    p = sub.add_parser('job')
    p.add_argument('--gpu', type=int)
    p.add_argument('--mib', type=int, default=1000)
    p.add_argument('--secs', type=float, default=0.5)
    p.add_argument('--uid', type=int)
    p.add_argument('--cid')
    p.add_argument('--cmd')
    p.add_argument('--ctx-ms', type=float, default=150)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--step-ms', type=float, default=30)
    p.add_argument('--late-mib', type=int, default=0)
    p.add_argument('--late-after', type=float, default=1.0)
    a = ap.parse_args()
    if a.what == 'create':
        create(a.dir, a.gpus, a.mib)
    elif a.what == 'show':
        print(json.dumps(snapshot(a.dir), indent=1))
    elif a.what == 'job':
        job(a)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
