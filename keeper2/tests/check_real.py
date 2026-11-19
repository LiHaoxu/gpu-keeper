#!/usr/bin/env python3
"""Check keeper2 on a real GPU. Holds ONE idle GPU for about a minute, runs the cases below on it and gives it back.

    python3 tests/check_real.py --gpu 4                 docker mode (as `up` would choose on this machine)
    python3 tests/check_real.py --gpu 0 --mode host     holders as plain processes

It uses a state directory of its own, so a keeper you already run is left alone (pick a GPU it does not hold).
Cases: seal / your job gets the GPU / sealed again / another user is kept out (docker mode: a root container) /
`run` / `run -m` limit / holder killed / down.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
K2 = os.path.join(os.path.dirname(HERE), 'keeper2.py')
results = []


def sh(cmd, timeout=300):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
    return p.returncode, p.stdout.strip()


def k2(*args):
    return sh([sys.executable, K2] + [str(a) for a in args])


def status(gpu):
    rc, out = k2('status', '--json')
    for row in json.loads(out)['gpus']:
        if row['gpu'] == gpu:
            return row
    raise SystemExit('GPU %d not found' % gpu)


def wait(cond, timeout, what):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return time.time() - t0
        time.sleep(0.05)
    return None


def check(name, ok, detail=''):
    results.append(ok)
    print('%-4s %s%s' % ('ok' if ok else 'FAIL', name, ('   [%s]' % detail) if detail else ''), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--mode', choices=['auto', 'docker', 'host'], default='auto')
    ap.add_argument('--image', default=None, help='docker image with python3 for the holder and the test jobs')
    a = ap.parse_args()
    g = a.gpu
    os.environ['KEEPER2_STATE'] = os.path.join('/dev/shm' if os.path.isdir('/dev/shm') else '/tmp',
                                               'gpu-keeper2-check-%d' % os.getuid())
    row = status(g)
    if row['held'] or row['used_mib'] >= 100:
        raise SystemExit('GPU %d is not idle (%d MiB in use): pick another one' % (g, row['used_mib']))
    total = row['total_mib']
    big = int(total * 0.85)
    up = ['up', '-n', 1, '-c', g, '--idle-secs', 2, '--mode', a.mode] + (['--image', a.image] if a.image else [])
    rc, out = k2(*up)
    if rc != 0:
        raise SystemExit(out)
    cfg = json.loads(k2('status', '--json')[1])['config']
    docker = cfg['mode'] == 'docker'
    print('keeper2 check on GPU %d (%d MiB), %s mode' % (g, total, cfg['mode']))

    def job(mib, secs=0.5, root=False):
        """The command line of a job outside keeper2's knowledge: yours, or (root=True) somebody else's."""
        if not docker:
            return ['env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES=%d' % g, sys.executable,
                    os.path.join(HERE, 'cudajob.py'), str(mib), str(secs)]
        return (['docker', 'run', '--rm', '--gpus', 'device=%d' % g, '--network', 'none'] +
                ([] if root else ['-u', '%d:%d' % (os.getuid(), os.getgid())]) +
                ['-v', '%s:/k2t:ro' % HERE, '--entrypoint', 'python3', cfg['image'], '/k2t/cudajob.py', str(mib), str(secs)])

    def mine():
        """The holder containers of this check (they carry its state directory as a label)."""
        return sh(['docker', 'ps', '-a', '--filter', 'label=keeper2.state=%s' % os.environ['KEEPER2_STATE'],
                   '--format', '{{.Names}}'])[1].split()

    def sealed():
        s = status(g)['holder']
        return bool(s and s['state'] == 'sealed' and s['free_mib'] <= s['landing_mib'] + 48)
    try:
        t = wait(sealed, 120, 'sealed')
        check('an idle GPU is claimed and sealed', t is not None, 'after %.1f s' % t if t else 'never sealed')
        s = status(g)
        check('nvidia-smi view: the GPU is full', s['free_mib'] <= 400, '%d MiB free of %d' % (s['free_mib'], total))

        rc, out = sh(job(big, 1.0))
        check('your job gets %d MiB in one allocation, without any wrapper' % big, rc == 0, out[-80:])
        t = wait(sealed, 10, 'sealed')
        check('sealed again after your job', t is not None, 'visible after %.2f s' % t if t is not None else '')

        if docker:
            rc, out = sh(job(1024, 0.5, root=True))
            check('a root container (= another user) is kept out', rc != 0, out[-80:])
            t = wait(lambda: (status(g)['holder'] or {}).get('blocked', 0) >= 1, 5, 'blocked')
            check('... and counted as blocked', t is not None)
            rc, out = k2('run', '--', *[x.replace('device=%d' % g, 'device={gpu}') for x in job(big, 0.5, root=True)])
            check('the same root container through `keeper2 run` gets %d MiB' % big, rc == 0, out[-80:])
        else:
            rc, out = k2('run', '--', sys.executable, os.path.join(HERE, 'cudajob.py'), big, 0.5)
            check('`keeper2 run` gives the command the GPU', rc == 0, out[-80:])
        wait(sealed, 10, 'sealed')
        inner = job(3000, 0.5, root=True) if docker else [sys.executable, os.path.join(HERE, 'cudajob.py'), '3000', '0.5']
        rc, out = k2('run', '-m', 4, '--', *inner)
        check('run -m 4: a 3000 MiB job fits', rc == 0, out[-80:])
        inner[-2] = '6000'
        rc, out = k2('run', '-m', 4, '--', *inner)
        check('run -m 4: a 6000 MiB job does not', rc == 4, out[-80:])
        t = wait(sealed, 10, 'sealed')
        check('sealed again after the runs', t is not None)

        pid = status(g)['holder']['pid']
        if docker:
            sh(['docker', 'kill'] + mine())
        else:
            os.kill(pid, 9)
        t0 = time.time()
        t = wait(lambda: sealed() and status(g)['holder']['pid'] != pid, 120, 'new holder')
        check('holder killed: a new one seals the GPU', t is not None, 'after %.1f s' % (time.time() - t0))
    finally:
        rc, out = k2('down')
        time.sleep(1.0)
        row = status(g)
        check('down: the GPU is free again', not row['held'] and row['used_mib'] < 100, '%d MiB in use' % row['used_mib'])
        if docker:
            check('down: no holder container left', not mine(), ' '.join(mine()))
    print('%d of %d checks passed' % (sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
