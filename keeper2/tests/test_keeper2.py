#!/usr/bin/env python3
"""Tests of keeper2 on simulated GPUs (tests/simgpu.py): no NVIDIA hardware and no docker needed.

    python3 tests/test_keeper2.py            all tests, about two minutes
    python3 tests/test_keeper2.py -v TestRun

Every test starts the real keeper2.py (keeper, holders, `run`) as separate processes against a fake machine with four
32 GiB GPUs, and looks at what ends up on those GPUs.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
K2 = os.path.join(os.path.dirname(HERE), 'keeper2.py')
SIM = os.path.join(HERE, 'simgpu.py')
sys.path.insert(0, HERE)
import simgpu  # noqa: E402

MIB = 1 << 20
TOTAL = 32110            # MiB per simulated GPU
LANDING = 250            # half of the simulated CUDA context
SEALED_MAX = LANDING + 32    # free MiB on a sealed GPU: the landing strip plus rounding
OTHER_UID = 4242


class Base(unittest.TestCase):
    GPUS = 4

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='k2-')
        self.sim = os.path.join(self.tmp, 'sim')
        self.state = os.path.join(self.tmp, 'state')
        simgpu.create(self.sim, self.GPUS, TOTAL)
        self.env = dict(os.environ, KEEPER2_BACKEND=SIM, KEEPER2_SIM_DIR=self.sim, KEEPER2_STATE=self.state)
        self.env.pop('CUDA_VISIBLE_DEVICES', None)
        self.children = []

    def tearDown(self):
        for p in self.children:
            if p.poll() is None:
                p.kill()
            p.wait()
            for f in (p.stdout, p.stderr):
                if f:
                    f.close()
        self.k2('down')
        left = [(i, pid) for i, g in simgpu.snapshot(self.sim)['gpus'].items() for pid, e in g['procs'].items()
                if not e.get('fake')]
        logs = self.logs() if left else ''
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.assertEqual(left, [], 'processes left on the simulated GPUs after `down`\n' + logs)

    # -- running things
    def k2(self, *args, **kw):
        timeout = kw.pop('timeout', 60)
        p = subprocess.run([sys.executable, K2] + [str(a) for a in args], env=self.env, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout, **kw)
        return p

    def up(self, *args):
        base = ['up', '--mode', 'host', '--idle-secs', '0.2', '--poll', '0.1']
        p = self.k2(*(base + list(args)))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p

    def spawn(self, argv, **kw):
        p = subprocess.Popen(argv, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, **kw)
        self.children.append(p)
        return p

    def job(self, *args, **kw):
        """A fake GPU job, started directly (not through `keeper2 run`)."""
        return self.spawn([sys.executable, SIM, 'job'] + [str(a) for a in args], **kw)

    def run_job(self, run_args, job_args):
        """A fake GPU job started through `keeper2 run`."""
        return self.spawn([sys.executable, K2, 'run'] + [str(a) for a in run_args] + ['--', sys.executable, SIM, 'job'] +
                          [str(a) for a in job_args])

    def finish(self, p, timeout=30):
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err

    # -- looking at things
    def free(self, gpu=0):
        g = simgpu.snapshot(self.sim)['gpus'][str(gpu)]
        return (g['total'] - sum(e['used'] for e in g['procs'].values())) // MIB

    def hstat(self, gpu=0):
        try:
            with open(os.path.join(self.state, 'gpu%d' % gpu, 'status.json')) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def state_of(self, gpu=0):
        return self.hstat(gpu).get('state')

    def held(self):
        return json.loads(self.k2('status', '--json').stdout)['held']

    def logs(self):
        return self.k2('log', '-n', '200').stdout

    def wait(self, cond, timeout=10, what='condition'):
        end = time.time() + timeout
        while time.time() < end:
            v = cond()
            if v:
                return v
            time.sleep(0.02)
        self.fail('timed out after %ss waiting for %s\n%s' % (timeout, what, self.logs()))

    def wait_sealed(self, gpu=0, timeout=10):
        self.wait(lambda: self.state_of(gpu) == 'sealed' and self.free(gpu) <= SEALED_MAX, timeout,
                  'GPU %d to be sealed' % gpu)

    def fake_proc(self, gpu, mib, pid=999990, uid=OTHER_UID, **kw):
        """A process of another user that just sits on a GPU (hidden=True: its owner cannot be seen)."""
        ok = simgpu.take(gpu, mib * MIB, pid=pid, root=self.sim, fake=True, uid=uid, cid=None, cmd='their job', **kw)
        self.assertTrue(ok, 'no room for the fake process')

    def drop_proc(self, gpu, pid=999990):
        simgpu.leave(gpu, pid=pid, root=self.sim)

    def wipe_state(self):
        """Delete the state directory under the running holders. They keep writing into it, so a file can appear
        between listing and removing the directory: try again until it is gone."""
        for _ in range(100):
            try:
                shutil.rmtree(self.state)
                return
            except FileNotFoundError:
                return
            except OSError:
                time.sleep(0.01)
        self.fail('could not remove the state directory')

    def set_gpu(self, gpu, **kw):
        s = simgpu.snapshot(self.sim)
        s['gpus'][str(gpu)].update(kw)
        with open(os.path.join(self.sim, 'sim.json'), 'w') as f:
            json.dump(s, f)


class TestHolder(Base):
    """The holder alone: seal, step aside for you, stand firm against anybody else."""

    def test_seals_an_idle_gpu(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        self.assertGreaterEqual(self.free(0), LANDING - 16)
        self.assertEqual(self.held(), [0])
        s = self.hstat(0)
        self.assertEqual(s['held_mib'] + s['ctx_mib'] + s['free_mib'], TOTAL)

    def test_your_job_gets_the_whole_gpu(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        # 30000 MiB in a single allocation right after the context: only fits if the holder got out of the way in time
        rc, out, _ = self.finish(self.job('--gpu', 0, '--mib', 30000, '--secs', 0.6))
        self.assertEqual(rc, 0, out + self.logs())
        self.wait_sealed(0, timeout=2)

    def test_gpu_is_open_while_your_job_runs_and_sealed_right_after(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        p = self.job('--gpu', 0, '--mib', 4000, '--secs', 1.5)
        self.wait(lambda: self.state_of(0) == 'open', 3, 'state open')
        self.assertEqual(self.hstat(0)['held_mib'], 0)
        self.assertEqual(self.finish(p)[0], 0)
        t0 = time.time()
        self.wait_sealed(0, timeout=3)
        self.assertLess(time.time() - t0, 1.5)

    def test_another_user_is_kept_out(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        held = self.hstat(0)['held_mib']
        rc, out, _ = self.finish(self.job('--gpu', 0, '--mib', 100, '--uid', OTHER_UID))
        self.assertEqual(rc, 3, 'the other user\'s job must fail while it creates its context: ' + out)
        self.wait(lambda: self.hstat(0).get('blocked') == 1, 3, 'the blocked counter')
        self.assertEqual(self.hstat(0)['held_mib'], held)
        self.assertEqual(self.state_of(0), 'sealed')

    def test_memory_taken_by_another_user_is_never_given_back_to_them(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        held = self.hstat(0)['held_mib']
        self.fake_proc(0, 200)                           # squats in the landing strip
        time.sleep(0.6)
        self.assertEqual(self.hstat(0)['held_mib'], held, 'the holder retreated for another user')
        self.assertLessEqual(self.free(0), SEALED_MAX - 200 + 16)
        self.drop_proc(0)
        self.wait_sealed(0)

    def test_a_hidden_process_of_another_user_counts_as_theirs(self):
        # its owner cannot be looked up (hidepid, another pid namespace): it must not pass for one of yours
        self.up('-n', 1, '--headroom', 2, '--settle-secs', 0.3, '--intruder-secs', 0.4)
        self.wait_sealed(0)
        p = self.job('--gpu', 0, '--mib', 6000, '--secs', 4)
        self.wait(lambda: self.state_of(0) == 'tracking' and abs(self.free(0) - 2048) <= 48, 4, 'tracking')
        held = self.hstat(0)['held_mib']
        self.fake_proc(0, 1500, hidden=True)             # takes most of your headroom
        self.wait(lambda: self.hstat(0).get('intruded'), 3, 'GPU 0 to be marked')
        self.assertGreaterEqual(self.hstat(0)['held_mib'], held - 16, 'the holder retreated for another user')
        self.assertAlmostEqual(self.hstat(0)['own_mib'], 6500, delta=16)
        self.drop_proc(0)
        self.assertEqual(self.finish(p)[0], 0)
        self.wait_sealed(0)

    def test_holder_stays_visible_when_the_state_directory_is_wiped(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        pid = self.hstat(0)['pid']
        with open(os.path.join(self.state, 'keeper.pid')) as f:
            os.kill(int(f.read()), signal.SIGTERM)       # nobody is there to restart anything
        self.wait(lambda: 'NOT running' in self.k2('status').stdout, 5, 'the keeper to stop')
        self.wipe_state()
        self.wait(lambda: self.hstat(0).get('pid') == pid and self.held() == [0], 5, 'the holder to show up again')
        self.assertLessEqual(self.free(0), SEALED_MAX)
        rc, out, err = self.finish(self.run_job([], ['--mib', 30000, '--uid', 0]))
        self.assertEqual(rc, 0, out + err)               # leases work in the new directory
        self.wait_sealed(0)

    def test_holder_survives_a_restart_while_your_job_runs(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        p = self.job('--gpu', 0, '--mib', 20000, '--secs', 3)
        self.wait(lambda: self.state_of(0) == 'open', 3, 'state open')
        os.kill(self.hstat(0)['pid'], signal.SIGKILL)
        self.wait(lambda: 'holder was not running, restarted' in self.logs(), 5, 'the holder to be restarted')
        self.assertEqual(self.finish(p)[0], 0)           # the job never noticed
        self.wait_sealed(0)


class TestRun(Base):
    """`keeper2 run`: explicit leases."""

    def test_run_gives_the_command_the_gpu(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        # uid 0: like a job in a root container, which the holder cannot tell from another user's
        rc, out, err = self.finish(self.run_job([], ['--mib', 30000, '--uid', 0]))
        self.assertEqual(rc, 0, out + err + self.logs())
        self.wait_sealed(0, timeout=2)

    def test_same_job_without_run_is_kept_out(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        self.assertEqual(self.finish(self.job('--gpu', 0, '--mib', 30000, '--uid', 0))[0], 3)

    def test_run_sets_the_gpu_for_the_command(self):
        self.up('-n', 2)
        self.wait_sealed(1)
        p = self.k2('run', '-g', 1, '--', 'sh', '-c', 'echo "$CUDA_VISIBLE_DEVICES $KEEPER2_GPU $CUDA_DEVICE_ORDER {gpu}"')
        self.assertEqual(p.stdout.strip(), '1 1 PCI_BUS_ID 1', p.stderr)
        p = self.k2('run', '-n', 2, '--', 'sh', '-c', 'echo $CUDA_VISIBLE_DEVICES device={gpu}')
        self.assertEqual(p.stdout.strip(), '0,1 device=0,1', p.stderr)

    def test_run_passes_the_exit_code_and_output(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        p = self.k2('run', '--', 'sh', '-c', 'echo out; echo err >&2; exit 7')
        self.assertEqual((p.returncode, p.stdout, p.stderr), (7, 'out\n', 'err\n'))
        p = self.k2('run', 'sh', '-c', 'exit 3')         # also without the --
        self.assertEqual(p.returncode, 3)
        p = self.k2('run', '--', '/no/such/command')
        self.assertEqual(p.returncode, 127, p.stderr)
        self.wait_sealed(0, timeout=3)                   # the lease of the failed start is gone

    def test_run_m_opens_exactly_that_much(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        rc, out, _ = self.finish(self.run_job(['-m', 8], ['--mib', 7000, '--uid', 0]))
        self.assertEqual(rc, 0, out + self.logs())
        rc, out, _ = self.finish(self.run_job(['-m', 8], ['--mib', 9000, '--uid', 0]))
        self.assertEqual(rc, 4, 'a job over its lease must not get the memory: ' + out)
        rc, out, _ = self.finish(self.run_job(['-m', '600m'], ['--mib', 50, '--uid', 0]))
        self.assertEqual(rc, 0, out)
        self.wait_sealed(0, timeout=3)

    def test_rest_of_the_gpu_stays_sealed_during_a_lease(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        p = self.run_job(['-m', 8], ['--mib', 6000, '--uid', 0, '--secs', 2.5])
        self.wait(lambda: self.state_of(0) == 'leased' and self.hstat(0)['own_mib'] >= 6000, 5, 'the leased job')
        s = self.hstat(0)
        self.assertAlmostEqual(s['held_mib'], TOTAL - 500 - 8192 - LANDING, delta=32)
        self.assertFalse(s['intruded'])
        # 1.9 GiB of the lease are unused: a 3 GiB job of another user does not fit
        self.assertNotEqual(self.finish(self.job('--gpu', 0, '--mib', 3000, '--uid', OTHER_UID))[0], 0)
        self.assertEqual(self.finish(p)[0], 0)
        self.wait_sealed(0, timeout=3)

    def test_lease_ends_when_the_command_is_killed(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        p = self.spawn([sys.executable, K2, 'run', '--', 'sleep', '60'])
        self.wait(lambda: self.state_of(0) == 'leased', 5, 'the lease')
        self.assertGreater(self.free(0), 30000)
        p.send_signal(signal.SIGKILL)
        t0 = time.time()
        self.wait_sealed(0, timeout=3)
        self.assertLess(time.time() - t0, 1.5)
        self.assertEqual(os.listdir(os.path.join(self.state, 'gpu0', 'leases')), [])

    def test_lease_next_to_a_job_started_without_run(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        p = self.job('--gpu', 0, '--mib', 20000, '--secs', 3)
        self.wait(lambda: self.state_of(0) == 'open', 3, 'state open')
        rc, out, err = self.finish(self.run_job(['-m', 4], ['--mib', 3000, '--uid', 0]))
        self.assertEqual(rc, 0, 'the holder took the room of the leased job: ' + out + err + self.logs())
        self.assertEqual(self.finish(p)[0], 0)
        self.wait_sealed(0)

    @unittest.skipUnless(os.path.exists('/proc/self/ns/pid'), 'needs /proc')
    def test_lease_ends_with_the_command_not_with_its_background_children(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        p = self.k2('run', '--', 'sh', '-c', 'sleep 4 >/dev/null 2>&1 & exit 0')   # the child keeps the lock fd
        self.assertEqual(p.returncode, 0, p.stderr)
        self.wait_sealed(0, timeout=1.5)

    def test_command_gets_default_signal_handling(self):
        if not shutil.which('perl'):
            self.skipTest('needs perl')
        self.up('-n', 1)
        self.wait_sealed(0)
        p = self.k2('run', '--', 'perl', '-e', 'print defined $SIG{PIPE} ? $SIG{PIPE} : "DEFAULT"')
        self.assertEqual(p.stdout, 'DEFAULT', 'SIGPIPE must not stay ignored in the command ' + p.stderr)

    def test_two_runs_share_a_gpu(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        a = self.run_job(['-m', 6], ['--mib', 5000, '--uid', 0, '--secs', 2])
        b = self.run_job(['-m', 6], ['--mib', 5000, '--uid', 0, '--secs', 1])
        self.wait(lambda: len(self.hstat(0).get('leases', [])) == 2, 5, 'two leases')
        self.assertEqual(self.finish(b)[0], 0)
        self.wait(lambda: len(self.hstat(0).get('leases', [])) == 1, 5, 'one lease left')
        self.assertAlmostEqual(self.hstat(0)['held_mib'], TOTAL - 500 - 6144 - LANDING, delta=32)
        self.assertEqual(self.finish(a)[0], 0)
        self.wait_sealed(0, timeout=3)

    def test_run_without_a_held_gpu(self):
        p = self.k2('run', '--', 'true')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('no GPU is held', p.stderr)
        t0 = time.time()
        p = self.k2('run', '--wait', '0.6', '--', 'true')
        self.assertNotEqual(p.returncode, 0)
        self.assertGreaterEqual(time.time() - t0, 0.5)

    def test_run_wait_starts_once_a_gpu_is_held(self):
        p = self.run_job(['--wait'], ['--mib', 20000, '--uid', 0])
        time.sleep(0.5)
        self.assertIsNone(p.poll())
        self.up('-n', 1)
        rc, out, err = self.finish(p)
        self.assertEqual(rc, 0, out + err)

    def test_exclusive_runs_do_not_overlap(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        script = 'import time; print(time.time()); time.sleep(0.6); print(time.time())'
        ps = [self.spawn([sys.executable, K2, 'run', '-x', '--', sys.executable, '-c', script]) for _ in range(3)]
        spans = sorted(tuple(float(x) for x in self.finish(p)[1].split()) for p in ps)
        self.assertEqual(len(spans), 3)
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            self.assertLessEqual(a1, b0, 'exclusive runs overlapped: %s' % (spans,))

    def test_manual_lease_and_release(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        p = self.k2('lease', '-m', 4, '--ttl', 1.0)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.state_of(0), 'leased')
        self.assertAlmostEqual(self.free(0), 4096 + LANDING, delta=48)
        self.wait_sealed(0, timeout=4)                   # runs out by itself
        self.k2('lease', '--ttl', 600)
        self.wait(lambda: self.free(0) > 31000, 3, 'the whole GPU')
        self.assertEqual(self.k2('release', '--all').returncode, 0)
        self.wait_sealed(0, timeout=3)


class TestTracking(Base):
    """--headroom: close in again once your job has settled."""

    def test_headroom(self):
        self.up('-n', 1, '--headroom', 2, '--settle-secs', 0.5)
        self.wait_sealed(0)
        p = self.job('--gpu', 0, '--mib', 6000, '--secs', 4, '--late-mib', 1500, '--late-after', 1.8)
        self.wait(lambda: self.state_of(0) == 'open', 3, 'state open')
        self.wait(lambda: self.state_of(0) == 'tracking', 3, 'state tracking')
        self.wait(lambda: abs(self.free(0) - 2048) <= 48, 2, '2 GiB of headroom')
        # another user finds 2 GiB, not 25
        self.assertNotEqual(self.finish(self.job('--gpu', 0, '--mib', 3000, '--uid', OTHER_UID))[0], 0)
        # your job grows by 1.5 GiB: fits into the headroom, and the holder moves back to keep 2 GiB free
        self.wait(lambda: self.hstat(0).get('own_mib', 0) >= 7900, 4, 'the late allocation')
        self.wait(lambda: abs(self.free(0) - 2048) <= 48, 2, '2 GiB of headroom again')
        rc, out, _ = self.finish(p)
        self.assertEqual(rc, 0, out)
        self.assertIn('LATE-OK', out)
        self.wait_sealed(0, timeout=3)

    def test_headroom_can_be_switched_while_running(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        p = self.job('--gpu', 0, '--mib', 6000, '--secs', 5)
        self.wait(lambda: self.state_of(0) == 'open', 3, 'state open')
        time.sleep(0.5)
        self.assertEqual(self.state_of(0), 'open')
        self.up('--headroom', 3, '--settle-secs', 0.2)
        self.wait(lambda: self.state_of(0) == 'tracking' and abs(self.free(0) - 3072) <= 48, 5, 'tracking at 3 GiB')
        self.up('--headroom', 'off')
        self.wait(lambda: self.state_of(0) == 'open' and self.hstat(0)['held_mib'] == 0, 5, 'open again')
        p.kill()


class TestOwnContainers(Base):
    """Jobs in root containers count as yours when the container is declared yours."""
    MINE, THEIRS = 'a' * 64, 'b' * 64

    def setUp(self):
        Base.setUp(self)
        simgpu.add_container(self.MINE, 'lhx-train-3', ['/data/me/proj', '/etc/localtime'], root=self.sim)
        simgpu.add_container(self.THEIRS, 'vllm-engine', ['/data/them/models'], root=self.sim)

    def check(self):
        self.wait_sealed(0)
        rc, out, _ = self.finish(self.job('--gpu', 0, '--mib', 20000, '--uid', 0, '--cid', self.MINE))
        self.assertEqual(rc, 0, out + self.logs())
        self.wait_sealed(0)
        self.assertEqual(self.finish(self.job('--gpu', 0, '--mib', 20000, '--uid', 0, '--cid', self.THEIRS))[0], 3)
        self.assertEqual(self.finish(self.job('--gpu', 0, '--mib', 20000, '--uid', 0))[0], 3)

    def test_by_container_name(self):
        self.up('-n', 1, '--own-containers', '^lhx-')
        self.check()

    def test_by_mounted_path(self):
        self.up('-n', 1, '--own-mounts', '/data/me,/somewhere/else')
        self.check()

    def test_a_path_prefix_is_not_enough(self):
        self.up('-n', 1, '--own-mounts', '/data/m')      # /data/me is not below /data/m
        self.wait_sealed(0)
        self.assertEqual(self.finish(self.job('--gpu', 0, '--mib', 20000, '--uid', 0, '--cid', self.MINE))[0], 3)


class TestKeeper(Base):
    """The keeper: claiming, restarting, replacing."""

    def test_claims_only_idle_gpus(self):
        self.fake_proc(0, 2000)
        self.fake_proc(2, 150, pid=999991)
        self.up('-n', 2)
        self.wait(lambda: self.held() == [1, 3], 5, 'GPUs 1 and 3')
        time.sleep(0.5)
        self.assertEqual(self.held(), [1, 3])
        self.drop_proc(0)
        self.drop_proc(2, pid=999991)

    def test_waits_until_a_gpu_was_idle_long_enough(self):
        self.k2('up', '--mode', 'host', '-n', 1, '--idle-secs', 2.5, '--poll', 0.1)
        self.assertEqual(self.held(), [])
        self.wait(lambda: self.held() == [0], 6, 'GPU 0')

    def test_idle_utilisation_limit(self):
        s = simgpu.snapshot(self.sim)
        s['gpus']['0']['util'] = 100                     # busy without memory, as seen on real machines
        with open(os.path.join(self.sim, 'sim.json'), 'w') as f:
            json.dump(s, f)
        self.up('-n', 1, '--idle-util', 20)
        self.wait(lambda: self.held() == [1], 5, 'GPU 1')

    def test_quiet_gpus_are_claimed_first(self):
        s = simgpu.snapshot(self.sim)
        s['gpus']['0']['util'] = 100
        with open(os.path.join(self.sim, 'sim.json'), 'w') as f:
            json.dump(s, f)
        self.up('-n', 1)
        self.wait(lambda: self.held() == [1], 5, 'GPU 1')

    def test_target_and_candidates_can_change_while_running(self):
        self.up('-n', 2)
        self.wait(lambda: self.held() == [0, 1], 5, 'two GPUs')
        self.up('-n', 1)
        self.wait(lambda: self.held() == [0], 5, 'one GPU')
        self.wait(lambda: self.free(1) > 32000, 3, 'GPU 1 to be free')
        self.up('-n', 3, '-c', '0,2-3')
        self.wait(lambda: self.held() == [0, 2, 3], 5, 'GPUs 0, 2, 3')
        self.up('-c', '2,3')
        self.wait(lambda: self.held() == [2, 3], 5, 'GPUs 2, 3')
        self.up('-n', 0)
        self.wait(lambda: self.held() == [], 5, 'nothing held')

    def test_a_gpu_in_use_by_you_is_not_given_back(self):
        self.up('-n', 2)
        self.wait(lambda: self.held() == [0, 1], 5, 'two GPUs')
        p = self.job('--gpu', 1, '--mib', 4000, '--secs', 2.5)
        self.wait(lambda: self.state_of(1) == 'open', 3, 'state open')
        self.up('-n', 1)                                 # the surplus is the one without your job
        self.wait(lambda: self.held() == [1], 5, 'GPU 1 kept')
        self.assertEqual(self.finish(p)[0], 0)

    def test_dead_holder_is_restarted(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        pid = self.hstat(0)['pid']
        os.kill(pid, signal.SIGKILL)
        self.wait(lambda: self.hstat(0).get('pid') not in (None, pid) and self.free(0) <= SEALED_MAX, 6, 'a new holder')

    def test_keeper_restart_keeps_the_holders(self):
        self.up('-n', 2)
        self.wait_sealed(1)
        pids = [self.hstat(0)['pid'], self.hstat(1)['pid']]
        with open(os.path.join(self.state, 'keeper.pid')) as f:
            os.kill(int(f.read()), signal.SIGTERM)
        self.wait(lambda: 'NOT running' in self.k2('status').stdout, 5, 'the keeper to stop')
        self.assertLessEqual(self.free(0), SEALED_MAX)   # GPUs stay held without the keeper
        self.assertEqual(self.k2('run', '--', 'true').returncode, 0)     # and leases still work
        self.up()
        time.sleep(0.5)
        self.assertEqual(self.held(), [0, 1])
        self.assertEqual([self.hstat(0)['pid'], self.hstat(1)['pid']], pids)
        self.assertEqual(json.loads(self.k2('status', '--json').stdout)['config']['target'], 2)

    def test_intruded_gpu_is_replaced_and_the_spare_given_back(self):
        self.up('-n', 1, '--intruder-secs', 0.4)
        self.wait_sealed(0)
        mine = self.job('--gpu', 0, '--mib', 4000, '--secs', 1.5)
        self.wait(lambda: self.state_of(0) == 'open', 3, 'state open')
        theirs = self.job('--gpu', 0, '--mib', 5000, '--uid', OTHER_UID, '--secs', 5)    # slips in while it is open
        self.assertEqual(self.finish(mine)[0], 0)
        self.wait(lambda: self.hstat(0).get('intruded'), 5, 'GPU 0 to be marked')
        # they keep what they have and get nothing more: the holder takes the rest as soon as your job is gone
        self.wait(lambda: self.state_of(0) == 'sealed' and self.free(0) <= SEALED_MAX, 3, 'the rest to be sealed')
        self.assertAlmostEqual(self.hstat(0)['foreign_mib'], 5500, delta=16)
        self.wait(lambda: self.held() == [0, 1], 5, 'the spare GPU')
        self.assertEqual(self.k2('gpus').stdout.strip(), '1')
        self.assertEqual(self.k2('gpus', '--all').stdout.strip(), '0,1')
        p = self.k2('run', '--', 'sh', '-c', 'echo $CUDA_VISIBLE_DEVICES')
        self.assertEqual(p.stdout.strip(), '1')          # new runs avoid the shared GPU
        self.assertIn('INTRUDED', self.k2('status').stdout)
        self.assertEqual(self.finish(theirs)[0], 0)      # nobody touched their job
        self.wait(lambda: self.held() == [0], 8, 'the spare to be given back')
        self.wait_sealed(0)

    def test_an_old_held_list_is_not_trusted(self):
        os.makedirs(self.state, mode=0o700)
        with open(os.path.join(self.state, 'held'), 'w') as f:
            f.write('2\n')                              # left by a keeper that died long ago
        self.fake_proc(2, 3000)                          # GPU 2 is somebody else's by now
        self.up('-n', 1)
        self.wait(lambda: self.held() == [0], 5, 'GPU 0')
        time.sleep(0.5)
        self.assertEqual(self.held(), [0])
        self.assertEqual(self.free(2), TOTAL - 3000)
        self.drop_proc(2)

    def test_a_holder_that_cannot_start_does_not_stall_the_keeper(self):
        self.set_gpu(0, broken=True)
        self.up('-n', 1, '-c', 0)
        self.wait(lambda: 'did not come up' in self.logs(), 4, 'the failure to be noticed')
        self.assertIn('simulated broken GPU', self.logs())
        time.sleep(2.5)
        self.assertLessEqual(self.logs().count('did not come up'), 3, 'no back-off between attempts')
        t0 = time.time()
        self.k2('down')
        self.assertLess(time.time() - t0, 5)

    def test_keeper_survives_a_wiped_state_directory(self):
        self.up('-n', 2)
        self.wait(lambda: self.held() == [0, 1], 5, 'two GPUs')
        pids = [self.hstat(0)['pid'], self.hstat(1)['pid']]
        self.wipe_state()
        self.wait(lambda: 'keeper: running' in self.k2('status').stdout and self.held() == [0, 1], 6,
                  'keeper and holders to be visible again')
        time.sleep(2.5)
        self.assertEqual(self.held(), [0, 1])
        self.assertEqual([self.hstat(0)['pid'], self.hstat(1)['pid']], pids, 'holders were replaced')
        self.assertEqual(json.loads(self.k2('status', '--json').stdout)['config']['target'], 2)

    def test_down_gives_everything_back(self):
        self.up('-n', 3)
        self.wait(lambda: self.held() == [0, 1, 2], 5, 'three GPUs')
        self.k2('lease', '--ttl', 600)
        p = self.k2('down')
        self.assertIn('released GPU(s): 0, 1, 2', p.stdout)
        for i in range(4):
            self.assertEqual(self.free(i), TOTAL)
        self.assertEqual(self.held(), [])
        self.assertEqual(self.k2('gpus').returncode, 1)
        self.assertIn('none were held', self.k2('down').stdout)

    def test_fresh_up_does_not_inherit_old_settings(self):
        self.up('-n', 2, '--headroom', 4)
        self.wait_sealed(1)
        self.k2('down')
        self.up()
        cfg = json.loads(self.k2('status', '--json').stdout)['config']
        self.assertEqual((cfg['target'], cfg['headroom_gib']), (1, None))

    def test_status_and_log_output(self):
        self.up('-n', 1)
        self.wait_sealed(0)
        p = self.run_job(['-m', 4], ['--mib', 1000, '--uid', 0, '--secs', 1.5])
        self.wait(lambda: self.state_of(0) == 'leased' and self.hstat(0)['own_mib'] > 1000, 5, 'the leased job')
        out = self.k2('status').stdout
        self.assertIn('keeper: running', out)
        self.assertRegex(out, r'(?m)^0 +leased +\d+\.\dG +1\.\dG +- +\S+ +lease 4\.0G')
        self.assertRegex(out, r'(?m)^1 +- ')
        self.finish(p)
        log = self.k2('log').stdout
        for needle in ('GPU 0: claimed', 'starting -> sealed', 'sealed -> leased', 'leased -> sealed'):
            self.assertIn(needle, log)

    def test_bad_arguments(self):
        self.assertNotEqual(self.k2('up', '--mode', 'host', '-c', '9').returncode, 0)
        self.assertNotEqual(self.k2('up', '--mode', 'host', '--own-containers', '(').returncode, 0)
        self.up('-n', 1)
        self.wait_sealed(0)
        p = self.k2('run', '-m', 'lots', '--', 'true')
        self.assertEqual(p.returncode, 2)
        self.assertIn('memory size', p.stderr)
        self.assertNotEqual(self.k2('run').returncode, 0)
        p = self.k2('run', '-n', 0, '--', 'true')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('-n', p.stderr)
        p = self.k2('up', '--headroom', 'max')
        self.assertNotEqual(p.returncode, 0)
        self.assertNotIn('Traceback', p.stderr)


class TestMix(Base):
    """A minute of what a day of agent-driven experiments looks like, compressed: your jobs come and go, with and
    without `run`, while another user keeps trying to get onto the GPU."""

    def test_your_jobs_always_fit_and_theirs_never_do(self):
        import random
        rnd = random.Random(7)
        self.up('-n', 1)
        self.wait_sealed(0)
        mine, theirs = [], []
        for kind in ['plain', 'run', 'run-m', 'idle'] * 6:
            if kind == 'plain':
                mine.append(self.job('--gpu', 0, '--mib', rnd.choice([2000, 12000, 30000]), '--secs', rnd.uniform(0.1, 0.5)))
            elif kind == 'run':
                mine.append(self.run_job([], ['--mib', rnd.choice([2000, 30000]), '--uid', 0, '--secs', rnd.uniform(0.1, 0.5)]))
            elif kind == 'run-m':
                mine.append(self.run_job(['-m', 10], ['--mib', rnd.choice([2000, 9000]), '--uid', 0, '--secs', rnd.uniform(0.1, 0.5)]))
            self.assertEqual(self.finish(mine[-1])[0] if kind != 'idle' else 0, 0, kind + '\n' + self.logs())
            # between two of your jobs the GPU is idle: that is when they come
            time.sleep(rnd.uniform(0.05, 0.25))
            for _ in range(rnd.randint(1, 3)):
                theirs.append(self.job('--gpu', 0, '--mib', rnd.choice([300, 6000, 19000]), '--uid', OTHER_UID))
            for p in theirs[-3:]:
                if p.poll() is None:
                    self.assertNotEqual(self.finish(p)[0], 0, 'another user got onto the GPU\n' + self.logs())
        self.assertEqual(len(mine), 18)
        self.wait(lambda: self.hstat(0).get('blocked', 0) >= len(theirs) - 2, 5, 'the blocked counter')
        self.assertFalse(self.hstat(0)['intruded'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
