"""Memory-aware pool over a file of shell commands (one per line; '#' lines ignored; 'EXCLUSIVE ' prefix = run alone).

A command starts only when fewer than --workers jobs run and the GPU has at least --min_free_gb free memory.
Commands that exited with 0 are recorded in <jobs>.done and skipped on restart.

    python scripts/jobpool.py jobs/phase1.txt --workers 6 --min_free_gb 6
"""
import argparse
import hashlib
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_gb():
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'])
        return float(out.decode().split()[0]) / 1024
    except Exception:
        return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('jobs')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--min_free_gb', type=float, default=6.0)
    p.add_argument('--ramp_seconds', type=float, default=20.0)
    args = p.parse_args()
    lines = [l.strip() for l in open(args.jobs) if l.strip() and not l.startswith('#')]
    done_path = args.jobs + '.done'
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    key = lambda c: hashlib.sha1(c.encode()).hexdigest()[:16]
    todo = [c for c in lines if key(c) not in done]
    logdir = os.path.join(ROOT, 'logs', 'jobs', os.path.splitext(os.path.basename(args.jobs))[0])
    os.makedirs(logdir, exist_ok=True)
    print(f'{time.strftime("%H:%M")} {len(todo)} of {len(lines)} jobs to run', flush=True)
    running, finished, failed, last_launch = {}, 0, 0, 0.0
    env = dict(os.environ, OMP_NUM_THREADS='3', MKL_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3')
    while todo or running:
        for pr, (cmd, log) in list(running.items()):
            if pr.poll() is not None:
                log.close()
                del running[pr]
                if pr.returncode == 0:
                    finished += 1
                    with open(done_path, 'a') as f:
                        f.write(key(cmd) + '\n')
                else:
                    failed += 1
                    print(f'{time.strftime("%H:%M")} FAILED ({pr.returncode}): {cmd}', flush=True)
        exclusive_running = any(c.startswith('EXCLUSIVE ') for c, _ in running.values())
        nxt = todo[0] if todo else ''
        can_start = (todo and len(running) < args.workers and time.time() - last_launch > args.ramp_seconds
                     and not exclusive_running and (not nxt.startswith('EXCLUSIVE ') or not running)
                     and (not running or free_gb() >= args.min_free_gb))
        if can_start:
            cmd = todo.pop(0)
            log = open(os.path.join(logdir, key(cmd) + '.log'), 'w')
            log.write(cmd + '\n')
            log.flush()
            shell_cmd = cmd[len('EXCLUSIVE '):] if cmd.startswith('EXCLUSIVE ') else cmd
            running[subprocess.Popen(shell_cmd, shell=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)] = (cmd, log)
            last_launch = time.time()
            if (finished + failed + len(running)) % 25 == 0:
                print(f'{time.strftime("%H:%M")} running {len(running)}, finished {finished}, failed {failed}, '
                      f'left {len(todo)}, free {free_gb():.1f} GB', flush=True)
        else:
            time.sleep(5)
    print(f'{time.strftime("%H:%M")} JOBPOOL_DONE finished {finished}, failed {failed}', flush=True)


if __name__ == '__main__':
    main()
