"""Writes config.json (hparams, git state, env), metrics.jsonl and final.json for every run."""
import datetime
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def git_info():
    diff = _git('diff', 'HEAD')
    return {
        'commit': _git('rev-parse', 'HEAD'),
        'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty': bool(diff),
        'diff_sha1': hashlib.sha1(diff.encode()).hexdigest() if diff else None,
    }


def env_info():
    info = {
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'host': socket.gethostname(),
        'platform': platform.platform(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import torch_geometric
        info['torch_geometric'] = torch_geometric.__version__
    except ImportError:
        pass
    try:
        import sklearn
        info['sklearn'] = sklearn.__version__
    except ImportError:
        pass
    return info


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def default_logdir(pipeline, dataset, seed, root='runs'):
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    return os.path.join(REPO_ROOT, root, pipeline, dataset, f'seed{seed}-{stamp}')


class RunLogger:
    def __init__(self, logdir, config, pipeline):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self.config = {
            'pipeline': pipeline,
            'argv': sys.argv,
            'started': datetime.datetime.now().isoformat(timespec='seconds'),
            'git': git_info(),
            'env': env_info(),
            'hparams': _to_jsonable(config),
        }
        self._write('config.json', self.config)
        self._metrics = open(os.path.join(logdir, 'metrics.jsonl'), 'a')
        print(f'[runlog] writing to {logdir} (commit {self.config["git"]["commit"]}, dirty={self.config["git"]["dirty"]})')

    def _write(self, name, obj):
        with open(os.path.join(self.logdir, name), 'w') as f:
            json.dump(_to_jsonable(obj), f, indent=2)

    def log(self, record):
        self._metrics.write(json.dumps(_to_jsonable(record)) + '\n')
        self._metrics.flush()

    def update_config(self, **extra):
        self.config.setdefault('extra', {}).update(_to_jsonable(extra))
        self._write('config.json', self.config)

    def finish(self, summary):
        summary = dict(summary)
        summary['finished'] = datetime.datetime.now().isoformat(timespec='seconds')
        self._write('final.json', summary)
        self._metrics.close()
        print(f'[runlog] final results written to {os.path.join(self.logdir, "final.json")}')


def flags_to_dict(flags_obj):
    out = {}
    for module, module_flags in flags_obj.flags_by_module_dict().items():
        if module.startswith('absl'):
            continue
        for fl in module_flags:
            out[fl.name] = fl.value
    return out
