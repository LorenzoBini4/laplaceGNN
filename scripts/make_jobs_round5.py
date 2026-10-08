"""Round 5 job lists: jobs/s1_phase1.txt (GraphSAGE control tuning, robustness attribution, open-gate seeds) and
jobs/s1_phase2.txt (GraphSAGE seeds)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HET = ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']
HOMO = ['cora', 'citeseer', 'pubmed', 'amazon-photos']

tuning = [f'python scripts/tune_node.py --dataset {d} --method bgrl_sage --trials 10 --workers 10 --shard {i}'
          for i in range(10) for d in HET]

robust = []
variants = {'nohid': ['adv_m=1'], 'nostruct': ['sadv_every=0'], 'noadv': ['adv_m=1', 'sadv_every=0']}
for d in ('cora', 'citeseer'):
    for cond in ('clean', 'random-0.2', 'dice-0.2', 'prbcd-0.2'):
        for v, sets in variants.items():
            extra = sets + ([] if cond == 'clean' else [f'edge_index_file=data/attacks/{d}/{cond}.pt'])
            setargs = ' '.join(f'--set {x}' for x in extra)
            for s in (1, 2, 3):
                robust.append(f'python scripts/run_best.py --dataset {d} --methods laplacegnn_full --tag {cond}-{v} {setargs} '
                              f'--job laplacegnn_full:{s}')

opengate = [f'python scripts/run_best.py --dataset {d} --methods {m} --tag clean --job {m}:{s}'
            for s in (1, 2, 3) for d in HOMO for m in ('bgrl_dualfreq', 'ccassg_dualfreq')]

phase1, others = [], robust + opengate
for j, cmd in enumerate(tuning):
    phase1.append(cmd)
    if others:
        phase1.append(others.pop(0))
    if others and j % 2:
        phase1.append(others.pop(0))
phase1 += others
phase2 = [f'python scripts/run_best.py --dataset {d} --methods bgrl_sage --tag clean --job bgrl_sage:{s}' for s in (1, 2, 3) for d in HET]

os.makedirs(os.path.join(ROOT, 'jobs'), exist_ok=True)
for name, lines in (('s1_phase1', phase1), ('s1_phase2', phase2)):
    with open(os.path.join(ROOT, 'jobs', f'{name}.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(name, len(lines), 'jobs')
