"""Final round job lists: jobs/final_phase1.txt (tuning: GraphMAE on heterophilous graphs, PolyGCL on homophilous graphs,
GraphSAGE / MLP encoder controls on homophilous graphs) and jobs/final_phase2.txt (3 fresh seeds of every new study)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HET = ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']
HOMO = ['cora', 'citeseer', 'pubmed', 'amazon-photos']
BUDGET = {'cora': 60, 'citeseer': 60, 'pubmed': 60, 'amazon-photos': 30}


def tune(d, m, n):
    return [f'python scripts/tune_node.py --dataset {d} --method {m} --trials {n} --workers {n} --shard {i}' for i in range(n)]


long = [j for i in range(10) for d in HET for j in tune(d, 'graphmae', 10)[i:i + 1]]
short = [j for d in HOMO for j in tune(d, 'polygcl', BUDGET[d])]
short += [j for d in HOMO for m in ('bgrl_sage', 'bgrl_mlp') for j in tune(d, m, 20)]

phase1 = []
ratio = len(short) // len(long) + 1
for j, cmd in enumerate(long):
    phase1.append(cmd)
    phase1 += short[j * ratio:(j + 1) * ratio]
phase1 += short[len(long) * ratio:]

studies = [(d, 'graphmae') for d in HET] + [(d, m) for d in HOMO for m in ('polygcl', 'bgrl_sage', 'bgrl_mlp')]
phase2 = [f'python scripts/run_best.py --dataset {d} --methods {m} --tag clean --job {m}:{s}' for s in (1, 2, 3) for d, m in studies]

os.makedirs(os.path.join(ROOT, 'jobs'), exist_ok=True)
for name, lines in (('final_phase1', phase1), ('final_phase2', phase2)):
    with open(os.path.join(ROOT, 'jobs', f'{name}.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(name, len(lines), 'jobs')
