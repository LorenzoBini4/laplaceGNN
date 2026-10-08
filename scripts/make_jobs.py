"""Writes jobs/phase1.txt (tuning trials, dose-response, attacks, TU) and jobs/phase2.txt (seeds) for scripts/jobpool.py."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOMO = ['cora', 'citeseer', 'pubmed', 'amazon-photos']
HET_OLD = ['roman-empire', 'amazon-ratings']
HET_NEW = ['minesweeper', 'tolokers', 'questions']
ALL_HET = ['bgrl', 'bgrl_dualfreq', 'bgrl_dualfreq_gate', 'bgrl_mlp', 'ccassg', 'ccassg_dualfreq', 'ccassg_dualfreq_gate',
           'polygcl', 'laplacegnn_full']
PAIRED = ['bgrl', 'ccassg', 'bgrl_dualfreq', 'ccassg_dualfreq', 'bgrl_dualfreq_gate', 'ccassg_dualfreq_gate']

studies = [(d, m, 10) for d in HET_NEW for m in ALL_HET]
studies += [(d, m, 10) for d in HET_OLD for m in PAIRED + ['bgrl_mlp']]
studies += [(d, m, 20) for d in HOMO for m in PAIRED]
studies += [('amazon-photos', m, 30) for m in ('laplacegnn_full', 'laplacegnn_v2', 'laplacegnn_cca')]

tuning = []
for i in range(30):
    for d, m, n in studies:
        if i < n:
            heavy = (m.startswith('laplacegnn') and d in HET_OLD + HET_NEW) or (m == 'polygcl' and d == 'questions')
            tuning.append(('EXCLUSIVE ' if heavy else '') + f'python scripts/tune_node.py --dataset {d} --method {m} --trials {n} --workers {n} --shard {i}')

other = []
for d in ('pubmed', 'amazon-photos'):
    for lvl in ('alpha_0.0', 'alpha_0.25', 'alpha_0.5', 'alpha_0.75', 'alpha_1.0', 'random_same_budget'):
        for s in range(3):
            other.append(f'python scripts/dose_response.py --dataset {d} --method laplacegnn_full --job {lvl}:{s}')
for d in ('cora', 'citeseer'):
    for attack in ('random', 'dice', 'prbcd'):
        for b in ('0.05', '0.1', '0.2'):
            for m in ('bgrl', 'ccassg', 'graphmae', 'laplacegnn_full'):
                for s in (1, 2, 3):
                    other.append(f'python scripts/run_best.py --dataset {d} --methods {m} --tag {attack}-{b} '
                                 f'--set edge_index_file=data/attacks/{d}/{attack}-{b}.pt --job {m}:{s}')
for d in ('MUTAG', 'PROTEINS', 'IMDB-BINARY'):
    for vm in ('spectral', 'random'):
        for s in range(5):
            other.append(f'python -m ssl_adv_graph.tudataset.run_adv_graph --dataset {d} --view_mode {vm} --seed {s} --lr 1e-3 '
                         f'--epoch 100 --gnn1_num_layers 2 --gnn1_dim 512 --gnn2_num_layers 2 --gnn2_dim 512 --mlp_dim 512 '
                         f'--sv_budget_ratio 0.2 --logdir ./runs/tu_replication/{d}-{vm}')

phase1, k = [], 0
for j, cmd in enumerate(tuning):
    phase1.append(cmd)
    if j % 4 == 3 and k < len(other):
        phase1.append(other[k])
        k += 1
phase1 += other[k:]

seeds = []
for d in HOMO:
    for m in ('bgrl', 'ccassg', 'graphmae', 'laplacegnn_full', 'bgrl_dualfreq_gate', 'ccassg_dualfreq_gate'):
        seeds.append((d, m))
for d in HET_OLD + HET_NEW:
    for m in ('bgrl', 'ccassg', 'polygcl', 'laplacegnn_full', 'bgrl_dualfreq', 'ccassg_dualfreq', 'bgrl_dualfreq_gate',
              'ccassg_dualfreq_gate', 'bgrl_mlp'):
        seeds.append((d, m))
phase2 = [('EXCLUSIVE ' if (m.startswith('laplacegnn') and d in HET_OLD + HET_NEW) or (m == 'polygcl' and d == 'questions') else '')
          + f'python scripts/run_best.py --dataset {d} --methods {m} --tag clean --job {m}:{s}' for s in (1, 2, 3) for d, m in seeds]

os.makedirs(os.path.join(ROOT, 'jobs'), exist_ok=True)
for name, lines in (('phase1', phase1), ('phase2', phase2)):
    with open(os.path.join(ROOT, 'jobs', f'{name}.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(name, len(lines), 'jobs')
print('tuning trials', len(tuning), '| dose', 36, '| attacks', 216, '| TU', 30)
