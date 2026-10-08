"""Markdown tables of every result in results/*.csv (printed to stdout)."""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda name: list(csv.DictReader(open(os.path.join(ROOT, 'results', name))))

HOMO = ['cora', 'citeseer', 'pubmed', 'amazon-photos']
HET = ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']
NAME = {'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed', 'amazon-photos': 'Amazon-Photos', 'roman-empire': 'Roman-empire',
        'amazon-ratings': 'Amazon-ratings', 'minesweeper': 'Minesweeper', 'tolokers': 'Tolokers', 'questions': 'Questions'}
METHOD = {'bgrl': 'BGRL', 'ccassg': 'CCA-SSG', 'graphmae': 'GraphMAE', 'polygcl': 'PolyGCL', 'laplacegnn_full': 'LaplaceGNN v2 (full)',
          'bgrl_dualfreq': 'BGRL + dual-freq', 'ccassg_dualfreq': 'CCA-SSG + dual-freq', 'bgrl_dualfreq_gate': 'BGRL + dual-freq (closed gate)',
          'ccassg_dualfreq_gate': 'CCA-SSG + dual-freq (closed gate)', 'bgrl_mlp': 'BGRL, MLP encoder', 'bgrl_sage': 'BGRL, GraphSAGE encoder',
          'bgrl_adv': 'BGRL + hidden adversary', 'laplacegnn': 'LaplaceGNN v1', 'laplacegnn_base': 'LaplaceGNN, views only',
          'laplacegnn_noadv': 'LaplaceGNN v1, no adversary', 'laplacegnn_v2': 'LaplaceGNN v2 (+ structural adversary)',
          'laplacegnn_cca': 'LaplaceGNN, CCA objective', 'laplacegnn_v0': 'LaplaceGNN v0 (old dense views)'}


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def main_table():
    best = {(r['dataset'], r['method']): r for r in R('best_runs.csv') if r['tag'] == 'clean'}
    cell = lambda d, m: f"{float(best[(d, m)]['test']):.1f} ± {float(best[(d, m)]['test_std']):.1f}" if (d, m) in best else '—'
    order = ['bgrl', 'ccassg', 'graphmae', 'polygcl', 'bgrl_dualfreq', 'bgrl_dualfreq_gate', 'ccassg_dualfreq', 'ccassg_dualfreq_gate',
             'bgrl_sage', 'bgrl_mlp', 'laplacegnn_full']
    parts = []
    for group in (HOMO, HET):
        ms = [m for m in order if any((d, m) in best for d in group)]
        parts.append(table(['Method'] + [NAME[d] for d in group], [[METHOD[m]] + [cell(d, m) for d in group] for m in ms]))
    return '\n\n'.join(parts)


def tuning_table():
    rows = R('tuning_summary.csv')
    by = {(r['dataset'], r['method']): r for r in rows}
    ds = [d for d in HOMO + HET if any(r['dataset'] == d for r in rows)]
    ms = [m for m in METHOD if any(r['method'] == m for r in rows)]
    cell = lambda d, m: f"{float(by[(d, m)]['test']):.1f} ({by[(d, m)]['trials']})" if (d, m) in by else '—'
    return table(['Method'] + [NAME[d] for d in ds], [[METHOD[m]] + [cell(d, m) for d in ds] for m in ms])


def round1_table():
    rows = []
    for d in ('cora', 'citeseer', 'pubmed', 'amazon-photos'):
        path = os.path.join(ROOT, 'results', f'final_{d}.csv')
        if os.path.exists(path):
            for r in R(f'final_{d}.csv'):
                m = r['runs'].split()[0].split('/')[3]
                rows.append((d, m, f"{float(r['test_mean']):.2f} ± {float(r['test_std']):.2f}", r['n_seeds']))
    ms = sorted({m for _, m, _, _ in rows}, key=list(METHOD).index)
    ds = sorted({d for d, _, _, _ in rows}, key=(HOMO + HET).index)
    by = {(d, m): v for d, m, v, _ in rows}
    return table(['Method'] + [NAME[d] for d in ds], [[METHOD[m]] + [by.get((d, m), '—') for d in ds] for m in ms])


def paired_table():
    rows = R('dualfreq_vs_homophily.csv')
    ds = sorted({r['dataset'] for r in rows}, key=lambda d: float(next(r for r in rows if r['dataset'] == d)['adjusted_homophily']))
    combos = [(b, v) for b in ('bgrl', 'ccassg') for v in ('dualfreq', 'dualfreq_gate', 'sage', 'mlp') if any(r['base'] == b and r['variant'] == v for r in rows)]
    by = {(r['dataset'], r['base'], r['variant']): r for r in rows}
    head = ['Dataset', 'Adj. homophily', 'Trials'] + [f"{METHOD[b]} → {METHOD[b + '_' + v].replace(METHOD[b] + ' ', '').replace(METHOD[b] + ', ', '')}" for b, v in combos]
    out = []
    for d in ds:
        any_r = next(r for r in rows if r['dataset'] == d)
        out.append([NAME[d], f"{float(any_r['adjusted_homophily']):+.2f}", any_r['trials']] +
                   [f"{float(by[(d, b, v)]['mean_delta']):+.1f} ± {float(by[(d, b, v)]['sem']):.1f} ({float(by[(d, b, v)]['frac_better']):.0%})"
                    if (d, b, v) in by else '—' for b, v in combos])
    return table(head, out)


def ablation_table():
    ds = [d for d in HOMO if os.path.exists(os.path.join(ROOT, 'results', f'ablation_{d}.csv'))]
    data = {d: {r['variant']: r for r in R(f'ablation_{d}.csv')} for d in ds}
    variants = []
    for d in ds:
        variants += [v for v in data[d] if v not in variants]
    out = []
    for v in variants:
        row = [v.replace('_', ' ')]
        for d in ds:
            r = data[d].get(v)
            row.append('—' if r is None else (f"{float(r['test']):.2f} ± {float(r['test_std']):.2f}" if v == 'full'
                                              else f"{float(r['delta_test']):+.2f}"))
        out.append(row)
    return table(['Variant (full = test; others = Δ vs full)'] + [NAME[d] for d in ds], out)


def dose_table():
    out = []
    for d in HOMO:
        path = os.path.join(ROOT, 'results', f'dose_{d}.csv')
        if os.path.exists(path):
            for r in R(f'dose_{d}.csv'):
                out.append([NAME[d], r['level'].replace('alpha_', 'α = ').replace('random_same_budget', 'random, same budget'),
                            f"{float(r['spectral_change']):.2e}", f"{float(r['test']):.2f} ± {float(r['test_std']):.2f}"])
    return table(['Dataset', 'Views', 'Sampled spectral change', 'Test'], out)


def mechanism_table():
    rows = R('spectral_mechanism.csv')
    return table(['Dataset', 'Views', 'Expected flips', 'Relaxed change', 'Sampled change'],
                 [[NAME[r['dataset']], r['views'], f"{float(r['expected_flips']):.0f}", f"{float(r['relaxed_change']):.2e}",
                   f"{float(r['sampled_change']):.2e} ± {float(r['sampled_std']):.1e}"] for r in rows])


def attack_table():
    best = {(r['dataset'], r['tag'], r['method']): r for r in R('best_runs.csv')}
    ms = ['bgrl', 'ccassg', 'graphmae', 'laplacegnn_full']
    tags = ['clean'] + [f'{a}-{b}' for a in ('random', 'dice', 'prbcd') for b in ('0.05', '0.1', '0.2')]
    parts = []
    for d in ('cora', 'citeseer'):
        out = []
        for t in tags:
            row = [t]
            for m in ms:
                r = best.get((d, t, m))
                row.append('—' if r is None else f"{float(r['test']):.1f} ± {float(r['test_std']):.1f}")
            out.append(row)
        parts.append(f'**{NAME[d]}**\n\n' + table(['Poisoning'] + [METHOD[m] for m in ms], out))
    return '\n\n'.join(parts)


def tu_table():
    rows = R('tu_replication.csv')
    return table(['Dataset', 'Views', 'Seeds', 'Test (10-fold CV)'],
                 [[r['dataset'], r['view_mode'], r['n_seeds'], f"{float(r['test_mean']):.2f} ± {float(r['test_std']):.2f}"]
                  for r in sorted(rows, key=lambda r: (r['dataset'], r['view_mode']))])


def scaling_table():
    rows = R('scaling.csv')
    fmt = lambda v: '—' if v in ('', 'nan') else f'{float(v):.1f}'
    return table(['Graph', 'Nodes', 'Edges', 'Method', 'Seconds', 'Peak GB', 'Status'],
                 [[r['graph'], r['nodes'], r['edges'], r['method'], fmt(r['seconds']), fmt(r['peak_gb']), r['status']] for r in rows])


def homophily_table():
    rows = R('raw_and_homophily.csv')
    return table(['Dataset', 'Nodes', 'Edges', 'Metric', 'Edge homophily', 'Adj. homophily', 'Raw-feature probe', 'SGC-feature probe'],
                 [[NAME[r['dataset']], r['nodes'], r['edges'], r['metric'], r['edge_homophily'], r['adjusted_homophily'],
                   r['raw_features_test'], r['sgc_features_test']] for r in rows])


if __name__ == '__main__':
    for title, fn in (('main', main_table), ('tuning', tuning_table), ('round1', round1_table), ('paired', paired_table),
                      ('ablation', ablation_table), ('dose', dose_table), ('mechanism', mechanism_table), ('attack', attack_table),
                      ('tu', tu_table), ('scaling', scaling_table), ('homophily', homophily_table)):
        print(f'<<<{title}>>>')
        print(fn())
