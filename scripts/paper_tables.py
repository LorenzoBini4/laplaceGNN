"""LaTeX tables for the paper, generated from results/*.csv and the tuning search spaces -> paper/tables/*.tex."""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
OUT = os.path.join(ROOT, 'paper', 'tables')
R = lambda name: list(csv.DictReader(open(os.path.join(ROOT, 'results', name))))

HOMO = ['cora', 'citeseer', 'pubmed', 'amazon-photos']
HET = ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']
NAME = {'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed', 'amazon-photos': 'Am.-Photos', 'roman-empire': 'Roman-emp.',
        'amazon-ratings': 'Am.-ratings', 'minesweeper': 'Minesweeper', 'tolokers': 'Tolokers', 'questions': 'Questions'}
FULLNAME = {'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed', 'amazon-photos': 'Amazon-Photos', 'roman-empire': 'Roman-empire',
            'amazon-ratings': 'Amazon-ratings', 'minesweeper': 'Minesweeper', 'tolokers': 'Tolokers', 'questions': 'Questions'}
METHOD = {'bgrl': 'BGRL', 'ccassg': 'CCA-SSG', 'graphmae': 'GraphMAE', 'polygcl': 'PolyGCL', 'laplacegnn_full': 'LaplaceGNN (full)',
          'bgrl_dualfreq': 'BGRL + dual-freq.', 'ccassg_dualfreq': 'CCA-SSG + dual-freq.', 'bgrl_dualfreq_gate': 'BGRL + dual-freq., closed gate',
          'ccassg_dualfreq_gate': 'CCA-SSG + dual-freq., closed gate', 'bgrl_mlp': 'BGRL, MLP encoder', 'bgrl_sage': 'BGRL, GraphSAGE encoder',
          'bgrl_adv': 'BGRL + hidden adversary', 'laplacegnn': 'Spectral views + hidden adv.\\ + aux.\\ terms', 'laplacegnn_base': 'Spectral views + hidden adv.',
          'laplacegnn_noadv': 'Spectral views only', 'laplacegnn_v2': '+ structural adv., BYOL objective',
          'laplacegnn_cca': '+ structural adv., CCA objectives'}


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name + '.tex'), 'w').write(text)
    print('wrote', name)


def rank_marks(values):
    """Bold for the best mean in a column, underline for the second best."""
    vals = sorted({v for v in values if v is not None}, reverse=True)
    return (vals[0] if vals else None), (vals[1] if len(vals) > 1 else None)


def main_tables():
    best = {(r['dataset'], r['method']): r for r in R('best_runs.csv') if r['tag'] == 'clean'}
    groups = [('Baselines', ['bgrl', 'ccassg', 'graphmae', 'polygcl']),
              ('Encoder controls', ['bgrl_dualfreq', 'bgrl_dualfreq_gate', 'ccassg_dualfreq', 'ccassg_dualfreq_gate', 'bgrl_sage', 'bgrl_mlp']),
              ('Full method', ['laplacegnn_full'])]
    for tag, ds in (('main_homo', HOMO), ('main_het', HET)):
        cols = {d: [round(float(best[(d, m)]['test']), 1) if (d, m) in best else None for _, ms in groups for m in ms] for d in ds}
        marks = {d: rank_marks(cols[d]) for d in ds}
        lines = ['\\begin{tabular}{l' + 'c' * len(ds) + '}', '\\toprule', 'Method & ' + ' & '.join(FULLNAME[d] for d in ds) + ' \\\\']
        for g, ms in groups:
            lines.append('\\midrule')
            lines.append(f'\\multicolumn{{{len(ds) + 1}}}{{l}}{{\\textit{{{g}}}}} \\\\')
            for m in ms:
                cells = []
                for d in ds:
                    if (d, m) not in best:
                        cells.append(r'$\cdot$')
                        continue
                    mu, sd = round(float(best[(d, m)]['test']), 1), float(best[(d, m)]['test_std'])
                    s = f'{mu:.1f}\\,{{\\scriptsize$\\pm${sd:.1f}}}'
                    if mu == marks[d][0]:
                        s = f'\\textbf{{{mu:.1f}}}\\,{{\\scriptsize$\\pm${sd:.1f}}}'
                    elif mu == marks[d][1]:
                        s = f'\\underline{{{mu:.1f}}}\\,{{\\scriptsize$\\pm${sd:.1f}}}'
                    cells.append(s)
                lines.append(METHOD[m] + ' & ' + ' & '.join(cells) + ' \\\\')
        lines += ['\\bottomrule', '\\end{tabular}']
        write(tag, '\n'.join(lines) + '\n')


def paired_table():
    rows = R('dualfreq_vs_homophily.csv')
    by = {(r['dataset'], r['base'], r['variant']): r for r in rows}
    hom = {r['dataset']: float(r['adjusted_homophily']) for r in rows}
    ds = sorted(hom, key=hom.get)
    combos = [('bgrl', 'dualfreq', 'dual-freq.'), ('bgrl', 'dualfreq_gate', 'closed gate'), ('bgrl', 'sage', 'GraphSAGE'),
              ('bgrl', 'mlp', 'MLP'), ('ccassg', 'dualfreq', 'dual-freq.'), ('ccassg', 'dualfreq_gate', 'closed gate')]
    lines = ['\\begin{tabular}{lrr' + 'c' * len(combos) + '}', '\\toprule',
             ' & & & \\multicolumn{4}{c}{BGRL, GCN $\\rightarrow$} & \\multicolumn{2}{c}{CCA-SSG, GCN $\\rightarrow$} \\\\',
             '\\cmidrule(lr){4-7}\\cmidrule(lr){8-9}',
             'Dataset & $h_{\\mathrm{adj}}$ & $n$ & ' + ' & '.join(c[2] for c in combos) + ' \\\\', '\\midrule']
    for i, d in enumerate(ds):
        if i == 5:
            lines.append('\\midrule')
        n = next(r['trials'] for r in rows if r['dataset'] == d)
        cells = []
        for b, v, _ in combos:
            r = by.get((d, b, v))
            if r is None:
                cells.append(r'$\cdot$')
                continue
            mu, se, fr = float(r['mean_delta']), float(r['sem']), float(r['frac_better'])
            cells.append(f'${mu:+.1f}$\\,{{\\scriptsize$\\pm${se:.1f}}}\\,{{\\scriptsize({100 * fr:.0f}\\%)}}')
        lines.append(f'{FULLNAME[d]} & ${hom[d]:+.2f}$ & {n} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write('paired', '\n'.join(lines) + '\n')


def tuning_table():
    rows = R('tuning_summary.csv')
    by = {(r['dataset'], r['method']): r for r in rows}
    ds = HOMO + HET
    ms = ['bgrl', 'ccassg', 'graphmae', 'polygcl', 'laplacegnn_full', 'bgrl_dualfreq', 'ccassg_dualfreq', 'bgrl_dualfreq_gate',
          'ccassg_dualfreq_gate', 'bgrl_sage', 'bgrl_mlp', 'bgrl_adv', 'laplacegnn_noadv', 'laplacegnn_base', 'laplacegnn',
          'laplacegnn_v2', 'laplacegnn_cca']
    lines = ['\\begin{tabular}{l' + 'c' * len(ds) + '}', '\\toprule', 'Method & ' + ' & '.join(NAME[d] for d in ds) + ' \\\\', '\\midrule']
    for m in ms:
        if not any((d, m) in by for d in ds):
            continue
        cells = [f"{float(by[(d, m)]['test']):.1f}\\,{{\\scriptsize({by[(d, m)]['trials']})}}" if (d, m) in by else r'$\cdot$' for d in ds]
        lines.append(METHOD[m] + ' & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write('tuning', '\n'.join(lines) + '\n')


def datasets_table():
    rows = {r['dataset']: r for r in R('raw_and_homophily.csv')}
    info = {'cora': ('citation', 7, 'public Planetoid, 140/500/1000'), 'citeseer': ('citation', 6, 'public Planetoid, 120/500/1000'),
            'pubmed': ('citation', 3, 'public Planetoid, 60/500/1000'), 'amazon-photos': ('co-purchase', 8, '10 random, 10/10/80\\%'),
            'roman-empire': ('word dependency', 18, '10 official, 50/25/25\\%'), 'amazon-ratings': ('co-purchase', 5, '10 official, 50/25/25\\%'),
            'minesweeper': ('synthetic grid', 2, '10 official, 50/25/25\\%'), 'tolokers': ('crowd-sourcing', 2, '10 official, 50/25/25\\%'),
            'questions': ('Q\\&A users', 2, '10 official, 50/25/25\\%')}
    lines = ['\\begin{tabular}{llrrrcrrrl}', '\\toprule',
             'Dataset & Domain & Nodes & Edges & Classes & Metric & $h_{\\mathrm{edge}}$ & $h_{\\mathrm{adj}}$ & Raw / SGC probe & Splits \\\\', '\\midrule']
    for i, d in enumerate(HOMO + HET):
        if i == 4:
            lines.append('\\midrule')
        r = rows[d]
        dom, k, sp = info[d]
        metric = 'ROC-AUC' if r['metric'] == 'auc' else 'Acc.'
        lines.append(f"{FULLNAME[d]} & {dom} & {int(r['nodes']):,} & {int(r['edges']):,} & {k} & {metric} & {float(r['edge_homophily']):.2f} & "
                     f"${float(r['adjusted_homophily']):+.2f}$ & {float(r['raw_features_test']):.1f} / {float(r['sgc_features_test']):.1f} & {sp} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    write('datasets', '\n'.join(lines).replace(',', '{,}') + '\n')


def attack_table():
    best = {(r['dataset'], r['tag'], r['method']): r for r in R('best_runs.csv')}
    ms = ['bgrl', 'ccassg', 'graphmae', 'laplacegnn_full']
    tags = ['clean'] + [f'{a}-{b}' for a in ('random', 'dice', 'prbcd') for b in ('0.05', '0.1', '0.2')]
    label = lambda t: 'clean' if t == 'clean' else {'random': 'Random', 'dice': 'DICE', 'prbcd': 'PR-BCD'}[t.split('-')[0]] + f" {int(round(100 * float(t.split('-')[1])))}\\%"
    lines = ['\\begin{tabular}{l' + 'c' * 8 + '}', '\\toprule',
             ' & \\multicolumn{4}{c}{Cora} & \\multicolumn{4}{c}{CiteSeer} \\\\', '\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}',
             'Poisoning & ' + ' & '.join(['BGRL', 'CCA-SSG', 'GraphMAE', 'LaplaceGNN'] * 2) + ' \\\\', '\\midrule']
    for t in tags:
        cells = []
        for d in ('cora', 'citeseer'):
            vals = [round(float(best[(d, t, m)]['test']), 1) for m in ms]
            top = max(vals)
            for m, v in zip(ms, vals):
                sd = float(best[(d, t, m)]['test_std'])
                s = f'{v:.1f}\\,{{\\scriptsize$\\pm${sd:.1f}}}'
                cells.append(f'\\textbf{{{v:.1f}}}\\,{{\\scriptsize$\\pm${sd:.1f}}}' if v == top else s)
        lines.append(label(t) + ' & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write('attacks', '\n'.join(lines) + '\n')


def attribution_table():
    best = {(r['dataset'], r['tag'], r['method']): r for r in R('best_runs.csv')}
    conds = ['clean', 'random-0.2', 'dice-0.2', 'prbcd-0.2']
    head = ['clean', 'Rand.', 'DICE', 'PR-BCD']
    variants = {'attribution': [('', 'full method'), ('-nohid', 'no hidden-layer adversary'), ('-noadv', 'no adversarial component')],
                'attribution_full': [('', 'full method'), ('-nohid', 'no hidden-layer adversary'), ('-nostruct', 'no encoder-aware view update'),
                                     ('-noadv', 'no adversarial component')]}
    for name, vs in variants.items():
        lines = ['\\begin{tabular}{l' + 'c' * 8 + '}', '\\toprule',
                 ' & \\multicolumn{4}{c}{Cora} & \\multicolumn{4}{c}{CiteSeer} \\\\', '\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}',
                 'Variant & ' + ' & '.join(head * 2) + ' \\\\', '\\midrule']
        for suf, lab in vs:
            cells = []
            for d in ('cora', 'citeseer'):
                for c in conds:
                    r = best[(d, c + suf, 'laplacegnn_full')]
                    cells.append(f"{float(r['test']):.1f}\\,{{\\scriptsize$\\pm${float(r['test_std']):.1f}}}")
            lines.append(lab + ' & ' + ' & '.join(cells) + ' \\\\')
        lines += ['\\bottomrule', '\\end{tabular}']
        write(name, '\n'.join(lines) + '\n')


def dose_table():
    lines = ['\\begin{tabular}{lcccc}', '\\toprule', 'Views & Cora & CiteSeer & PubMed & Amazon-Photos \\\\', '\\midrule']
    data = {d: {r['level']: r for r in R(f'dose_{d}.csv')} for d in HOMO}
    for lv in ['alpha_0.0', 'alpha_0.25', 'alpha_0.5', 'alpha_0.75', 'alpha_1.0', 'random_same_budget']:
        lab = f"$\\alpha={lv.split('_')[1]}$" if lv.startswith('alpha') else 'random, same budget'
        if lv.startswith('random'):
            lines.append('\\midrule')
        cells = [f"{float(data[d][lv]['test']):.2f}\\,{{\\scriptsize$\\pm${float(data[d][lv]['test_std']):.2f}}}" for d in HOMO]
        lines.append(lab + ' & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write('dose', '\n'.join(lines) + '\n')


def sci(v):
    m, e = f'{v:.2e}'.split('e')
    return f'${m}\\times10^{{{int(e)}}}$'


def mechanism_table():
    rows = R('spectral_mechanism.csv')
    lab = {'ours-max': 'ours, max', 'ours-min': 'ours, min', 'span-style-max': 'SPAN-style, max', 'span-style-min': 'SPAN-style, min',
           'random-drop': 'random'}
    lines = ['\\begin{tabular}{llrcc}', '\\toprule', 'Dataset & Views & Expected flips & Relaxed change & Sampled change \\\\', '\\midrule']
    prev = None
    for r in rows:
        if prev and r['dataset'] != prev:
            lines.append('\\midrule')
        rel = float(r['relaxed_change'])
        rel_s = '$<10^{-20}$' if rel < 1e-20 else sci(rel)
        lines.append(f"{FULLNAME[r['dataset']] if r['dataset'] != prev else ''} & {lab[r['views']]} & {float(r['expected_flips']):.0f} & {rel_s} & "
                     f"{sci(float(r['sampled_change']))} \\\\")
        prev = r['dataset']
    lines += ['\\bottomrule', '\\end{tabular}']
    write('mechanism', '\n'.join(lines) + '\n')


def tu_table():
    tuned = {(r['dataset'], r['mode']): r for r in R('tu_tuned.csv')}
    rep = {(r['dataset'], r['view_mode']): r for r in R('tu_replication.csv')}
    lines = ['\\begin{tabular}{lcccc}', '\\toprule', ' & \\multicolumn{2}{c}{One fixed configuration} & \\multicolumn{2}{c}{Tuned (12 shared configurations)} \\\\',
             '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}', 'Dataset & Spectral & Random & Spectral & Random \\\\', '\\midrule']
    for d in ('MUTAG', 'PROTEINS', 'IMDB-BINARY'):
        c = [f"{float(rep[(d, m)]['test_mean']):.2f}\\,{{\\scriptsize$\\pm${float(rep[(d, m)]['test_std']):.2f}}}" for m in ('spectral', 'random')]
        c += [f"{float(tuned[(d, m)]['test']):.2f}\\,{{\\scriptsize$\\pm${float(tuned[(d, m)]['test_std']):.2f}}}" for m in ('spectral', 'random')]
        lines.append(f'{d} & ' + ' & '.join(c) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write('tu', '\n'.join(lines) + '\n')


def scaling_table():
    dense = {r['nodes']: r for r in R('scaling.csv') if r['method'].startswith('dense')}
    lines = ['\\begin{tabular}{lrrrcccc}', '\\toprule',
             ' & & & & \\multicolumn{2}{c}{Sparse (ours)} & \\multicolumn{2}{c}{Dense (original)} \\\\', '\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}',
             'Graph & Nodes & Edges & Candidates & Time (s) & Peak (GB) & Time (s) & Peak (GB) \\\\', '\\midrule']
    for r in R('scaling_sparse.csv'):
        dr = dense.get(r['nodes'])
        if dr and dr['status'] == 'ok':
            dc = f"{float(dr['seconds']):.1f} & {float(dr['peak_gb']):.2f}"
        elif dr and dr['status'] == 'OOM':
            dc = '\\multicolumn{2}{c}{out of memory}'
        else:
            dc = '\\multicolumn{2}{c}{not attempted}'
        g = 'ogbn-arXiv' if r['graph'] == 'ogbn-arxiv' else 'random'
        lines.append(f"{g} & {int(r['nodes']):,} & {int(r['edges']):,} & {int(r['candidates']):,} & {float(r['seconds']):.1f} & {float(r['peak_gb']):.2f} & {dc} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    write('scaling', '\n'.join(lines).replace(',', '{,}') + '\n')


def ablation_table():
    data = {d: {r['variant']: r for r in R(f'ablation_{d}.csv')} for d in HOMO}
    order = ['full', 'no_latent_prediction', 'random_views_same_budget', 'no_hidden_adversary', 'no_structural_adversary',
             'no_centrality_preservation', 'no_masked_reconstruction', 'no_spectral_positions', 'no_variance_covariance', 'no_curriculum',
             'pair_max_min', 'no_frequency_filter', 'gcn_encoder', 'byol_only', 'cca_only']
    lab = {'full': 'full method (test accuracy)', 'no_latent_prediction': 'without latent prediction', 'random_views_same_budget': 'random views, same budget',
           'no_hidden_adversary': 'without hidden-layer adversary', 'no_structural_adversary': 'without encoder-aware view update',
           'no_centrality_preservation': 'without centrality preservation', 'no_masked_reconstruction': 'without masked reconstruction',
           'no_spectral_positions': 'without spectral positions', 'no_variance_covariance': 'without variance/covariance term',
           'no_curriculum': 'without flip curriculum', 'pair_max_min': 'max/min instead of max/max pairing',
           'no_frequency_filter': 'unfiltered hidden adversary', 'gcn_encoder': 'GCN instead of dual-freq.\\ encoder',
           'byol_only': 'BYOL only (no CCA term)', 'cca_only': 'CCA only (no BYOL term)'}
    lines = ['\\begin{tabular}{lcccc}', '\\toprule', 'Variant & Cora & CiteSeer & PubMed & Amazon-Photos \\\\', '\\midrule']
    for v in order:
        cells = []
        for d in HOMO:
            r = data[d].get(v)
            if r is None:
                cells.append(r'$\cdot$')
            elif v == 'full':
                cells.append(f"{float(r['test']):.2f}\\,{{\\scriptsize$\\pm${float(r['test_std']):.2f}}}")
            else:
                x = float(r['delta_test'])
                cells.append('$0.00$' if abs(x) < 0.005 else f'${x:+.2f}$')
        lines.append(lab[v] + ' & ' + ' & '.join(cells) + ' \\\\')
        if v == 'full':
            lines.append('\\midrule')
    lines += ['\\bottomrule', '\\end{tabular}']
    write('ablation', '\n'.join(lines) + '\n')


def search_space_table():
    import tune_node as T

    def num(v):
        if isinstance(v, float) and v != 0 and (abs(v) < 1e-3):
            m, e = f'{v:.0e}'.split('e')
            return (f'{m}\\cdot' if m != '1' else '') + f'10^{{{int(e)}}}'
        return f'{v:g}'

    def fmt(spec):
        kind, *a = spec
        if kind == 'log':
            return f'log-uniform $[{num(a[0])}, {num(a[1])}]$'
        if kind == 'uniform':
            return f'uniform $[{num(a[0])}, {num(a[1])}]$'
        return '$\\{' + ', '.join(num(v) if isinstance(v, (int, float)) else '\\text{' + v.replace('_', '\\_') + '}' for v in a[0]) + '\\}$'

    blocks = [('Training (BGRL family and LaplaceGNN)', T.TRAINING_SPACE), ('Random views', T.RANDOM_VIEWS),
              ('Spectral views (LaplaceGNN)', T.SPECTRAL), ('Hidden-layer adversary', T.ADVERSARIAL),
              ('Objective and auxiliary terms (LaplaceGNN)', T.OBJECTIVE), ('Encoder-aware view update (LaplaceGNN)', T.STRUCTURAL),
              ('CCA-SSG', T.CCASSG), ('GraphMAE', T.GRAPHMAE), ('PolyGCL', T.POLYGCL)]
    full = T.METHODS['laplacegnn_full'][1]
    extra = {k: full[k] for k in ('epochs', 'objective', 'cca_lambda', 'cca_weight', 'latent_weight', 'latent_pe_dim')}
    blocks.insert(6, ('LaplaceGNN (full), overrides and additions', extra))
    lines = ['\\begin{tabular}{lll}', '\\toprule', 'Group & Hyperparameter & Distribution \\\\', '\\midrule']
    for i, (g, space) in enumerate(blocks):
        if i:
            lines.append('\\midrule')
        for j, (k, spec) in enumerate(space.items()):
            if g == 'CCA-SSG' and k.startswith('drop_'):
                continue
            key = k.replace('_', '\\_')
            lines.append(f"{g if j == 0 else ''} & \\texttt{{{key}}} & {fmt(spec)} \\\\")
        if g == 'CCA-SSG':
            lines.append(' & \\texttt{drop\\_*} & as in random views \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write('search_space', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    for fn in (main_tables, paired_table, tuning_table, datasets_table, attack_table, attribution_table, dose_table, mechanism_table,
               tu_table, scaling_table, ablation_table, search_space_table):
        fn()
