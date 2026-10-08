"""Vector figures for the paper, generated from results/*.csv and runs/tune -> paper/figures/*.pdf."""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'paper', 'figures')
R = lambda name: list(csv.DictReader(open(os.path.join(ROOT, 'results', name))))

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
NAME = {'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed', 'amazon-photos': 'Amazon-Photos', 'roman-empire': 'Roman-empire',
        'amazon-ratings': 'Amazon-ratings', 'minesweeper': 'Minesweeper', 'tolokers': 'Tolokers', 'questions': 'Questions'}
HOMO = ['cora', 'citeseer', 'pubmed', 'amazon-photos']
HET = ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']
W = 6.5

plt.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.6,
    'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'lines.linewidth': 1.4, 'lines.markersize': 4.5, 'errorbar.capsize': 1.8, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42, 'text.latex.preamble': r'\usepackage{amsmath}',
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    plt.close(fig)
    print('wrote', name)


def trials(d, m):
    out = {}
    for f in glob.glob(os.path.join(ROOT, 'runs', 'tune', d, m, 'trials-shard*.jsonl')):
        for line in open(f):
            t = json.loads(line)
            if 'error' not in t:
                out[t['trial']] = 100 * t['test']
    return out


def fig_encoder_homophily():
    rows = R('dualfreq_vs_homophily.csv')
    by = {(r['dataset'], r['base'], r['variant']): r for r in rows}
    hom = {r['dataset']: float(r['adjusted_homophily']) for r in rows}
    ds = sorted(hom, key=hom.get)
    x = np.arange(len(ds))
    fig, axes = plt.subplots(2, 1, figsize=(W, 3.6), sharex=True, gridspec_kw={'height_ratios': [1.6, 1]})
    series = [('dualfreq', 'dual-frequency, open gate', BLUE, 'o'), ('dualfreq_gate', 'dual-frequency, closed gate', AQUA, 's'),
              ('sage', 'GraphSAGE', ORANGE, 'D')]
    ax = axes[0]
    for k, (v, lab, c, mk) in enumerate(series):
        off = (k - 1) * 0.2
        ys = [float(by[(d, 'bgrl', v)]['mean_delta']) for d in ds]
        es = [float(by[(d, 'bgrl', v)]['sem']) for d in ds]
        ax.errorbar(x + off, ys, yerr=es, fmt=mk, color=c, mec='white', mew=0.6, ms=4.5, elinewidth=0.9, label=lab)
    ax.set_ylim(-11, 29)
    ax.set_ylabel(r'paired $\Delta$ (points)')
    ax.set_title('(a) Encoders that keep the node apart from its neighbours', loc='left')
    ax.legend(loc='upper center', ncol=3, handletextpad=0.3, borderaxespad=0.1, bbox_to_anchor=(0.62, 1.0))
    ax2 = axes[1]
    ys = [float(by[(d, 'bgrl', 'mlp')]['mean_delta']) for d in ds]
    es = [float(by[(d, 'bgrl', 'mlp')]['sem']) for d in ds]
    ax2.errorbar(x, ys, yerr=es, fmt='v', color=VIOLET, mec='white', mew=0.6, ms=5, elinewidth=0.9)
    ax2.set_ylim(-44, 26)
    ax2.set_ylabel(r'paired $\Delta$ (points)')
    ax2.set_title('(b) Encoder that ignores the graph (MLP)', loc='left')
    for a in axes:
        a.axvspan(-0.5, 4.5, color='#f0efec', zorder=0, lw=0)
        a.axhline(0, color=INK2, lw=0.7)
        a.grid(axis='x', visible=False)
        a.set_xlim(-0.6, len(ds) - 0.4)
    axes[0].text(2, -9.5, 'heterophilous graphs', ha='center', color=INK2, fontsize=7)
    axes[0].text(6.5, 19, 'homophilous graphs', ha='center', color=INK2, fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{NAME[d]} ({hom[d]:+.2f})' for d in ds], rotation=25, ha='right', fontsize=7)
    fig.tight_layout(h_pad=0.6)
    save(fig, 'encoder_homophily')


def fig_trial_scatter():
    ds = ['roman-empire', 'minesweeper', 'cora', 'citeseer']
    variants = [('bgrl_dualfreq', 'dual-frequency', BLUE, 'o'), ('bgrl_dualfreq_gate', 'closed gate', AQUA, 's'),
                ('bgrl_sage', 'GraphSAGE', ORANGE, 'D'), ('bgrl_mlp', 'MLP', VIOLET, 'v')]
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.9))
    for ax, d in zip(axes, ds):
        base = trials(d, 'bgrl')
        lo, hi = np.inf, -np.inf
        for m, lab, c, mk in variants:
            t = trials(d, m)
            common = sorted(set(base) & set(t))
            xs, ys = [base[i] for i in common], [t[i] for i in common]
            ax.scatter(xs, ys, s=11, marker=mk, color=c, edgecolors='white', linewidths=0.4, label=lab, zorder=3)
            lo, hi = min(lo, *xs, *ys), max(hi, *xs, *ys)
        pad = 0.04 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, lw=0.7, ls='--', zorder=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect('equal')
        ax.set_title(NAME[d])
        ax.set_xlabel('GCN encoder, test')
    axes[0].set_ylabel('variant, test')
    handles = [Line2D([], [], marker=mk, color=c, ls='', mec='white', mew=0.4, ms=4.5, label=lab) for _, lab, c, mk in variants]
    fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.6)
    save(fig, 'trial_scatter')


def fig_dose():
    fig, axes = plt.subplots(2, 4, figsize=(W, 3.1), sharex=True)
    for j, d in enumerate(HOMO):
        rows = R(f'dose_{d}.csv')
        al = [r for r in rows if r['level'].startswith('alpha_')]
        rnd = next(r for r in rows if r['level'] == 'random_same_budget')
        a = np.array([float(r['level'].split('_')[1]) for r in al])
        acc, sd = np.array([float(r['test']) for r in al]), np.array([float(r['test_std']) for r in al])
        ch = np.array([float(r['spectral_change']) for r in al])
        ax = axes[0, j]
        ax.axhspan(float(rnd['test']) - float(rnd['test_std']), float(rnd['test']) + float(rnd['test_std']), color=ORANGE, alpha=0.13, lw=0)
        ax.axhline(float(rnd['test']), color=ORANGE, lw=1.0, ls='--')
        ax.errorbar(a, acc, yerr=sd, color=BLUE, marker='o', mec='white', mew=0.6, elinewidth=0.9)
        ax.set_title(NAME[d])
        ax2 = axes[1, j]
        ax2.plot(a, ch, color=BLUE, marker='o', mec='white', mew=0.6)
        ax2.axhline(float(rnd['spectral_change']), color=ORANGE, lw=1.0, ls='--')
        ax2.set_yscale('log')
        ax2.set_xlabel(r'mixing weight $\alpha$')
        ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax2.set_xticklabels(['0', '', '0.5', '', '1'])
        lo, hi = min(ch.min(), float(rnd['spectral_change'])), max(ch.max(), float(rnd['spectral_change']))
        ax2.set_ylim(lo / 2.5, hi * 2.5)
    axes[0, 0].set_ylabel('test accuracy (\\%)')
    axes[1, 0].set_ylabel('sampled spectral change')
    handles = [Line2D([], [], color=BLUE, marker='o', mec='white', label=r'spectral views, $\alpha\Delta_{\max}+(1-\alpha)\Delta_{\min}$'),
               Line2D([], [], color=ORANGE, ls='--', label='random views, same expected flips')]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    save(fig, 'dose_response')


def fig_mechanism():
    rows = R('spectral_mechanism.csv')
    ds = ['cora', 'citeseer', 'pubmed', 'amazon-photos', 'roman-empire', 'amazon-ratings']
    views = [('ours-max', 'ours, max'), ('ours-min', 'ours, min'), ('random-drop', 'random'), ('span-style-max', 'SPAN-style, max'),
             ('span-style-min', 'SPAN-style, min')]
    colors = {'ours-max': BLUE, 'ours-min': AQUA, 'random-drop': ORANGE, 'span-style-max': VIOLET, 'span-style-min': MAGENTA}
    floor = 1e-7
    fig, axes = plt.subplots(1, 6, figsize=(W, 2.3), sharey=True)
    for ax, d in zip(axes, ds):
        have = {r['views']: r for r in rows if r['dataset'] == d}
        vs = [v for v, _ in views if v in have]
        for i, v in enumerate(vs):
            r = have[v]
            rel, smp = max(float(r['relaxed_change']), floor), float(r['sampled_change'])
            c = colors[v]
            ax.plot([i, i], [rel, smp], color=c, lw=1.0, alpha=0.7)
            ax.scatter([i], [rel], s=22, facecolors='white', edgecolors=c, linewidths=1.1, zorder=3)
            ax.scatter([i], [smp], s=22, color=c, edgecolors='white', linewidths=0.5, zorder=4)
            if float(r['relaxed_change']) < floor:
                ax.annotate(r'$\approx 0$', (i, floor), xytext=(5, -1), textcoords='offset points', ha='left', va='center', fontsize=6, color=INK2)
        ax.set_yscale('log')
        ax.set_xticks(range(len(vs)))
        ax.set_xticklabels([dict(views)[v] for v in vs], rotation=90, fontsize=6.5)
        ax.set_xlim(-0.6, len(vs) - 0.4)
        ax.set_title(NAME[d], fontsize=7.5)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylim(floor / 3, 0.2)
    axes[0].set_ylabel('relative spectral change')
    handles = [Line2D([], [], marker='o', ls='', mfc='white', mec=INK2, label='relaxed (expected) graph'),
               Line2D([], [], marker='o', ls='', color=INK2, mec='white', label='sampled views (mean of 10)')]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=0.4)
    save(fig, 'relaxed_vs_sampled')


def fig_attacks():
    best = {(r['dataset'], r['tag'], r['method']): r for r in R('best_runs.csv')}
    ms = [('bgrl', 'BGRL', BLUE, 'o'), ('ccassg', 'CCA-SSG', ORANGE, 's'), ('graphmae', 'GraphMAE', AQUA, 'D'),
          ('laplacegnn_full', 'LaplaceGNN', VIOLET, '^')]
    rates = [0, 0.05, 0.1, 0.2]
    fig, axes = plt.subplots(2, 3, figsize=(W, 3.3), sharex=True)
    for i, d in enumerate(['cora', 'citeseer']):
        for j, (atk, title) in enumerate([('random', 'random insertion'), ('dice', 'DICE'), ('prbcd', 'PR-BCD')]):
            ax = axes[i, j]
            for m, lab, c, mk in ms:
                tags = ['clean'] + [f'{atk}-{r}' for r in rates[1:]]
                y = [float(best[(d, t, m)]['test']) for t in tags]
                e = [float(best[(d, t, m)]['test_std']) for t in tags]
                ax.errorbar(np.array(rates) * 100, y, yerr=e, color=c, marker=mk, mec='white', mew=0.5, ms=4, elinewidth=0.8, label=lab)
            ax.set_title(f'{NAME[d]}, {title}')
            if i == 1:
                ax.set_xlabel('perturbed edges (\\% of $|E|$)')
            ax.set_xticks([0, 5, 10, 20])
        axes[i, 0].set_ylabel('test accuracy (\\%)')
    handles = [Line2D([], [], color=c, marker=mk, mec='white', label=lab) for _, lab, c, mk in ms]
    fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(h_pad=0.7, w_pad=0.6)
    save(fig, 'attacks')


def fig_attribution():
    best = {(r['dataset'], r['tag'], r['method']): r for r in R('best_runs.csv')}
    conds = [('clean', 'clean'), ('random-0.2', 'random'), ('dice-0.2', 'DICE'), ('prbcd-0.2', 'PR-BCD')]
    vars_ = [('', 'full method', VIOLET, '^'), ('-nohid', 'no hidden-layer adversary', BLUE, 'o'),
             ('-noadv', 'no adversarial component', ORANGE, 'D')]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.1), sharey=False)
    for ax, d in zip(axes, ['cora', 'citeseer']):
        for k, (suf, lab, c, mk) in enumerate(vars_):
            off = (k - 1) * 0.2
            y = [float(best[(d, t + suf, 'laplacegnn_full')]['test']) for t, _ in conds]
            e = [float(best[(d, t + suf, 'laplacegnn_full')]['test_std']) for t, _ in conds]
            ax.errorbar(np.arange(4) + off, y, yerr=e, fmt=mk, color=c, mec='white', mew=0.5, ms=4.5, elinewidth=0.9, label=lab)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f'{n}' + ('' if t == 'clean' else ' 20\\%') for t, n in conds])
        ax.set_title(NAME[d])
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('test accuracy (\\%)')
    handles = [Line2D([], [], marker=mk, color=c, ls='', mec='white', label=lab) for _, lab, c, mk in vars_]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=1.0)
    save(fig, 'attack_attribution')


def fig_scaling():
    dense = [r for r in R('scaling.csv') if r['method'].startswith('dense') and r['status'] == 'ok']
    sparse = R('scaling_sparse.csv')
    rnd = [r for r in sparse if r['graph'] != 'ogbn-arxiv']
    arx = next(r for r in sparse if r['graph'] == 'ogbn-arxiv')
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3))
    for ax, key, lab in ((axes[0], 'seconds', 'time for 10 iterations (s)'), (axes[1], 'peak_gb', 'peak GPU memory (GB)')):
        xs = [int(r['nodes']) for r in rnd]
        ax.plot(xs, [float(r[key]) for r in rnd], color=BLUE, marker='o', mec='white', mew=0.6, label='sparse candidate set (ours)')
        ax.plot([int(r['nodes']) for r in dense], [float(r[key]) for r in dense], color=ORANGE, marker='s', mec='white', mew=0.6,
                label='dense $n\\times n$ (original)')
        ax.scatter([int(arx['nodes'])], [float(arx[key])], marker='*', s=70, color=BLUE, edgecolors='white', linewidths=0.5, zorder=4)
        ax.annotate('ogbn-arXiv', (int(arx['nodes']), float(arx[key])), xytext=(-6, 6), textcoords='offset points', ha='right', fontsize=6.5, color=INK2)
        ax.axvline(20000, color=RED, lw=0.8, ls=':')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('nodes (average degree 10)')
        ax.set_ylabel(lab)
    axes[1].axhline(24, color=INK2, lw=0.7, ls='--')
    axes[1].text(1100, 26, '24 GB card', fontsize=6.5, color=INK2, va='bottom')
    axes[1].set_ylim(0.02, 60)
    axes[0].text(21500, 0.12, 'dense runs out\nof memory', fontsize=6.5, color=RED, va='bottom')
    axes[0].legend(loc='upper left')
    fig.tight_layout(w_pad=1.5)
    save(fig, 'scaling')


def fig_ablation():
    labels = {'no_latent_prediction': 'latent prediction', 'random_views_same_budget': 'spectral views (vs.\\ random)',
              'no_hidden_adversary': 'hidden-layer adversary',
              'no_centrality_preservation': 'centrality preservation', 'no_masked_reconstruction': 'masked reconstruction',
              'no_spectral_positions': 'spectral positions', 'no_variance_covariance': 'variance/covariance',
              'no_curriculum': 'flip curriculum', 'pair_max_min': 'max/max pairing (vs.\\ max/min)',
              'no_frequency_filter': 'frequency filter of adversary', 'gcn_encoder': 'dual-frequency encoder (vs.\\ GCN)',
              'byol_only': 'CCA term (vs.\\ BYOL only)', 'cca_only': 'BYOL term (vs.\\ CCA only)'}
    data = {d: {r['variant']: float(r['delta_test']) for r in R(f'ablation_{d}.csv') if r['variant'] != 'full'} for d in HOMO}
    order = list(labels)
    M = np.full((len(order), len(HOMO)), np.nan)
    for i, v in enumerate(order):
        for j, d in enumerate(HOMO):
            if v in data[d]:
                M[i, j] = -data[d][v]
    lim = 2.5
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('div', ['#e34948', '#f3b7b1', '#f0efec', '#9ec5f4', '#2a78d6'])
    cmap.set_bad('white')
    fig, ax = plt.subplots(figsize=(W * 0.62, 3.3))
    ax.imshow(np.clip(M, -lim, lim), cmap=cmap, vmin=-lim, vmax=lim, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, r'$\cdot$', ha='center', va='center', color=MUTED, fontsize=8)
            else:
                ax.text(j, i, f'{M[i, j] + 0.0:+.2f}'.replace('-0.00', '0.00').replace('+0.00', '0.00'), ha='center', va='center', fontsize=6.5, color='white' if abs(M[i, j]) > 1.6 else INK)
    ax.set_xticks(range(len(HOMO)))
    ax.set_xticklabels([NAME[d] for d in HOMO])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[v] for v in order])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    save(fig, 'ablation')


def fig_tu():
    tuned = R('tu_tuned.csv')
    rep = R('tu_replication.csv')
    ds = ['MUTAG', 'PROTEINS', 'IMDB-BINARY']
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.9))
    for ax, d in zip(axes, ds):
        for k, (rows, key, mkey, lab) in enumerate(((rep, 'test_mean', 'view_mode', 'one fixed configuration'),
                                                     (tuned, 'test', 'mode', 'tuned (12 shared configurations)'))):
            for m, c, off in (('spectral', BLUE, -0.08), ('random', ORANGE, 0.08)):
                r = next(r for r in rows if r['dataset'] == d and r[mkey] == m)
                ax.errorbar([k + off], [float(r[key])], yerr=[float(r['test_std'])], fmt='o', color=c, mec='white', mew=0.5, ms=5, elinewidth=0.9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['fixed config.', 'tuned'])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(d)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('10-fold CV accuracy (\\%)')
    handles = [Line2D([], [], marker='o', ls='', color=BLUE, mec='white', label='spectral views'),
               Line2D([], [], marker='o', ls='', color=ORANGE, mec='white', label='random views, same budget')]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout(w_pad=1.0)
    save(fig, 'tu_views')


def fig_probes():
    rows = {r['dataset']: r for r in R('raw_and_homophily.csv')}
    best = {(r['dataset'], r['method']): float(r['test']) for r in R('best_runs.csv') if r['tag'] == 'clean'}
    ds = sorted(rows, key=lambda d: float(rows[d]['adjusted_homophily']))
    fig, ax = plt.subplots(figsize=(W, 2.2))
    x = np.arange(len(ds))
    pts = [('raw_features_test', 'raw features, no graph', VIOLET, 'v'), ('sgc_features_test', 'two-hop averaged features (SGC)', ORANGE, 's')]
    for key, lab, c, mk in pts:
        ax.scatter(x, [float(rows[d][key]) for d in ds], marker=mk, s=24, color=c, edgecolors='white', linewidths=0.5, zorder=3, label=lab)
    ax.scatter(x, [best[(d, 'bgrl')] for d in ds], marker='o', s=24, color=BLUE, edgecolors='white', linewidths=0.5, zorder=3, label='BGRL, GCN encoder')
    ax.scatter(x, [best[(d, 'bgrl_dualfreq')] for d in ds], marker='D', s=22, color=AQUA, edgecolors='white', linewidths=0.5, zorder=3,
               label='BGRL, dual-frequency encoder')
    for i, d in enumerate(ds):
        ys = [float(rows[d]['raw_features_test']), float(rows[d]['sgc_features_test']), best[(d, 'bgrl')], best[(d, 'bgrl_dualfreq')]]
        ax.plot([i, i], [min(ys), max(ys)], color=GRID, lw=2.5, zorder=1, solid_capstyle='round')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{NAME[d]} ({float(rows[d]["adjusted_homophily"]):+.2f})' for d in ds], rotation=20, ha='right')
    ax.set_ylabel('test score (\\%)')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, 'feature_probes')


if __name__ == '__main__':
    for fn in (fig_encoder_homophily, fig_trial_scatter, fig_dose, fig_mechanism, fig_attacks, fig_attribution, fig_scaling,
               fig_ablation, fig_tu, fig_probes):
        fn()
