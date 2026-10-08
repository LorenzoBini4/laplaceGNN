"""Builds results/registry.csv (every number in the paper and rebuttal) and results/inconsistencies.csv."""
import csv
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, 'tmlr_submission/LaplaceGNN_TMLR/tmlr-style-file-main/main.tex')
REBUTTAL = os.path.join(ROOT, 'rebutall_and_reviews.md')
OUT = os.path.join(ROOT, 'results')

DATASETS = {
    'wikics': 'WikiCS', 'am. computers': 'Amazon-Computers', 'am. photos': 'Amazon-Photos', 'coauthor cs': 'Coauthor-CS',
    'coauthor-cs': 'Coauthor-CS', 'coauthor physics': 'Coauthor-Physics', 'ogbn-arxiv': 'ogbn-arXiv', 'mutag': 'MUTAG',
    'proteins': 'PROTEINS', 'imdb-b': 'IMDB-B', 'imdb-m': 'IMDB-M', 'collab': 'COLLAB', 'nci1': 'NCI1', 'hiv': 'HIV',
    'tox21': 'Tox21', 'toxcast': 'ToxCast', 'bbbp': 'BBBP', 'ppi': 'PPI', 'cora': 'Cora', 'citeseer': 'CiteSeer',
    'pubmed': 'PubMed', 'polblogs': 'Polblogs',
}

# label -> (task, column names or None to read them from the header row)
PAPER_TABLES = {
    'tab3:node_class': ('node', None),
    'appendix:tab3:node_class': ('node', None),
    'tab1:graph_class_mutag': ('tu-graph', None),
    'appendix:tab1:graph_class_mutag': ('tu-graph', None),
    'tab7:transfer_learning': ('transfer', ['BBBP', 'Tox21', 'HIV', 'ToxCast', 'PPI']),
    'appendix:tab7:transfer_learning': ('transfer', ['BBBP', 'Tox21', 'HIV', 'ToxCast', 'PPI']),
    'tab4:ablation_centralities-node-classification': ('node', None),
    'tab4b:ablation_centralities_on_graph_classification': ('ogb-mol', None),
    'appendix:tab2:graph_class_ogb': ('ogb-mol', None),
    'tab5:node_class_ogn-arXiv': ('node', ['ogbn-arXiv (val)', 'ogbn-arXiv']),
    'tab6:grap_class_PPI': ('ppi-inductive', ['PPI']),
    'appendix:tab:memory_experiment': ('memory-gb', None),
    'appendix:tab8:adv_attacks': ('attack', ['Cora clean', 'Cora random-0.05', 'Cora random-0.2', 'Cora dice-0.05', 'Cora dice-0.2',
                                            'Cora gfattack-0.05', 'Cora gfattack-0.2', 'Cora mettack-0.05', 'Cora mettack-0.2']),
}

# rebuttal tables in order of appearance -> (task, key columns forming the method name, dataset columns renamed)
REBUTTAL_TABLES = [
    ('R1', 'Table 1 decoupling', 'node', 2, {}),
    ('R1', 'Table 2 CCA-SSG', 'node', 1, {'Training Time (s/epoch)': 'time s/epoch', 'Neg. Samples': None}),
    ('R1', 'Wasserstein (first reply)', 'wasserstein-cora', 1, {'Interpretation': None}),
    ('R1', 'Wasserstein G1 vs G2', 'wasserstein-cora', 1, {'Interpretation': None}),
    ('R2', 'SPAN comparison', 'node|tu-graph', 1, {}),
    ('R3', 'Memory (copy of Table 7)', 'memory-gb', 1, {}),
    ('R3', 'Efficiency ogbn-arXiv', 'node', 1, {'Sampling Strategy': None, 'Pairs per Epoch (approx.)': None,
                                                 'GPU Memory (GB)': 'ogbn-arXiv memory-gb', 'Time to Convergence (min)': 'ogbn-arXiv minutes',
                                                 'Test Acc (%)': 'ogbn-arXiv'}),
    ('R3', 'Mettack sigma=0.2', 'attack', 1, {'Citeseer (Small)': 'CiteSeer mettack-0.2', 'Polblogs (Heterophilous)': 'Polblogs mettack-0.2',
                                             'Pubmed (Large)': 'PubMed mettack-0.2', 'Coauthor-CS (Dense)': 'Coauthor-CS mettack-0.2'}),
]

NODE_MAIN = ('ssl_adv_node/run_adv_node.py',
             'trained on random DropEdges/DropFeatures views in every committed version (spectral views computed, never used); '
             'encoder forward skipped all norm/PReLU (linear 2-layer GCN); no validation-based checkpoint selection '
             '(test printed every eval, Amazon/Coauthor splits had no validation set); cfg weight_decay ignored (8e-4 used)')
NODE_ALT = ('ssl_adv_graph/tudataset/run_adv_node.py (cached Delta dated 2025-02-21)',
            'single random 10/10/80 split (fixed seed 15) instead of the public split; features masked with p=0.4; '
            'default m=1 means no adversarial ascent step; encoder forward was the linear legacy one; '
            'centrality initialisation ~1/n^2 (sum-normalised)')
TOO_LARGE = ('none (spectral module is dense)', 'graph exceeds what the dense spectral module fits on a 24 GB GPU; '
             'cannot come from spectral views; ogbn-arXiv branch of the committed script also crashes (transform undefined) '
             'and its 3-layer config ran as 2 layers')
TU = ('ssl_adv_graph/tudataset/run_adv_graph.py',
      'all flip probabilities are exactly 0 (Delta initialised at 0, zero gradient) so views = input graph; '
      'centrality unused; batched views mis-mapped (moot); evaluation = one random 80/10/10 split, one seed '
      '(paper: 10-fold CV x 5 runs); encoder GCN (paper: GIN)')
TU_NOFEAT = (TU[0], TU[1] + '; featureless dataset: the committed training loop needs node features, so it could not run as committed')
OGB = ('ssl_adv_graph/ogb/run_adv_graph.py',
       'spectral views overwritten by random drops of the original batch (and all probabilities are 0 anyway); '
       'online/target view-swap bug; MLP probe (paper: linear) kept training across evaluations; EMA before optimizer step; '
       'metric is ROC-AUC but the paper says accuracy')
MISSING = lambda what: ('none in repo', f'no {what} code in the repository')


def classify(task, dataset, method, location):
    m = method.lower()
    if location.startswith('rebuttal'):
        return MISSING('code for this rebuttal experiment')
    if task in ('transfer', 'ppi-inductive'):
        return MISSING('transfer / PPI (run_adv_node_ppi referenced in main_node.sh does not exist)')
    if task == 'attack':
        return MISSING('adversarial attack (DeepRobust/Mettack/DICE/GF-Attack)')
    if task == 'memory-gb':
        return MISSING('memory profiling')
    if task == 'node':
        if '-dc' in m or '-pc' in m or '-kc' in m:
            return ('ssl_adv_node/run_adv_node.py', 'Katz centrality not implemented; centrality weights hardcoded [0.2,0.3,0.5]; ' + NODE_MAIN[1])
        if dataset in ('Coauthor-Physics', 'ogbn-arXiv', 'ogbn-arXiv (val)'):
            return TOO_LARGE
        if dataset in ('Cora', 'CiteSeer', 'PubMed', 'WikiCS') and dataset != 'WikiCS':
            return NODE_ALT
        if dataset == 'WikiCS':
            return (NODE_MAIN[0] + ' or ' + NODE_ALT[0], 'cached Delta exists from the alternate script; ' + NODE_MAIN[1])
        return NODE_MAIN
    if task == 'tu-graph':
        return TU_NOFEAT if dataset in ('IMDB-B', 'IMDB-M', 'COLLAB') else TU
    if task == 'ogb-mol':
        extra = 'Katz centrality not implemented; graph augmentor ignores centrality (Delta starts at 0); ' if ('-dc' in m or '-pc' in m or '-kc' in m) else ''
        return (OGB[0], extra + OGB[1])
    return ('?', '')


def clean_cell(c):
    c = re.sub(r'\\citeyearpar\{[^}]*\}|\\cite[pt]?\{[^}]*\}', '', c)
    c = re.sub(r'\\multicolumn\{\d+\}\{\w+\}\{(.*)\}', r'\1', c)
    for macro in ('textbf', 'underline', 'textit'):
        c = re.sub(r'\\%s\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}' % macro, r'\1', c)
    c = c.replace('$\\pm$', '±').replace('\\pm', '±').replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
    return re.sub(r'\s+', ' ', c).strip()


def parse_value(c):
    c = c.replace('*', '').replace('%', '').strip()
    m = re.match(r'^([-+]?\d[\d,]*\.?\d*)\s*(?:±\s*(\d+\.?\d*))?$', c)
    if not m:
        return None, None
    return float(m.group(1).replace(',', '')), (float(m.group(2)) if m.group(2) else None)


def norm_dataset(name):
    key = name.strip().lower()
    return DATASETS.get(key, name.strip())


def norm_method(name):
    m = re.sub(r'\s+', ' ', name.replace('*', '')).strip()
    low = m.lower()
    if 'laplacegnn' in low:
        if 'random' in low:
            return 'LaplaceGNN + random augmentation'
        variant = re.search(r'-(dc|pc|kc)\b', low)
        return f'LaplaceGNN-{variant.group(1).upper()}' if variant else 'LaplaceGNN'
    if low.startswith('grace') and 'specaug' in low:
        return 'GRACE + SpecAug'
    if low.startswith('grace-subsampling'):
        return 'GRACE (subsampled)'
    if low.startswith('span'):
        return 'SPAN'
    if low in ('gcn (supervised)', 'gcn'):
        return 'Supervised GCN'
    m = re.sub(r'\s*/\s*random \(edge/feat\)$', '', m, flags=re.I)
    m = re.sub(r'\s*\((base|contrastive|bootstrap|non-contrastive)\)', '', m, flags=re.I)
    return m.strip()


def paper_tables():
    tex = open(PAPER).read()
    tex = '\n'.join(l for l in tex.split('\n') if not l.lstrip().startswith('%'))
    for block in re.findall(r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}', tex, re.S):
        label = re.search(r'\\label\{([^}]*)\}', block)
        if not label or label.group(1) not in PAPER_TABLES:
            continue
        label = label.group(1)
        task, cols = PAPER_TABLES[label]
        body = re.search(r'\\toprule(.*?)\\bottomrule', block, re.S).group(1)
        header, rows = body.split('\\midrule', 1)
        if cols is None:
            cols = [clean_cell(c) for c in header.split('\\\\')[0].split('&')][1:]
        for line in rows.replace('\\midrule', '').split('\\\\'):
            if '&' not in line:
                continue
            cells = [clean_cell(c) for c in line.split('&')]
            yield f'paper:{label}', task, cells[0], cols, cells[1:]


def rebuttal_tables():
    lines = open(REBUTTAL).read().split('\n')
    tables, cur = [], []
    for l in lines:
        if l.strip().startswith('|'):
            cur.append(l)
        elif cur:
            tables.append(cur)
            cur = []
    for (reviewer, name, task, nkey, rename), t in zip(REBUTTAL_TABLES, tables):
        header = [c.strip() for c in t[0].strip().strip('|').split('|')]
        section = None
        for row in t[2:]:
            cells = [c.strip() for c in row.strip().strip('|').split('|')]
            if all(not c for c in cells[1:]):
                section = cells[0].strip('* ')
                continue
            method = ' / '.join(c.strip('* ') for c in cells[:nkey])
            cols, vals = [], []
            for h, v in zip(header[nkey:], cells[nkey:]):
                h2 = rename.get(h, h)
                if h2 is None:
                    continue
                cols.append(h2)
                vals.append(v)
            if task == 'node|tu-graph':
                row_task = 'node' if section == 'Node Classification' else 'tu-graph'
            else:
                row_task = task
            yield f'rebuttal:{reviewer}:{name}', row_task, method, cols, vals


def split_column(task, col):
    """Returns (task, dataset) for a column header."""
    if task == 'attack' or ' mettack' in col or col.endswith(('memory-gb', 'minutes', 'clean')) or 'random-' in col or '-0.' in col:
        parts = col.split(' ', 1)
        ds, cond = norm_dataset(parts[0]), parts[1] if len(parts) > 1 else ''
        if cond == 'clean':
            return 'node', ds
        if cond == 'memory-gb':
            return 'memory-gb', ds
        if cond == 'minutes':
            return 'time-min', ds
        return f'attack:{cond}', ds
    if col == 'time s/epoch':
        return 'time-s-per-epoch', 'Cora'
    return task, norm_dataset(col)


def main():
    os.makedirs(OUT, exist_ok=True)
    records = []
    for source in (paper_tables(), rebuttal_tables()):
        for location, task, method_raw, cols, vals in source:
            for col, raw in zip(cols, vals):
                value, std = parse_value(raw)
                row_task, dataset = split_column(task, col)
                if task == 'wasserstein-cora':
                    row_task, dataset = f'wasserstein:{method_raw}', 'Cora'
                    method = col
                else:
                    method = norm_method(method_raw)
                if value is None and raw.strip() not in ('--', '-', 'OOM'):
                    continue
                if method.lower().startswith('improvement'):
                    continue
                if method == 'GRACE' and dataset == 'ogbn-arXiv':
                    method = 'GRACE (subsampled)'
                ours = 'laplacegnn' in method_raw.lower() or 'ours' in method_raw.lower() or 'ours' in col.lower() or 'view $g' in col.lower()
                pipeline, issues = classify(row_task, dataset, method_raw, location) if ours else ('baseline', 'record source: rerun under protocol.py or cite the exact paper/table and protocol')
                records.append({
                    'id': len(records) + 1, 'location': location, 'task': row_task, 'dataset': dataset, 'method': method,
                    'method_raw': method_raw, 'column_raw': col, 'value': value if value is not None else raw.strip(),
                    'std': std if std is not None else '', 'ours': ours, 'pipeline_in_repo': pipeline,
                    'traceability': 'untraced: no logs in repo' if ours else 'unverified',
                    'known_issues': issues, 'new_value': '', 'new_run_dir': '',
                })
    fields = list(records[0].keys())
    with open(os.path.join(OUT, 'registry.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    groups = defaultdict(list)
    for r in records:
        groups[(r['task'], r['dataset'], r['method'])].append(r)
    conflicts = []
    for (task, ds, method), rs in sorted(groups.items()):
        vals = sorted({r['value'] for r in rs if isinstance(r['value'], float)})
        mixed = len(vals) > 0 and len(vals) < len({str(r['value']) for r in rs}) and any(not isinstance(r['value'], float) for r in rs)
        if mixed or (len(vals) > 1 and vals[-1] - vals[0] > 0.051):
            conflicts.append({'task': task, 'dataset': ds, 'method': method,
                              'values': ' | '.join(f"{r['value']} ({r['location']})" for r in rs),
                              'spread': round(vals[-1] - vals[0], 3) if len(vals) > 1 else 'OOM vs value'})
    with open(os.path.join(OUT, 'inconsistencies.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['task', 'dataset', 'method', 'spread', 'values'])
        w.writeheader()
        w.writerows(sorted(conflicts, key=lambda c: -c['spread'] if isinstance(c['spread'], float) else 0))
    ours = [r for r in records if r['ours']]
    print(f'{len(records)} numbers ({len(ours)} ours), {len(conflicts)} inconsistent (task, dataset, method) groups')


if __name__ == '__main__':
    main()
