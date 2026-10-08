"""Best validation trial per (dataset, method) from runs/tune -> results/tuning_summary.csv."""
import csv
import glob
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIAL_CAP = {'amazon-photos': 30, 'roman-empire': 10, 'amazon-ratings': 10, 'minesweeper': 10, 'tolokers': 10, 'questions': 10}


def capped(dataset, trials):
    cap = TRIAL_CAP.get(dataset)
    return [t for t in trials if cap is None or t['trial'] < cap]


def main():
    rows = []
    for study in sorted(glob.glob(os.path.join(ROOT, 'runs', 'tune', '*', '*'))):
        dataset, method = study.split(os.sep)[-2:]
        trials = capped(dataset, [json.loads(l) for f in glob.glob(os.path.join(study, 'trials-shard*.jsonl')) for l in open(f)])
        ok = [t for t in trials if 'error' not in t]
        if not ok:
            continue
        best = max(ok, key=lambda t: t['val'])
        rows.append({'dataset': dataset, 'method': method, 'trials': len(trials), 'errors': len(trials) - len(ok),
                     'best_trial': best['trial'], 'val': round(100 * best['val'], 2), 'test': round(100 * best['test'], 2)})
    snapshot = os.path.join(ROOT, 'results', 'tuning_summary_pre_deletion.csv')
    if os.path.exists(snapshot):
        have = {(r['dataset'], r['method']): r for r in rows}
        for r in csv.DictReader(open(snapshot)):
            key = (r['dataset'], r['method'])
            if key in have and have[key]['trials'] - have[key]['errors'] >= int(r['trials']):
                continue
            if key in have:
                rows.remove(have[key])
            rows.append({k: (int(v) if k in ('trials', 'errors', 'best_trial') else float(v) if k in ('val', 'test') else v)
                         for k, v in r.items()})
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'results', 'tuning_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(list)
    for r in rows:
        by[r['dataset']].append(r)
    for d, rs in by.items():
        print(f'== {d}')
        for r in sorted(rs, key=lambda r: -r['test']):
            print(f"   {r['method']:<18} {r['trials']:>3} trials  val {r['val']:6.2f}  test {r['test']:6.2f}")


if __name__ == '__main__':
    main()
