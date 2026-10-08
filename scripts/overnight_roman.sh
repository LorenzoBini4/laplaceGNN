#!/bin/bash
# Roman-empire overnight: round-robin equal-budget tuning (2 trials per method per round), stopped by scripts/deadline.sh.
until grep -q DOSE_FINISHED logs/tmux-dose.log 2>/dev/null && [ "$(date +%H%M)" -ge 1231 ]; do sleep 60; done
for k in 2 4 6 8 10; do
    for method in bgrl ccassg laplacegnn_full; do
        python -u scripts/tune_node.py --dataset roman-empire --method ${method} --trials ${k} --workers 2 >> logs/tune-roman-empire-${method}.out 2>> logs/tune-roman-empire-${method}.err
    done
    echo "round of ${k} trials done $(date)"
done
python scripts/summarize_tuning.py
