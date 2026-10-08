#!/bin/bash
# Encoder-fairness and heterophily study: round-robin equal budget, stopped by scripts/deadline.sh.
rr() {
    local dataset=$1 max=$2; shift 2
    for k in $(seq 3 3 $max) $max; do
        for method in "$@"; do
            python -u scripts/tune_node.py --dataset ${dataset} --method ${method} --trials ${k} --workers 3 >> logs/tune-${dataset}-${method}.out 2>> logs/tune-${dataset}-${method}.err
        done
        echo "${dataset}: ${k} trials per method done $(date)"
    done
}
rr roman-empire 10 bgrl_dualfreq ccassg_dualfreq polygcl
rr cora 20 bgrl_dualfreq ccassg_dualfreq
rr citeseer 20 bgrl_dualfreq ccassg_dualfreq
rr amazon-ratings 10 bgrl ccassg laplacegnn_full bgrl_dualfreq ccassg_dualfreq polygcl
python scripts/summarize_tuning.py
