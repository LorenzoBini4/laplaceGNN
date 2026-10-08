# Multi-dataset queue: equal-budget random search on validation, then optional final seeds (FINAL_SEEDS, default 0).
mkdir -p logs
TRIALS=${TRIALS:-30}
# #### JOB ARRAY ###
DATASETS=${DATASETS:-"cora"}
# DATASETS="cora citeseer pubmed amazon-photos"
for dataset in $DATASETS; do
METHODS=${METHODS:-"bgrl bgrl_adv laplacegnn laplacegnn_noadv laplacegnn_v0"}
for method in $METHODS; do
    python -u scripts/tune_node.py --dataset ${dataset} --method ${method} --trials ${TRIALS} --workers 4 > logs/tune-${dataset}-${method}.out 2> logs/tune-${dataset}-${method}.err
    if [ "${FINAL_SEEDS:-0}" -gt 0 ]; then
        python -u scripts/tune_node.py --dataset ${dataset} --method ${method} --final_seeds ${FINAL_SEEDS} > logs/final-${dataset}-${method}.out 2> logs/final-${dataset}-${method}.err
    fi
done
done
python scripts/summarize_tuning.py
