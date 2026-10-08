# Round 4: heterophilous datasets (tuning + ablation), then accuracy vs spectral change at a fixed budget on every dataset.
mkdir -p logs
# #### JOB ARRAY ###
for dataset in roman-empire amazon-ratings; do
    for method in bgrl ccassg graphmae laplacegnn_full; do
        python -u scripts/tune_node.py --dataset ${dataset} --method ${method} --trials 30 --workers 3 > logs/tune-${dataset}-${method}.out 2> logs/tune-${dataset}-${method}.err
    done
    python -u scripts/ablate.py --dataset ${dataset} --method laplacegnn_full --seeds 3 --workers 3 > logs/ablate-${dataset}.out 2> logs/ablate-${dataset}.err
done
python scripts/summarize_tuning.py
for dataset in cora citeseer pubmed amazon-photos roman-empire amazon-ratings; do
    python -u scripts/dose_response.py --dataset ${dataset} --method laplacegnn_full --seeds 3 --workers 3 > logs/dose-${dataset}.out 2> logs/dose-${dataset}.err
done
