# Deadline sprint: dose-response on the tuned datasets (dose) and a reduced equal-budget heterophilous study (hetero).
mkdir -p logs
# #### JOB ARRAY ###
if [ "$1" = "dose" ]; then
    for dataset in cora citeseer pubmed amazon-photos; do
        python -u scripts/dose_response.py --dataset ${dataset} --method laplacegnn_full --seeds 3 --workers 3 > logs/dose-${dataset}.out 2> logs/dose-${dataset}.err
    done
else
    for dataset in roman-empire amazon-ratings; do
        for method in bgrl ccassg laplacegnn_full; do
            python -u scripts/tune_node.py --dataset ${dataset} --method ${method} --trials 10 --workers 2 > logs/tune-${dataset}-${method}.out 2> logs/tune-${dataset}-${method}.err
        done
    done
fi
