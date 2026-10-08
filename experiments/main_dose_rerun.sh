mkdir -p logs
for dataset in cora citeseer pubmed amazon-photos; do
    python -u scripts/dose_response.py --dataset ${dataset} --method laplacegnn_full --seeds 3 --workers 3 > logs/dose-${dataset}.out 2> logs/dose-${dataset}.err
done
