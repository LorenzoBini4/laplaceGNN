mkdir -p logs
# #### JOB ARRAY ###
DATASETS='citeseer pubmed cora' METHODS='laplacegnn_full' TRIALS=60 bash experiments/main_queue.sh
DATASETS='amazon-photos' METHODS='laplacegnn_full' TRIALS=30 bash experiments/main_queue.sh
for dataset in citeseer pubmed cora amazon-photos; do
    python -u scripts/ablate.py --dataset ${dataset} --method laplacegnn_full --seeds 3 --workers 3 > logs/ablate-${dataset}.out 2> logs/ablate-${dataset}.err
done
