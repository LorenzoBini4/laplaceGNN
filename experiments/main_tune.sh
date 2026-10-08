# Equal-budget random search on validation (same trials and search seed for every method), then 10 seeds of the best config.
mkdir -p logs
TRIALS=${TRIALS:-60}
# #### JOB ARRAY ###
dataset=cora
# dataset=citeseer
# dataset=pubmed
# dataset=amazon-photos
METHODS=${METHODS:-"bgrl bgrl_adv laplacegnn laplacegnn_noadv laplacegnn_v0"}
for method in $METHODS; do
    python -u scripts/tune_node.py --dataset ${dataset} --method ${method} --trials ${TRIALS} --workers 4 > logs/tune-${dataset}-${method}.out 2> logs/tune-${dataset}-${method}.err
    python -u scripts/tune_node.py --dataset ${dataset} --method ${method} --final_seeds 10 > logs/final-${dataset}-${method}.out 2> logs/final-${dataset}-${method}.err
done
python scripts/aggregate.py --runs ./runs/final --out ./results/final_${dataset}.csv
