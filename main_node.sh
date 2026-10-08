mkdir -p logs
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
# #### JOB ARRAY ###
for dataset in coauthor-cs; do
# for dataset in amazon-computers amazon-photos coauthor-cs wiki-cs cora citeseer pubmed; do
# for dataset in coauthor-physics ogbn-arxiv; do   # --view_mode=random only until the sparse spectral backend exists
    for seed in $SEEDS; do
        python -u -m ssl_adv_node.run_adv_node --flagfile=config_node/${dataset}.cfg --model_seed=${seed} > logs/run-${dataset}-s${seed}.out 2> logs/run-${dataset}-s${seed}.err
    done
done
python scripts/aggregate.py

################## TRANSFER LEARNING ##################
# not in the repository yet (run_adv_node_ppi does not exist)
