# Phase-0 gate: the current method with the spectral views actually used, under laplaceGNN/protocol.py,
# next to the random-drop views the original node script trained on.
mkdir -p logs
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
# #### JOB ARRAY ###
for dataset in cora amazon-photos coauthor-cs; do
    for seed in $SEEDS; do
        python -u -m ssl_adv_node.run_adv_node --flagfile=config_node/${dataset}.cfg --model_seed=${seed} --logdir=./runs/gate/${dataset}-spectral > logs/gate-${dataset}-spectral-s${seed}.out 2> logs/gate-${dataset}-spectral-s${seed}.err
        python -u -m ssl_adv_node.run_adv_node --flagfile=config_node/${dataset}.cfg --model_seed=${seed} --view_mode=random --logdir=./runs/gate/${dataset}-random > logs/gate-${dataset}-random-s${seed}.out 2> logs/gate-${dataset}-random-s${seed}.err
    done
done
python scripts/aggregate.py --runs ./runs/gate --out ./results/gate_summary.csv
