mkdir -p logs
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
# #### JOB ARRAY ###

######################## OGB DATASET ########################
dataset="ogbg-molbbbp"
# dataset="ogbg-molhiv"
# dataset="ogbg-moltox21"
# dataset="ogbg-moltoxcast"
for seed in $SEEDS; do
    python -u -m ssl_adv_graph.ogb.run_adv_graph --dataset $dataset --gnn gin --num_layer 3 --m 3 --seed ${seed} --lr 1e-3 --pp H --emb_dim 128 --hidden_channel 256 --projection_hidden_size 256 --projection_size 512 --prediction_size 512 --epochs 200 --device 0 > logs/run_${dataset}-s${seed}.out 2> logs/run_${dataset}-s${seed}.err
done
python scripts/aggregate.py
