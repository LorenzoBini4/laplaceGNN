mkdir -p logs
SEEDS=${SEEDS:-"0 1 2 3 4"}
# #### JOB ARRAY ###
############### GRAPH CLASSIFICATION ###############
dataset=PROTEINS
# dataset=MUTAG
# dataset=NCI1
# dataset=IMDB-BINARY
# dataset=IMDB-MULTI
# dataset=COLLAB
for seed in $SEEDS; do
    python -u -m ssl_adv_graph.tudataset.run_adv_graph --dataset ${dataset} --seed ${seed} --lr 1e-5 --epoch 500 --gnn1_num_layers 2 --gnn1_dim 512 --gnn2_num_layers 2 --gnn2_dim 512 --mlp_dim 512 > logs/run_${dataset}-s${seed}.out 2> logs/run_${dataset}-s${seed}.err
done
python scripts/aggregate.py
############### OTHER TYPE OF NODE CLASSIFICATION ###############
# Planetoid datasets now run through the node pipeline: see config_node/cora.cfg and main_node.sh
