# #### JOB ARRAY ###

######################## OGB DATASET ########################
dataset="ogbg-molbbbp"
# dataset="ogbg-molhiv"
# dataset="ogbg-moltox21"
# dataset="ogbg-moltoxcasta"
python -u -m ssl_adv_graph.ogb.run_adv_graph --dataset $dataset --gnn gin --num_layer 3 --m 3 --seed 77 --lr 1e-3 --pp H --emb_dim 128 --hidden_channel 256 --projection_hidden_size 256 --projection_size 512 --prediction_size 512 --epochs 200 --device 0 > run_${dataset}.out 2> run_${dataset}.err 