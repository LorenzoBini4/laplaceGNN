# #### JOB ARRAY ###
######################## OGB DATASET ########################
# dataset="ogbg-molbbbp"
# python -u run_adv_graph.py --dataset $dataset --gnn gin --m 3 --seed 77 --lr 1e-4 --pp H --hidden_channel 512 --projection_hidden_size 512 --projection_size 512 --prediction_size 512 --epochs 100  --device 0 > perturb-${dataset}-lapacian.out 2> perturb-${dataset}-lapacian.err
# dataset="ogbg-molhiv"
# python -u run_adv_graph.py --dataset $dataset --gnn gin --m 3 --seed 77 --lr 1e-4 --pp H --hidden_channel 512 --projection_hidden_size 512 --projection_size 512 --prediction_size 512 --epochs 100 --device 0 > perturb-${dataset}-lapacian.out 2> perturb-${dataset}-lapacian.err
# dataset="ogbg-moltox21"
# python -u run_adv_graph.py --dataset $dataset --gnn gin --m 1 --seed 77 --lr 1e-5 --pp H --hidden_channel 512 --projection_hidden_size 512 --projection_size 512 --prediction_size 512 --epochs 100 --device 0 > perturb-${dataset}-lapacian.out 2> perturb-${dataset}-lapacian.err
dataset="ogbg-moltoxcast"
python -u run_adv_graph.py --dataset $dataset --gnn gin --m 1 --seed 77 --lr 1e-3 --pp H --hidden_channel 512 --projection_hidden_size 512 --projection_size 512 --prediction_size 512 --epochs 100 --device 0 > perturb-${dataset}-lapacian.out 2> perturb-${dataset}-lapacian.err
