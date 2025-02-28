# #### JOB ARRAY ###
############### GRAPH CLASSIFICATION ###############
dataset=PROTEINS
python -u -m ssl_adv_graph.tudataset.run_adv_graph --dataset ${dataset} --lr 1e-5 --epoch 500 --gnn1_num_layers 2 --gnn1_dim 512 --gnn2_num_layers 2 --gnn2_dim 512 --mlp_dim 512 > run_${dataset}.out 2> run_${dataset}.err
############### OTHER TYPE OF NODE CLASSIFICATION ###############
# dataset=CiteSeer
# python -u -m ssl_adv_graph.tudataset.run_adv_node --epoch 500 --lapl_epoch 40 --lr 1e-3 --weight_decay 1e-5 --delta 1e-3 --step_size 1e-3 --m 3 --graph_encoder_layer 512 512 --predictor_hidden_size 512 --dataset ${dataset} > run_${dataset}.o 2> run_${dataset}.e
