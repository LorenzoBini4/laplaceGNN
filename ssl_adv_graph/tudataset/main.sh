# #### JOB ARRAY ###
############### GRAPH CLASSIFICATION ###############
dataset=PROTEINS
python -u run_adv_graph.py --dataset ${dataset} > laplaceGNN_${dataset}.out 2> laplacenGNN_${dataset}.err
############### OTHER TYPE OF NODE CLASSIFICATION ###############
# dataset=Wiki-CS
# python -u run_adv_node.py --dataset ${dataset} > laplaceGNN_${dataset}.out 2> laplaceGNN_${dataset}.err
