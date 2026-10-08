# Light stream: analysis without training, remaining dose-response, attacks, TU replication.
python -u scripts/raw_and_homophily.py > logs/raw_and_homophily.out 2> logs/raw_and_homophily.err
python -u scripts/spectral_mechanism.py > logs/mechanism.out 2> logs/mechanism.err
for d in pubmed amazon-photos; do
    python -u scripts/dose_response.py --dataset $d --method laplacegnn_full --seeds 3 --workers 2 > logs/dose-$d.out 2> logs/dose-$d.err
done
python -u scripts/make_attacks.py --datasets cora,citeseer > logs/make_attacks.out 2> logs/make_attacks.err
for d in cora citeseer; do
    for attack in random dice prbcd; do
        for b in 0.05 0.1 0.2; do
            python -u scripts/run_best.py --dataset $d --methods bgrl,ccassg,graphmae,laplacegnn_full --seeds 3 --workers 2 \
                --tag $attack-$b --set edge_index_file=data/attacks/$d/$attack-$b.pt >> logs/attacks.out 2>> logs/attacks.err
        done
    done
done
for d in MUTAG PROTEINS IMDB-BINARY; do
    for vm in spectral random; do
        for seed in 0 1 2 3 4; do
            python -u -m ssl_adv_graph.tudataset.run_adv_graph --dataset $d --view_mode $vm --seed $seed --lr 1e-3 --epoch 100 \
                --gnn1_num_layers 2 --gnn1_dim 512 --gnn2_num_layers 2 --gnn2_dim 512 --mlp_dim 512 --sv_budget_ratio 0.2 \
                --logdir ./runs/tu_replication/$d-$vm > logs/tu-$d-$vm-$seed.out 2> logs/tu-$d-$vm-$seed.err
        done
    done
done
echo "LIGHT_DONE $(date)"
