# Heavy stream: one interleaved pool of tuning trials, then 3 fresh seeds of every best config.
python -u scripts/pool.py --workers 7 \
    --study minesweeper:bgrl:10 \
    --study minesweeper:bgrl_dualfreq:10 \
    --study minesweeper:bgrl_dualfreq_gate:10 \
    --study minesweeper:bgrl_mlp:10 \
    --study minesweeper:ccassg:10 \
    --study minesweeper:ccassg_dualfreq:10 \
    --study minesweeper:ccassg_dualfreq_gate:10 \
    --study minesweeper:polygcl:10 \
    --study minesweeper:laplacegnn_full:10 \
    --study tolokers:bgrl:10 \
    --study tolokers:bgrl_dualfreq:10 \
    --study tolokers:bgrl_dualfreq_gate:10 \
    --study tolokers:bgrl_mlp:10 \
    --study tolokers:ccassg:10 \
    --study tolokers:ccassg_dualfreq:10 \
    --study tolokers:ccassg_dualfreq_gate:10 \
    --study tolokers:polygcl:10 \
    --study tolokers:laplacegnn_full:10 \
    --study questions:bgrl:10 \
    --study questions:bgrl_dualfreq:10 \
    --study questions:bgrl_dualfreq_gate:10 \
    --study questions:bgrl_mlp:10 \
    --study questions:ccassg:10 \
    --study questions:ccassg_dualfreq:10 \
    --study questions:ccassg_dualfreq_gate:10 \
    --study questions:polygcl:10 \
    --study questions:laplacegnn_full:10 \
    --study roman-empire:bgrl_dualfreq_gate:10 \
    --study roman-empire:ccassg_dualfreq_gate:10 \
    --study roman-empire:bgrl_mlp:10 \
    --study amazon-ratings:bgrl_dualfreq_gate:10 \
    --study amazon-ratings:ccassg_dualfreq_gate:10 \
    --study amazon-ratings:bgrl_mlp:10 \
    --study pubmed:bgrl_dualfreq:20 \
    --study pubmed:ccassg_dualfreq:20 \
    --study pubmed:bgrl_dualfreq_gate:20 \
    --study pubmed:ccassg_dualfreq_gate:20 \
    --study amazon-photos:bgrl_dualfreq:20 \
    --study amazon-photos:ccassg_dualfreq:20 \
    --study amazon-photos:bgrl_dualfreq_gate:20 \
    --study amazon-photos:ccassg_dualfreq_gate:20 \
    --study cora:bgrl_dualfreq_gate:20 \
    --study cora:ccassg_dualfreq_gate:20 \
    --study citeseer:bgrl_dualfreq_gate:20 \
    --study citeseer:ccassg_dualfreq_gate:20 \
    > logs/pool.out 2> logs/pool.err
echo "pool done $(date)"
HOMO="bgrl,ccassg,graphmae,laplacegnn_full,bgrl_dualfreq_gate,ccassg_dualfreq_gate"
HETM="bgrl,ccassg,polygcl,laplacegnn_full,bgrl_dualfreq,ccassg_dualfreq,bgrl_dualfreq_gate,ccassg_dualfreq_gate,bgrl_mlp"
for d in cora citeseer pubmed amazon-photos; do python -u scripts/run_best.py --dataset $d --methods $HOMO --seeds 3 --workers 7 --tag clean >> logs/seeds.out 2>> logs/seeds.err; done
for d in roman-empire amazon-ratings minesweeper tolokers questions; do python -u scripts/run_best.py --dataset $d --methods $HETM --seeds 3 --workers 7 --tag clean >> logs/seeds.out 2>> logs/seeds.err; done
echo "HEAVY_DONE $(date)"
