# Round 5, session 1: GraphSAGE control, robustness attribution, open-gate seeds; then GraphSAGE seeds and summaries.
mkdir -p logs
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
# #### JOB ARRAY ###
python -u scripts/jobpool.py jobs/s1_phase1.txt --workers 2 --min_free_gb 6 > logs/jobpool-s1-phase1.out 2>&1
python -u scripts/jobpool.py jobs/s1_phase2.txt --workers 2 --min_free_gb 6 > logs/jobpool-s1-phase2.out 2>&1
python -u scripts/paired_encoder.py > logs/paired-encoder.out 2>&1
for d in roman-empire amazon-ratings minesweeper tolokers questions; do python -u scripts/run_best.py --dataset $d --methods bgrl_sage --seeds 3 --workers 1 --tag clean >> logs/s1-summary.out 2>&1; done
for d in cora citeseer pubmed amazon-photos; do python -u scripts/run_best.py --dataset $d --methods bgrl_dualfreq,ccassg_dualfreq --seeds 3 --workers 1 --tag clean >> logs/s1-summary.out 2>&1; done
for d in cora citeseer; do for c in clean random-0.2 dice-0.2 prbcd-0.2; do for v in nohid nostruct noadv; do
    s="--set adv_m=1"; [ $v = nostruct ] && s="--set sadv_every=0"; [ $v = noadv ] && s="--set adv_m=1 --set sadv_every=0"
    f=""; [ $c != clean ] && f="--set edge_index_file=data/attacks/$d/$c.pt"
    python -u scripts/run_best.py --dataset $d --methods laplacegnn_full --seeds 3 --workers 1 --tag $c-$v $s $f >> logs/s1-summary.out 2>&1
done; done; done
echo "S1_FINISHED $(date)"
