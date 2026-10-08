# Option A resumed with 2 workers and exclusive heavy jobs: phase 1, phase 2, phase 3.
mkdir -p logs
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
# #### JOB ARRAY ###
python -u scripts/jobpool.py jobs/phase1.txt --workers 2 --min_free_gb 6 > logs/jobpool-phase1.out 2>&1
echo "phase 1 done $(date)"
python -u scripts/jobpool.py jobs/phase2.txt --workers 2 --min_free_gb 6 > logs/jobpool-phase2.out 2>&1
echo "phase 2 done $(date)"
python -u scripts/summarize_tuning.py > logs/summary-tuning.out 2>&1
for d in pubmed amazon-photos; do python -u scripts/dose_response.py --dataset $d --method laplacegnn_full --seeds 3 --workers 1 > logs/dose-summary-$d.out 2>&1; done
for d in cora citeseer pubmed amazon-photos; do python -u scripts/run_best.py --dataset $d --methods bgrl,ccassg,graphmae,laplacegnn_full,bgrl_dualfreq_gate,ccassg_dualfreq_gate --seeds 3 --workers 1 --tag clean >> logs/seeds-summary.out 2>&1; done
for d in roman-empire amazon-ratings minesweeper tolokers questions; do python -u scripts/run_best.py --dataset $d --methods bgrl,ccassg,polygcl,laplacegnn_full,bgrl_dualfreq,ccassg_dualfreq,bgrl_dualfreq_gate,ccassg_dualfreq_gate,bgrl_mlp --seeds 3 --workers 1 --tag clean >> logs/seeds-summary.out 2>&1; done
for d in cora citeseer; do for a in random dice prbcd; do for b in 0.05 0.1 0.2; do
    python -u scripts/run_best.py --dataset $d --methods bgrl,ccassg,graphmae,laplacegnn_full --seeds 3 --workers 1 --tag $a-$b --set edge_index_file=data/attacks/$d/$a-$b.pt >> logs/attacks-summary.out 2>&1
done; done; done
python -u scripts/aggregate.py --runs ./runs/tu_replication --out ./results/tu_replication.csv > logs/tu-summary.out 2>&1
python -u scripts/scaling.py > logs/scaling-rerun.out 2> logs/scaling-rerun.err
echo "FULL_FINISHED $(date)"
