# Final round: GraphMAE on heterophilous graphs, PolyGCL on homophilous graphs, GraphSAGE / MLP controls on homophilous graphs.
mkdir -p logs
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
# #### JOB ARRAY ###
python scripts/make_jobs_final.py
python -u scripts/jobpool.py jobs/final_phase1.txt --workers 3 --min_free_gb 6 > logs/jobpool-final-phase1.out 2>&1
python -u scripts/jobpool.py jobs/final_phase2.txt --workers 3 --min_free_gb 6 > logs/jobpool-final-phase2.out 2>&1
python -u scripts/summarize_tuning.py > logs/final-tuning.out 2>&1
for d in roman-empire amazon-ratings minesweeper tolokers questions; do
    python -u scripts/run_best.py --dataset $d --methods graphmae --seeds 3 --workers 1 --tag clean >> logs/final-summary.out 2>&1
done
for d in cora citeseer pubmed amazon-photos; do
    python -u scripts/run_best.py --dataset $d --methods polygcl,bgrl_sage,bgrl_mlp --seeds 3 --workers 1 --tag clean >> logs/final-summary.out 2>&1
done
python -u scripts/paired_encoder.py > logs/paired-encoder.out 2>&1
python -u scripts/results_markdown.py > logs/final-tables.md 2>&1
echo "FINAL_FINISHED $(date)"
