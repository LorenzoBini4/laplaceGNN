# Round 5, session 2: generator memory profile (done), paired TU tuning, then clean sparse scaling once session 1 is done.
mkdir -p logs
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
# #### JOB ARRAY ###
[ -f results/generator_memory_profile.csv ] || python -u scripts/profile_generator.py > logs/profile-generator.out 2>&1
for d in MUTAG PROTEINS IMDB-BINARY; do
    python -u scripts/tune_tu.py --dataset $d --trials 12 --seeds 5 > logs/tune-tu-$d.out 2>&1
done
until grep -q S1_FINISHED logs/tmux-round5-s1.log 2>/dev/null; do sleep 120; done
python -u scripts/scaling_sparse.py > logs/scaling-sparse.out 2>&1
echo "S2_FINISHED $(date)"
