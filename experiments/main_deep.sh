# Deep experiments: scaling (alone), then a heavy GPU stream (tuning pool + seeds) and a light stream in parallel.
mkdir -p logs
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
# #### JOB ARRAY ###
python -u scripts/scaling.py > logs/scaling.out 2> logs/scaling.err
echo "scaling done $(date)"
bash experiments/main_deep_heavy.sh > logs/deep-heavy.log 2>&1 &
bash experiments/main_deep_light.sh > logs/deep-light.log 2>&1 &
wait
echo "DEEP_FINISHED $(date)"
