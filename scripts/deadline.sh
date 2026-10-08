#!/bin/bash
# Usage: deadline.sh "<date expression>" session... — at that time kill the given tmux sessions and any leftover workers.
when=$1; shift
sleep $(( $(date -d "$when" +%s) - $(date +%s) ))
for s in "$@"; do tmux kill-session -t "$s" 2>/dev/null; done
sleep 10
for p in $(ps -eo pid,args | grep -E "scripts/(tune_node|dose_response|ablate|pool|run_best|spectral_mechanism|make_attacks|scaling|raw_and_homophily)\.py|ssl_adv_(node|graph)" | grep -v grep | awk '{print $1}'); do
    kill "$p" 2>/dev/null
done
echo "DEADLINE_REACHED $when $(date)" >> logs/deadline.log
