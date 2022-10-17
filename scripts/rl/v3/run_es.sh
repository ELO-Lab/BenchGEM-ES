#!/bin/bash

for ((i=2016; i<=2025; i++))
do
    python run_rl_es.py \
    --env_name "HalfCheetah-v2" \
    --seed $i \
    --max_ep_len 1000 \
    --num_workers 8 \
    --max_iters 600 \
    --sigma 0.05 \
    --learning_rate 0.003 \
    --pop_size 20 \
    --elite_size 20
done

for ((i=2016; i<=2025; i++))
do
    python run_rl_es.py \
    --env_name "Ant-v2" \
    --seed $i \
    --max_ep_len 1000 \
    --num_workers 8 \
    --max_iters 600 \
    --sigma 0.02 \
    --learning_rate 0.001 \
    --pop_size 100 \
    --elite_size 100
done

for ((i=2016; i<=2025; i++))
do
    python run_rl_es.py \
    --env_name "Swimmer-v2" \
    --seed $i \
    --max_ep_len 1000 \
    --num_workers 8 \
    --max_iters 400 \
    --sigma 0.15 \
    --learning_rate 0.005 \
    --pop_size 10 \
    --elite_size 10
done