#!/bin/bash

for ((i=2016; i<=2025; i++))
do
    python run_rl_sges.py \
    --env_name "HalfCheetah-v2" \
    --seed $i \
    --max_ep_len 1000 \
    --num_workers 8 \
    --max_iters 600 \
    --sigma 0.05 \
    --min_alpha 0.1 \
    --max_alpha 0.8 \
    --alpha_step 1.05 \
    --learning_rate 0.003 \
    --pop_size 20 \
    --elite_size 20 \
    --warm_up 20 \
    --k 10
done

for ((i=2016; i<=2025; i++))
do
    python run_rl_sges.py \
    --env_name "Ant-v2" \
    --seed $i \
    --max_ep_len 1000 \
    --num_workers 8 \
    --max_iters 600 \
    --sigma 0.02 \
    --min_alpha 0.1 \
    --max_alpha 0.8 \
    --alpha_step 1.05 \
    --learning_rate 0.001 \
    --pop_size 100 \
    --elite_size 100 \
    --warm_up 40 \
    --k 20
done

for ((i=2016; i<=2025; i++))
do
    python run_rl_sges.py \
    --env_name "Swimmer-v2" \
    --seed $i \
    --max_ep_len 1000 \
    --num_workers 8 \
    --max_iters 400 \
    --sigma 0.15 \
    --min_alpha 0.1 \
    --max_alpha 0.8 \
    --alpha_step 1.05 \
    --learning_rate 0.005 \
    --pop_size 10 \
    --elite_size 10 \
    --warm_up 2 \
    --k 1
done